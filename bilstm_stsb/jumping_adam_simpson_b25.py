"""jumping_adam_simpson_b25.py — JumpingAdamSimpsonB25.

Clean single-file optimizer: Adam + macro-jumps along an EMA-of-updates direction,
jump length from a 3-point (Simpson-type) slope model, fresh-batch undo.

No inheritance from the flaw stack. Subclasses torch.optim.Adam only.
decay_mode="as_b25" reproduces the current Gaps Fixing B25 ordinary-step
and jump path (JumpingAdamP2JumpB + Flaw3Fresh undo semantics).
"""

from __future__ import annotations

import collections
import math
import statistics
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "JumpingAdamSimpsonB25",
    "make_closures",
    "DEFAULT_MOMENTUM_AFTER_JUMP",
    "DEFAULT_MAX_GRAD_NORM",
    "DEFAULT_GATE",
]

# ---------------------------------------------------------------------------
# Options (final default: THREE ON). Each can be switched off:
#   momentum_after_jump="keep" | max_grad_norm=None | gate=None
# stagewise / clip / gn_median must equal b25_options.B25Opt / B25_all.
# ---------------------------------------------------------------------------
DEFAULT_MOMENTUM_AFTER_JUMP = "stagewise"  # "keep" | "stagewise"
DEFAULT_MAX_GRAD_NORM = 1.0  # None = off (1.0 = standard for transformer fine-tuning)
DEFAULT_GATE = "gn_median"  # None (off) | "gn_median" (the near-minima gate)

_GATE_CHOICES = (None, "gn_median")
# Elements per chunk in the momentum refill (keeps the float64 temporaries near 0.4 GB).
_REFILL_CHUNK = 1 << 22

_DEQUE_MAXLEN = {
    "loss_history": 5,
    "_att_pflug_hist": 10,
    "_att_cos_hist": 10,
}


def make_closures(model, loss_fn, next_batch=None, next_fresh_batch=None):
    """Return (closure, closure_forward_only, set_batch).

    Closures NEVER pull a batch. The training loop sets the step batch, then
    every probe / landing / fresh-undo forward on that step uses it — same
    pattern as task_mini_bert_sst2.train_one_trial (one batch closed over per
    outer step; the next step's set_batch is the fresh batch for undo).

    next_batch / next_fresh_batch are unused (kept for call-site compatibility).

    Loop::

        closure, cfo, set_batch = make_closures(model, loss_fn)
        for batch in data_iter:
            set_batch(batch)          # or set_batch(next(it))
            opt.step(closure, closure_forward_only=cfo)
    """
    _ = (next_batch, next_fresh_batch)
    box = {"batch": None}

    def set_batch(b):
        box["batch"] = b

    def closure():
        if box["batch"] is None:
            raise RuntimeError("call set_batch(batch) before opt.step(...)")
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        loss = loss_fn(model, box["batch"])
        loss.backward()
        return loss

    def closure_forward_only():
        if box["batch"] is None:
            raise RuntimeError("call set_batch(batch) before opt.step(...)")
        with torch.no_grad():
            return loss_fn(model, box["batch"])

    return closure, closure_forward_only, set_batch


class JumpingAdamSimpsonB25(torch.optim.Adam):
    """Adam + Simpson-type macro-jumps (EMA direction, 3-point slope model for the length, fresh-batch undo)."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        decay_mode: str = "as_b25",
        total_budget: float = 1000.0,
        jump_every: int = 10,
        window: Tuple[float, float] = (0.05, 0.80),
        N_max: float = 25,
        ema_rho: float = 0.9,
        jb_min_hist: int = 10,
        jb_c: float = 0.1,
        jb_N_min: float = 2.0,
        warmup_steps: int = 15,
        enable_jumps: bool = True,
        fresh_undo: bool = True,
        log_attempts: bool = False,
        max_grad_norm: Optional[float] = DEFAULT_MAX_GRAD_NORM,
        momentum_after_jump: Optional[str] = None,
        gate: Optional[str] = DEFAULT_GATE,
        clip_transfer: bool = True,
        *,
        amsgrad: bool = False,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        if decay_mode not in ("as_b25", "l2", "decoupled"):
            raise ValueError(
                "decay_mode must be 'as_b25', 'l2', or 'decoupled', got %r"
                % (decay_mode,)
            )
        mom = (
            DEFAULT_MOMENTUM_AFTER_JUMP
            if momentum_after_jump is None
            else str(momentum_after_jump)
        )
        if mom not in ("keep", "stagewise"):
            raise ValueError(
                "momentum_after_jump must be 'keep' or 'stagewise', got %r"
                % (mom,)
            )
        if gate not in _GATE_CHOICES:
            raise ValueError("bad gate %r" % (gate,))

        # Parent Adam weight_decay:
        #   l2        → parent sees the real wd (L2-in-gradient)
        #   as_b25    → parent wd=0; exp(-wd·lr) applied manually
        #   decoupled → parent wd=0; p.mul_(1-lr*wd) applied manually (AdamW)
        # Never rely on param-group decoupled_weight_decay.
        if decay_mode == "l2":
            parent_wd = float(weight_decay)
        else:
            parent_wd = 0.0

        raw = list(params)
        if len(raw) == 0:
            raise ValueError("JumpingAdamSimpsonB25 got an empty parameter list")

        if isinstance(raw[0], dict):
            groups_in = []
            for g in raw:
                gg = dict(g)
                logical_wd = float(gg.get("weight_decay", weight_decay))
                gg["b25_weight_decay"] = logical_wd
                if decay_mode == "l2":
                    gg["weight_decay"] = logical_wd
                else:
                    gg["weight_decay"] = 0.0
                groups_in.append(gg)
        else:
            groups_in = [
                {
                    "params": raw,
                    "weight_decay": parent_wd,
                    "b25_weight_decay": float(weight_decay),
                }
            ]

        super().__init__(
            groups_in,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=parent_wd,
            amsgrad=amsgrad,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
        for g in self.param_groups:
            if "b25_weight_decay" not in g:
                g["b25_weight_decay"] = float(weight_decay)
            if decay_mode == "l2":
                g["weight_decay"] = float(g["b25_weight_decay"])
            else:
                g["weight_decay"] = 0.0
            # Force classical Adam path; we implement AdamW ourselves for decoupled.
            g["decoupled_weight_decay"] = False

        self._raise_if_half()

        self.decay_mode = decay_mode
        self.total_budget = float(total_budget)
        self.jump_every = int(jump_every)
        self.window = (float(window[0]), float(window[1]))
        self.N_max = float(N_max)
        self.ema_rho = float(ema_rho)
        self.jb_min_hist = int(jb_min_hist)
        self.jb_c = float(jb_c)
        self.jb_N_min = float(jb_N_min)
        self.warmup_steps = int(warmup_steps)
        self.warmup_remaining = self.warmup_steps
        self.warmup_complete = self.warmup_steps <= 0
        self.enable_jumps = bool(enable_jumps)
        self.fresh_undo = bool(fresh_undo)
        self.log_attempts = bool(log_attempts)
        self.max_grad_norm = (
            None if max_grad_norm is None else float(max_grad_norm)
        )
        self.momentum_after_jump = mom
        self.gate = gate
        self.clip_transfer = bool(clip_transfer)

        # Aliases matching the current stack names (tests / harness).
        self.jb_N_max = self.N_max
        self.jb_every = self.jump_every
        self.jb_rho = self.ema_rho
        self.jb_budget = self.total_budget
        self.jb_window = self.window

        # Legacy P1 time bookkeeping (as_b25 ordinary step advances it).
        b1, b2 = self.param_groups[0]["betas"]
        self.beta1_0 = float(b1)
        self.beta2_0 = float(b2)
        self.tau = 0.0
        self.log_B1 = 0.0
        self.log_B2 = 0.0
        self.h0 = 1.0

        # Counters
        self.compute_calls = 0.0
        self.virtual_step = 0
        self.jump_attempts = 0
        self.jumps_accepted = 0
        self.jumps_failed = 0
        self.skipped_uphill = 0
        self.fu_checked = 0
        self.fu_undone = 0
        self.steps_since_last_jump = 0
        self.loss_history: collections.deque = collections.deque(maxlen=5)
        self._fu_last_N = -1
        self._flaw1_L0 = None

        # EMA of realized ordinary updates
        self._ema_d: Optional[List[torch.Tensor]] = None
        self._ema_count = 0
        self._jb_d = None  # alias updated whenever EMA mutates
        self._jb_count = 0

        # Fresh-undo pending snapshot
        self._fu_pending: Optional[Dict[str, Any]] = None

        # Optional stagewise / clip stats (_fu_ prefix: not restored by undo)
        self._fu_opt_stats = {
            "gate_skipped": 0,
            "momentum_transfers": 0,
            "clip_fired": 0,
            "_clip_norm_sum": 0.0,
            "_clip_norm_count": 0,
        }
        self._opt_g0 = None
        self._opt_g_mid = None
        self._opt_g_end = None


        # Live gate histories (mirror B25Opt: NOT _fu_ → snapshotted/restored).
        # gate=None → allocate nothing.
        if self.gate is not None:
            self._gate_gn_hist = []

        # Passive attempt log (must not affect training path)
        self._att_rows: List[Dict[str, Any]] = []
        self._att_pending_idx: Optional[int] = None
        self._att_step = 0
        self._att_s0_abs_hist: List[float] = []
        self._att_prev_grads = None
        self._att_pflug_hist: collections.deque = collections.deque(maxlen=10)
        self._att_cos_hist: collections.deque = collections.deque(maxlen=10)

    @classmethod
    def plain(cls, params, **kw):
        """Force momentum keep / no clip / no gate (plain B25) for equivalence tests."""
        kw = dict(kw)
        kw["momentum_after_jump"] = "keep"
        kw["max_grad_norm"] = None
        kw["gate"] = None
        return cls(params, **kw)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def memory_report(self) -> Dict[str, int]:
        """Bytes of extra jump state (EMA + pending undo + log rows meta)."""
        ema_bytes = 0
        if self._ema_d is not None:
            for t in self._ema_d:
                ema_bytes += int(t.numel() * t.element_size())
        pend_bytes = 0
        pend = self._fu_pending
        if pend is not None:
            snap = pend.get("snap") or {}
            for t in snap.get("theta") or []:
                if torch.is_tensor(t):
                    pend_bytes += int(t.numel() * t.element_size())
            for st in (snap.get("opt_state") or {}).values():
                for v in st.values():
                    if torch.is_tensor(v):
                        pend_bytes += int(v.numel() * v.element_size())
        log_bytes = 0  # row dicts are Python objects; report count separately
        return {
            "ema_bytes": ema_bytes,
            "pending_undo_bytes": pend_bytes,
            "attempt_log_rows": len(self._att_rows),
            "attempt_log_bytes_est": log_bytes,
            "extra_state_bytes": ema_bytes + pend_bytes,
        }

    @property
    def gate_skipped(self) -> int:
        return int(self._fu_opt_stats["gate_skipped"])

    @property
    def momentum_transfers(self) -> int:
        return int(self._fu_opt_stats["momentum_transfers"])

    @property
    def clip_fired(self) -> int:
        return int(self._fu_opt_stats["clip_fired"])

    @property
    def transfer_clipped(self) -> int:
        return int(self._fu_opt_stats.get("transfer_clipped", 0))

    # ------------------------------------------------------------------
    # Internals: params / dtype / time / decay / clip
    # ------------------------------------------------------------------

    def _raise_if_half(self) -> None:
        for g in self.param_groups:
            for p in g["params"]:
                if p.dtype in (torch.float16, torch.bfloat16):
                    raise RuntimeError(
                        "JumpingAdamSimpsonB25 is fp32-only; got param dtype %s"
                        % (p.dtype,)
                    )

    def _flat_params(self) -> List[torch.nn.Parameter]:
        out: List[torch.nn.Parameter] = []
        for g in self.param_groups:
            out.extend(list(g["params"]))
        return out

    def _params_with_grad(self) -> List[torch.nn.Parameter]:
        return [p for p in self._flat_params() if p.grad is not None]

    def _advance_time(self, r: float) -> None:
        self.tau += float(r)
        self.log_B1 += float(r) * math.log(self.beta1_0)
        self.log_B2 += float(r) * math.log(self.beta2_0)

    def _group_logical_wd(self, group: Dict[str, Any]) -> float:
        return float(group.get("b25_weight_decay", group.get("weight_decay", 0.0)))

    def _apply_as_b25_decay(self, params: Sequence[torch.nn.Parameter]) -> None:
        """Exponential multiplicative decay before Adam (current-stack path)."""
        want = {id(p) for p in params}
        for group in self.param_groups:
            wd = self._group_logical_wd(group)
            if wd <= 0.0:
                continue
            lr = float(group["lr"])
            factor = math.exp(-wd * lr)
            for p in group["params"]:
                if id(p) in want:
                    p.data.mul_(factor)

    def _apply_adamw_decay(self, params: Sequence[torch.nn.Parameter]) -> None:
        """AdamW: p.mul_(1 - lr*wd) per group. Parent Adam wd stays 0."""
        want = {id(p) for p in params}
        for group in self.param_groups:
            wd = self._group_logical_wd(group)
            if wd <= 0.0:
                continue
            lr = float(group["lr"])
            factor = 1.0 - lr * wd
            for p in group["params"]:
                if id(p) in want:
                    p.data.mul_(factor)

    def _clip_adam_grads(self, params: Sequence[torch.nn.Parameter]) -> None:
        """Global L2 clip of Adam-update grads only. Probes never call this."""
        if self.max_grad_norm is None:
            return
        c = float(self.max_grad_norm)
        sq = 0.0
        for p in params:
            if p.grad is None:
                continue
            sq += float(p.grad.detach().double().pow(2).sum().item())
        if not math.isfinite(sq) or sq <= 0.0:
            return
        norm = math.sqrt(sq)
        st = self._fu_opt_stats
        st["_clip_norm_sum"] += float(norm)
        st["_clip_norm_count"] += 1
        coef = min(1.0, c / (norm + 1e-6))
        if coef < 1.0:
            st["clip_fired"] += 1
            for p in params:
                if p.grad is not None:
                    p.grad.mul_(coef)

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------

    def _ema_record(self, before: Sequence[torch.Tensor]) -> None:
        params = self._flat_params()
        with torch.no_grad():
            if self._ema_d is None:
                self._ema_d = [torch.zeros_like(p) for p in params]
            for i, (p, b) in enumerate(zip(params, before)):
                u = p.detach() - b
                self._ema_d[i].mul_(self.ema_rho).add_(u, alpha=1.0 - self.ema_rho)
        self._ema_count += 1
        self._jb_d = self._ema_d
        self._jb_count = self._ema_count

    # ------------------------------------------------------------------
    # Ordinary Adam step (warmup / fallback)
    # ------------------------------------------------------------------

    def _run_adam_update(self, params: Sequence[torch.nn.Parameter]) -> None:
        if self.decay_mode == "as_b25":
            self._apply_as_b25_decay(params)
        elif self.decay_mode == "decoupled":
            self._apply_adamw_decay(params)
        # l2: parent Adam applies weight_decay into the grad (group wd set).
        self._advance_time(1.0)
        self._clip_adam_grads(params)
        torch.optim.Adam.step(self)
        self.virtual_step += 1

    def _warmup_step(self, closure: Callable) -> Any:
        before = [p.detach().clone() for p in self._flat_params()]
        with torch.enable_grad():
            loss = closure()
        self.compute_calls += 1.0
        if not torch.isfinite(loss):
            self._ema_record(before)
            return loss
        self.loss_history.append(float(loss.item()))
        params = self._params_with_grad()
        if params:
            self._run_adam_update(params)
        else:
            self.virtual_step += 1
        self.warmup_remaining -= 1
        if self.warmup_remaining <= 0:
            self.warmup_complete = True
        self._ema_record(before)
        return loss

    def _fallback_adam_step(
        self,
        params: Sequence[torch.nn.Parameter],
        grads_before: Sequence[torch.Tensor],
    ) -> None:
        before = [p.detach().clone() for p in self._flat_params()]
        for i, p in enumerate(params):
            p.grad.copy_(grads_before[i])
        self._run_adam_update(params)
        self._ema_record(before)

    # ------------------------------------------------------------------
    # Jump eligibility / geometry
    # ------------------------------------------------------------------

    def _can_jump(self) -> bool:
        if not self.enable_jumps or self.N_max < self.jb_N_min:
            return False
        if self._ema_d is None or self._ema_count < self.jb_min_hist:
            return False
        if self.steps_since_last_jump < self.jump_every:
            return False
        if self.total_budget <= 0.0:
            return False
        frac = float(self.compute_calls) / float(self.total_budget)
        return self.window[0] <= frac <= self.window[1]

    def _jb_slope(
        self, plist: Sequence[torch.nn.Parameter], d: Sequence[torch.Tensor]
    ) -> float:
        total = 0.0
        for p, di in zip(plist, d):
            if p.grad is None:
                continue
            total += float((p.grad.double() * di.double()).sum().item())
        return total

    def _jb_place(
        self,
        plist: Sequence[torch.nn.Parameter],
        theta0: Sequence[torch.Tensor],
        d: Sequence[torch.Tensor],
        n: float,
    ) -> None:
        with torch.no_grad():
            for p, t, di in zip(plist, theta0, d):
                p.data.copy_(t).add_(di, alpha=n)

    def _jb_nstar(self, a: float, b: float, c: float, n_max: float) -> float:
        hi = float(n_max)
        if abs(c) < 1e-30:
            if abs(b) < 1e-30:
                return hi
            n = -a / b
            if math.isfinite(n) and 0.0 < n <= hi:
                return n
            return hi
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return hi
        sqrt_d = math.sqrt(disc)
        roots = []
        for n in ((-b - sqrt_d) / (2.0 * c), (-b + sqrt_d) / (2.0 * c)):
            if math.isfinite(n) and 0.0 < n <= hi:
                roots.append(n)
        if not roots:
            return hi
        return min(roots)

    def _jb_dL(self, a: float, b: float, c: float, n: float) -> float:
        return a * n + b * n * n / 2.0 + c * n ** 3 / 3.0

    def _clip_list(self, grads) -> bool:
        """Scale a list of gradient tensors IN PLACE so their global L2 norm is at most
        max_grad_norm. Same rule as _clip_adam_grads. Returns True if it scaled them."""
        if self.max_grad_norm is None:
            return False
        c = float(self.max_grad_norm)
        sq = 0.0
        for g in grads:
            sq += float(g.detach().double().pow(2).sum().item())
        if not math.isfinite(sq) or sq <= 0.0:
            return False
        coef = min(1.0, c / (math.sqrt(sq) + 1e-6))
        if coef < 1.0:
            for g in grads:
                g.mul_(coef)
            return True
        return False

    def _transfer_after_jump(self, n_int, g0, g_mid, g_end) -> None:
        """Momentum transfer after an accepted jump. With clip_transfer on, each of the
        three probe gradients first gets the same global-norm clip as a real Adam step."""
        if self.clip_transfer and self.max_grad_norm is not None:
            fired = 0
            for gl in (g0, g_mid, g_end):
                fired += int(self._clip_list(gl))
            if fired:
                st = self._fu_opt_stats
                st["transfer_clipped"] = st.get("transfer_clipped", 0) + fired
        self._stagewise_transfer(n_int, g0, g_mid, g_end)

    # ------------------------------------------------------------------
    # Stagewise momentum (equals b25_options.B25Opt when "stagewise")
    # ------------------------------------------------------------------

    def _stagewise_transfer(
        self,
        n_int: int,
        g0: Sequence[torch.Tensor],
        g_mid: Sequence[torch.Tensor],
        g_end: Sequence[torch.Tensor],
    ) -> None:
        n = int(n_int)
        if n <= 0:
            return
        if n > 50:
            n = 50
        n_max = float(self.N_max)
        h = n_max / 2.0
        # Map each param → its own group's betas.
        p_to_betas = {}
        for group in self.param_groups:
            b1g, b2g = group["betas"]
            for p in group["params"]:
                p_to_betas[id(p)] = (float(b1g), float(b2g))
        params = self._flat_params()
        chunk = max(1, int(_REFILL_CHUNK))
        for i, p in enumerate(params):
            b1, b2 = p_to_betas[id(p)]
            b1_n = b1 ** n
            b2_n = b2 ** n
            state = self.state[p]
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)
            m = state["exp_avg"]
            v = state["exp_avg_sq"]
            # Work on flat views, one chunk at a time. Every operation is elementwise, so the
            # result is bit-identical to doing the whole tensor at once, but the float64
            # temporaries stay small (this matters for the huge embedding matrices of
            # multilingual models).
            m_f = m.view(-1) if m.is_contiguous() else m.reshape(-1)
            v_f = v.view(-1) if v.is_contiguous() else v.reshape(-1)
            g0f = g0[i].detach().reshape(-1)
            gmf = g_mid[i].detach().reshape(-1)
            gef = g_end[i].detach().reshape(-1)
            total = m_f.numel()
            for s in range(0, total, chunk):
                e = min(total, s + chunk)
                a0 = g0f[s:e].double()
                am = gmf[s:e].double()
                ae = gef[s:e].double()
                c = (ae - 2.0 * am + a0) / (2.0 * h * h)
                b = (am - a0) / h - c * h
                sum_m = torch.zeros_like(a0)
                sum_v = torch.zeros_like(a0)
                for k in range(1, n + 1):
                    t = float(k)
                    gk = a0 + b * t + c * (t * t)
                    w1 = b1 ** (n - k)
                    w2 = b2 ** (n - k)
                    sum_m = sum_m + w1 * gk
                    sum_v = sum_v + w2 * (gk * gk)
                m_new = b1_n * m_f[s:e].double() + (1.0 - b1) * sum_m
                v_new = b2_n * v_f[s:e].double() + (1.0 - b2) * sum_v
                m_f[s:e].copy_(m_new.to(dtype=m_f.dtype))
                v_f[s:e].copy_(v_new.to(dtype=v_f.dtype))
            if m_f.data_ptr() != m.data_ptr():  # only for odd memory layouts (reshape made a copy)
                m.copy_(m_f.view_as(m))
            if v_f.data_ptr() != v.data_ptr():
                v.copy_(v_f.view_as(v))
            if "step" in state:
                step = state["step"]
                if torch.is_tensor(step):
                    step.add_(n)
                else:
                    state["step"] = int(step) + n
            else:
                state["step"] = n
        self._advance_time(float(n))
        self._fu_opt_stats["momentum_transfers"] += 1

    # ------------------------------------------------------------------
    # Fresh-batch undo (Flaw3Fresh semantics)
    # ------------------------------------------------------------------

    _FU_SKIP_EXACT = frozenset(
        {
            "fresh_undo",
            "param_groups",
            "state",
            "defaults",
            "compute_calls",
            "jump_attempts",
            "_att_rows",
            "_att_pending_idx",
            "_att_step",
            "_att_s0_abs_hist",
            "_att_prev_grads",
            "_att_pflug_hist",
            "_att_cos_hist",
            "log_attempts",
            "_jb_d",  # alias of _ema_d: saved once (as _ema_d); re-linked at the end of _fu_restore
        }
    )

    def _fu_copy_attr(self, value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().clone()
        if value is None or isinstance(value, (int, float, bool, str)):
            return value
        if isinstance(value, collections.deque):
            if all(isinstance(x, (int, float, bool)) for x in value):
                return collections.deque(value, maxlen=value.maxlen)
            return None
        if isinstance(value, (list, tuple)):
            if all((v is None) or torch.is_tensor(v) for v in value):
                cloned = [
                    v.detach().clone() if v is not None else None for v in value
                ]
                return type(value)(cloned)
            if all(isinstance(v, (int, float, bool)) for v in value):
                return type(value)(value)
            return None
        if isinstance(value, dict):
            if all(
                isinstance(v, (int, float, bool, str, type(None)))
                for v in value.values()
            ):
                return dict(value)
            return None
        return None

    def _fu_copy_state_value(self, value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().clone()
        if isinstance(value, (int, float, bool, str, type(None))):
            return value
        if isinstance(value, dict) and all(
            isinstance(v, (int, float, bool, str, type(None)))
            for v in value.values()
        ):
            return dict(value)
        return value

    def _fu_snapshot(self, params: Sequence[torch.nn.Parameter]) -> Optional[Dict]:
        try:
            opt_state = {}
            for i, p in enumerate(params):
                if p in self.state:
                    opt_state[i] = {
                        k: self._fu_copy_state_value(v)
                        for k, v in self.state[p].items()
                    }
            attrs = {}
            for name, value in vars(self).items():
                if name.startswith("_fu_") or name.startswith("fu_"):
                    continue
                if name in self._FU_SKIP_EXACT:
                    continue
                copied = self._fu_copy_attr(value)
                if copied is None and value is not None:
                    continue
                attrs[name] = copied
            return {
                "theta": [p.detach().clone() for p in params],
                "opt_state": opt_state,
                "attrs": attrs,
            }
        except Exception:
            return None

    def _fu_restore(self, snap: Dict, params: Sequence[torch.nn.Parameter]) -> None:
        with torch.no_grad():
            for p, t in zip(params, snap["theta"]):
                p.data.copy_(t)
        for i, saved_state in snap["opt_state"].items():
            idx = int(i)
            if idx >= len(params):
                continue
            p = params[idx]
            if p not in self.state:
                self.state[p] = {}
            st = self.state[p]
            for k, saved in saved_state.items():
                if torch.is_tensor(saved):
                    cur = st.get(k)
                    if (
                        torch.is_tensor(cur)
                        and tuple(cur.shape) == tuple(saved.shape)
                    ):
                        cur.copy_(saved)
                    else:
                        st[k] = saved.detach().clone()
                else:
                    st[k] = saved
        for name, value in snap["attrs"].items():
            if name in _DEQUE_MAXLEN and not isinstance(
                value, collections.deque
            ):
                value = collections.deque(
                    value or [], maxlen=_DEQUE_MAXLEN[name]
                )
            setattr(self, name, value)
        # Keep EMA aliases coherent after attr restore.
        self._jb_d = self._ema_d
        self._jb_count = self._ema_count

    def _maybe_fresh_undo(self, closure_forward_only: Optional[Callable]) -> None:
        pend = self._fu_pending
        self._fu_pending = None
        if pend is None or not self.fresh_undo:
            return
        if closure_forward_only is None:
            return
        snap = pend["snap"]
        params = self._flat_params()
        # Snap was taken on the flat param list at accept time.
        with torch.no_grad():
            out_new = closure_forward_only()
            L_new = float(
                out_new.item() if torch.is_tensor(out_new) else out_new
            )
            landing_weights = [p.detach().clone() for p in params]
            for p, t in zip(params, snap["theta"]):
                p.data.copy_(t)
            out_pre = closure_forward_only()
            L_pre = float(
                out_pre.item() if torch.is_tensor(out_pre) else out_pre
            )
            for p, t in zip(params, landing_weights):
                p.data.copy_(t)
        self.compute_calls += 0.6
        self.fu_checked += 1
        keep = (
            math.isfinite(L_new)
            and math.isfinite(L_pre)
            and (L_new <= L_pre)
        )
        if not keep:
            self._fu_restore(snap, params)
            self.fu_undone += 1
            self.steps_since_last_jump = 0
            if pend["N"] > 0:
                # Mirror Flaw3Fresh next_jump_size floor behaviour (unused by B25
                # length selection, but kept for state parity).
                pass
        if self._att_pending_idx is not None and self.log_attempts:
            self._att_rows[self._att_pending_idx]["undone"] = (not keep)
            self._att_pending_idx = None

    # ------------------------------------------------------------------
    # Jump attempt
    # ------------------------------------------------------------------

    def _params_grad_norm(self, params: Sequence[torch.nn.Parameter]) -> Optional[float]:
        sq = 0.0
        any_g = False
        for p in params:
            if p.grad is None:
                continue
            any_g = True
            sq += float(p.grad.detach().double().pow(2).sum().item())
        if not any_g:
            return None
        if not math.isfinite(sq) or sq <= 0.0:
            return 0.0
        return math.sqrt(sq)

    def _att_mean10(self, hist) -> Optional[float]:
        if hist is None or len(hist) < 10:
            return None
        return float(sum(hist) / 10.0)

    def _att_grad_clones(self):
        params = self._flat_params()
        if not params or any(p.grad is None for p in params):
            return None
        return [p.grad.detach().clone() for p in params]

    def _att_update_pflug(self) -> None:
        """Passive consecutive-grad dot/cos (b25_attempt_log._att_update_pflug)."""
        if not self.log_attempts:
            return
        curr = self._att_grad_clones()
        if curr is None:
            return
        prev = self._att_prev_grads
        if prev is not None and len(prev) == len(curr):
            dot = 0.0
            na = 0.0
            nb = 0.0
            for x, y in zip(prev, curr):
                xd = x.detach().double().reshape(-1)
                yd = y.detach().double().reshape(-1)
                dot += float((xd * yd).sum().item())
                na += float((xd * xd).sum().item())
                nb += float((yd * yd).sum().item())
            if math.isfinite(dot):
                self._att_pflug_hist.append(float(dot))
                if na > 0.0 and nb > 0.0:
                    cos = float(dot / (math.sqrt(na) * math.sqrt(nb)))
                    if math.isfinite(cos):
                        self._att_cos_hist.append(cos)
        self._att_prev_grads = curr

    def _att_losses3(self):
        hist = list(self.loss_history)
        if len(hist) >= 3:
            a, b, c = float(hist[-3]), float(hist[-2]), float(hist[-1])
            return [a, b, c]
        pad = [None, None, None]
        recent = hist[-3:]
        for i, v in enumerate(recent):
            pad[3 - len(recent) + i] = float(v)
        return pad

    def _exp_avg_norm(self) -> Optional[float]:
        sq = 0.0
        any_m = False
        for p in self._flat_params():
            st = self.state.get(p)
            if not st or "exp_avg" not in st:
                continue
            any_m = True
            sq += float(st["exp_avg"].detach().double().pow(2).sum().item())
        if not any_m:
            return None
        if not math.isfinite(sq) or sq <= 0.0:
            return 0.0
        return math.sqrt(sq)

    def _log_attempt(
        self,
        outcome: str,
        s0: float,
        grad_norm: Optional[float],
        compute_spent: float,
        handled: bool,
    ) -> None:
        losses3 = self._att_losses3()
        row = {
            "step_index": int(self._att_step),
            "compute_calls": float(self.compute_calls) - float(compute_spent),
            "steps_since_last_jump": int(self.steps_since_last_jump),
            "attempts_so_far": int(self.jump_attempts) - 1,
            "s0": float(s0),
            "grad_norm": None if grad_norm is None else float(grad_norm),
            "exp_avg_norm": self._exp_avg_norm(),
            "loss_m2": losses3[0],
            "loss_m1": losses3[1],
            "loss_m0": losses3[2],
            "pflug_mean10": self._att_mean10(self._att_pflug_hist),
            "cos_mean10": self._att_mean10(self._att_cos_hist),
            "outcome": outcome,
            "compute_spent": float(compute_spent),
            "undone": None,
            "handled": bool(handled),
        }
        self._att_rows.append(row)
        self._att_s0_abs_hist.append(
            float(abs(s0)) if math.isfinite(s0) else float("nan")
        )
        if outcome == "accepted":
            self._att_pending_idx = len(self._att_rows) - 1

    # ------------------------------------------------------------------
    # step / state_dict
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None, closure_forward_only=None):
        if closure is None:
            raise RuntimeError("JumpingAdamSimpsonB25 requires a closure.")
        self._raise_if_half()
        if self.log_attempts:
            self._att_step += 1

        try:
            self._maybe_fresh_undo(closure_forward_only)

            if not self.warmup_complete:
                return self._warmup_step(closure)

            with torch.enable_grad():
                loss = closure()
            self.compute_calls += 1.0
            if not torch.isfinite(loss):
                return loss

            self.loss_history.append(float(loss.item()))
            params = self._params_with_grad()
            if not params:
                torch.optim.Adam.step(self)
                self.virtual_step += 1
                return loss

            grads_before = [p.grad.clone() for p in params]
            self.steps_since_last_jump += 1
            if self._can_jump():
                handled, jump_loss = self._attempt_macro_jump(
                    params, closure, closure_forward_only
                )
                if handled:
                    return jump_loss

            self._fallback_adam_step(params, grads_before)
            return loss
        finally:
            # Passive: update after the step so logging never changes the path.
            if self.log_attempts:
                self._att_update_pflug()


    # ------------------------------------------------------------------
    # Live gate: gn_median only
    # ------------------------------------------------------------------

    def _gate_should_skip(self):
        """The near-minima gate (gn_median), the only gate in this code.

        Skip a jump attempt when the current gradient norm is below 0.9 x the
        median of the previous 10 attempt-time norms (needs >= 10 values).
        The history includes the attempts the gate itself skipped.
        """
        if self.gate is None:
            return False
        gn = self._params_grad_norm(self._flat_params())
        hist = self._gate_gn_hist
        fire = False
        if gn is not None and math.isfinite(gn):
            if len(hist) >= 10:
                med = float(statistics.median(hist[-10:]))
                if math.isfinite(med) and gn < 0.9 * med:
                    fire = True
            hist.append(float(gn))
        return bool(fire)

    def _attempt_macro_jump(
        self,
        params: Sequence[torch.nn.Parameter],
        closure: Callable,
        closure_forward_only: Optional[Callable],
    ) -> Tuple[bool, Any]:
        # LIVE gate at top, before any probe: free-uphill skip counters/no bill.
        if self.gate is not None and self._gate_should_skip():
            self.jump_attempts += 1
            self.steps_since_last_jump = 0
            self._fu_opt_stats["gate_skipped"] += 1
            return (False, None)

        self.jump_attempts += 1
        self.steps_since_last_jump = 0
        plist = self._flat_params()
        d = self._ema_d
        assert d is not None
        L0 = float(self.loss_history[-1])
        self._flaw1_L0 = L0
        s0 = self._jb_slope(plist, d)
        grad_norm = self._params_grad_norm(plist) if self.log_attempts else None
        cc0 = float(self.compute_calls)

        if (not math.isfinite(s0)) or s0 >= 0.0:
            self.skipped_uphill += 1
            if self.log_attempts:
                self._log_attempt(
                    "skip_uphill" if math.isfinite(s0) else "nonfinite",
                    s0,
                    grad_norm,
                    0.0,
                    handled=False,
                )
            return (False, None)

        capture = self.momentum_after_jump == "stagewise"
        g0 = g_mid = g_end = None
        if capture:
            g0 = [
                p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                for p in plist
            ]

        N = float(self.N_max)
        h = N / 2.0
        theta0 = [p.detach().clone() for p in plist]

        # Snapshot for fresh undo (pre-jump)
        snap = self._fu_snapshot(plist) if self.fresh_undo else None
        acc0 = self.jumps_accepted

        def place(n: float) -> None:
            self._jb_place(plist, theta0, d, n)

        place(h)
        with torch.enable_grad():
            loss_mid = closure()
        self.compute_calls += 1.0
        if capture:
            g_mid = [
                p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                for p in plist
            ]
        s_mid = self._jb_slope(plist, d)
        place(N)
        with torch.enable_grad():
            loss_end = closure()
        self.compute_calls += 1.0
        if capture:
            g_end = [
                p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                for p in plist
            ]
        s_end = self._jb_slope(plist, d)
        _ = loss_mid
        _ = loss_end

        a = s0
        c = (s_end - 2.0 * s_mid + s0) / (2.0 * h * h)
        b = (s_mid - s0) / h - c * h
        n_star = self._jb_nstar(a, b, c, N)
        pred = -self._jb_dL(a, b, c, n_star)

        if (
            (not math.isfinite(pred))
            or pred <= 0.0
            or n_star < self.jb_N_min
        ):
            place(0)
            self.jumps_failed += 1
            if self.log_attempts:
                self._log_attempt(
                    "stage_reject",
                    s0,
                    grad_norm,
                    float(self.compute_calls) - cc0,
                    handled=False,
                )
            return (False, None)

        place(n_star)
        if closure_forward_only is not None:
            with torch.no_grad():
                L_land = closure_forward_only()
            self.compute_calls += 0.3
        else:
            with torch.enable_grad():
                L_land = closure()
            self.compute_calls += 1.0
        L_land_f = float(L_land.item() if torch.is_tensor(L_land) else L_land)

        if math.isfinite(L_land_f) and L_land_f <= L0 - self.jb_c * pred:
            n_int = int(round(n_star))
            self.virtual_step += n_int
            self.jumps_accepted += 1
            self._fu_last_N = n_int
            if (
                capture
                and g0 is not None
                and g_mid is not None
                and g_end is not None
            ):
                self._transfer_after_jump(n_int, g0, g_mid, g_end)
            if self.fresh_undo and snap is not None and self.jumps_accepted > acc0:
                landing = L_land
                self._fu_pending = {
                    "snap": snap,
                    "N": n_int,
                    "own_L0": float(L0),
                    "own_landing": float(
                        landing.item() if torch.is_tensor(landing) else landing
                    ),
                }
            if self.log_attempts:
                self._log_attempt(
                    "accepted",
                    s0,
                    grad_norm,
                    float(self.compute_calls) - cc0,
                    handled=True,
                )
            if torch.is_tensor(L_land):
                return (True, L_land)
            return (True, torch.tensor(L_land_f))

        place(0)
        self.jumps_failed += 1
        if self.log_attempts:
            self._log_attempt(
                "landing_reject",
                s0,
                grad_norm,
                float(self.compute_calls) - cc0,
                handled=False,
            )
        return (False, None)

    def state_dict(self):
        sd = super().state_dict()
        ema = None
        if self._ema_d is not None:
            ema = [t.detach().cpu().clone() for t in self._ema_d]
        pending = None
        if self._fu_pending is not None:
            snap = self._fu_pending["snap"]
            pending = {
                "N": int(self._fu_pending["N"]),
                "own_L0": float(self._fu_pending["own_L0"]),
                "own_landing": float(self._fu_pending["own_landing"]),
                "snap": {
                    "theta": [t.detach().cpu().clone() for t in snap["theta"]],
                    "opt_state": {
                        str(k): {
                            sk: (
                                sv.detach().cpu().clone()
                                if torch.is_tensor(sv)
                                else sv
                            )
                            for sk, sv in st.items()
                        }
                        for k, st in snap["opt_state"].items()
                    },
                    "attrs": {
                        k: (
                            [
                                x.detach().cpu().clone() if torch.is_tensor(x) else x
                                for x in v
                            ]
                            if isinstance(v, list)
                            and v
                            and any(torch.is_tensor(x) for x in v if x is not None)
                            else (
                                list(v)
                                if isinstance(v, collections.deque)
                                else v
                            )
                        )
                        for k, v in snap["attrs"].items()
                        if k not in ("_ema_d", "_jb_d")
                    },
                    "ema": ema,
                },
            }
        b25 = {
            "decay_mode": self.decay_mode,
            "total_budget": self.total_budget,
            "jump_every": self.jump_every,
            "window": self.window,
            "N_max": self.N_max,
            "ema_rho": self.ema_rho,
            "jb_min_hist": self.jb_min_hist,
            "jb_c": self.jb_c,
            "jb_N_min": self.jb_N_min,
            "warmup_steps": self.warmup_steps,
            "warmup_remaining": self.warmup_remaining,
            "warmup_complete": self.warmup_complete,
            "enable_jumps": self.enable_jumps,
            "fresh_undo": self.fresh_undo,
            "log_attempts": self.log_attempts,
            "max_grad_norm": self.max_grad_norm,
            "momentum_after_jump": self.momentum_after_jump,
            "gate": self.gate,
            "clip_transfer": self.clip_transfer,
            "_gate_gn_hist": list(getattr(self, "_gate_gn_hist", []) or []),
            "tau": self.tau,
            "log_B1": self.log_B1,
            "log_B2": self.log_B2,
            "compute_calls": self.compute_calls,
            "virtual_step": self.virtual_step,
            "jump_attempts": self.jump_attempts,
            "jumps_accepted": self.jumps_accepted,
            "jumps_failed": self.jumps_failed,
            "skipped_uphill": self.skipped_uphill,
            "fu_checked": self.fu_checked,
            "fu_undone": self.fu_undone,
            "steps_since_last_jump": self.steps_since_last_jump,
            "loss_history": list(self.loss_history),
            "loss_history_maxlen": self.loss_history.maxlen,
            "_fu_last_N": self._fu_last_N,
            "_flaw1_L0": self._flaw1_L0,
            "_ema_count": self._ema_count,
            "ema": ema,
            "pending": pending,
            "_fu_opt_stats": dict(self._fu_opt_stats),
            "_att_rows": list(self._att_rows) if self.log_attempts else [],
            "_att_pending_idx": self._att_pending_idx,
            "_att_step": self._att_step,
            "_att_s0_abs_hist": list(self._att_s0_abs_hist),
            "b25_weight_decay_per_group": [
                g.get("b25_weight_decay", 0.0) for g in self.param_groups
            ],
        }
        sd["b25"] = b25
        return sd

    def load_state_dict(self, state_dict):
        sd = dict(state_dict)
        b25 = sd.pop("b25", None)
        super().load_state_dict(sd)
        if b25 is None:
            return
        self.decay_mode = b25["decay_mode"]
        self.total_budget = float(b25["total_budget"])
        self.jb_budget = self.total_budget
        self.jump_every = int(b25["jump_every"])
        self.jb_every = self.jump_every
        self.window = tuple(b25["window"])
        self.jb_window = self.window
        self.N_max = float(b25["N_max"])
        self.jb_N_max = self.N_max
        self.ema_rho = float(b25["ema_rho"])
        self.jb_rho = self.ema_rho
        self.jb_min_hist = int(b25["jb_min_hist"])
        self.jb_c = float(b25["jb_c"])
        self.jb_N_min = float(b25["jb_N_min"])
        self.warmup_steps = int(b25["warmup_steps"])
        self.warmup_remaining = int(b25["warmup_remaining"])
        self.warmup_complete = bool(b25["warmup_complete"])
        self.enable_jumps = bool(b25["enable_jumps"])
        self.fresh_undo = bool(b25["fresh_undo"])
        self.log_attempts = bool(b25["log_attempts"])
        self.max_grad_norm = b25["max_grad_norm"]
        self.momentum_after_jump = b25["momentum_after_jump"]
        self.clip_transfer = bool(b25.get("clip_transfer", True))
        self.gate = b25.get("gate", None)
        if self.gate is not None:
            self._gate_gn_hist = list(b25.get("_gate_gn_hist") or [])
        elif hasattr(self, "_gate_gn_hist"):
            delattr(self, "_gate_gn_hist")
        self.tau = float(b25["tau"])
        self.log_B1 = float(b25["log_B1"])
        self.log_B2 = float(b25["log_B2"])
        self.compute_calls = float(b25["compute_calls"])
        self.virtual_step = int(b25["virtual_step"])
        self.jump_attempts = int(b25["jump_attempts"])
        self.jumps_accepted = int(b25["jumps_accepted"])
        self.jumps_failed = int(b25["jumps_failed"])
        self.skipped_uphill = int(b25["skipped_uphill"])
        self.fu_checked = int(b25["fu_checked"])
        self.fu_undone = int(b25["fu_undone"])
        self.steps_since_last_jump = int(b25["steps_since_last_jump"])
        ml = b25.get("loss_history_maxlen", 5)
        self.loss_history = collections.deque(b25["loss_history"], maxlen=ml)
        self._fu_last_N = int(b25["_fu_last_N"])
        self._flaw1_L0 = b25["_flaw1_L0"]
        self._ema_count = int(b25["_ema_count"])
        self._jb_count = self._ema_count
        ema = b25.get("ema")
        if ema is None:
            self._ema_d = None
        else:
            params = self._flat_params()
            self._ema_d = [
                t.to(device=p.device, dtype=p.dtype) for t, p in zip(ema, params)
            ]
        self._jb_d = self._ema_d
        self._fu_opt_stats = dict(b25.get("_fu_opt_stats") or {})
        self._att_rows = list(b25.get("_att_rows") or [])
        self._att_pending_idx = b25.get("_att_pending_idx")
        self._att_step = int(b25.get("_att_step") or 0)
        self._att_s0_abs_hist = list(b25.get("_att_s0_abs_hist") or [])
        for g, wd in zip(
            self.param_groups, b25.get("b25_weight_decay_per_group") or []
        ):
            g["b25_weight_decay"] = float(wd)

        pending = b25.get("pending")
        if pending is None:
            self._fu_pending = None
        else:
            params = self._flat_params()
            snap_in = pending["snap"]
            theta = [
                t.to(device=p.device, dtype=p.dtype)
                for t, p in zip(snap_in["theta"], params)
            ]
            opt_state = {}
            for k, st in snap_in["opt_state"].items():
                opt_state[int(k)] = {
                    sk: (
                        sv.to(device=params[0].device)
                        if torch.is_tensor(sv)
                        else sv
                    )
                    for sk, sv in st.items()
                }
            attrs = dict(snap_in.get("attrs") or {})
            # Restore deque attrs as deques (loss_history / att hists).
            for dk, ml in _DEQUE_MAXLEN.items():
                if dk in attrs and not isinstance(attrs[dk], collections.deque):
                    attrs[dk] = collections.deque(attrs[dk] or [], maxlen=ml)
            if snap_in.get("ema") is not None:
                attrs["_ema_d"] = [
                    t.to(device=p.device, dtype=p.dtype)
                    for t, p in zip(snap_in["ema"], params)
                ]
                attrs["_jb_d"] = attrs["_ema_d"]
            self._fu_pending = {
                "snap": {
                    "theta": theta,
                    "opt_state": opt_state,
                    "attrs": attrs,
                },
                "N": int(pending["N"]),
                "own_L0": float(pending["own_L0"]),
                "own_landing": float(pending["own_landing"]),
            }
