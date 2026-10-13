"""JumpingAdamSimpsonB25 builder for this fine-tune folder (used by hpo_gp_bayesopt.py and retrain_five_seeds.py)."""
import hashlib
import os
import sys

_B25_DIR = os.path.dirname(os.path.abspath(__file__))  # the verified optimizer file sits in this same folder
_B25_FILE = "jumping_adam_simpson_b25.py"
_B25_SHA16 = "0b63021273376f1e"  # verified file: all acceptance tests pass on this exact hash


def _load_class():
    path = os.path.join(_B25_DIR, _B25_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError("verified optimizer file not found: " + path)
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    if got != _B25_SHA16:
        raise RuntimeError("optimizer file hash %s != verified %s" % (got, _B25_SHA16))
    if _B25_DIR not in sys.path:
        sys.path.insert(0, _B25_DIR)
    from jumping_adam_simpson_b25 import JumpingAdamSimpsonB25

    return JumpingAdamSimpsonB25


def build_b25(params, lr, betas, weight_decay, *, plain=False):
    """plain=False: stagewise momentum (clipped refill) + gn_median gate + clip.
    plain=True: the jump mechanism only (momentum keep, no gate) with the SAME clip as AdamW."""
    from task_glue import CLIP_NORM, DEFAULT_BUDGET

    cls = _load_class()
    kw = dict(
        lr=float(lr),
        betas=tuple(betas),
        weight_decay=float(weight_decay),
        decay_mode="decoupled",
        total_budget=float(DEFAULT_BUDGET),
        N_max=25,
        jump_every=10,
        window=(0.05, 0.80),
        ema_rho=0.9,
        fresh_undo=True,
        max_grad_norm=CLIP_NORM,
    )
    if plain:
        return cls(params, momentum_after_jump="keep", gate=None, **kw)
    return cls(params, momentum_after_jump="stagewise", gate="gn_median", **kw)
