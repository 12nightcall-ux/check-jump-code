"""B25 checks library: parts W (wide box), M (mechanism), R (revisit), D (jump diagnostics) and their analysis.

Everything reuses the experiment folders' OWN task / tuning / builder modules. Nothing here changes the audited training loop,
the verified optimizer file or any finished result: new runs are written under B25_Experiments/_checks/ or into NEW clone folders.
"""
from __future__ import annotations

import gc
import glob
import hashlib
import importlib
import json
import math
import os
import shutil
import statistics as st
import sys
import time

ROOT = os.environ.get("B25_ROOT", "/content/drive/MyDrive/Optimiser Formula Creation - Working/B25_Experiments")
SEEDS5 = (42, 137, 256, 512, 1024)
SEEDS10 = SEEDS5 + (2048, 4096, 8192, 16384, 32768)
# (momentum_after_jump, gate, random_direction): the B25 builder's own settings are ("stagewise", "gn_median"); plain = ("keep", None)
VARIANTS = {
    "b25_default": ("stagewise", "gn_median", False),
    "b25_nogate": ("stagewise", None, False),
    "b25_norefill": ("keep", "gn_median", False),
    "b25_randdir": ("stagewise", "gn_median", True),
}
WIDE_PATCH = (
    ("LR_LO = 1e-6", "LR_LO = 1e-5"),
    ("LR_HI = 3e-3", "LR_HI = 1e-1"),
    ("BETA1_LO = 0.80", "BETA1_LO = 0.0"),
    ("BETA2_LO = 0.9\n", "BETA2_LO = 0.5\n"),
    ('SPACE_VERSION = "box5prime-v1"', 'SPACE_VERSION = "wide-v1"'),
)
BOX_NARROW = {"lr": (1e-6, 3e-3), "momentum": (0.1, 0.99), "beta1": (0.80, 0.9995), "beta2": (0.9, 0.99999)}
BOX_WIDE = {"lr": (1e-5, 1e-1), "momentum": (0.1, 0.99), "beta1": (0.0, 0.9995), "beta2": (0.5, 0.99999)}


def chk_dir() -> str:
    return os.path.join(ROOT, "_checks")


def sha16(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def _atomic_write(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read(path: str):
    with open(path) as f:
        return json.load(f)


def _valid(path: str, keys=("val_loss", "curve")) -> bool:
    try:
        r = _read(path)
        return all(k in r for k in keys)
    except Exception:
        return False


def _jsonable(h: dict) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in h.items()}


# ======================================================================================================================
# access to an experiment folder
# ======================================================================================================================
def open_exp(exp: str):
    """chdir into the experiment folder and import ITS task_glue and hpo_gp_bayesopt (never a module cached from another folder)."""
    d = os.path.join(ROOT, exp)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"experiment folder not found: {d}")
    for k, m in list(sys.modules.items()):
        f = (getattr(m, "__file__", None) or "").replace("\\", "/")
        if "/B25_Experiments/" in f and k != __name__:
            del sys.modules[k]
    sys.path[:] = [d] + [p for p in sys.path if "/B25_Experiments/" not in p.replace("\\", "/")]
    os.chdir(d)
    task = importlib.import_module("task_glue")
    hpo = importlib.import_module("hpo_gp_bayesopt")
    if os.path.dirname(os.path.abspath(task.__file__)) != os.path.abspath(d):
        raise RuntimeError(f"loaded {task.__file__}, not the {exp} folder")
    return task, hpo


def tuned(exp: str, arm: str) -> dict:
    """The setting the 5-seed retrain of this arm used (the winner of its original 5-trial tuning)."""
    h = dict(_read(os.path.join(ROOT, exp, "results", f"{arm}_five_seed", "summary.json"))["hparams"])
    if isinstance(h.get("betas"), list):
        h["betas"] = tuple(h["betas"])
    return h


def _random_dir_class(cls):
    import torch

    class B25RandomDirection(cls):
        """CONTROL: identical to B25, but at every attempt the averaged-move direction is replaced by a random vector of the SAME norm."""

        def _attempt_macro_jump(self, params, closure, closure_forward_only):
            real = self._ema_d
            if real is None:
                return super()._attempt_macro_jump(params, closure, closure_forward_only)
            with torch.no_grad():
                nrm = math.sqrt(sum(float(t.detach().double().pow(2).sum()) for t in real))
                rnd = [torch.randn_like(t) for t in real]
                rn = math.sqrt(sum(float(t.double().pow(2).sum()) for t in rnd))
                sc = nrm / rn if rn > 0 else 0.0
                rnd = [t * sc for t in rnd]
            self._ema_d, self._jb_d = rnd, rnd
            try:
                return super()._attempt_macro_jump(params, closure, closure_forward_only)
            finally:
                self._ema_d, self._jb_d = real, real

    return B25RandomDirection


def b25_variant(params, hps: dict, variant: str, *, log_attempts: bool = False):
    """The B25 builder's own constants (b25_builder.build_b25), with the momentum / gate / direction switches of VARIANTS."""
    import b25_builder as B
    from task_glue import CLIP_NORM, DEFAULT_BUDGET

    mom, gate, rnd = VARIANTS[variant]
    cls = B._load_class()
    if rnd:
        cls = _random_dir_class(cls)
    opt = cls(
        params, momentum_after_jump=mom, gate=gate,
        lr=float(hps["lr"]), betas=tuple(hps["betas"]), weight_decay=float(hps["weight_decay"]), decay_mode="decoupled",
        total_budget=float(DEFAULT_BUDGET), N_max=25, jump_every=10, window=(0.05, 0.80), ema_rho=0.9, fresh_undo=True, max_grad_norm=CLIP_NORM,
    )
    if log_attempts:
        opt.log_attempts = True
    return opt


def run_one(task, hpo, arm: str, hps: dict, seed: int, *, variant=None, log_attempts: bool = False, n_steps=None) -> dict:
    """One run with exactly the per-seed procedure of retrain_five_seeds.py (same model init, same loop, same checkpoint rule)."""
    try:
        import torch

        cuda = torch.cuda.is_available()
    except ImportError:
        torch, cuda = None, False
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    model = task.build_model(seed=seed)
    if variant:
        opt = b25_variant(model.parameters(), hps, variant, log_attempts=log_attempts)
    else:
        opt = hpo.build_optimizer(arm, model.parameters(), hps)
        if log_attempts and hasattr(opt, "log_attempts"):
            opt.log_attempts = True
    kw = {"smoke": False, "seed": seed, "score_test": True}
    if n_steps is not None:
        kw["n_steps"] = int(n_steps)
    else:
        kw["budget"] = int(task.DEFAULT_BUDGET)
    result = task.train_one_trial(model, opt, **kw)
    locked = task.global_best_checkpoint(result["curve"])
    row = {
        "arm": arm, "variant": variant, "seed": seed, "hparams": _jsonable(hps), "experiment_name": task.EXPERIMENT_NAME,
        "gpu_name": result.get("gpu_name"), "budget": result["budget"], "n_outer_steps": result["n_outer_steps"],
        "clip_rate": float(result.get("clip_rate", 0.0)), "val_loss": locked["val_loss"], "val_acc": locked["val_acc"],
        "compute_calls": locked["compute_calls"], "wall_s": locked["wall_s"], "end_val_loss": result["val_loss"],
        "end_val_acc": result["val_acc"], "test_loss": result.get("test_loss"), "test_acc": result.get("test_acc"),
        "curve": result["curve"], "b25_stats": result.get("b25_stats"), "checkpoint_rule": "global_best",
    }
    if cuda:
        row["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
    if log_attempts and hasattr(opt, "_att_rows"):
        row["att_rows"] = list(opt._att_rows)
    del model, opt, result
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    return row


class override:
    """Temporarily change the batch size and / or train on a fixed random subset of the training windows (one fixed subset per size)."""

    def __init__(self, task, *, frac=None, batch=None, sub_seed: int = 0):
        self.task, self.frac, self.batch, self.sub_seed = task, frac, batch, sub_seed
        self.n_train = None

    def __enter__(self):
        t = self.task
        self.old = (t.BATCH_SIZE, t._CACHE)
        if self.batch is not None:
            t.BATCH_SIZE = int(self.batch)
        ds = t._datasets()
        if self.frac is not None and self.frac < 1.0:
            import numpy as np

            tr = ds["train"]
            ids = tr.encodings["input_ids"]
            n = len(ids)
            k = max(int(t.BATCH_SIZE), int(round(self.frac * n)))
            pick = np.sort(np.random.default_rng(1000 + int(self.sub_seed)).permutation(n)[:k])
            try:
                import torch

                pick = torch.from_numpy(pick)
            except ImportError:
                pass
            sub = t._TokenDS({"input_ids": ids[pick], "attention_mask": tr.encodings["attention_mask"][pick]}, tr.labels[pick])
            t._CACHE = {**ds, "train": sub}
            self.n_train = k
        else:
            self.n_train = len(ds["train"])
        return self

    def __exit__(self, *a):
        self.task.BATCH_SIZE, self.task._CACHE = self.old
        return False


def _p(part: str, exp: str, name: str) -> str:
    return os.path.join(chk_dir(), part, exp, name + ".json")


# ======================================================================================================================
# part runners (resumable: a finished valid file is never run again)
# ======================================================================================================================
def run_M(exp="lstm_warpeace", lrs=(3e-4, 1e-3, 3e-3, 1e-2), seeds=SEEDS5, variants=("b25_nogate", "b25_norefill", "b25_randdir"), log=print):
    task, hpo = open_exp(exp)
    for arm in ("adamw", "b25"):
        base = tuned(exp, arm)
        for lr in lrs:
            for s in seeds:
                p = _p("M", exp, f"{arm}_lr{lr:g}_seed{s}")
                if _valid(p):
                    continue
                hps = dict(base)
                hps["lr"] = lr
                row = run_one(task, hpo, arm, hps, s)
                row["lr"] = lr
                _atomic_write(p, row)
                log(f"M {arm} lr={lr:g} seed={s} val_loss={row['val_loss']:.5f}")
    base = tuned(exp, "b25")
    for var in variants:
        for s in seeds:
            p = _p("M", exp, f"{var}_seed{s}")
            if _valid(p):
                continue
            row = run_one(task, hpo, "b25", base, s, variant=var)
            _atomic_write(p, row)
            log(f"M {var} seed={s} val_loss={row['val_loss']:.5f}")


def run_R(exp="lstm_warpeace", fracs=(0.01, 0.05, 0.25), batches=(5, 10, 25), seeds=SEEDS5, log=print):
    task, hpo = open_exp(exp)
    base_batch = int(task.BATCH_SIZE)
    n_full = len(task._datasets()["train"])
    settings = [("revisit", f, base_batch) for f in fracs] + [("batch", min(1.0, 1000.0 * b / n_full), b) for b in batches]
    for kind, frac, b in settings:
        with override(task, frac=frac, batch=b, sub_seed=0) as ov:
            for arm in ("adamw", "b25"):
                hps = tuned(exp, arm)
                for s in seeds:
                    p = _p("R", exp, f"{arm}_{kind}_f{frac:.4f}_b{b}_seed{s}")
                    if _valid(p):
                        continue
                    row = run_one(task, hpo, arm, hps, s)
                    row.update({"kind": kind, "frac": frac, "batch": b, "n_train": ov.n_train, "epochs": 1000.0 * b / ov.n_train, "targets_per_batch": b * int(task.SEQ_LEN)})
                    _atomic_write(p, row)
                    log(f"R {kind} frac={frac:.4f} batch={b} {arm} seed={s} val_loss={row['val_loss']:.5f}")


def run_D(exp: str, arms=("b25", "b25_plain"), seeds=(42, 137, 256), mem_arm="adamw", log=print):
    task, hpo = open_exp(exp)
    for arm in arms:
        hps = tuned(exp, arm)
        for s in seeds:
            p = _p("D", exp, f"{arm}_seed{s}")
            if _valid(p):
                continue
            row = run_one(task, hpo, arm, hps, s, log_attempts=True)
            _atomic_write(p, row)
            log(f"D {exp} {arm} seed={s} attempt rows={len(row.get('att_rows', []))}")
    p = _p("D", exp, f"{mem_arm}_seed{seeds[0]}")
    if mem_arm and not _valid(p):
        row = run_one(task, hpo, mem_arm, tuned(exp, mem_arm), seeds[0])
        _atomic_write(p, row)
        log(f"D {exp} {mem_arm} (memory reference) seed={seeds[0]}")


# ======================================================================================================================
# the wide-box clones (W) and new-corpus clones
# ======================================================================================================================
_SKIP = {"results", "logs", "Notebooks", "data", "__pycache__", "MANIFEST.json", "SELFTEST_PASS.txt", "GPU_LOCK.txt", "CLONE_OF.txt"}


def make_clone(src: str, dst: str, *, wide: bool = False, harness_text=None, glue_task=None, extra_files=None, log=print) -> str:
    s, d = os.path.join(ROOT, src), os.path.join(ROOT, dst)
    if os.path.exists(os.path.join(d, "CLONE_OF.txt")):
        log(f"{dst}: clone already made")
        return d
    if os.path.exists(d):
        raise RuntimeError(f"{d} exists but is not a finished clone made by this library; rename or remove it by hand")
    os.makedirs(d)
    for name in sorted(os.listdir(s)):
        p = os.path.join(s, name)
        if name not in _SKIP and os.path.isfile(p):
            shutil.copy2(p, os.path.join(d, name))
    for sub in ("results", "logs", "Notebooks"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    qs, qd = f'"{src}"', f'"{dst}"'
    for name in sorted(os.listdir(d)):
        if name.endswith((".py", ".md")):
            p = os.path.join(d, name)
            t = open(p, encoding="utf-8").read()
            if qs in t:
                open(p, "w", encoding="utf-8").write(t.replace(qs, qd))
    if harness_text is not None:
        open(os.path.join(d, "task_glue.py"), "w", encoding="utf-8").write(harness_text)
    patched = []
    if wide:
        p = os.path.join(d, "official_fields.py")
        t = open(p, encoding="utf-8").read()
        for old, new in WIDE_PATCH:
            if t.count(old) != 1:
                raise RuntimeError(f"cannot patch the search space: {old!r} occurs {t.count(old)} times in official_fields.py")
            t = t.replace(old, new)
            patched.append(f"{old.strip()} -> {new.strip()}")
        open(p, "w", encoding="utf-8").write(t)
    for fn, txt in (extra_files or {}).items():
        open(os.path.join(d, fn), "w", encoding="utf-8").write(txt)
    for fn in ("task_glue.py", "hpo_gp_bayesopt.py"):
        if f'EXPERIMENT_NAME = "{dst}"' not in open(os.path.join(d, fn), encoding="utf-8").read():
            raise RuntimeError(f"{fn} does not carry the experiment name {dst}")
    man = _read(os.path.join(s, "MANIFEST.json"))
    names = sorted(set(man["files"]) | set(extra_files or {}))
    man["files"] = {n: {"sha256_16": sha16(os.path.join(d, n)), "edited_vs_staging": True} for n in names}
    if glue_task is not None:
        man["glue_task"] = glue_task
    man.update({"experiment": dst, "clone_of": src, "space_sha16": sha16(os.path.join(d, "official_fields.py")), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    _atomic_write(os.path.join(d, "MANIFEST.json"), man)
    open(os.path.join(d, "CLONE_OF.txt"), "w").write(f"clone of {src} on {time.ctime()}\nwide box: {wide}\n" + "\n".join(patched) + "\n")
    log(f"built clone {dst} from {src} ({'wide search box' if wide else 'same box'}; harness replaced: {harness_text is not None})")
    return d


RETRAIN_WIDE = '''"""Retrain the best COMPLETE tuning trial of each arm on K seeds. Same per-seed procedure as retrain_five_seeds.py, but K and the trial count are arguments."""
import argparse, gc, importlib, json, os, statistics, sys
from hpo_gp_bayesopt import DEFAULT_BUDGET, build_optimizer, complete_trial_numbers, load_existing_study
from time_to_target import best_by_val_loss, load_trial_jsons

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS10 = (42, 137, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)


def jsonable(h):
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in h.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--task", default="task_glue")
    a = ap.parse_args(argv)
    task = importlib.import_module(a.task)
    seeds = SEEDS10[: a.seeds]
    budget = int(DEFAULT_BUDGET)
    if budget != int(task.DEFAULT_BUDGET):
        raise SystemExit("budget mismatch")
    for arm in a.arms:
        hpo_arm = "b25" if arm == "b25_plain" else arm
        study = load_existing_study(hpo_arm, smoke=False)
        complete = complete_trial_numbers(study)
        if len(complete) != a.n_trials:
            raise SystemExit(f"{hpo_arm}: {len(complete)} COMPLETE trials, need exactly {a.n_trials}; finish the tuning first")
        trials = [t for t in load_trial_jsons(hpo_arm, budget) if int(t.get("trial_number", -1)) in complete]
        if len(trials) != a.n_trials:
            raise SystemExit(f"{hpo_arm}: {len(trials)} trial files for {a.n_trials} COMPLETE trials")
        best = best_by_val_loss(trials)
        hps = dict(best["hparams"])
        if isinstance(hps.get("betas"), list):
            hps["betas"] = tuple(hps["betas"])
        dest = os.path.join(HERE, "results", f"{arm}_wide_seed")
        os.makedirs(dest, exist_ok=True)
        rows = []
        for seed in seeds:
            path = os.path.join(dest, f"seed_{seed}.json")
            if os.path.isfile(path):
                try:
                    prev = json.load(open(path))
                    if prev.get("hparams") == best["hparams"] and prev.get("seed") == seed and prev.get("curve"):
                        rows.append(prev)
                        print(f"{arm} seed={seed}: valid result exists, skipping", flush=True)
                        continue
                except Exception:
                    pass
            model = task.build_model(seed=seed)
            opt = build_optimizer(arm, model.parameters(), hps)
            res = task.train_one_trial(model, opt, smoke=False, seed=seed, score_test=True, budget=budget)
            locked = task.global_best_checkpoint(res["curve"])
            row = {"optimizer": arm, "seed": seed, "hparams": best["hparams"], "source_trial": best.get("trial_number"), "gpu_name": res.get("gpu_name"),
                   "budget": res["budget"], "n_outer_steps": res["n_outer_steps"], "clip_rate": float(res.get("clip_rate", 0.0)), "checkpoint_rule": "global_best",
                   "val_loss": locked["val_loss"], "val_acc": locked["val_acc"], "compute_calls": locked["compute_calls"], "wall_s": locked["wall_s"],
                   "end_val_loss": res["val_loss"], "end_val_acc": res["val_acc"], "test_loss": res["test_loss"], "test_acc": res["test_acc"],
                   "curve": res["curve"], "b25_stats": res.get("b25_stats"), "experiment_name": task.EXPERIMENT_NAME}
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(row, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            rows.append(row)
            del model, opt, res
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            print(f"{arm} seed={seed} locked_val_loss={row['val_loss']:.6f} Ng={row['compute_calls']} wall_s={row['wall_s']:.1f}", flush=True)
        vals = [r["val_loss"] for r in rows]
        summ = {"optimizer": arm, "n": len(rows), "seeds": list(seeds), "hparams": best["hparams"], "source_trial": best.get("trial_number"), "n_trials": a.n_trials,
                "val_loss": {"mean": statistics.mean(vals), "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}, "experiment_name": task.EXPERIMENT_NAME}
        json.dump(summ, open(os.path.join(dest, "summary.json"), "w"), indent=2)
        print(f"{arm} wide locked_val_loss={summ['val_loss']['mean']:.6f}+-{summ['val_loss']['std']:.6f} over {len(rows)} seeds hp={best['hparams']}", flush=True)


if __name__ == "__main__":
    main()
'''


# ======================================================================================================================
# analysis (CPU only)
# ======================================================================================================================
def paired(a, b) -> dict:
    """a - b over equal-length lists (same seeds); negative = a lower (better for losses)."""
    from scipy import stats

    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        return {"n": n, "mean": (d[0] if d else None)}
    m, s = st.mean(d), st.stdev(d)
    if s == 0:
        return {"n": n, "mean": m, "ci": (m, m), "p": 1.0 if m == 0 else 0.0, "wins": sum(1 for x in d if x < 0)}
    se = s / math.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    return {"n": n, "mean": m, "ci": (m - h, m + h), "p": float(2 * (1 - stats.t.cdf(abs(m / se), n - 1))), "wins": sum(1 for x in d if x < 0)}


def _f(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _pr(p) -> str:
    return "n/a" if not p or p.get("mean") is None or "ci" not in p else f"{p['mean']:+.4f} (CI {p['ci'][0]:+.4f} to {p['ci'][1]:+.4f}, p={p['p']:.4f}, better in {p['wins']}/{p['n']})"


def _ms(xs) -> str:
    return "n/a" if not xs else (f"{st.mean(xs):.4f} +- {st.stdev(xs):.4f} (n={len(xs)})" if len(xs) > 1 else f"{xs[0]:.4f} (n=1)")


def load_rows(part: str, exp: str, prefix: str = "") -> list:
    out = []
    for f in sorted(glob.glob(os.path.join(chk_dir(), part, exp, prefix + "*.json"))):
        if _valid(f):
            out.append(_read(f))
    return out


def orig_rows(exp: str, arm: str) -> dict:
    return {r["seed"]: r for r in (_read(f) for f in sorted(glob.glob(os.path.join(ROOT, exp, "results", f"{arm}_five_seed", "seed_*.json"))) if _valid(f, ("val_loss", "curve", "seed")))}


def bsf_at(curve, c):
    v = [float(p["val_loss"]) for p in curve if float(p["compute_calls"]) <= c + 1e-9]
    return min(v) if v else None


def reach(curve, level):
    for p in curve:
        if float(p["val_loss"]) <= level:
            return float(p["compute_calls"])
    return None


def kept_share(b25_stats):
    if not b25_stats:
        return None
    acc = float(b25_stats.get("kept", 0)) + float(b25_stats.get("undone", 0))
    return None if acc == 0 else float(b25_stats.get("kept", 0)) / acc


def saving(rows_a: dict, rows_b: dict, q: float):
    """billed units AdamW-like arm A needs to reach ITS OWN mean best-so-far loss at q of the budget, minus the units arm B needs (B faster = positive)."""
    budget = float(next(iter(rows_a.values()))["budget"])
    lv = [bsf_at(r["curve"], q * budget) for r in rows_a.values()]
    lv = [x for x in lv if x is not None]
    if not lv:
        return None
    target = st.mean(lv)
    seeds = sorted(set(rows_a) & set(rows_b))
    ha = {s: reach(rows_a[s]["curve"], target) for s in seeds}
    hb = {s: reach(rows_b[s]["curve"], target) for s in seeds}
    both = [s for s in seeds if ha[s] is not None and hb[s] is not None]
    if len(both) < 3:
        return {"target": target, "n_both": len(both)}
    pr = paired([ha[s] for s in both], [hb[s] for s in both])
    return {"target": target, "n_both": len(both), "saving": pr}


def _sav(sv) -> str:
    if not sv or "saving" not in sv:
        return "n/a (fewer than 3 common seeds reach it)"
    p = sv["saving"]
    return f"{p['mean']:+.0f} units (CI {p['ci'][0]:+.0f} to {p['ci'][1]:+.0f}; {sv['n_both']} seeds)"


def edge_flags(h: dict, box: dict, tol: float = 0.02) -> list:
    out = []

    def where(name, v):
        lo, hi = box[name]
        if name == "lr":
            f = (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))
        elif name == "momentum":
            f = (v - lo) / (hi - lo)
        else:
            f = 1 - (math.log(1 - v) - math.log(1 - hi)) / (math.log(1 - lo) - math.log(1 - hi)) if v < 1 else 1.0
        return "top" if f >= 1 - tol else ("bottom" if f <= tol else None)

    vals = {"lr": h.get("lr")}
    if "momentum" in h:
        vals["momentum"] = h["momentum"]
    if "betas" in h:
        vals["beta1"], vals["beta2"] = h["betas"]
    for k, v in vals.items():
        if v is not None:
            w = where(k, float(v))
            if w:
                out.append(f"{k}-{w}")
    return out


def report_M(exp="lstm_warpeace", lrs=(3e-4, 1e-3, 3e-3, 1e-2)) -> str:
    L = [f"## M - fixed learning rates ({exp}; each arm keeps its original tuned beta1, beta2 and weight decay; only the learning rate changes)", ""]
    rows = {}
    for arm in ("adamw", "b25"):
        for lr in lrs:
            rows[(arm, lr)] = {r["seed"]: r for r in load_rows("M", exp, f"{arm}_lr{lr:g}_seed")}
    L += ["| lr | AdamW best val loss | B25 best val loss | B25 - AdamW (paired by seed) | B25 jumps per run: accepted / kept |", "|---|---|---|---|---|"]
    for lr in lrs:
        a, b = rows[("adamw", lr)], rows[("b25", lr)]
        cs = sorted(set(a) & set(b))
        acc = [float(b[s]["b25_stats"]["kept"]) + float(b[s]["b25_stats"]["undone"]) for s in cs if b[s].get("b25_stats")]
        kp = [float(b[s]["b25_stats"]["kept"]) for s in cs if b[s].get("b25_stats")]
        L.append(f"| {lr:g} | {_ms([a[s]['val_loss'] for s in cs])} | {_ms([b[s]['val_loss'] for s in cs])} | {_pr(paired([b[s]['val_loss'] for s in cs], [a[s]['val_loss'] for s in cs])) if len(cs) > 1 else 'n/a'} | {(f'{st.mean(acc):.1f} / {st.mean(kp):.1f}' if acc else 'n/a')} |")
    L += ["", "**Effective-learning-rate comparison (paired by seed; negative = the first is better):**", ""]
    for (a_arm, a_lr), (b_arm, b_lr) in ((("b25", 3e-3), ("adamw", 1e-2)), (("b25", 1e-2), ("adamw", 1e-2)), (("b25", 3e-3), ("adamw", 3e-3)), (("b25", 1e-3), ("adamw", 3e-3))):
        A, B = rows.get((a_arm, a_lr), {}), rows.get((b_arm, b_lr), {})
        cs = sorted(set(A) & set(B))
        if len(cs) > 1:
            L.append(f"- {a_arm} at {a_lr:g} minus {b_arm} at {b_lr:g}: {_pr(paired([A[s]['val_loss'] for s in cs], [B[s]['val_loss'] for s in cs]))}")
    L += ["", "**Control (lr 3e-3 must reproduce the original 5-seed runs):**", ""]
    for arm in ("adamw", "b25"):
        o = orig_rows(exp, arm)
        d = [rows[(arm, 3e-3)][s]["val_loss"] - o[s]["val_loss"] for s in sorted(set(o) & set(rows[(arm, 3e-3)]))]
        if d:
            L.append(f"- {arm}: new minus original best val loss per seed: max |diff| {max(abs(x) for x in d):.6f} over {len(d)} seeds")
    L += ["", "**Components (B25 at its tuned lr, 5 seeds):**", "", "| variant | best val loss | minus AdamW | minus B25 | accepted / kept per run |", "|---|---|---|---|---|"]
    ref_a, ref_b = rows[("adamw", 3e-3)], rows[("b25", 3e-3)]
    for var in ("b25_nogate", "b25_norefill", "b25_randdir"):
        V = {r["seed"]: r for r in load_rows("M", exp, f"{var}_seed")}
        cs_a, cs_b = sorted(set(V) & set(ref_a)), sorted(set(V) & set(ref_b))
        if not V:
            continue
        acc = [float(r["b25_stats"]["kept"]) + float(r["b25_stats"]["undone"]) for r in V.values() if r.get("b25_stats")]
        kp = [float(r["b25_stats"]["kept"]) for r in V.values() if r.get("b25_stats")]
        L.append(f"| {var} | {_ms([r['val_loss'] for r in V.values()])} | {_pr(paired([V[s]['val_loss'] for s in cs_a], [ref_a[s]['val_loss'] for s in cs_a])) if len(cs_a) > 1 else 'n/a'} | {_pr(paired([V[s]['val_loss'] for s in cs_b], [ref_b[s]['val_loss'] for s in cs_b])) if len(cs_b) > 1 else 'n/a'} | {f'{st.mean(acc):.1f} / {st.mean(kp):.1f}' if acc else 'n/a'} |")
    return "\n".join(L) + "\n"


def report_R(exp="lstm_warpeace") -> str:
    L = [f"## R - why do jumps survive? ({exp}; AdamW and B25 at their original tuned settings, 1000 units, 5 seeds, one fixed subset per size)", "",
         "revisit = fewer training windows at batch 50 (more passes); batch = smaller batches with the training set cut so that about ONE pass is made.", "",
         "| kind | train windows | batch | targets per batch | passes in 1000 steps | AdamW loss | B25 loss | B25 - AdamW | accepted | kept | kept share | saving at 50% / 75% |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    allr = load_rows("R", exp)
    keys = sorted({(r["kind"], r["frac"], r["batch"]) for r in allr}, key=lambda k: (k[0], -k[1], k[2]))
    base_a, base_b = orig_rows(exp, "adamw"), orig_rows(exp, "b25")
    groups = [("base", None, None, base_a, base_b)] + [(k[0], k[1], k[2], {r["seed"]: r for r in allr if (r["kind"], r["frac"], r["batch"]) == k and r["arm"] == "adamw"}, {r["seed"]: r for r in allr if (r["kind"], r["frac"], r["batch"]) == k and r["arm"] == "b25"}) for k in keys]
    meta = {(r["kind"], r["frac"], r["batch"]): r for r in allr}
    for kind, frac, b, A, B in groups:
        cs = sorted(set(A) & set(B))
        if not cs:
            continue
        if kind == "base":
            info = ("original", "all", "50", "2500", f"{1000 * 50 / 51332:.2f}")
            m = None
        else:
            m = meta[(kind, frac, b)]
            info = (kind, str(m["n_train"]), str(b), str(m["targets_per_batch"]), f"{m['epochs']:.1f}")
        ks = [kept_share(B[s].get("b25_stats")) for s in cs]
        ks = [x for x in ks if x is not None]
        acc = [float(B[s]["b25_stats"]["kept"]) + float(B[s]["b25_stats"]["undone"]) for s in cs if B[s].get("b25_stats")]
        kp = [float(B[s]["b25_stats"]["kept"]) for s in cs if B[s].get("b25_stats")]
        L.append(f"| {info[0]} | {info[1]} | {info[2]} | {info[3]} | {info[4]} | {_ms([A[s]['val_loss'] for s in cs])} | {_ms([B[s]['val_loss'] for s in cs])} | {_pr(paired([B[s]['val_loss'] for s in cs], [A[s]['val_loss'] for s in cs])) if len(cs) > 1 else 'n/a'} | {f'{st.mean(acc):.1f}' if acc else 'n/a'} | {f'{st.mean(kp):.1f}' if kp else 'n/a'} | {f'{st.mean(ks):.2f}' if ks else 'n/a'} | {_sav(saving(A, B, 0.5))} ; {_sav(saving(A, B, 0.75))} |")
    return "\n".join(L) + "\n"


def _auc(pos, neg):
    """Mann-Whitney AUROC of 'pos' scores above 'neg' scores (None if either is empty)."""
    pos, neg = [x for x in pos if x is not None], [x for x in neg if x is not None]
    if not pos or not neg:
        return None
    gt = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return gt / (len(pos) * len(neg))


def report_D(exps) -> str:
    L = ["## D - per-attempt jump diagnostics (B25 with the attempt log on; the log is passive: it adds no gradient evaluation)", "",
         "The attempt log is written after the gate, so gate skips are NOT in these rows; they are in b25_stats (Table E).", "",
         "| experiment | arm | runs | logged attempts | skipped uphill | accepted | kept | undone | kept share | AUROC: consecutive-gradient cosine separates kept from undone | accepted share, early / mid / late third |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    mem = []
    for exp in exps:
        for arm in ("b25", "b25_plain"):
            rows = [r for r in load_rows("D", exp, arm + "_seed") if r.get("att_rows") is not None]
            if not rows:
                continue
            att = [a for r in rows for a in r["att_rows"]]
            n_up = sum(1 for a in att if a["outcome"] in ("skip_uphill", "nonfinite"))
            acc = [a for a in att if a["outcome"] == "accepted"]
            kept = [a for a in acc if a.get("undone") is False]
            und = [a for a in acc if a.get("undone") is True]
            auc = _auc([a.get("cos_mean10") for a in kept], [a.get("cos_mean10") for a in und])
            budget = float(rows[0]["budget"])
            thirds = []
            for lo, hi in ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.01)):
                seg = [a for a in att if lo * budget <= float(a["compute_calls"]) < hi * budget]
                thirds.append(f"{sum(1 for a in seg if a['outcome'] == 'accepted') / len(seg):.2f}" if seg else "n/a")
            L.append(f"| {exp} | {arm} | {len(rows)} | {len(att)} | {n_up} | {len(acc)} | {len(kept)} | {len(und)} | {len(kept) / len(acc):.2f} | {_f(auc, 3) if acc else 'n/a'} | {' / '.join(thirds)} |" if acc else f"| {exp} | {arm} | {len(rows)} | {len(att)} | {n_up} | 0 | 0 | 0 | n/a | n/a | {' / '.join(thirds)} |")
        ra = load_rows("D", exp, "adamw_seed")
        rb = [r for r in load_rows("D", exp, "b25_seed") if r.get("peak_mem_mb")]
        if ra and ra[0].get("peak_mem_mb") and rb:
            mem.append((exp, ra[0]["peak_mem_mb"], st.mean(r["peak_mem_mb"] for r in rb)))
    if mem:
        L += ["", "**Peak GPU memory (MB), AdamW against B25 (the attempt log is on in the B25 runs, so this is an upper bound):**", "", "| experiment | AdamW | B25 | ratio |", "|---|---|---|---|"]
        L += [f"| {e} | {a:.0f} | {b:.0f} | {b / a:.2f} |" for e, a, b in mem]
    return "\n".join(L) + "\n"


def report_W(clones=("lstm_warpeace_wide", "lstm_warpeace_wide2"), orig="lstm_warpeace") -> str:
    L = ["## W - re-tuning in a WIDE search box (lr 1e-5 to 1e-1, beta1 0 to 0.9995, beta2 0.5 to 0.99999; same weight-decay list; 20 trials; the Optuna sampler and default first trial are the original ones)", ""]
    for clone in clones:
        d = os.path.join(ROOT, clone, "results")
        if not os.path.isdir(d):
            continue
        rows = {}
        for arm in ("adamw", "b25", "b25_plain", "rmsprop"):
            fs = sorted(glob.glob(os.path.join(d, f"{arm}_wide_seed", "seed_*.json")))
            if fs:
                rows[arm] = {r["seed"]: r for r in (_read(f) for f in fs if _valid(f, ("val_loss", "curve", "seed", "hparams")))}
                if not rows[arm]:
                    del rows[arm]
        if not rows:
            continue
        L += [f"### {clone}", "", "| arm | seeds | best val loss | tuned setting | on a box edge (wide box)? |", "|---|---|---|---|---|"]
        for arm, R in rows.items():
            h = next(iter(R.values()))["hparams"]
            hh = dict(h)
            if isinstance(hh.get("betas"), list):
                hh["betas"] = tuple(hh["betas"])
            L.append(f"| {arm} | {len(R)} | {_ms([r['val_loss'] for r in R.values()])} | {json.dumps(h)} | {', '.join(edge_flags(hh, BOX_WIDE)) or 'no'} |")
        L.append("")

        def pair(a, b, label):
            if a in rows and b in rows:
                cs = sorted(set(rows[a]) & set(rows[b]))
                if len(cs) > 1:
                    L.append(f"- {label}: {_pr(paired([rows[a][s]['val_loss'] for s in cs], [rows[b][s]['val_loss'] for s in cs]))}")

        pair("b25", "adamw", "B25 - AdamW (primary, wide box)")
        pair("b25_plain", "adamw", "B25-plain - AdamW")
        pair("b25", "b25_plain", "B25 - B25-plain")
        pair("b25", "rmsprop", "B25 - RMSprop")
        pair("adamw", "rmsprop", "AdamW - RMSprop")
        if "b25" in rows and "adamw" in rows:
            L.append(f"- saving, billed units to reach AdamW's loss at 50% / 75% / 95% of the budget: {_sav(saving(rows['adamw'], rows['b25'], 0.5))} ; {_sav(saving(rows['adamw'], rows['b25'], 0.75))} ; {_sav(saving(rows['adamw'], rows['b25'], 0.95))}")
        for arm in ("adamw", "b25", "rmsprop"):
            o = orig_rows(orig, arm)
            if arm in rows and o:
                cs = sorted(set(o) & set(rows[arm]))
                if len(cs) > 1:
                    L.append(f"- {arm}: wide box minus original box, same seeds: {_pr(paired([rows[arm][s]['val_loss'] for s in cs], [o[s]['val_loss'] for s in cs]))}")
        L.append("")
    return "\n".join(L) + "\n"


def write_report(exps_D=("lstm_warpeace",), path=None) -> str:
    parts = [f"# B25 checks report ({time.ctime()})", "", report_M(), report_R(), report_D(exps_D), report_W()]
    txt = "\n".join(parts)
    path = path or os.path.join(chk_dir(), "CHECKS_REPORT.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(txt)
    return txt
