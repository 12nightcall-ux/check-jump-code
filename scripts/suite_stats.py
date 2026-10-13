"""Suite-level statistics for the CheckJump paper, recomputed from the raw per-seed files,
plus integrity and coverage checks.  Usage: python suite_stats.py <folder with the 30 experiment folders>
Arms: `adamw` and `b25` (b25 = CheckJump). Nothing is tuned; the script only reads files."""
import os, re, sys, json, math, collections
import numpy as np
from scipy import stats

SEEDS = [42, 137, 256, 512, 1024]
SKIP = ("_checks", "_live", "smoke", "trial", "wide", "analysis", "__pycache__",
        "revisit", "nogate", "norefill", "randdir", "fixed_lr", "batch")
MARGIN = 1.0  # percent
ARMS = ["adam", "adamw", "b25", "b25_plain", "nadam", "rmsprop", "sgd"]


def seed_files(root):
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if not any(s in d.lower() for s in SKIP) and not d.startswith(".")]
        for f in sorted(fn):
            if re.fullmatch(r"seed_\d+\.json", f):
                p = os.path.join(dp, f)
                try:
                    j = json.load(open(p))
                except Exception:
                    print("UNREADABLE:", p); continue
                if all(k in j for k in ("optimizer", "experiment_name", "val_loss", "seed")):
                    yield p, j


def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); m = len(ps)
    adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj


def main(root):
    data, seen, problems = {}, collections.Counter(), []
    cov = collections.defaultdict(lambda: collections.Counter())
    for p, j in seed_files(root):
        e, a, s = j["experiment_name"], j["optimizer"], j["seed"]
        seen[(e, a, s)] += 1
        data.setdefault((e, a), {}).setdefault(s, float(j["val_loss"]))
        c = cov[a]; c["files"] += 1
        cv = j.get("curve") or []
        if cv: c["curve"] += 1
        if "test_loss" in j or "test_acc" in j: c["test"] += 1
        if j.get("b25_stats"): c["b25_stats"] += 1
        if any("param" in k.lower() and k != "hparams" for k in j): c["nparams"] += 1
        v = j["val_loss"]
        if not (isinstance(v, (int, float)) and math.isfinite(v)): problems.append(f"non-finite val_loss {p}")
        if cv:
            vals = [x["val_loss"] for x in cv]
            if abs(min(vals) - v) > 1e-6: problems.append(f"val_loss != curve minimum: {p}")
            last = cv[-1]["compute_calls"]
            if not (999 <= last <= 1000 + 1e-6): problems.append(f"curve ends at {last}: {p}")
    for k, n in seen.items():
        if n > 1: problems.append(f"duplicate {k} x{n}")
    exps = sorted({k[0] for k in data})
    print(f"experiments found: {len(exps)} (expected 30)")
    if len(exps) != 30: problems.append("experiment count != 30")
    for e in exps:
        for a in ARMS:
            got = sorted(data.get((e, a), {}))
            if got != sorted(SEEDS): problems.append(f"{e}/{a}: seeds {got}")
    print("\nCOVERAGE (files per arm, expected 150 = 30 x 5):")
    for a in ARMS:
        c = cov[a]
        print(f"  {a:10s} files {c['files']:4d} | curve {c['curve']:4d} | test metric {c['test']:4d} | b25_stats {c['b25_stats']:4d} | param count {c['nparams']:4d}")
    print(f"\nINTEGRITY problems: {len(problems)}")
    for x in problems[:25]: print("  -", x)

    rows = []
    for e in exps:
        a, b = data.get((e, "adamw"), {}), data.get((e, "b25"), {})
        seeds = [s for s in SEEDS if s in a and s in b]
        if len(seeds) < 2: continue
        va = np.array([a[s] for s in seeds]); vb = np.array([b[s] for s in seeds])
        rows.append((e, len(seeds), va.mean(), vb.mean(), (vb - va).mean(),
                     100.0 * (vb.mean() - va.mean()) / va.mean(), float(stats.ttest_rel(vb, va).pvalue)))
    n = len(rows); adj = holm([r[6] for r in rows])
    print("\nexperiment         seeds  AdamW    CheckJump  gap       gap%    raw p    Holm p")
    for r, ap in zip(rows, adj):
        print(f"{r[0]:18s} {r[1]:3d}  {r[2]:.4f}  {r[3]:.4f}   {r[4]:+.5f}  {r[5]:+.2f}  {r[6]:.4f}  {ap:.4f}")
    g = np.array([r[5] for r in rows]); mu, se = g.mean(), stats.sem(g)
    h = stats.t.ppf(0.975, n - 1) * se
    tp = max(stats.t.sf((mu + MARGIN) / se, n - 1), stats.t.sf((MARGIN - mu) / se, n - 1))
    sig = [r[0] for r, ap in zip(rows, adj) if ap < 0.05]
    print(f"\npooled gap: {mu:+.3f}%  95% CI [{mu - h:+.2f}, {mu + h:+.2f}]  (n = {n}, df = {n - 1})")
    print(f"TOST vs +/-{MARGIN:g}%: p = {tp:.6f}")
    print(f"significant after Holm: {len(sig)} -> {sig} | not significant: {n - len(sig)} of {n}")


if __name__ == "__main__":
    if len(sys.argv) != 2: raise SystemExit("usage: python suite_stats.py <folder>")
    main(sys.argv[1])
