"""smoke_compare.py - AdamW vs B25 (and the plain-jump ablation) in the REAL harness.

Same model init, same data order (seed), same budget, same settings (dropout off, clip 1.0),
same hyper-parameters (trial-0 default). A pipeline check, NOT a result: 200 units is far too short
to compare optimizers.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch

import task_distilbert_sst2 as task
from b25_builder import build_b25
from official_adamw import build_adamw

SEED = int(os.environ.get("SMOKE_SEED", "11"))
UNITS = int(os.environ.get("SMOKE_UNITS", "200"))
LR, BETAS, WD = 2e-5, (0.9, 0.999), 0.01  # trial-0 default shared by all arms
ARMS = ("adamw", "b25", "b25_plain")


def run_arm(name):
    model = task.build_model(seed=SEED)
    params = model.parameters()
    if name == "adamw":
        opt = build_adamw(params, LR, BETAS, WD)
    elif name == "b25":
        opt = build_b25(params, LR, BETAS, WD)
    else:
        opt = build_b25(params, LR, BETAS, WD, plain=True)
    res = task.train_one_trial(model, opt, smoke=False, seed=SEED, n_steps=UNITS)
    del model, opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def summarize(results):
    """results: {arm: train_one_trial result}. Prints the side-by-side table and returns True if all checks pass."""
    ok = True
    print("\nSMOKE_COMPARE seed=%d units=%d lr=%g betas=%s wd=%g" % (SEED, UNITS, LR, BETAS, WD))
    print("%-10s %7s %6s %9s %9s %9s %8s %9s %7s  %s" % (
        "arm", "units", "steps", "first_val", "best_val", "last_val", "best_acc", "clip_rate", "wall_s", "jumps att/kept/undone"))
    for arm in ARMS:
        r = results.get(arm)
        if r is None:
            continue
        curve = r["curve"]
        gb = task.global_best_checkpoint(curve)
        st = r.get("b25_stats")
        jumps = "-" if st is None else "%d/%d/%d" % (st["attempts"], st["kept"], st["undone"])
        print("%-10s %7.1f %6d %9.4f %9.4f %9.4f %8.4f %9.3f %7.1f  %s" % (
            arm, r["compute_calls"], r["n_outer_steps"], curve[0]["val_loss"], gb["val_loss"],
            r["val_loss"], gb["val_acc"], r["clip_rate"], r["wall_s"], jumps))
        in_budget = r["compute_calls"] <= r["budget"] + 1e-9
        learned = math.isfinite(gb["val_loss"]) and gb["val_loss"] < 0.60
        same_recipe = r.get("dropout_mode") == "eval_all" and r.get("clip_norm") == 1.0
        cov = True if st is None else bool(st["undo_coverage_ok"])
        if not (in_budget and learned and same_recipe and cov):
            ok = False
            print("  CHECK %s: in_budget=%s learned(best_val<0.60)=%s same_recipe=%s undo_coverage_ok=%s" % (
                arm, in_budget, learned, same_recipe, cov))
    print("\nval_loss at each checkpoint (units:val_loss)")
    for arm in ARMS:
        r = results.get(arm)
        if r is not None:
            print("%-10s %s" % (arm, " ".join("%d:%.3f" % (round(p["compute_calls"]), p["val_loss"]) for p in r["curve"])))
    if "adamw" in results and "b25" in results:
        a = task.global_best_checkpoint(results["adamw"]["curve"])["val_loss"]
        b = task.global_best_checkpoint(results["b25"]["curve"])["val_loss"]
        print("\nbest_val: adamw=%.4f  b25=%.4f  (b25 - adamw = %+.4f; one seed, %d units: a pipeline check, not a result)" % (a, b, b - a, UNITS))
    return ok


if __name__ == "__main__":
    results = {}
    for arm in ARMS:
        print("=== running %s ===" % arm, flush=True)
        results[arm] = run_arm(arm)
    ok = summarize(results)
    print("SMOKE_COMPARE PASS" if ok else "SMOKE_COMPARE CHECK - paste me the lines above")
