"""Post-HPO 5-seed DistilBERT SST-2 retrain. Not part of T=50.

Freeze best COMPLETE HPO trial per arm (lowest hpo_objective_val_loss,
never last_val_loss). Retrain seeds [42, 137, 256, 512, 1024] at the
task DEFAULT_BUDGET. Report global-best val_loss and last-at-stop.
Training itself has no early stopping. clip_rate=0.

Completeness joins JSON to Optuna COMPLETE trials in results/<opt>.db.
FAIL/RUNNING JSON is ignored. Does not delete files.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
from collections import Counter

from hpo_gp_bayesopt import (
    CLI_ALL,
    DEFAULT_BUDGET,
    DEFAULT_TASK,
    N_TRIALS,
    build_optimizer,
    complete_trial_numbers,
    fail_or_running_trial_numbers,
    load_existing_study,
)
from time_to_target import TEST_SEEDS, best_by_val_loss, load_trial_jsons

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_task(module_name: str):
    return importlib.import_module(module_name)


def _hps_from_json(hparams: dict) -> dict:
    hps = dict(hparams)
    if "betas" in hps and isinstance(hps["betas"], list):
        hps["betas"] = tuple(hps["betas"])
    if "jump_size" in hps and "min_jump_size" not in hps:
        hps["min_jump_size"] = 10
    if "jump_size" in hps and "adaptive_tolerance" not in hps:
        hps["adaptive_tolerance"] = False
    return hps


def _mean_std(xs: list[float]) -> dict:
    if not xs:
        return {"mean": None, "std": None}
    if len(xs) == 1:
        return {"mean": xs[0], "std": 0.0}
    return {"mean": statistics.mean(xs), "std": statistics.stdev(xs)}


def _require_protocol_on_result(result: dict, task) -> None:
    if "experiment_name" not in result or "protocol_early_stop" not in result:
        raise SystemExit(
            "train_one_trial must set experiment_name and protocol_early_stop"
        )
    if result["experiment_name"] != task.EXPERIMENT_NAME:
        raise SystemExit(
            f"experiment_name={result['experiment_name']!r} != "
            f"{task.EXPERIMENT_NAME!r}"
        )
    if bool(result["protocol_early_stop"]) != bool(task.PROTOCOL_EARLY_STOP):
        raise SystemExit(
            f"protocol_early_stop={result['protocol_early_stop']!r} != "
            f"{task.PROTOCOL_EARLY_STOP!r}"
        )


def _row_from_locked(opt_name, seed, hparams, source_trial, result, locked, task) -> dict:
    _require_protocol_on_result(result, task)
    row = {
        "optimizer": opt_name,
        "seed": seed,
        "hparams": hparams,
        "source_trial": source_trial,
        "gpu_name": result["gpu_name"],
        "budget": result["budget"],
        "n_outer_steps": result["n_outer_steps"],
        "clip_rate": float(result.get("clip_rate", 0.0)),
        "checkpoint_rule": "global_best",
        "locked_outer_step": locked["outer_step"],
        "compute_calls": locked["compute_calls"],
        "wall_s": locked["wall_s"],
        "val_loss": locked["val_loss"],
        "val_acc": locked["val_acc"],
        "end_val_loss": result["val_loss"],
        "end_val_acc": result["val_acc"],
        "end_compute_calls": result["compute_calls"],
        "test_loss": result["test_loss"],
        "test_acc": result["test_acc"],
        "protocol_early_stop": bool(task.PROTOCOL_EARLY_STOP),
        "experiment_name": task.EXPERIMENT_NAME,
        "curve": result["curve"],
        "b25_stats": result.get("b25_stats"),
        "dropout_mode": result.get("dropout_mode"),
        "clip_norm": result.get("clip_norm"),
    }
    if "train_loss" in locked:
        row["train_loss"] = locked["train_loss"]
        row["train_acc"] = locked["train_acc"]
    return row


def _require_complete_hpo_trials(opt_name: str, trials: list[dict], task) -> list[dict]:
    """Join JSON to Optuna COMPLETE. Ignore FAIL/RUNNING JSON. Do not delete files."""
    study = load_existing_study(opt_name, smoke=False)
    complete = complete_trial_numbers(study)
    ignored = fail_or_running_trial_numbers(study)
    if len(complete) != N_TRIALS:
        raise SystemExit(
            f"{opt_name}: Optuna COMPLETE count={len(complete)} numbers="
            f"{sorted(complete)}; need exactly {N_TRIALS} COMPLETE. "
            f"FAIL/RUNNING numbers={sorted(ignored)}. Do not delete JSON. Finish: "
            f"python hpo_gp_bayesopt.py --optimizer {opt_name} --seed 11 "
            f"--task {task.__name__}"
        )
    numbers = []
    missing_field = 0
    smoke_hits = []
    kept = []
    for t in trials:
        if "trial_number" not in t:
            missing_field += 1
            continue
        n = int(t["trial_number"])
        if n in ignored or n not in complete:
            continue
        if "protocol_early_stop" not in t or "experiment_name" not in t:
            raise SystemExit(
                f"COMPLETE trial {n} missing protocol_early_stop or experiment_name"
            )
        if bool(t["protocol_early_stop"]) != bool(task.PROTOCOL_EARLY_STOP):
            raise SystemExit(
                f"{opt_name}: trial {n} protocol_early_stop="
                f"{t['protocol_early_stop']} != {task.PROTOCOL_EARLY_STOP}."
            )
        if t["experiment_name"] != task.EXPERIMENT_NAME:
            raise SystemExit(
                f"{opt_name}: trial {n} experiment_name="
                f"{t['experiment_name']!r} != {task.EXPERIMENT_NAME!r}."
            )
        numbers.append(n)
        kept.append(t)
        if t.get("smoke") is True:
            smoke_hits.append(n)
    counts = Counter(numbers)
    present = set(numbers)
    duplicates = sorted(n for n, c in counts.items() if c > 1)
    bits = []
    if present != complete:
        bits.append(
            f"JSON COMPLETE numbers {sorted(present)} are not the Optuna "
            f"COMPLETE set {sorted(complete)}"
        )
    if duplicates:
        bits.append(f"duplicate COMPLETE trial numbers: {duplicates}")
    if smoke_hits:
        bits.append(f"smoke=true on COMPLETE trial numbers: {smoke_hits}")
    if missing_field:
        bits.append(f"{missing_field} JSON file(s) missing trial_number")
    if bits:
        raise SystemExit(
            f"{opt_name}: HPO completeness check failed ({'; '.join(bits)}). "
            f"Do not delete files. Finish: python hpo_gp_bayesopt.py "
            f"--optimizer {opt_name} --seed 11 --task {task.__name__}"
        )
    return kept


def _seed_path(dest: str, seed: int) -> str:
    return os.path.join(dest, f"seed_{seed}.json")


def _load_valid_seed_row(path: str, opt_name: str, seed: int, task) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        row = json.load(f)
    need = (
        "val_loss",
        "val_acc",
        "compute_calls",
        "wall_s",
        "curve",
        "end_val_loss",
        "end_val_acc",
        "test_loss",
        "test_acc",
        "checkpoint_rule",
        "experiment_name",
        "protocol_early_stop",
    )
    missing = [k for k in need if k not in row]
    if missing:
        return None
    if row.get("checkpoint_rule") != "global_best":
        return None
    if int(row.get("seed", -1)) != int(seed):
        return None
    if row.get("optimizer") != opt_name:
        return None
    try:
        _require_protocol_on_result(row, task)
    except SystemExit:
        return None
    if not row["curve"]:
        return None
    return row


def _all_five_seed_rows_if_complete(dest: str, opt_name: str, task) -> list[dict] | None:
    rows = []
    for seed in TEST_SEEDS:
        row = _load_valid_seed_row(_seed_path(dest, seed), opt_name, seed, task)
        if row is None:
            return None
        rows.append(row)
    return rows


def _write_summary(dest: str, opt_name: str, seed_rows: list[dict], hparams, source_trial, budget, task) -> dict:
    keys = (
        "val_loss",
        "val_acc",
        "compute_calls",
        "wall_s",
        "n_outer_steps",
        "train_loss",
        "train_acc",
        "test_loss",
        "test_acc",
    )
    summary = {
        "optimizer": opt_name,
        "seeds": list(TEST_SEEDS),
        "hparams": hparams,
        "source_trial": source_trial,
        "gpu_name": seed_rows[0].get("gpu_name", "unknown"),
        "budget": budget,
        "checkpoint_rule": "global_best",
        "clip_rate": float(sum(float(r.get("clip_rate", 0.0)) for r in seed_rows) / max(1, len(seed_rows))),
        "n": len(seed_rows),
        "protocol_early_stop": bool(task.PROTOCOL_EARLY_STOP),
        "experiment_name": task.EXPERIMENT_NAME,
    }
    for k in keys:
        vals = [float(r[k]) for r in seed_rows if k in r and r[k] is not None]
        if vals:
            summary[k] = _mean_std(vals)
    for k in ("end_val_loss", "end_val_acc", "end_compute_calls"):
        vals = [float(r[k]) for r in seed_rows if k in r and r[k] is not None]
        summary[k] = _mean_std(vals)
    summary_path = os.path.join(dest, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print(
        f"{opt_name} locked_val_loss={summary['val_loss']['mean']:.6f}"
        f"±{summary['val_loss']['std']:.6f} "
        f"Ng={summary['compute_calls']['mean']:.3f}"
        f"±{summary['compute_calls']['std']:.3f} "
        f"n_outer_steps={summary['n_outer_steps']['mean']:.3f}"
        f"±{summary['n_outer_steps']['std']:.3f}"
    )
    return summary


def run_one_arm(opt_name: str, budget: int, task) -> dict:
    dest = os.path.join(_HERE, "results", f"{opt_name}_five_seed")
    os.makedirs(dest, exist_ok=True)
    hpo_arm = "b25" if opt_name == "b25_plain" else opt_name  # b25_plain reuses the b25 tuning
    trials = load_trial_jsons(hpo_arm, budget)
    trials = _require_complete_hpo_trials(hpo_arm, trials, task)
    best = best_by_val_loss(trials)
    if best is None:
        raise SystemExit(f"no HPO trial JSON for {opt_name}; run HPO first")

    existing = _all_five_seed_rows_if_complete(dest, opt_name, task)
    if existing is not None:
        print(
            f"{opt_name}: 5 valid seed files already in {dest}; skip retrain",
            flush=True,
        )
        return _write_summary(
            dest,
            opt_name,
            existing,
            best["hparams"],
            best.get("trial_number"),
            budget,
            task,
        )

    hps = _hps_from_json(best["hparams"])
    seed_rows = []
    for seed in TEST_SEEDS:
        _prev = _load_valid_seed_row(_seed_path(dest, seed), opt_name, seed, task)
        if _prev is not None and _prev.get("hparams") == best["hparams"]:
            print(f"{opt_name} seed={seed}: valid result file exists, skipping (resume)", flush=True)
            seed_rows.append(_prev)
            continue
        model = task.build_model(seed=seed)
        optimizer = build_optimizer(opt_name, model.parameters(), hps)
        result = task.train_one_trial(
            model,
            optimizer,
            smoke=False,
            seed=seed,
            score_test=True,
            budget=budget,
        )
        locked = task.global_best_checkpoint(result["curve"])
        row = _row_from_locked(
            opt_name, seed, best["hparams"], best.get("trial_number"), result, locked, task
        )
        path = _seed_path(dest, seed)
        with open(path, "w") as f:
            json.dump(row, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        seed_rows.append(row)
        del model, optimizer, result
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        print(
            f"{opt_name} seed={seed} locked_val_loss={row['val_loss']:.6f} "
            f"locked_val_acc={row['val_acc']:.4f} "
            f"end_val_loss={row['end_val_loss']:.6f} "
            f"Ng={row['compute_calls']} wall_s={row['wall_s']:.3f} "
            f"checkpoint_rule=global_best"
        )
    return _write_summary(
        dest,
        opt_name,
        seed_rows,
        best["hparams"],
        best.get("trial_number"),
        budget,
        task,
    )


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="5-seed retrain; report global-best val_loss checkpoint."
    )
    p.add_argument("--optimizer", choices=(*CLI_ALL, "b25_plain", "all"), required=True)
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    p.add_argument("--task", default=DEFAULT_TASK)
    args = p.parse_args(argv)
    if tuple(TEST_SEEDS) != (42, 137, 256, 512, 1024):
        raise RuntimeError("TEST_SEEDS must stay [42, 137, 256, 512, 1024]")
    task = load_task(args.task)
    budget = int(args.budget)
    if budget != int(task.DEFAULT_BUDGET):
        raise SystemExit(
            f"--budget {budget} != task.DEFAULT_BUDGET={task.DEFAULT_BUDGET}"
        )
    names = (list(CLI_ALL) + ["b25_plain"]) if args.optimizer == "all" else [args.optimizer]
    summaries = [run_one_arm(name, budget, task) for name in names]
    out = os.path.join(_HERE, "results", "five_seed_summary.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
