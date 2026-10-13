"""Post-hoc Ng/time-to-target: each Jumping arm vs each official target.

Not HPO. Five official targets stay separate. Ten crossing rows.
Official target = mean of the 5-seed global-best val_loss checkpoints
(from results/<opt>_five_seed/, not HPO trial JSON).
Each Jumping seed's logged curve is checked against that mean
(no interpolation; CURVE_INTERVAL=25 Ng points only).

Headline ng_to_loss / time_to_loss fields are seed 42 (suite shape).
jumper_seed_hits lists all 5 seeds.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os

from hpo_gp_bayesopt import (
    CLI_JUMPING,
    CLI_OPTIMIZERS,
    DEFAULT_BUDGET,
    DEFAULT_TASK,
)

CHECKPOINT_RULE = "global_best"
TEST_SEEDS = (42, 137, 256, 512, 1024)
_HERE = os.path.dirname(os.path.abspath(__file__))


def load_task(module_name: str):
    return importlib.import_module(module_name)


def _refuse_mixed_protocol(run: dict, opt_name: str, task) -> None:
    flag = run.get("protocol_early_stop")
    exp = run.get("experiment_name")
    if flag is None or exp is None:
        raise SystemExit(
            f"{opt_name}: missing protocol_early_stop or experiment_name. "
            "Re-run 5-seed in this DistilBERT SST-2 folder."
        )
    if bool(flag) != bool(task.PROTOCOL_EARLY_STOP):
        raise SystemExit(
            f"{opt_name}: protocol_early_stop={flag} != {task.PROTOCOL_EARLY_STOP}."
        )
    if exp != task.EXPERIMENT_NAME:
        raise SystemExit(
            f"{opt_name}: experiment_name={exp!r} != {task.EXPERIMENT_NAME!r}."
        )


def _trial_dir(opt_name: str) -> str:
    return os.path.join(_HERE, "results", f"{opt_name}_trials")


def _five_seed_dir(opt_name: str) -> str:
    return os.path.join(_HERE, "results", f"{opt_name}_five_seed")


def load_trial_jsons(opt_name: str, budget: int = DEFAULT_BUDGET) -> list[dict]:
    del budget
    folder = _trial_dir(opt_name)
    if not os.path.isdir(folder):
        return []
    rows = []
    for name in sorted(os.listdir(folder)):
        if not name.startswith("trial_") or not name.endswith(".json"):
            continue
        path = os.path.join(folder, name)
        with open(path) as f:
            rows.append(json.load(f))
    return rows


_BEST_REQUIRED = ("trial_number", "hparams", "optimizer", "curve")


def _hpo_score(trial: dict) -> float:
    if "hpo_objective_val_loss" in trial:
        return float(trial["hpo_objective_val_loss"])
    return float(trial["val_loss"])


def best_by_val_loss(trials: list[dict]) -> dict | None:
    """Lowest HPO objective. Never last_val_loss."""
    if not trials:
        return None
    for t in trials:
        missing = [k for k in _BEST_REQUIRED if k not in t]
        if missing:
            raise SystemExit(
                f"trial {t.get('trial_number', '?')} missing fields: {missing}"
            )
        if "hpo_objective_val_loss" not in t and "val_loss" not in t:
            raise SystemExit(
                f"trial {t.get('trial_number', '?')} missing "
                "hpo_objective_val_loss and val_loss"
            )
    finite = [t for t in trials if math.isfinite(_hpo_score(t))]
    if not finite:
        raise SystemExit(
            "no trial with finite hpo_objective_val_loss; cannot pick HPO best"
        )
    return min(finite, key=lambda t: (_hpo_score(t), int(t["trial_number"])))


def _load_five_seed_one(opt_name: str, seed: int) -> dict | None:
    path = os.path.join(_five_seed_dir(opt_name), f"seed_{seed}.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_five_seed_runs(opt_name: str, task) -> list[dict]:
    rows = []
    missing = []
    for seed in TEST_SEEDS:
        run = _load_five_seed_one(opt_name, seed)
        if run is None:
            missing.append(seed)
            continue
        _refuse_mixed_protocol(run, opt_name, task)
        rows.append(run)
    if missing:
        raise SystemExit(
            f"missing results/{opt_name}_five_seed/seed_{{s}}.json for "
            f"seeds {missing}. Run first: python retrain_five_seeds.py "
            f"--optimizer {opt_name} --task {task.__name__}"
        )
    if len(rows) != len(TEST_SEEDS):
        raise SystemExit(f"{opt_name}: expected {len(TEST_SEEDS)} seed files")
    return rows


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs))


def _official_mean_target(runs: list[dict], task) -> dict:
    locked = [task.global_best_checkpoint(r["curve"]) for r in runs]
    return {
        "target_val_loss": _mean([float(p["val_loss"]) for p in locked]),
        "target_val_acc": _mean([float(p["val_acc"]) for p in locked]),
        "official_ng": _mean([float(p["compute_calls"]) for p in locked]),
        "official_wall_s": _mean([float(p["wall_s"]) for p in locked]),
    }


def _crossing_point(curve, threshold, *, y_key, higher_is_better):
    """First Ng-sorted logged point that crosses threshold, or None."""
    pts = sorted(curve, key=lambda p: float(p["compute_calls"]))
    for p in pts:
        y = float(p[y_key])
        hit = y >= threshold if higher_is_better else y <= threshold
        if hit:
            return p
    return None


def _closest_point(curve, threshold, *, y_key, higher_is_better):
    pts = sorted(curve, key=lambda p: float(p["compute_calls"]))
    if not pts:
        return None, None

    def gap(y: float) -> float:
        return threshold - y if higher_is_better else y - threshold

    best = min(pts, key=lambda p: abs(gap(float(p[y_key]))))
    return best, gap(float(best[y_key]))


def _hits_from_ng_point(point, near, near_gap):
    if point is not None:
        return (
            {"status": "CROSSED", "value": float(point["compute_calls"])},
            {"status": "CROSSED", "value": float(point["wall_s"])},
        )
    if near is None:
        empty = {"status": "NO_CROSS", "closest_value": None, "closest_gap": None}
        return empty, dict(empty)
    return (
        {
            "status": "NO_CROSS",
            "closest_value": float(near["compute_calls"]),
            "closest_gap": near_gap,
        },
        {
            "status": "NO_CROSS",
            "closest_value": float(near["wall_s"]),
            "closest_gap": near_gap,
        },
    )


def _pct_saved(jumper_x, official_x):
    if official_x is None or float(official_x) == 0.0:
        return None
    if jumper_x is None:
        return None
    return (1.0 - float(jumper_x) / float(official_x)) * 100.0


def _fmt_hit(hit: dict | None) -> str:
    if not hit:
        return "n/a"
    if hit.get("status") == "CROSSED":
        return f"CROSSED {hit['value']}"
    return (
        f"NO_CROSS closest={hit.get('closest_value')} "
        f"gap={hit.get('closest_gap')}"
    )


def _one_seed_cross(curve, thr_loss, thr_acc) -> dict:
    hit_pt = _crossing_point(
        curve, thr_loss, y_key="val_loss", higher_is_better=False
    )
    near_pt, near_gap = (None, None)
    if hit_pt is None:
        near_pt, near_gap = _closest_point(
            curve, thr_loss, y_key="val_loss", higher_is_better=False
        )
    ng_hit, time_hit = _hits_from_ng_point(hit_pt, near_pt, near_gap)
    acc_pt = _crossing_point(
        curve, thr_acc, y_key="val_acc", higher_is_better=True
    )
    acc_near, acc_gap = (None, None)
    if acc_pt is None:
        acc_near, acc_gap = _closest_point(
            curve, thr_acc, y_key="val_acc", higher_is_better=True
        )
    ng_acc, time_acc = _hits_from_ng_point(acc_pt, acc_near, acc_gap)
    return {
        "ng_to_loss": ng_hit,
        "time_to_loss": time_hit,
        "ng_to_acc": ng_acc,
        "time_to_acc": time_acc,
        "jumper_ng": None if hit_pt is None else float(hit_pt["compute_calls"]),
        "jumper_wall_s": None if hit_pt is None else float(hit_pt["wall_s"]),
        "jumper_val_loss_at_cross": (
            None if hit_pt is None else float(hit_pt["val_loss"])
        ),
        "jumper_val_acc_at_cross": (
            None if hit_pt is None else float(hit_pt["val_acc"])
        ),
    }


def one_crossing(official_runs: list[dict], jumping_runs: list[dict], task) -> dict:
    tgt = _official_mean_target(official_runs, task)
    thr_loss = tgt["target_val_loss"]
    thr_acc = tgt["target_val_acc"]
    seed_hits = []
    headline = None
    n_crossed = 0
    for jrun in jumping_runs:
        seed = int(jrun["seed"])
        hit = _one_seed_cross(jrun["curve"], thr_loss, thr_acc)
        hit["seed"] = seed
        seed_hits.append(hit)
        if hit["ng_to_loss"].get("status") == "CROSSED":
            n_crossed += 1
        if seed == 42:
            headline = hit
    if headline is None:
        raise SystemExit("seed 42 missing from Jumping 5-seed files")
    jumper_ng = headline["jumper_ng"]
    jumper_wall = headline["jumper_wall_s"]
    official_ng = tgt["official_ng"]
    official_wall = tgt["official_wall_s"]
    return {
        "target_optimizer": official_runs[0].get("optimizer"),
        "jumper": jumping_runs[0].get("optimizer"),
        "checkpoint_rule": CHECKPOINT_RULE,
        "target_val_loss": thr_loss,
        "target_val_acc": thr_acc,
        "official_ng": official_ng,
        "official_wall_s": official_wall,
        "ng_to_loss": headline["ng_to_loss"],
        "time_to_loss": headline["time_to_loss"],
        "jumper_ng": jumper_ng,
        "jumper_wall_s": jumper_wall,
        "jumper_val_loss_at_cross": headline["jumper_val_loss_at_cross"],
        "jumper_val_acc_at_cross": headline["jumper_val_acc_at_cross"],
        "compute_saved_pct": _pct_saved(jumper_ng, official_ng),
        "time_saved_pct": _pct_saved(jumper_wall, official_wall),
        "ng_to_acc": headline["ng_to_acc"],
        "time_to_acc": headline["time_to_acc"],
        "n_seeds_crossed": n_crossed,
        "n_seeds": len(TEST_SEEDS),
        "jumper_seed_hits": seed_hits,
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="10 crossings: 2 Jumping × 5 officials.")
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    p.add_argument("--task", default=DEFAULT_TASK)
    args = p.parse_args(argv)
    task = load_task(args.task)
    budget = int(args.budget)
    if budget != int(task.DEFAULT_BUDGET):
        raise SystemExit(
            f"--budget {budget} != task.DEFAULT_BUDGET={task.DEFAULT_BUDGET}"
        )
    official_runs = {
        name: load_five_seed_runs(name, task) for name in CLI_OPTIMIZERS
    }
    jumpers = {name: load_five_seed_runs(name, task) for name in CLI_JUMPING}
    rows = []
    for jname, jruns in jumpers.items():
        for oname, oruns in official_runs.items():
            row = one_crossing(oruns, jruns, task)
            rows.append(row)
            print(
                f"{jname} vs {oname}: "
                f"target_loss={row['target_val_loss']:.6f} "
                f"Ng={_fmt_hit(row['ng_to_loss'])} "
                f"wall_s={_fmt_hit(row['time_to_loss'])} "
                f"n_seeds_crossed={row['n_seeds_crossed']}/{row['n_seeds']} "
                f"compute_saved_pct={row['compute_saved_pct']} "
                f"time_saved_pct={row['time_saved_pct']}"
            )
    output = {
        "note": (
            "Ten rows: JumpingAdam and JumpingSGD each vs sgd/adam/adamw/"
            "rmsprop/nadam. Official target is the mean of 5-seed "
            "global-best val_loss (lowest logged val_loss on each completed "
            "5-seed run), not a pooled HPO best and not last-at-stop. "
            "Each Jumping seed is checked on logged CURVE_INTERVAL points "
            "(no interpolation). Headline ng_to_loss is seed 42. "
            "compute_saved_pct = 100 * (1 - Jumping_Ng / Official_Ng). "
            "time_saved_pct = 100 * (1 - Jumping_wall / Official_wall)."
        ),
        "budget": budget,
        "checkpoint_rule": CHECKPOINT_RULE,
        "task": args.task,
        "experiment_name": task.EXPERIMENT_NAME,
        "rows": rows,
    }
    ttt_path = os.path.join(_HERE, "results", "time_to_target.json")
    cross_path = os.path.join(
        _HERE, "crossing_results", f"budget_{budget}.json"
    )
    os.makedirs(os.path.dirname(ttt_path), exist_ok=True)
    os.makedirs(os.path.dirname(cross_path), exist_ok=True)
    for path in (ttt_path, cross_path):
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"wrote {path}")
    if len(rows) != 10 and official_runs and jumpers:
        print(
            f"warning: expected 10 crossing rows, got {len(rows)} "
            f"(need 5 officials and 2 jumpers present)"
        )


if __name__ == "__main__":
    main()
