"""Retrain the best COMPLETE tuning trial of each arm on K seeds. Same per-seed procedure as retrain_five_seeds.py, but K and the trial count are arguments."""
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
