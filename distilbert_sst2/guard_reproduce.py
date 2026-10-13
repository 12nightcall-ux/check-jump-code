"""Reproduction guard. Runs this folder with the OLD settings (dropout on for AdamW, no clip) and
compares with the stored AdamW seed-42 run of the original distilbert_sst2 folder."""
import json
import os
import sys

os.environ["B25_DROPOUT_MODE"] = "official"
os.environ["B25_CLIP"] = "none"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ORIG = os.environ.get(
    "B25_GUARD_ORIG",
    "/content/drive/MyDrive/Optimiser Formula Creation - Working/experiments/distilbert_sst2",
)

import task_distilbert_sst2 as task
from official_adamw import build_adamw

assert task.DROPOUT_MODE == "official" and task.CLIP_NORM is None, "guard needs the old settings"
summary = json.load(open(os.path.join(ORIG, "results", "adamw_five_seed", "summary.json")))
stored = json.load(open(os.path.join(ORIG, "results", "adamw_five_seed", "seed_42.json")))
hp = summary["hparams"]
print("stored hparams:", hp, "| stored gpu:", stored.get("gpu_name"), "| this gpu:", task.gpu_name(), flush=True)
model = task.build_model(seed=42)
opt = build_adamw(model.parameters(), hp["lr"], tuple(hp["betas"]), hp["weight_decay"])
result = task.train_one_trial(
    model, opt, smoke=False, seed=42, score_test=True, budget=int(task.DEFAULT_BUDGET)
)
gb = task.global_best_checkpoint(result["curve"])
diff = abs(float(gb["val_loss"]) - float(stored["val_loss"]))
sc = {round(float(p["compute_calls"]), 3): float(p["val_loss"]) for p in stored["curve"]}
rc = {round(float(p["compute_calls"]), 3): float(p["val_loss"]) for p in result["curve"]}
common = sorted(set(sc) & set(rc))
cd = [abs(sc[k] - rc[k]) for k in common]
print(
    "GUARD stored_best_val=%.6f reproduced_best_val=%.6f abs_diff=%.6f"
    % (float(stored["val_loss"]), float(gb["val_loss"]), diff),
    flush=True,
)
print(
    "GUARD curve points compared=%d max_abs_diff=%.6f first5=%s"
    % (len(common), max(cd) if cd else float("nan"), [round(x, 5) for x in cd[:5]]),
    flush=True,
)
if diff <= 2e-3:
    print("GUARD PASS")
elif diff <= 1e-2:
    print("GUARD WARN (small drift; compare the GPU names above)")
else:
    print("GUARD FAIL")
