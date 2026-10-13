"""Quick check of the B25 arm inside the real harness: ~200 compute units, prints the jump counters."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import task_glue as task
from b25_builder import build_b25

model = task.build_model(seed=11)
opt = build_b25(model.parameters(), 2e-5, (0.9, 0.999), 0.01)
res = task.train_one_trial(model, opt, smoke=False, seed=11, n_steps=200)
s = res.get("b25_stats")
print("SMOKE_B25 stats:", s, flush=True)
print(
    "SMOKE_B25 compute_calls=%.1f budget=%.1f clip_rate=%.3f best_val=%.4f"
    % (
        res["compute_calls"],
        res["budget"],
        res["clip_rate"],
        min(float(p["val_loss"]) for p in res["curve"]),
    ),
    flush=True,
)
ok = (
    s is not None
    and s["attempts"] >= 3
    and res["compute_calls"] <= res["budget"] + 1e-9
    and s["undo_coverage_ok"]
)
print("SMOKE_B25 PASS" if ok else "SMOKE_B25 CHECK - paste me the lines above")
