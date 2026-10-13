"""Official PyTorch NAdam. Protocol NAdam; momentum_decay locked 0.004;
decoupled_weight_decay locked False if present."""

from __future__ import annotations

import torch.nn as nn
from torch.optim import NAdam

from official_fields import filter_locked, verify_one


def build_nadam(params, lr, betas, weight_decay):
    verify_one("NAdam")
    return NAdam(
        params,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        **filter_locked(NAdam),
    )


if __name__ == "__main__":
    model = nn.Linear(4, 2)
    opt = build_nadam(
        model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.0
    )
    g0 = opt.param_groups[0]
    print(type(opt).__name__)
    for k in ("lr", "betas", "weight_decay", "eps", "momentum_decay"):
        print(f"{k}={g0[k]}")
    if "decoupled_weight_decay" in g0:
        print(f"decoupled_weight_decay={g0['decoupled_weight_decay']}")
