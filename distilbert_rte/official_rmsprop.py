"""Official PyTorch RMSprop. Protocol RMSprop; momentum locked 0."""

from __future__ import annotations

import torch.nn as nn
from torch.optim import RMSprop

from official_fields import filter_locked, verify_one


def build_rmsprop(params, lr, weight_decay):
    verify_one("RMSprop")
    return RMSprop(
        params,
        lr=lr,
        weight_decay=weight_decay,
        **filter_locked(RMSprop),
    )


if __name__ == "__main__":
    model = nn.Linear(4, 2)
    opt = build_rmsprop(model.parameters(), lr=0.01, weight_decay=0.0)
    g0 = opt.param_groups[0]
    print(type(opt).__name__)
    for k in ("lr", "weight_decay", "momentum", "alpha", "centered"):
        print(f"{k}={g0[k]}")
