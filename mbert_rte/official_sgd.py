"""Official PyTorch SGD. Protocol SGD+momentum; nesterov locked False."""

from __future__ import annotations

import torch.nn as nn
from torch.optim import SGD

from official_fields import filter_locked, verify_one


def build_sgd(params, lr, momentum, weight_decay):
    verify_one("SGD")
    return SGD(
        params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        **filter_locked(SGD),
    )


if __name__ == "__main__":
    model = nn.Linear(4, 2)
    opt = build_sgd(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)
    g0 = opt.param_groups[0]
    print(type(opt).__name__)
    for k in ("lr", "momentum", "weight_decay", "nesterov", "dampening"):
        print(f"{k}={g0[k]}")
