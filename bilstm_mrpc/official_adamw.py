"""Official PyTorch AdamW. Protocol AdamW; weight_decay searched, not frozen."""

from __future__ import annotations

import torch.nn as nn
from torch.optim import AdamW

from official_fields import filter_locked, verify_one


def build_adamw(params, lr, betas, weight_decay):
    verify_one("AdamW")
    return AdamW(
        params,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        **filter_locked(AdamW),
    )


if __name__ == "__main__":
    model = nn.Linear(4, 2)
    opt = build_adamw(
        model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.0
    )
    g0 = opt.param_groups[0]
    print(type(opt).__name__)
    for k in ("lr", "betas", "weight_decay", "eps", "amsgrad"):
        print(f"{k}={g0[k]}")
