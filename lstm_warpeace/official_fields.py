"""Canonical SEARCHED vs LOCKED constructor-field tables for five official
PyTorch Tier-1 optimizers (SGD, Adam, AdamW, RMSprop, NAdam).

This file is DistilBERT SST-2 fine-tune only. Narrower LR / beta ranges
than from-scratch experiments. Do not copy these bounds into mnist_mlp,
bert_mini, or CIFAR. WD list and jump_size stay the suite defaults.

Source of constructors: torch.optim SGD, Adam, AdamW, RMSprop, NAdam only.
No custom optimizer math. No gradient clipping. No Jumping. No cosine.

carps (Benjamins et al. 2025, arXiv:2506.06143v2) is CITE-ONLY inventory #35
— HPO framework, not a training optimizer; n_trials ≠ Ng.
"""

from __future__ import annotations

import inspect

from torch.optim import SGD, Adam, AdamW, RMSprop, NAdam

# DistilBERT SST-2 fine-tune box (this folder only).
LR_LO = 1e-6
LR_HI = 3e-3
MOMENTUM_LO = 0.1
MOMENTUM_HI = 0.99
BETA1_LO = 0.80
BETA1_HI = 0.9995
BETA2_LO = 0.9
BETA2_HI = 0.99999
# Box 5': shared by ALL transformer experiments (every model, every dataset, every arm).
# Range (not the optimum) around the tuned values of all five baselines found with the original box:
#   lr 2.6e-5..5.0e-4 | beta1 0.85 and 0.999 | beta2 0.9999 | momentum 0.969 | wd 0..1e-4.
# Edges moved only where a best value sat ON an edge (beta1, beta2) or close to it (SGD lr).
SPACE_VERSION = "box5prime-v1"
WD_CHOICES = (0.0, 1e-4, 5e-4, 0.001, 0.01, 0.05, 0.1, 0.3, 5e-5, 1e-5, 5e-6, 1e-6)  # 12 values; unchanged suite list

OPTIMIZERS = {
    "SGD": SGD,
    "Adam": Adam,
    "AdamW": AdamW,
    "RMSprop": RMSprop,
    "NAdam": NAdam,
}

# Field name -> how BayesOpt draws it. Keys must appear in some SEARCHED row.
SPACE = {
    "lr": ("log_uniform", LR_LO, LR_HI),
    "momentum": ("uniform", MOMENTUM_LO, MOMENTUM_HI),
    "weight_decay": ("categorical", WD_CHOICES),
    "betas": ("uniform_pair", (BETA1_LO, BETA1_HI), (BETA2_LO, BETA2_HI)),
}

# RMSProp momentum is LOCKED 0 (Hinton/torch default) — not in SEARCHED.
SEARCHED = {
    "SGD": ("lr", "momentum", "weight_decay"),
    "Adam": ("lr", "betas", "weight_decay"),
    "AdamW": ("lr", "betas", "weight_decay"),
    "RMSprop": ("lr", "weight_decay"),
    "NAdam": ("lr", "betas", "weight_decay"),
}

# Frozen constructor kwargs. SGD is momentum-SGD (nesterov False), not Nesterov.
# AdamW official default wd=0.01 is NOT frozen — wd is SEARCHED.
# Adam decoupled_weight_decay=False keeps Adam ≠ AdamW (dropped if absent).
LOCKED = {
    "SGD": {
        "dampening": 0,
        "nesterov": False,
        "maximize": False,
        "foreach": None,
        "differentiable": False,
        "fused": None,
    },
    "Adam": {
        "eps": 1e-8,
        "amsgrad": False,
        "foreach": None,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "decoupled_weight_decay": False,
    },
    "AdamW": {
        "eps": 1e-8,
        "amsgrad": False,
        "foreach": None,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
        "fused": None,
    },
    "RMSprop": {
        "alpha": 0.99,
        "eps": 1e-8,
        "momentum": 0,
        "centered": False,
        "capturable": False,
        "foreach": None,
        "maximize": False,
        "differentiable": False,
    },
    "NAdam": {
        "eps": 1e-8,
        "momentum_decay": 0.004,
        "decoupled_weight_decay": False,
        "foreach": None,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
    },
}

_SKIP = frozenset({"self", "params"})


def official_param_names(cls):
    """Constructor names on `cls` except self/params (and *args/**kwargs)."""
    names = set()
    for p in inspect.signature(cls).parameters.values():
        if p.name in _SKIP:
            continue
        if p.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        names.add(p.name)
    return names


def filter_locked(cls, locked=None):
    """Drop LOCKED keys this torch class does not accept. Never pass unofficial fields."""
    if locked is None:
        locked = LOCKED[cls.__name__]
    names = official_param_names(cls)
    return {k: v for k, v in locked.items() if k in names}


def verify_one(name, cls=None):
    """Assert SEARCHED/LOCKED cover the real class; SEARCHED extras fail.

    Extra LOCKED keys absent from this torch are OK (intersection / filter).
    Extra SEARCHED keys this torch does not have = fail.
    """
    if cls is None:
        cls = OPTIMIZERS[name]
    official = official_param_names(cls)
    searched = set(SEARCHED[name])
    locked = set(LOCKED[name])

    extra_searched = searched - official
    if extra_searched:
        raise AssertionError(
            f"{name}: SEARCHED keys not accepted by {cls.__name__}: "
            f"{sorted(extra_searched)}"
        )

    overlap = searched & locked
    if overlap:
        raise AssertionError(
            f"{name}: keys in both SEARCHED and LOCKED: {sorted(overlap)}"
        )

    locked_ok = locked & official
    missing = official - (searched | locked_ok)
    if missing:
        raise AssertionError(
            f"{name}: official fields missing from SEARCHED/LOCKED: "
            f"{sorted(missing)}"
        )
    return True


def verify_all():
    """Run verify_one on every Tier-1 class. LOCKED extras filtered, not failed."""
    return {name: verify_one(name, cls) for name, cls in OPTIMIZERS.items()}


if __name__ == "__main__":
    print(verify_all())
