"""glue_data.py - data layer for the B25 transformer experiments.

Pure data code: no model, no optimizer, no training loop. One spec per task. The split rules are
the ones this project already used:
  sst2 : DistilBERT folder   -> 2000 validation rows held out of train (seed 42); official validation = test
  mrpc : BiLSTM MRPC folder  -> official validation = validation; 10% of train (seed 42) = test
  rte, cola : the same rule as mrpc (small tasks: official validation = validation, 10% of train = test)
Extra metrics per task (logged on every curve point next to loss and accuracy): mrpc = F1, cola = Matthews correlation.
stsb is REGRESSION (one output, mean squared error): the 'accuracy' slot of every curve point then holds the Pearson correlation,
and Spearman is the extra metric. The harness objective stays 'best validation loss' = validation MSE.
Every row is normalised to {"a": text, "b": second text or None, "label": int}.
"""
from __future__ import annotations

import os

import torch

SPLIT_SEED = 42

TASKS = {
    "sst2": {
        "task_type": "classification", "glue": "sst2", "keys": ("sentence", None), "num_labels": 2,
        "split": "holdout_val", "holdout": 2000, "n_train": 67349, "n_official_val": 872, "f1": False, "extra_metrics": (),
    },
    "mrpc": {
        "task_type": "classification", "glue": "mrpc", "keys": ("sentence1", "sentence2"), "num_labels": 2,
        "split": "holdout_test", "holdout": 0.10, "n_train": 3668, "n_official_val": 408, "f1": True, "extra_metrics": ("f1",),
    },
    "rte": {
        "task_type": "classification", "glue": "rte", "keys": ("sentence1", "sentence2"), "num_labels": 2,
        "split": "holdout_test", "holdout": 0.10, "n_train": 2490, "n_official_val": 277, "f1": False, "extra_metrics": (),
    },
    "cola": {
        "task_type": "classification", "glue": "cola", "keys": ("sentence", None), "num_labels": 2,
        "split": "holdout_test", "holdout": 0.10, "n_train": 8551, "n_official_val": 1043, "f1": False, "extra_metrics": ("mcc",),
    },
    "stsb": {
        "task_type": "regression", "glue": "stsb", "keys": ("sentence1", "sentence2"), "num_labels": 1,
        "split": "holdout_test", "holdout": 0.10, "n_train": 5749, "n_official_val": 1500, "f1": False, "extra_metrics": ("spearman",),
    },
}


def spec(task: str) -> dict:
    if task not in TASKS:
        raise ValueError(f"unknown GLUE task {task!r}; enabled: {sorted(TASKS)}")
    return TASKS[task]


def normalize_row(row: dict, keys: tuple, num_labels: int, regression: bool = False) -> dict:
    if "label" not in row:
        raise RuntimeError(f"row has no label: {list(row)}")
    if regression:
        label = float(row["label"])
        if not 0.0 <= label <= 5.0:  # STS-B similarity scores live on 0..5; -1 would be a hidden test label
            raise RuntimeError(f"regression label {label} outside 0..5")
    else:
        label = int(row["label"])
        if not 0 <= label < num_labels:
            raise RuntimeError(f"label {label} outside 0..{num_labels - 1}")
    a = row[keys[0]]
    b = row[keys[1]] if keys[1] is not None else None
    return {"a": str(a), "b": None if b is None else str(b), "label": label}


def load_glue_rows(task: str, cache_dir: str):
    s = spec(task)
    from datasets import load_dataset

    try:
        raw = load_dataset("glue", s["glue"], cache_dir=cache_dir)
    except Exception as first:  # the hub name changed once; same fallback as the audited harness
        print(f"glue/{s['glue']} load failed ({type(first).__name__}: {first}); retry nyu-mll/glue", flush=True)
        raw = load_dataset("nyu-mll/glue", s["glue"], cache_dir=cache_dir)
    reg = s["task_type"] == "regression"
    train = [normalize_row(r, s["keys"], s["num_labels"], reg) for r in raw["train"]]
    val = [normalize_row(r, s["keys"], s["num_labels"], reg) for r in raw["validation"]]
    return train, val


def _held_out_indices(n: int, k: int, seed: int) -> set:
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g).tolist()
    return set(perm[:k])


def split_rows(task: str, train_rows: list, official_val_rows: list, seed: int = SPLIT_SEED):
    """Return (train, val, test)."""
    s = spec(task)
    n = len(train_rows)
    if n != s["n_train"] or len(official_val_rows) != s["n_official_val"]:
        raise RuntimeError(
            f"{task}: expected train/val = {s['n_train']}/{s['n_official_val']}, got {n}/{len(official_val_rows)}"
        )
    if s["split"] == "holdout_val":
        k = int(s["holdout"])
    elif s["split"] == "holdout_test":
        k = int(round(n * float(s["holdout"])))
    else:
        raise ValueError(s["split"])
    if k < 1 or k >= n:
        raise RuntimeError(f"{task}: bad holdout size {k}")
    held = _held_out_indices(n, k, seed)
    keep = [row for i, row in enumerate(train_rows) if i not in held]
    out = [row for i, row in enumerate(train_rows) if i in held]
    if len(out) != k or len(keep) != n - k:
        raise RuntimeError(f"{task}: split sizes keep={len(keep)} held={len(out)}; want {n - k}/{k}")
    if s["split"] == "holdout_val":
        return keep, out, list(official_val_rows)
    return keep, list(official_val_rows), out


def encode_rows(tokenizer, rows: list, max_len: int):
    """Tokenize with static padding. Pair tasks use the tokenizer's native sentence-pair mode."""
    if not rows:
        raise RuntimeError("no rows to encode")
    a = [r["a"] for r in rows]
    bs = [r["b"] for r in rows]
    is_pair = any(b is not None for b in bs)
    if is_pair and not all(b is not None for b in bs):
        raise RuntimeError("mixed single/pair rows")
    kw = dict(padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    enc = tokenizer(a, bs, **kw) if is_pair else tokenizer(a, **kw)
    is_float = isinstance(rows[0]["label"], float)
    labels = torch.tensor([r["label"] for r in rows], dtype=torch.float32 if is_float else torch.long)
    return enc, labels, is_pair


def _as_list(x):
    return x.view(-1).tolist() if hasattr(x, "view") else list(x)


def binary_f1(y_true, y_pred) -> float:
    """Binary F1 for the positive class 1 (identical to the MRPC BiLSTM harness)."""
    yt, yp = _as_list(y_true), _as_list(y_pred)
    tp = sum(1 for t, p in zip(yt, yp) if p == 1 and t == 1)
    fp = sum(1 for t, p in zip(yt, yp) if p == 1 and t == 0)
    fn = sum(1 for t, p in zip(yt, yp) if p == 0 and t == 1)
    if tp == 0 and fp == 0 and fn == 0:
        return 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    if prec + rec == 0.0:
        return 0.0
    return float(2.0 * prec * rec / (prec + rec))


def matthews_corrcoef(y_true, y_pred) -> float:
    """Binary Matthews correlation coefficient (the standard CoLA metric); 0.0 when it is undefined."""
    yt, yp = _as_list(y_true), _as_list(y_pred)
    tp = sum(1 for t, p in zip(yt, yp) if p == 1 and t == 1)
    tn = sum(1 for t, p in zip(yt, yp) if p == 0 and t == 0)
    fp = sum(1 for t, p in zip(yt, yp) if p == 1 and t == 0)
    fn = sum(1 for t, p in zip(yt, yp) if p == 0 and t == 1)
    den = float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
    if den == 0.0:
        return 0.0
    return float((tp * tn - fp * fn) / den ** 0.5)


def pearson(y_true, y_pred) -> float:
    """Pearson correlation; 0.0 when undefined (a constant input)."""
    yt, yp = [float(v) for v in _as_list(y_true)], [float(v) for v in _as_list(y_pred)]
    n = len(yt)
    if n < 2 or n != len(yp):
        return 0.0
    mt, mp = sum(yt) / n, sum(yp) / n
    sxy = sum((a - mt) * (b - mp) for a, b in zip(yt, yp))
    sxx = sum((a - mt) ** 2 for a in yt)
    syy = sum((b - mp) ** 2 for b in yp)
    if sxx == 0.0 or syy == 0.0:
        return 0.0
    return float(sxy / (sxx * syy) ** 0.5)


def _average_ranks(v) -> list:
    """Ranks starting at 1; tied values share the average of the ranks they occupy (what scipy.stats.rankdata does)."""
    vals = [float(x) for x in v]
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(y_true, y_pred) -> float:
    """Spearman rank correlation (Pearson of the average ranks); 0.0 when undefined."""
    return pearson(_average_ranks(_as_list(y_true)), _average_ranks(_as_list(y_pred)))


EXTRA_METRICS = {"f1": binary_f1, "mcc": matthews_corrcoef, "pearson": pearson, "spearman": spearman}
