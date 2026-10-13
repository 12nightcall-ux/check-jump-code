"""Tabular benchmarks (California Housing, Adult, Covertype) x (MLP, FT-Transformer). Stop = billed Ng budget.

Derived from the audited experiments/distilbert_sst2/task_distilbert_sst2.py. Everything that trains, bills, clips,
evaluates and logs is IDENTICAL to that file (verified function by function when this file was built); the regression
evaluation is the same twin that ran on STS-B (mean squared error and Pearson r, with Spearman as an extra). What changed:
  * the data: California Housing (regression, 20,640 rows), Adult (binary classification, 48,842 rows) and a fixed stratified
    100,000-row subsample of Covertype (7 classes). One fixed 70 / 15 / 15 train / validation / test split (seed 42, stratified
    for classification). Numeric columns are standardised and categorical columns one-hot encoded with statistics fitted on the
    TRAIN rows only; the regression target is standardised with the train mean and standard deviation, so the loss is a mean
    squared error in standardised units (R squared is about 1 minus that loss). Validation selects checkpoints and drives the
    tuning; the test rows are never used for either. The prepared table is FROZEN on its first use (data/frozen_<task>.npz with a
    SHA-256 check), so every arm and every resumed session trains on exactly the same arrays.
  * the model: an MLP (3 hidden layers of 256, ReLU) or an FT-Transformer-style network (a linear token per feature plus a
    CLS token, 3 pre-norm blocks, 8 heads of 24, d = 192, ReGLU feed-forward), trained FROM SCRATCH (random initialisation,
    every weight trained, nothing frozen). Dropout is 0 for every arm (the same dropout-off protocol as the other experiments).
  * a "batch" is a set of row ids (BATCH_SIZE = 256, the usual size for tabular data; the NLP experiments use 32): the model
    holds the whole table and returns the outputs of those rows, so the audited loop, which only passes input_ids,
    attention_mask and labels, is untouched.

Stop is compute_calls <= DEFAULT_BUDGET (1000 Ng), 20 Ng if smoke. Gradient clip: CLIP_NORM (default 1.0).
HPO objective is global_best_checkpoint val_loss (scored in hpo_gp_bayesopt.py).
"""

from __future__ import annotations

import inspect
import os
import random
import time
import hashlib
import math
import types

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import numpy as np

import glue_data  # metrics only (F1 / Matthews / Pearson / Spearman); the tabular tasks are described by TAB_SPEC below

DEFAULT_BUDGET = 1000
SMOKE_BUDGET = 20  # Jumping warmup is 15 Ng; 10 Ng smoke never finishes warmup
CURVE_INTERVAL = 25
EXPERIMENT_NAME = "ftt_covertype"
PROTOCOL_EARLY_STOP = False
WORST_BISECTION_TRIES = 4
BISECTION_TRY_NG = 2.3
MODEL_NAME = "ftt"
TAB_TASK = "covertype"  # key of TAB_SPEC
MAX_LEN = 0  # unused (tables have no sequences); the audited loop prints it
BATCH_SIZE = 256  # rows per step: the usual size for tabular data (the NLP experiments use 32)
SPLIT_SEED = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.15
COVERTYPE_ROWS = 100_000  # fixed stratified subsample of the 581,012 rows
MLP_WIDTH = 256
FTT_DIM = 192
FTT_HEADS = 8
FTT_BLOCKS = 3
FTT_FF = 256  # width of the ReGLU feed-forward (about 4/3 of the token size)
# task_type, number of outputs (1 = regression), extra metrics (names in glue_data.EXTRA_METRICS), raw table shape
TAB_SPEC = {
    "california": {"task_type": "regression", "num_labels": 1, "extra_metrics": ("spearman",), "rows": 20640, "raw_features": 8},
    "adult": {"task_type": "classification", "num_labels": 2, "extra_metrics": ("f1",), "rows": 48842, "raw_features": 14},
    "covertype": {"task_type": "classification", "num_labels": 7, "extra_metrics": (), "rows": 100_000, "raw_features": 54},
}
NUM_LABELS = TAB_SPEC[TAB_TASK]["num_labels"]

# --- JumpingAdamSimpsonB25 experiment settings (the env overrides exist only for the reproduction guard) ---
DROPOUT_MODE = os.environ.get("B25_DROPOUT_MODE", "eval_all")  # "eval_all" = dropout off for EVERY optimizer; "official" = old behaviour
_CLIP_ENV = os.environ.get("B25_CLIP", "1.0")
CLIP_NORM = None if _CLIP_ENV.strip().lower() in ("none", "off", "") else float(_CLIP_ENV)
assert DROPOUT_MODE in ("eval_all", "official"), DROPOUT_MODE

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE = None


class _TokenDS(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return int(self.labels.size(0))

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def _seed_torch(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(int(seed))
    except ImportError:
        pass


def _data_dir() -> str:
    here_data = os.path.join(_HERE, "data")
    os.makedirs(here_data, exist_ok=True)
    return here_data










def _split_indices(y, task_type: str, seed: int):
    """Fixed 70 / 15 / 15 train / validation / test split of the row ids; stratified by class for classification."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    strat = y if task_type == "classification" else None
    rest, test = train_test_split(idx, test_size=TEST_FRAC, random_state=seed, stratify=strat)
    train, val = train_test_split(
        rest,
        test_size=VAL_FRAC / (1.0 - TEST_FRAC),
        random_state=seed,
        stratify=(y[rest] if strat is not None else None),
    )
    return np.sort(train), np.sort(val), np.sort(test)


def _prepare_table(frame, y, task_type: str, seed: int):
    """Features from a pandas frame with every statistic fitted on the TRAIN rows only.

    Returns (float32 features, targets, (train, val, test) row ids, regression-target statistics).
    """
    import pandas as pd

    train, val, test = _split_indices(y, task_type, seed)
    num_cols = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    cat_cols = [c for c in frame.columns if c not in num_cols]
    parts = []
    if num_cols:
        xn = frame[num_cols].to_numpy(dtype=np.float64)
        if np.isnan(xn).any():
            raise RuntimeError("missing numeric values are not expected in these datasets")
        mu = xn[train].mean(axis=0)
        sd = xn[train].std(axis=0)
        sd[sd == 0] = 1.0
        parts.append((xn - mu) / sd)
    for c in cat_cols:
        s = frame[c].astype(object).where(frame[c].notna(), "missing").astype(str).to_numpy()
        cats = sorted(set(s[train]))  # categories seen in the train rows; an unseen category becomes all zeros
        parts.append(np.stack([s == k for k in cats], axis=1).astype(np.float64))
    x = np.concatenate(parts, axis=1).astype(np.float32)
    stats = {}
    if task_type == "regression":
        t = np.asarray(y, dtype=np.float64)
        stats = {"target_mean": float(t[train].mean()), "target_std": float(t[train].std())}
        target = ((t - stats["target_mean"]) / stats["target_std"]).astype(np.float32)
    else:
        target = np.asarray(y, dtype=np.int64)
    return x, target, (train, val, test), stats


def _load_table(name: str):
    """(pandas feature frame, raw target) of a dataset; the downloads are cached in the experiment's data folder."""
    home = os.path.join(_data_dir(), "sk")
    if name == "california":
        from sklearn.datasets import fetch_california_housing

        d = fetch_california_housing(data_home=home, as_frame=True)
        return d.data, d.target.to_numpy(dtype=np.float64)
    if name == "adult":
        from sklearn.datasets import fetch_openml

        d = fetch_openml("adult", version=2, as_frame=True, data_home=home, parser="auto")
        labels = d.target.astype(str).str.strip()
        if set(labels.unique()) != {">50K", "<=50K"}:
            raise RuntimeError(f"unexpected Adult labels {sorted(labels.unique())}")
        return d.data, (labels == ">50K").to_numpy(dtype=np.int64)
    if name == "covertype":
        from sklearn.datasets import fetch_covtype
        from sklearn.model_selection import train_test_split

        d = fetch_covtype(data_home=home, as_frame=True)
        y = d.target.to_numpy(dtype=np.int64) - 1
        keep, _ = train_test_split(np.arange(len(y)), train_size=COVERTYPE_ROWS, random_state=SPLIT_SEED, stratify=y)
        keep = np.sort(keep)
        return d.data.iloc[keep].reset_index(drop=True), y[keep]
    raise ValueError(name)


def _table_sha16(x, target, tr, va, te) -> str:
    h = hashlib.sha256()
    for arr in (x, target, tr, va, te):
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:16]


def _datasets():
    global _CACHE
    if _CACHE is None:
        spec = TAB_SPEC[TAB_TASK]
        path = os.path.join(_data_dir(), f"frozen_{TAB_TASK}.npz")
        if os.path.exists(path):  # frozen on the first use: every arm and every resumed session reads the SAME arrays
            z = np.load(path, allow_pickle=False)
            x, target, tr, va, te = z["x"], z["target"], z["train"], z["val"], z["test"]
            stats = {"target_mean": float(z["target_mean"]), "target_std": float(z["target_std"])} if spec["task_type"] == "regression" else {}
            sha = _table_sha16(x, target, tr, va, te)
            if sha != str(z["sha16"]):
                raise RuntimeError(f"{path} is corrupt or was altered: hash {sha} != {str(z['sha16'])}")
            print(f"data: loaded the frozen table {os.path.basename(path)} (sha16 {sha})", flush=True)
        else:
            frame, y = _load_table(TAB_TASK)
            if (len(frame), int(frame.shape[1])) != (spec["rows"], spec["raw_features"]):
                raise RuntimeError(f"{TAB_TASK}: unexpected table shape {tuple(frame.shape)}")
            x, target, (tr, va, te), stats = _prepare_table(frame, y, spec["task_type"], SPLIT_SEED)
            sha = _table_sha16(x, target, tr, va, te)
            tmp = path + ".tmp.npz"
            np.savez(tmp, x=x, target=target, train=tr, val=va, test=te, sha16=np.array(sha),
                     target_mean=np.array(stats.get("target_mean", 0.0)), target_std=np.array(stats.get("target_std", 1.0)))
            os.replace(tmp, path)
            print(f"data: built and froze the table {os.path.basename(path)} (sha16 {sha})", flush=True)
        if x.shape[0] != spec["rows"] or len(tr) + len(va) + len(te) != spec["rows"]:
            raise RuntimeError("the split does not cover the table exactly")
        if len(set(tr.tolist()) & set(va.tolist())) or len(set(va.tolist()) & set(te.tolist())) or len(set(tr.tolist()) & set(te.tolist())):
            raise RuntimeError("train, validation and test rows overlap")
        if spec["task_type"] == "classification" and int(target.max()) + 1 != NUM_LABELS:
            raise RuntimeError(f"{TAB_TASK}: expected {NUM_LABELS} classes, found {int(target.max()) + 1}")
        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(target)

        def make(rows):
            ids = torch.from_numpy(rows.astype(np.int64))
            return _TokenDS({"input_ids": ids, "attention_mask": torch.ones_like(ids)}, y_t[ids])

        _CACHE = {
            "train": make(tr),
            "val": make(va),
            "test": make(te),
            "x": x_t,
            "num_features": int(x.shape[1]),
            "stats": stats,
            "data_sha16": sha,
        }
    return _CACHE


class _FTBlock(nn.Module):
    """One pre-norm Transformer block of the FT-Transformer: multi-head attention, then a ReGLU feed-forward; no dropout."""

    def __init__(self, d: int, heads: int, ff: int):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, 2 * ff)
        self.f2 = nn.Linear(ff, d)

    def forward(self, h):
        b, t, d = h.shape
        q, k, v = self.qkv(self.n1(h)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        att = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(d // self.heads), dim=-1)  # explicit attention: identical numerics in every mode, no kernel-dependent choice
        a = (att @ v).transpose(1, 2).reshape(b, t, d)
        h = h + self.proj(a)
        u, g = self.f1(self.n2(h)).chunk(2, dim=-1)
        return h + self.f2(u * torch.relu(g))


class TabularNet(nn.Module):
    """MLP or FT-Transformer with the call signature of a Hugging Face sequence model; a 'batch' is a set of row ids."""

    def __init__(self, x, kind: str, num_outputs: int):
        super().__init__()
        self.register_buffer("x", x)  # the table is data, not a parameter
        self.kind = kind
        self.regression = num_outputs == 1
        f = int(x.size(1))
        if kind == "mlp":
            self.net = nn.Sequential(
                nn.Linear(f, MLP_WIDTH), nn.ReLU(),
                nn.Linear(MLP_WIDTH, MLP_WIDTH), nn.ReLU(),
                nn.Linear(MLP_WIDTH, MLP_WIDTH), nn.ReLU(),
                nn.Linear(MLP_WIDTH, num_outputs),
            )
        elif kind == "ftt":
            d = FTT_DIM
            bound = 1.0 / math.sqrt(d)
            self.tok_w = nn.Parameter(torch.empty(f, d).uniform_(-bound, bound))  # one linear token per feature
            self.tok_b = nn.Parameter(torch.empty(f, d).uniform_(-bound, bound))
            self.cls = nn.Parameter(torch.empty(1, 1, d).uniform_(-bound, bound))
            self.blocks = nn.ModuleList([_FTBlock(d, FTT_HEADS, FTT_FF) for _ in range(FTT_BLOCKS)])
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, num_outputs)
        else:
            raise ValueError(kind)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        x = self.x[input_ids]
        if self.kind == "mlp":
            logits = self.net(x)
        else:
            tokens = x.unsqueeze(-1) * self.tok_w + self.tok_b
            h = torch.cat([self.cls.expand(tokens.size(0), -1, -1), tokens], dim=1)
            for block in self.blocks:
                h = block(h)
            logits = self.head(torch.relu(self.norm(h[:, 0])))
        loss = None
        if labels is not None:
            if self.regression:
                loss = nn.functional.mse_loss(logits.squeeze(-1), labels)
            else:
                loss = nn.functional.cross_entropy(logits, labels)
        return types.SimpleNamespace(loss=loss, logits=logits)


def build_model(seed: int = 42):
    _seed_torch(seed)
    ds = _datasets()
    model = TabularNet(ds["x"].clone(), MODEL_NAME, NUM_LABELS)
    n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    if n_frozen:
        raise RuntimeError(f"full training of every parameter is required; n_frozen={n_frozen}")
    return model.to(get_device())


def make_loaders(seed: int = 42):
    ds = _datasets()
    pin = torch.cuda.is_available()
    shuffle_g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        ds["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=shuffle_g,
        drop_last=True,
        pin_memory=pin,
        num_workers=0,
    )
    val_loader = DataLoader(
        ds["val"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        pin_memory=pin,
        num_workers=0,
    )
    test_loader = DataLoader(
        ds["test"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        pin_memory=pin,
        num_workers=0,
    )
    return train_loader, val_loader, test_loader


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def _sync_if_cuda(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _elapsed_wall_s(t0: float, device) -> float:
    _sync_if_cuda(device)
    return time.perf_counter() - t0


def _batch_to_device(batch: dict, device):
    return {
        "input_ids": batch["input_ids"].to(device, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
        "labels": batch["labels"].to(device, non_blocking=True),
    }


def _loader_metrics_base(model, loader, device, name: str) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = out.logits
            y = batch["labels"]
            loss = nn.functional.cross_entropy(logits, y, reduction="sum")
            bs = int(y.size(0))
            total_loss += float(loss.item())
            correct += int((logits.argmax(dim=1) == y).sum().item())
            n += bs
    if n == 0:
        raise RuntimeError(f"empty DistilBERT {name} loader")
    return total_loss / n, correct / n


def _loader_metrics_regression(model, loader, device, name: str) -> tuple[float, float]:
    """Regression twin of the audited evaluation: (mean squared error, Pearson r). Pearson takes the 'accuracy' slot."""
    model.eval()
    total_sq = 0.0
    n = 0
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            pred = out.logits.squeeze(-1)
            y = batch["labels"]
            total_sq += float(nn.functional.mse_loss(pred, y, reduction="sum").item())
            n += int(y.size(0))
            ys.append(y.cpu())
            ps.append(pred.cpu())
    if n == 0:
        raise RuntimeError(f"empty {name} loader")
    return total_sq / n, float(glue_data.pearson(torch.cat(ys), torch.cat(ps)))


def _loader_metrics(model, loader, device, name: str) -> tuple[float, float]:
    """Classification: the audited function, unchanged. Regression: mean squared error and Pearson r."""
    if TAB_SPEC[TAB_TASK]["task_type"] == "regression":
        return _loader_metrics_regression(model, loader, device, name)
    return _loader_metrics_base(model, loader, device, name)


def _billed_compute_calls(optimizer, fallback: float) -> float:
    if hasattr(optimizer, "compute_calls"):
        return float(optimizer.compute_calls)
    return float(fallback)


def _is_jumping(optimizer) -> bool:
    return hasattr(optimizer, "compute_calls") and hasattr(
        optimizer, "steps_since_last_jump"
    )


def _worst_case_step_ng(optimizer) -> float:
    if not _is_jumping(optimizer):
        return 1.0
    if not getattr(optimizer, "warmup_complete", True):
        return 1.0
    n = int(
        min(
            getattr(optimizer, "next_jump_size", 500),
            getattr(optimizer, "jump_size", 500),
        )
    )
    floor = int(getattr(optimizer, "min_jump_size", 10))
    tries = 0
    cur = n
    while cur >= floor and tries < WORST_BISECTION_TRIES:
        tries += 1
        if cur <= floor:
            break
        cur = max(int(cur / 2), floor)
    return 1.0 + tries * BISECTION_TRY_NG


def _suppress_jump(optimizer) -> None:
    optimizer.steps_since_last_jump = 0


def global_best_checkpoint(curve: list[dict]) -> dict:
    if not curve:
        raise RuntimeError("empty curve")
    best = dict(curve[0])
    best_loss = float(best["val_loss"])
    for point in curve[1:]:
        y = float(point["val_loss"])
        if y < best_loss:
            best = dict(point)
            best_loss = y
    return best


def _append_curve_point_base(
    curve,
    outer_step,
    optimizer,
    fallback,
    model,
    val_loader,
    test_loader,
    device,
    t0,
    train_loss: float,
    train_acc: float,
    *,
    include_test: bool = False,
) -> dict:
    billed = _billed_compute_calls(optimizer, fallback)
    vl, va = _loader_metrics(model, val_loader, device, "val")
    wall_s = _elapsed_wall_s(t0, device)
    point = {
        "outer_step": int(outer_step),
        "compute_calls": billed,
        "wall_s": float(wall_s),
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
        "val_loss": float(vl),
        "val_acc": float(va),
    }
    if include_test:
        tel, tea = _loader_metrics(model, test_loader, device, "test")
        point["test_loss"] = float(tel)
        point["test_acc"] = float(tea)
    curve.append(point)
    extra = ""
    if include_test:
        extra = (
            f" test_loss={point['test_loss']:.6f} "
            f"test_acc={point['test_acc']:.4f}"
        )
    print(
        f"step={point['outer_step']} Ng={point['compute_calls']} "
        f"wall_s={point['wall_s']:.3f} "
        f"train_loss={point['train_loss']:.6f} "
        f"train_acc={point['train_acc']:.4f} "
        f"val_loss={point['val_loss']:.6f} "
        f"val_acc={point['val_acc']:.4f}{extra}",
        flush=True,
    )
    model.train()
    return point


def _loader_extra_metrics(model, loader, device) -> dict:
    """The task's extra metrics (TAB_SPEC[...]["extra_metrics"]) on a loader; evaluation mode, no gradients."""
    model.eval()
    regression = TAB_SPEC[TAB_TASK]["task_type"] == "regression"
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            if regression:
                ps.append(out.logits.squeeze(-1).cpu())
            else:
                ps.append(out.logits.argmax(dim=1).cpu())
            ys.append(batch["labels"].cpu())
    y, p = torch.cat(ys), torch.cat(ps)
    return {
        name: float(glue_data.EXTRA_METRICS[name](y, p))
        for name in TAB_SPEC[TAB_TASK]["extra_metrics"]
    }


def _append_curve_point(
    curve,
    outer_step,
    optimizer,
    fallback,
    model,
    val_loader,
    test_loader,
    device,
    t0,
    train_loss: float,
    train_acc: float,
    *,
    include_test: bool = False,
) -> dict:
    """The audited curve point, plus the task's extra metrics (the audited function is unchanged)."""
    point = _append_curve_point_base(
        curve, outer_step, optimizer, fallback, model, val_loader, test_loader,
        device, t0, train_loss, train_acc, include_test=include_test,
    )
    names = TAB_SPEC[TAB_TASK]["extra_metrics"]
    if names:
        for k, v in _loader_extra_metrics(model, val_loader, device).items():
            point["val_" + k] = v
        if include_test:
            for k, v in _loader_extra_metrics(model, test_loader, device).items():
                point["test_" + k] = v
        shown = [k for k in sorted(point) if k in {p + n for p in ("val_", "test_") for n in names}]
        print("   extra: " + " ".join(f"{k}={point[k]:.4f}" for k in shown), flush=True)
        model.train()
    return point


def _marks_crossed(prev_ng: float, new_ng: float, budget: float, interval: int):
    marks = []
    mark = interval
    while mark <= budget + 1e-12:
        if prev_ng < mark <= new_ng + 1e-12:
            marks.append(mark)
        mark += interval
    return marks


def _dropout_off(use_fwd_only: bool) -> bool:
    """True -> run this closure under model.eval() (dropout off)."""
    return DROPOUT_MODE == "eval_all" or bool(use_fwd_only)


def _clip_rate(optimizer, clip_box) -> float:
    """Share of real steps whose gradient was longer than CLIP_NORM."""
    if hasattr(optimizer, "clip_fired"):  # B25: counted inside the optimizer
        n = int(optimizer._fu_opt_stats.get("_clip_norm_count", 0))
        return float(optimizer.clip_fired) / n if n else 0.0
    n = int(clip_box["n"])
    return float(clip_box["fired"]) / n if n else 0.0


def _b25_stats(optimizer):
    """Counters of the jumping optimizer (None for official optimizers)."""
    if not hasattr(optimizer, "jump_attempts"):
        return None
    return {
        "attempts": int(optimizer.jump_attempts),
        "kept": int(optimizer.jumps_accepted),
        "undone": int(optimizer.fu_undone),
        "fu_checked": int(optimizer.fu_checked),
        "gate_skipped": int(optimizer.gate_skipped),
        "skipped_uphill": int(optimizer.skipped_uphill),
        "jumps_failed": int(optimizer.jumps_failed),
        "clip_fired": int(optimizer.clip_fired),
        "momentum_transfers": int(optimizer.momentum_transfers),
        "transfer_clipped": int(optimizer.transfer_clipped),
        "undo_coverage_ok": bool(optimizer.fu_checked >= optimizer.jumps_accepted + optimizer.fu_undone - 1),
    }


def train_one_trial(
    model,
    optimizer,
    *,
    smoke: bool,
    seed: int = 42,
    score_test: bool = False,
    budget: int = DEFAULT_BUDGET,
    n_steps: int | None = None,
) -> dict:
    if smoke and score_test:
        raise ValueError("score_test is for the 5-seed paper script, not --smoke")
    if n_steps is not None:
        budget = int(n_steps)
    elif smoke:
        budget = SMOKE_BUDGET
    budget = float(budget)
    if budget <= 0:
        raise ValueError(f"budget must be > 0, got {budget}")
    _seed_torch(seed)
    train_loader, val_loader, test_loader = make_loaders(seed=seed)
    device = next(model.parameters()).device
    model.train()
    official_calls = 0.0
    curve = []
    step_params = inspect.signature(type(optimizer).step).parameters
    use_fwd_only = "closure_forward_only" in step_params
    _sync_if_cuda(device)
    t0 = time.perf_counter()
    print(
        f"budget={budget} clip={CLIP_NORM} dropout={DROPOUT_MODE} gpu_name={gpu_name()} "
        f"experiment={EXPERIMENT_NAME} model={MODEL_NAME} "
        f"batch={BATCH_SIZE} max_len={MAX_LEN}",
        flush=True,
    )
    data_iter = _cycle(train_loader)
    clip_box = {"n": 0, "fired": 0}
    n_outer_steps = 0
    outer_step = 0
    last_train_loss = 0.0
    last_train_acc = 0.0
    while True:
        billed = _billed_compute_calls(optimizer, official_calls)
        remaining = budget - billed
        if remaining < 1.0:
            break
        if _is_jumping(optimizer) and remaining < _worst_case_step_ng(optimizer):
            _suppress_jump(optimizer)
            if remaining < 1.0:
                break
        outer_step += 1
        batch = _batch_to_device(next(data_iter), device)
        box = {"loss": last_train_loss, "acc": last_train_acc}

        def closure():
            # DistilBERT uses LayerNorm, not BatchNorm. LayerNorm computes
            # stats per-example on every forward and has no running mean/var
            # that differ between train and eval, so there is no BN train/eval
            # risk. Jumping may call closure() with a closure_forward_only
            # helper; this task then runs probes under model.eval() to turn
            # Dropout off. That dropout-only switch is already safe here.
            # Do not edit jumping_adam_loss.py or jumping_sgd_loss.py.
            if _dropout_off(use_fwd_only):
                model.eval()
            try:
                optimizer.zero_grad()
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = out.loss
                with torch.no_grad():
                    pred = out.logits.argmax(dim=1)
                    box["acc"] = float(
                        (pred == batch["labels"]).sum().item()
                    ) / int(batch["labels"].size(0))
                    box["loss"] = float(loss.item())
                loss.backward()
                if CLIP_NORM is not None and not use_fwd_only:
                    # official optimizers are clipped here; B25 clips inside the optimizer (its probes stay raw)
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                    clip_box["n"] += 1
                    clip_box["fired"] += int(float(gn) > CLIP_NORM)
                return loss
            finally:
                if _dropout_off(use_fwd_only):
                    model.train()

        def closure_forward():
            # Forward-only landing probe. eval() turns Dropout off. LayerNorm
            # stats are per-example either way (see closure() comment).
            if _dropout_off(use_fwd_only):
                model.eval()
            try:
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                return out.loss
            finally:
                if _dropout_off(use_fwd_only):
                    model.train()

        prev_ng = billed
        if use_fwd_only:
            optimizer.step(closure, closure_forward_only=closure_forward)
        else:
            optimizer.step(closure)
        official_calls += 1.0
        n_outer_steps = outer_step
        last_train_loss = float(box["loss"])
        last_train_acc = float(box["acc"])
        billed = _billed_compute_calls(optimizer, official_calls)
        if billed > budget + 1e-9:
            raise AssertionError(
                f"compute_calls={billed} exceeded budget={budget}"
            )
        crossed = _marks_crossed(prev_ng, billed, budget, CURVE_INTERVAL)
        log_now = bool(crossed) or billed + 1e-12 >= budget
        if log_now:
            _append_curve_point(
                curve,
                outer_step,
                optimizer,
                official_calls,
                model,
                val_loader,
                test_loader,
                device,
                t0,
                last_train_loss,
                last_train_acc,
                include_test=bool(score_test),
            )
    if not curve:
        _append_curve_point(
            curve,
            n_outer_steps,
            optimizer,
            official_calls,
            model,
            val_loader,
            test_loader,
            device,
            t0,
            last_train_loss,
            last_train_acc,
            include_test=bool(score_test),
        )
    elif (
        abs(
            curve[-1]["compute_calls"]
            - _billed_compute_calls(optimizer, official_calls)
        )
        > 1e-9
    ):
        _append_curve_point(
            curve,
            n_outer_steps,
            optimizer,
            official_calls,
            model,
            val_loader,
            test_loader,
            device,
            t0,
            last_train_loss,
            last_train_acc,
            include_test=bool(score_test),
        )
    last = curve[-1]
    billed = float(last["compute_calls"])
    assert billed <= budget + 1e-9, (
        f"final compute_calls={billed} exceeded budget={budget}"
    )
    print(
        f"done budget={budget} compute_calls={billed} "
        f"n_outer_steps={n_outer_steps} clip_rate={_clip_rate(optimizer, clip_box):.4f}",
        flush=True,
    )
    out = {
        "val_loss": float(last["val_loss"]),
        "val_acc": float(last["val_acc"]),
        "train_loss": float(last["train_loss"]),
        "train_acc": float(last["train_acc"]),
        "wall_s": float(last["wall_s"]),
        "compute_calls": billed,
        "n_outer_steps": int(n_outer_steps),
        "budget": float(budget),
        "clip_rate": _clip_rate(optimizer, clip_box),
        "gpu_name": gpu_name(),
        "curve": curve,
        "protocol_early_stop": PROTOCOL_EARLY_STOP,
        "experiment_name": EXPERIMENT_NAME,
    }
    out["b25_stats"] = _b25_stats(optimizer)
    out["dropout_mode"] = DROPOUT_MODE
    out["clip_norm"] = CLIP_NORM
    if score_test:
        out["test_loss"] = float(last["test_loss"])
        out["test_acc"] = float(last["test_acc"])
    return out


if __name__ == "__main__":
    print(MODEL_NAME)
    print(
        f"{TAB_TASK} budget={DEFAULT_BUDGET} Ng "
        f"batch={BATCH_SIZE} max_len={MAX_LEN} clip={CLIP_NORM}"
    )
    model = build_model(seed=42)
    n_train = sum(p.requires_grad for p in model.parameters())
    n_all = sum(1 for _ in model.parameters())
    print(f"trainable={n_train}/{n_all} gpu_name={gpu_name()}")
    train_loader, val_loader, test_loader = make_loaders(seed=42)
    print(
        f"n_train={len(train_loader.dataset)} n_val={len(val_loader.dataset)} "
        f"n_test={len(test_loader.dataset)}"
    )
    opt = torch.optim.Adam(model.parameters(), lr=2e-5)
    result = train_one_trial(
        model, opt, smoke=False, seed=42, n_steps=1
    )
    print(
        f"one_step n_outer_steps={result['n_outer_steps']} "
        f"Ng={result['compute_calls']} val_loss={result['val_loss']:.6f} "
        f"gpu_name={result['gpu_name']}"
    )
