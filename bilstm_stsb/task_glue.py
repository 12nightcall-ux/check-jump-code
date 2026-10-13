"""BiLSTM + GloVe harness for the B25 comparison: one task. Stop = billed Ng budget.

Derived from the audited experiments/distilbert_sst2/task_distilbert_sst2.py. Everything that trains, bills, clips,
evaluates and logs is IDENTICAL to that file (verified function by function when this file was built). What changed:
  * the data layer (glue_data.py splits, the SAME splits as the transformer experiments) and a word-level pipeline:
    lower-cased words and punctuation as separate tokens, a vocabulary built from the training rows only, GloVe 840B 300d,
  * the model: the GLUE-baseline-style encoder - one shared 2-layer BiLSTM (hidden 1500 per direction) with max-pooling over
    time, features [u, v, |u-v|, u*v] for sentence pairs, then a 512-unit MLP. GloVe INITIALISES the embedding table and it is
    FINE-TUNED together with everything else: every weight is trained, nothing is frozen. Dropout is 0 for every arm (the same dropout-off
    protocol as the transformers).
  * both sentences travel packed in input_ids / attention_mask (first MAX_LEN columns = sentence A, last MAX_LEN = sentence B)
    so the audited loop, which only passes input_ids, attention_mask and labels, is untouched,
  * cuDNN needs the LSTM in training mode for backward, even when the loop asks for eval (dropout off): the model's train()
    keeps the LSTM in training mode; with dropout 0 this changes nothing numerically,
  * the task's extra metrics (F1 / Matthews / Spearman) and regression support (STS-B), as in the transformer harness.

Stop is compute_calls <= DEFAULT_BUDGET (1000 Ng), 20 Ng if smoke. Gradient clip: CLIP_NORM (default 1.0).
HPO objective is global_best_checkpoint val_loss (scored in hpo_gp_bayesopt.py).
"""

from __future__ import annotations

import inspect
import os
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import hashlib
import json
import re
import types

import numpy as np

import glue_data  # data layer: task spec, split rules, F1 / Matthews / Spearman

DEFAULT_BUDGET = 1000
SMOKE_BUDGET = 20  # Jumping warmup is 15 Ng; 10 Ng smoke never finishes warmup
CURVE_INTERVAL = 25
EXPERIMENT_NAME = "bilstm_stsb"
PROTOCOL_EARLY_STOP = False
WORST_BISECTION_TRIES = 4
BISECTION_TRY_NG = 2.3
MODEL_NAME = "bilstm-glove-840B-300d"
GLUE_TASK = "stsb"  # key of glue_data.TASKS
GLOVE_PATH = "/content/drive/MyDrive/Optimiser Formula Creation - Working/experiments/GLUE/data/glove.840B.300d.txt"
EMB_DIM = 300
LSTM_HIDDEN = 1500
LSTM_LAYERS = 2
MLP_HIDDEN = 512
MAX_LEN = 128
BATCH_SIZE = 32
NUM_LABELS = glue_data.spec(GLUE_TASK)["num_labels"]

# --- JumpingAdamSimpsonB25 experiment settings (the env overrides exist only for the reproduction guard) ---
DROPOUT_MODE = os.environ.get("B25_DROPOUT_MODE", "eval_all")  # "eval_all" = dropout off for EVERY optimizer; "official" = old behaviour
_CLIP_ENV = os.environ.get("B25_CLIP", "1.0")
CLIP_NORM = None if _CLIP_ENV.strip().lower() in ("none", "off", "") else float(_CLIP_ENV)
assert DROPOUT_MODE in ("eval_all", "official"), DROPOUT_MODE
SPLIT_SEED = glue_data.SPLIT_SEED

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


# ---- word-level data pipeline (replaces the transformer tokenizer path) ----
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)  # words and punctuation marks as separate tokens
_PAD, _UNK = "<pad>", "<unk>"


def _word_tokens(text: str) -> list:
    toks = _TOKEN_RE.findall(text.lower())[:MAX_LEN]
    return toks if toks else [_UNK]


def _build_vocab(rows: list):
    """Vocabulary from the TRAINING rows only (id 0 = padding, 1 = unknown), in a deterministic order."""
    counts = {}
    for r in rows:
        toks = _word_tokens(r["a"]) + (_word_tokens(r["b"]) if r["b"] is not None else [])
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
    stoi = {_PAD: 0, _UNK: 1}
    for t, _c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if t not in stoi:
            stoi[t] = len(stoi)
    return stoi, counts


def _glove_matrix(stoi: dict, counts: dict):
    """(len(stoi), EMB_DIM) float32 matrix from GloVe 840B 300d.

    Streams the file once, keeps only the words we need, and caches the result next to this file, so the 5.6 GB file is
    read once per task, not once per process. Unknown words get the mean of the found vectors, padding is zero.
    """
    key = hashlib.sha256("\n".join(sorted(stoi)).encode("utf-8")).hexdigest()[:16]
    cache = os.path.join(_data_dir(), f"glove_{GLUE_TASK}_{key}.npy")
    if os.path.isfile(cache) and os.path.isfile(cache + ".json"):
        return np.load(cache), json.load(open(cache + ".json"))
    if not os.path.isfile(GLOVE_PATH):
        raise RuntimeError(f"GloVe file not found: {GLOVE_PATH}")
    need = set(stoi) - {_PAD, _UNK}
    found = {}
    with open(GLOVE_PATH, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            word, _, rest = line.partition(" ")
            if word in need and word not in found:
                vals = rest.split()
                if len(vals) == EMB_DIM:  # a few GloVe tokens contain spaces; those lines are skipped
                    found[word] = np.asarray(vals, dtype=np.float32)
    if len(found) < 100:
        raise RuntimeError(f"too few GloVe hits ({len(found)}) - wrong file?")
    unk = np.mean(np.stack(list(found.values())), axis=0).astype(np.float32)
    mat = np.tile(unk, (len(stoi), 1))
    mat[stoi[_PAD]] = 0.0
    for w, v in found.items():
        mat[stoi[w]] = v
    total = sum(counts.values())
    stats = {
        "vocab_size": len(stoi),
        "words_found": len(found),
        "type_coverage": len(found) / max(1, len(need)),
        "token_coverage": sum(counts[w] for w in found) / max(1, total),
    }
    tmp = cache + ".tmp"
    with open(tmp, "wb") as fh:
        np.save(fh, mat)
    os.replace(tmp, cache)
    with open(cache + ".json", "w") as fh:
        json.dump(stats, fh)
    return mat, stats


def _tokenize_rows(rows: list, stoi: dict):
    """Rows -> dataset. Columns [0, MAX_LEN) hold sentence A, [MAX_LEN, 2*MAX_LEN) sentence B (empty for single-sentence tasks)."""
    n = len(rows)
    ids = np.zeros((n, 2 * MAX_LEN), dtype=np.int64)
    mask = np.zeros((n, 2 * MAX_LEN), dtype=np.int64)
    for i, r in enumerate(rows):
        for half, text in ((0, r["a"]), (1, r["b"])):
            if text is None:
                continue
            idx = [stoi.get(t, 1) for t in _word_tokens(text)]
            ids[i, half * MAX_LEN : half * MAX_LEN + len(idx)] = idx
            mask[i, half * MAX_LEN : half * MAX_LEN + len(idx)] = 1
    is_float = isinstance(rows[0]["label"], float)
    labels = torch.tensor([r["label"] for r in rows], dtype=torch.float32 if is_float else torch.long)
    return _TokenDS({"input_ids": torch.from_numpy(ids), "attention_mask": torch.from_numpy(mask)}, labels)


def _datasets():
    global _CACHE
    if _CACHE is None:
        train_rows, official_val_rows = glue_data.load_glue_rows(
            GLUE_TASK, os.path.join(_data_dir(), "hf")
        )
        train_keep, val_keep, test_keep = glue_data.split_rows(
            GLUE_TASK, train_rows, official_val_rows, SPLIT_SEED
        )
        stoi, counts = _build_vocab(train_keep)
        emb, stats = _glove_matrix(stoi, counts)
        print(
            f"GloVe coverage: {stats['words_found']}/{stats['vocab_size'] - 2} vocabulary words "
            f"({stats['type_coverage']:.1%}), {stats['token_coverage']:.1%} of training tokens",
            flush=True,
        )
        _CACHE = {
            "train": _tokenize_rows(train_keep, stoi),
            "val": _tokenize_rows(val_keep, stoi),
            "test": _tokenize_rows(test_keep, stoi),
            "emb_weight": torch.from_numpy(emb),
            "glove_stats": stats,
        }
    return _CACHE


class _BiLSTMEncoder(nn.Module):
    """Shared sentence encoder: 2-layer BiLSTM over packed sequences, max-pooled over time."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=EMB_DIM,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            bidirectional=True,
            batch_first=True,
            dropout=0.0,
        )

    def forward(self, x, mask):
        lengths = mask.sum(dim=1).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        out = out.masked_fill(~(mask > 0).unsqueeze(-1), torch.finfo(out.dtype).min)
        return out.max(dim=1).values


class BiLSTMClassifier(nn.Module):
    """GLUE-baseline-style classifier / regressor; same call signature and outputs as a Hugging Face sequence model."""

    def __init__(self, emb_weight, pair: bool, num_labels: int):
        super().__init__()
        # GloVe INITIALISES the embedding table and it is FINE-TUNED with everything else. clone(): the cached GloVe matrix is shared by
        # every model built in this process (all trials, seeds and arms), so training one model must never be able to change it.
        self.embedding = nn.Embedding.from_pretrained(emb_weight.clone(), freeze=False, padding_idx=0)
        self.pair = bool(pair)
        self.num_labels = int(num_labels)
        self.encoder = _BiLSTMEncoder()
        feat = (4 if self.pair else 1) * 2 * LSTM_HIDDEN
        self.fc1 = nn.Linear(feat, MLP_HIDDEN)
        self.fc2 = nn.Linear(MLP_HIDDEN, self.num_labels)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.lstm.train(True)  # cuDNN backward needs the LSTM in training mode; dropout is 0, so nothing else changes
        return self

    def _embed(self, ids):
        return self.embedding(ids)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        ids_a, ids_b = input_ids[:, :MAX_LEN], input_ids[:, MAX_LEN:]
        m_a, m_b = attention_mask[:, :MAX_LEN], attention_mask[:, MAX_LEN:]
        la = max(int(m_a.sum(dim=1).max()), 1)
        u = self.encoder(self._embed(ids_a[:, :la]), m_a[:, :la])
        if self.pair:
            lb = max(int(m_b.sum(dim=1).max()), 1)
            v = self.encoder(self._embed(ids_b[:, :lb]), m_b[:, :lb])
            feats = torch.cat([u, v, (u - v).abs(), u * v], dim=-1)
        else:
            feats = u
        logits = self.fc2(torch.relu(self.fc1(feats)))
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = nn.functional.mse_loss(logits.squeeze(-1), labels)
            else:
                loss = nn.functional.cross_entropy(logits, labels)
        return types.SimpleNamespace(loss=loss, logits=logits)


def build_model(seed: int = 42):
    _seed_torch(seed)
    ds = _datasets()
    pair = glue_data.spec(GLUE_TASK)["keys"][1] is not None
    model = BiLSTMClassifier(ds["emb_weight"], pair, NUM_LABELS)
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
        raise RuntimeError(f"empty DistilBERT {name} loader")
    return total_sq / n, float(glue_data.pearson(torch.cat(ys), torch.cat(ps)))


def _loader_metrics(model, loader, device, name: str) -> tuple[float, float]:
    """Classification: the audited function, unchanged. Regression: mean squared error and Pearson r."""
    if glue_data.spec(GLUE_TASK)["task_type"] == "regression":
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
    """The task's extra metrics (glue_data.TASKS[...]["extra_metrics"]) on a loader; evaluation mode, no gradients."""
    model.eval()
    regression = glue_data.spec(GLUE_TASK)["task_type"] == "regression"
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
        for name in glue_data.spec(GLUE_TASK)["extra_metrics"]
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
    names = glue_data.spec(GLUE_TASK)["extra_metrics"]
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
        f"{GLUE_TASK} budget={DEFAULT_BUDGET} Ng "
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
