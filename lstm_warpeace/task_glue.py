"""War and Peace character language model (2-layer LSTM). Stop = billed Ng budget.

A TRUNCATED, simplified "Tolstoi-style" CharRNN after the tolstoi workload of Choi et al. (2019) and DeepOBS (Schneider et al., 2019): 2-layer LSTM with 128 units,
windows of 50 characters, batch 50 (all as published) trained for 1000 gradient evaluations (about one epoch) instead of the published 200 epochs; it is NOT the
canonical benchmark. Rebuilt on the audited loop.
Derived from the audited experiments/distilbert_sst2/task_distilbert_sst2.py: everything that trains, bills, clips, evaluates
and logs is IDENTICAL to that file, with ONE evaluation difference (below). What changed:
  * the data: the text of War and Peace (Project Gutenberg #2600, header and footer removed), split by POSITION into the first
    80% (train), the next 10% (validation) and the last 10% (test); no shuffling, so no passage of the book is in two parts.
    Windows of 50 characters are cut without overlap inside each part; a window is trained to predict the next character at every
    position. The prepared text is FROZEN on first use (data/frozen_warpeace.npz with a SHA-256 check), so every arm and every
    resumed session reads the same characters. (The original workload carries the LSTM state across batches; here every window
    starts from a zero state, so that batches are independent. The exact text version and the preprocessing differ from DeepOBS.)
  * the model: 128-dimensional learned character embeddings (as in the published setup of Choi et al.), a 2-layer LSTM (hidden size 128) and a
    linear output layer, trained FROM SCRATCH (every weight trained, nothing frozen), dropout 0. cuDNN needs the LSTM in training mode for its backward pass; with dropout 0 this
    changes nothing numerically.
  * a "batch" is a set of window ids (BATCH_SIZE = 50, as in the original workload): the model holds the whole text and cuts the
    windows itself, so the audited loop, which only passes input_ids, attention_mask and labels, is untouched. logits are
    [batch, vocabulary, 50] so that the audited argmax over dim 1 is the predicted character at every position.
  * the evaluation: the audited _loader_metrics averages the loss and the accuracy over windows; here every window holds 50 labels,
    so the same function is used with the average taken over CHARACTERS (a documented one-line difference). The loss is therefore
    nats per character and the accuracy is next-character accuracy.
  * one audited DIAGNOSTIC is left exactly as it is: train_one_trial logs train_acc as correct characters per WINDOW (0 to 50; divide by 50 for a
    rate). Nothing reads it (selection, tuning and every reported metric use the validation and test values above), so it was not touched.

Stop is compute_calls <= DEFAULT_BUDGET (1000 Ng), 20 Ng if smoke. Gradient clip: CLIP_NORM (default 1.0).
HPO objective is global_best_checkpoint val_loss (scored in hpo_gp_bayesopt.py).
"""

from __future__ import annotations

import inspect
import os
import random
import time
import hashlib
import types
import urllib.request

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import numpy as np

DEFAULT_BUDGET = 1000
SMOKE_BUDGET = 20  # Jumping warmup is 15 Ng; 10 Ng smoke never finishes warmup
CURVE_INTERVAL = 25
EXPERIMENT_NAME = "lstm_warpeace"
PROTOCOL_EARLY_STOP = False
WORST_BISECTION_TRIES = 4
BISECTION_TRY_NG = 2.3
MODEL_NAME = "char_lstm"
LM_TASK = "warpeace"  # key of LM_SPEC
MAX_LEN = 50  # the window length (the audited loop prints it)
BATCH_SIZE = 50  # windows per step, as in the original workload
SEQ_LEN = 50  # characters per window
LSTM_HIDDEN = 128
EMB_DIM = 128  # learned character embedding size
LSTM_LAYERS = 2
LM_SPEC = {
    "warpeace": {
        "urls": (
            "https://www.gutenberg.org/cache/epub/2600/pg2600.txt",
            "https://www.gutenberg.org/files/2600/2600-0.txt",
            "https://www.gutenberg.org/ebooks/2600.txt.utf-8",
        ),
        "min_chars": 3_000_000, "max_chars": 3_800_000, "min_vocab": 50, "max_vocab": 130,
    }
}
NUM_LABELS = 0  # the vocabulary size is read from the text

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










def _load_text():
    """The text of War and Peace without the Project Gutenberg header and footer; downloaded once and kept in the data folder."""
    path = os.path.join(_data_dir(), "warpeace_raw.txt")
    if not os.path.exists(path):
        last = None
        for url in LM_SPEC[LM_TASK]["urls"]:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research benchmark)"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                if len(data) > 2_000_000:
                    with open(path, "wb") as f:
                        f.write(data)
                    break
            except Exception as e:  # try the next mirror
                last = e
        else:
            raise RuntimeError(f"could not download War and Peace from Project Gutenberg: {last}")
    raw = open(path, encoding="utf-8-sig").read().replace("\r\n", "\n")
    a, b = raw.find("*** START OF"), raw.find("*** END OF")
    if a < 0 or b < 0 or b <= a:
        raise RuntimeError("the Project Gutenberg start / end markers were not found")
    return raw[raw.index("\n", a) + 1 : b]


def _lm_sha16(tokens, vocab: str, bounds) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(tokens).tobytes())
    h.update(vocab.encode("utf-8"))
    h.update(np.asarray(bounds, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def _datasets():
    global _CACHE
    if _CACHE is None:
        spec = LM_SPEC[LM_TASK]
        path = os.path.join(_data_dir(), f"frozen_{LM_TASK}.npz")
        if os.path.exists(path):  # frozen on the first use: every arm and every resumed session reads the SAME characters
            z = np.load(path, allow_pickle=False)
            tokens, vocab, bounds = z["tokens"], str(z["vocab"]), z["bounds"]
            sha = _lm_sha16(tokens, vocab, bounds)
            if sha != str(z["sha16"]):
                raise RuntimeError(f"{path} is corrupt or was altered: hash {sha} != {str(z['sha16'])}")
            print(f"data: loaded the frozen text {os.path.basename(path)} (sha16 {sha})", flush=True)
        else:
            text = _load_text()
            vocab = "".join(sorted(set(text)))
            if not (spec["min_chars"] <= len(text) <= spec["max_chars"]) or not (spec["min_vocab"] <= len(vocab) <= spec["max_vocab"]):
                raise RuntimeError(f"unexpected text: {len(text)} characters, {len(vocab)} distinct")
            stoi = {c: i for i, c in enumerate(vocab)}
            tokens = np.fromiter((stoi[c] for c in text), dtype=np.int64, count=len(text)).astype(np.uint8)
            n = len(tokens)
            bounds = np.array([0, int(0.8 * n), int(0.9 * n), n], dtype=np.int64)
            sha = _lm_sha16(tokens, vocab, bounds)
            tmp = path + ".tmp.npz"
            np.savez(tmp, tokens=tokens, vocab=np.array(vocab), bounds=bounds, sha16=np.array(sha))
            os.replace(tmp, path)
            print(f"data: built and froze the text {os.path.basename(path)} (sha16 {sha}, {n} characters, {len(vocab)} distinct)", flush=True)
        tok = torch.from_numpy(tokens.astype(np.int64))
        parts = {}
        for k, name in enumerate(("train", "val", "test")):
            lo, hi = int(bounds[k]), int(bounds[k + 1])
            starts = torch.from_numpy(np.arange(lo, hi - SEQ_LEN, SEQ_LEN, dtype=np.int64))  # a window needs SEQ_LEN + 1 characters inside its part
            labels = tok[starts.unsqueeze(1) + 1 + torch.arange(SEQ_LEN).unsqueeze(0)]  # the next character at every position
            parts[name] = _TokenDS({"input_ids": starts, "attention_mask": torch.ones_like(starts)}, labels)
        _CACHE = {**parts, "tokens": tok, "vocab_size": len(vocab), "vocab": vocab, "bounds": bounds, "data_sha16": sha}
    return _CACHE


class CharLSTM(nn.Module):
    """2-layer LSTM on learned character embeddings with the call signature of a Hugging Face sequence model; a 'batch' is a set of window ids."""

    def __init__(self, tokens, vocab_size: int):
        super().__init__()
        self.register_buffer("tokens", tokens)  # the whole text is data, not a parameter
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, EMB_DIM)  # one learned row of EMB_DIM numbers per character
        self.rnn = nn.LSTM(EMB_DIM, LSTM_HIDDEN, num_layers=LSTM_LAYERS, batch_first=True)
        self.out = nn.Linear(LSTM_HIDDEN, vocab_size)

    def train(self, mode: bool = True):
        super().train(mode)
        self.rnn.train(True)  # cuDNN backward needs the LSTM in training mode; dropout is 0, so nothing else changes
        return self

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        pos = input_ids.unsqueeze(1) + torch.arange(SEQ_LEN, device=input_ids.device).unsqueeze(0)  # [batch, SEQ_LEN]
        x = self.embed(self.tokens[pos])
        h, _ = self.rnn(x)
        logits = self.out(h).transpose(1, 2)  # [batch, vocabulary, SEQ_LEN]: the audited argmax over dim 1 is the predicted character
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return types.SimpleNamespace(loss=loss, logits=logits)


def build_model(seed: int = 42):
    _seed_torch(seed)
    ds = _datasets()
    model = CharLSTM(ds["tokens"].clone(), ds["vocab_size"])
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


def _loader_metrics(model, loader, device, name: str) -> tuple[float, float]:
    """The audited evaluation with ONE change: loss and accuracy are averaged over characters (y.numel()), not over windows (y.size(0)),
    because every window holds SEQ_LEN labels. Everything else is the audited function."""
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
            total_loss += float(loss.item())
            correct += int((logits.argmax(dim=1) == y).sum().item())
            n += int(y.numel())
    if n == 0:
        raise RuntimeError(f"empty {name} loader")
    return total_loss / n, correct / n



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
        f"DistilBERT SST-2 budget={DEFAULT_BUDGET} Ng "
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
