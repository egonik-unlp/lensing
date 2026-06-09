"""Shared helpers for classifier predictors (binary + multiclass).

The dataset `target.f32` holds class labels (already integer-valued for a
classification build, e.g. a binary label ∈ {0,1}). A classifier predictor treats the
target as categorical: distinct sorted values define the class ids 0..K-1, so
these predictors work on any dataset whose target is a small set of labels,
regardless of whether the manifest yet carries a `task` field.

Predictions follow the framework contract: `predicted` is P(class==1) for binary
and the argmax class id for multiclass; `proba` carries the full probability
vector. Metrics mirror lensing-core's classification metrics (the server trusts
this metrics.json): accuracy, logloss, auc (binary ROC / multiclass macro-OvR),
plus brier (binary) / macro_f1 (multiclass).

Run from the shared predictor venv (predictors/.venv)."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def read_features(dir: Path, n_rows: int, n_cols: int) -> np.ndarray:
    x = np.fromfile(dir / "features.f32", dtype="<f4")
    assert x.size == n_rows * n_cols, (
        f"features.f32 has {x.size} values, manifest says {n_rows}x{n_cols}")
    return x.reshape(n_rows, n_cols).astype(np.float64)


def load_dataset(dataset: Path):
    """Returns (manifest, X, raw_target, row_ids, train_idx, test_idx)."""
    manifest = json.loads((dataset / "manifest.json").read_text())
    n_rows, n_cols = manifest["n_rows"], manifest["n_cols"]
    x = read_features(dataset, n_rows, n_cols)
    target = np.fromfile(dataset / "target.f32", dtype="<f4").astype(np.float64)
    row_ids = np.fromfile(dataset / "row_ids.u64", dtype="<u8")
    train_idx = np.fromfile(dataset / "train_idx.u32", dtype="<u4")
    test_idx = np.fromfile(dataset / "test_idx.u32", dtype="<u4")
    return manifest, x, target, row_ids, train_idx, test_idx


def encode_labels(target: np.ndarray):
    """Map raw target values → class ids 0..K-1 via sorted unique values.
    Returns (y_ids, classes, k). Labels are rounded to ints first (the target
    is stored as f32 class codes)."""
    raw = np.rint(target).astype(np.int64)
    classes = np.unique(raw)
    remap = {int(c): i for i, c in enumerate(classes)}
    y = np.array([remap[int(c)] for c in raw], dtype=np.int64)
    return y, [int(c) for c in classes], len(classes)


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, k: int) -> dict:
    """Target-space classification metrics; keys mirror lensing-core's
    Metrics (binary → accuracy/logloss/auc/brier; multiclass adds macro_f1)."""
    n = len(y_true)
    pred = proba.argmax(axis=1)
    labels = list(range(k))
    out = {
        "n_test": int(n),
        "accuracy": float(accuracy_score(y_true, pred)),
        "logloss": float(log_loss(y_true, proba, labels=labels)),
    }
    if k == 2:
        p1 = proba[:, 1]
        try:
            out["auc"] = float(roc_auc_score(y_true, p1))
        except ValueError:
            out["auc"] = 0.5  # one class absent in test
        out["brier"] = float(brier_score_loss(y_true, p1))
    else:
        try:
            out["auc"] = float(
                roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=labels))
        except ValueError:
            out["auc"] = 0.5
        out["macro_f1"] = float(f1_score(y_true, pred, average="macro", zero_division=0))
    return out


def predictions_payload(row_ids, idx, y_true, proba, k):
    """predictions.json rows: predicted = P(class1) (binary) / argmax id
    (multiclass); proba = full probability vector."""
    rows = []
    for j, i in enumerate(idx):
        p = [float(v) for v in proba[j]]
        predicted = p[1] if k == 2 else float(int(np.argmax(proba[j])))
        rows.append({
            "row_id": int(row_ids[i]),
            "actual": float(y_true[j]),
            "predicted": predicted,
            "proba": p,
        })
    return rows


def inference_payload(row_ids, proba, k):
    """predict-subcommand rows: predicted + proba, no ground truth."""
    rows = []
    for j in range(len(row_ids)):
        p = [float(v) for v in proba[j]]
        predicted = p[1] if k == 2 else float(int(np.argmax(proba[j])))
        rows.append({"row_id": int(row_ids[j]), "predicted": predicted, "proba": p})
    return rows


def align_proba(model_proba: np.ndarray, model_classes, k: int) -> np.ndarray:
    """Reorder a model's predict_proba columns into canonical 0..K-1 order.
    sklearn/xgboost order columns by `classes_`; we want column c = P(class id c)."""
    proba = np.zeros((model_proba.shape[0], k), dtype=np.float64)
    for col, cls in enumerate(model_classes):
        proba[:, int(cls)] = model_proba[:, col]
    return proba
