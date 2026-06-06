#!/usr/bin/env python3
"""Random forest predictor. Implements the predictor contract in README.md
with numpy + scikit-learn: a bagged tree ensemble fit in transformed
(log) target space, target-space metrics on the test split. Tree
ensembles are scale-invariant, so there is no standardization step and no
scaler.json.

sklearn has no version-stable native serialization, so the model is stored
with joblib (model.joblib). That's fine here: predict runs from the same
pinned venv that trained the model (predictors/.venv, see
`zig build py-setup`).

Training is a single fit() call — no epoch events, no STOP support."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

DEFAULTS = {
    "n_estimators": 300,
    "max_depth": 0,  # 0 = unlimited (sklearn's None)
    "min_samples_leaf": 1,
    "max_features": 1.0,
    "seed": 42,
}


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def read_features(dir: Path, n_rows: int, n_cols: int) -> np.ndarray:
    x = np.fromfile(dir / "features.f32", dtype="<f4")
    assert x.size == n_rows * n_cols, (
        f"features.f32 has {x.size} values, manifest says {n_rows}x{n_cols}")
    return x.reshape(n_rows, n_cols)


def invert_target(y: np.ndarray, transform: str) -> np.ndarray:
    """Map transformed-space targets back to target space."""
    if transform == "log1p":
        return np.expm1(y)
    return y


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Target-space metrics, formula identical to lensing-core::compute_metrics
    (medape for even n = mean of the two middle APEs)."""
    n = len(actual)
    err = predicted - actual
    apes = np.sort(np.abs(err / actual))
    medape = apes[n // 2] if n % 2 == 1 else (apes[n // 2 - 1] + apes[n // 2]) / 2.0
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    sq_err = float(np.sum(err**2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(sq_err / n)),
        "r2": 1.0 - sq_err / ss_tot,
        "mape": float(np.mean(apes)),
        "medape": float(medape),
        "n_test": n,
    }


def train(dataset: Path, output: Path, hp_path: Path) -> None:
    hp = {**DEFAULTS, **json.loads(hp_path.read_text())}

    manifest = json.loads((dataset / "manifest.json").read_text())
    n_rows, n_cols = manifest["n_rows"], manifest["n_cols"]
    features = read_features(dataset, n_rows, n_cols)
    target = np.fromfile(dataset / "target.f32", dtype="<f4")
    row_ids = np.fromfile(dataset / "row_ids.u64", dtype="<u8")
    train_idx = np.fromfile(dataset / "train_idx.u32", dtype="<u4")
    test_idx = np.fromfile(dataset / "test_idx.u32", dtype="<u4")

    max_depth = int(hp["max_depth"])
    emit({"event": "log", "msg": f"dataset {manifest['dataset_id']}: "
          f"{len(train_idx)} train / {len(test_idx)} test rows, "
          f"{n_cols} features"})
    emit({"event": "log", "msg": f"random forest, {hp['n_estimators']} trees, "
          f"depth {max_depth if max_depth > 0 else 'unlimited'}, "
          f"min leaf {hp['min_samples_leaf']}, "
          f"max features {hp['max_features']}"})

    # No standardization: trees split on raw feature values.
    model = RandomForestRegressor(
        n_estimators=int(hp["n_estimators"]),
        max_depth=max_depth if max_depth > 0 else None,
        min_samples_leaf=int(hp["min_samples_leaf"]),
        max_features=float(hp["max_features"]),
        random_state=int(hp["seed"]),
        n_jobs=-1,
    )
    model.fit(features[train_idx], target[train_idx])

    transform = manifest["target"]["transform"]
    predicted = np.maximum(
        invert_target(model.predict(features[test_idx]), transform), 0.0)
    actual = invert_target(target[test_idx].astype(np.float64), transform)

    metrics = compute_metrics(actual, predicted)
    predictions = [
        {"row_id": int(row_ids[i]), "actual": float(a), "predicted": float(p)}
        for i, a, p in zip(test_idx, actual, predicted)
    ]
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics))
    (output / "predictions.json").write_text(json.dumps(predictions))
    joblib.dump(model, output / "model.joblib", compress=3)

    emit({"event": "log", "msg": f"MAE {metrics['mae']:,.0f}  "
          f"RMSE {metrics['rmse']:,.0f}  medAPE {metrics['medape']:.1%}  "
          f"R² {metrics['r2']:.3f}"})
    emit({"event": "done"})


def predict(model_dir: Path, input_dir: Path, output: Path) -> None:
    """Contract v2 predict: load the joblib model, predict on the
    server-featurized mini-artifact, write target-space predictions."""
    model = joblib.load(model_dir / "model.joblib")

    manifest = json.loads((input_dir / "manifest.json").read_text())
    n_rows, n_cols = manifest["n_rows"], manifest["n_cols"]
    assert model.n_features_in_ == n_cols, (
        f"model was fit on {model.n_features_in_} columns, input has {n_cols}")
    features = read_features(input_dir, n_rows, n_cols)
    row_ids = np.fromfile(input_dir / "row_ids.u64", dtype="<u8")
    emit({"event": "log", "msg": f"predicting {n_rows} rows, {n_cols} "
          f"features, random forest {model.n_estimators} trees"})

    transform = manifest["target"]["transform"]
    predicted = np.maximum(
        invert_target(model.predict(features), transform), 0.0)

    output.write_text(json.dumps([
        {"row_id": int(rid), "predicted": float(p)}
        for rid, p in zip(row_ids, predicted)
    ]))
    emit({"event": "done"})


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--dataset", required=True, type=Path)
    tr.add_argument("--output", required=True, type=Path)
    tr.add_argument("--hyperparams", required=True, type=Path)
    pr = sub.add_parser("predict")
    pr.add_argument("--model", required=True, type=Path)
    pr.add_argument("--input", required=True, type=Path)
    pr.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    if args.cmd == "train":
        train(args.dataset, args.output, args.hyperparams)
    else:
        predict(args.model, args.input, args.output)


if __name__ == "__main__":
    main()
