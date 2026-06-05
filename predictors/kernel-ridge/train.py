#!/usr/bin/env python3
"""Kernel ridge regression predictor. Implements the predictor contract in
README.md with numpy + scikit-learn: standardize on the train split, fit
sklearn.kernel_ridge.KernelRidge in transformed (log-price) target space
with a selectable kernel (rbf/linear/poly/sigmoid), report price-space
metrics. The model is saved as plain JSON + the binary standardized train
matrix (KRR is dense — every train row carries a dual coefficient), so
predict needs no sklearn version pinning — the decision function is
recomputed with numpy, same philosophy as the svm predictor.

The interesting contrast with svm (epsilon-SVR): identical rbf kernel —
hence the same intrinsic protection against expm1 blowups (K -> 0 far from
the data) — but squared loss instead of the epsilon-insensitive loss, so
log-space tail residuals are penalized quadratically.

Run from the repo's shared predictor venv (predictors/.venv)."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.kernel_ridge import KernelRidge

DEFAULTS = {
    "kernel": "rbf",
    "alpha": 1.0,
    "gamma": 0.0,  # 0 = the svm predictor's "scale": 1 / (n_features * X.var())
    "degree": 3,
    "coef0": 1.0,
}


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def read_features(dir: Path, n_rows: int, n_cols: int) -> np.ndarray:
    x = np.fromfile(dir / "features.f32", dtype="<f4")
    assert x.size == n_rows * n_cols, (
        f"features.f32 has {x.size} values, manifest says {n_rows}x{n_cols}")
    return x.reshape(n_rows, n_cols).astype(np.float64)


def fit_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Train-split standardization matching the Rust/Julia predictors'
    scaler: population std, near-constant columns get std 1.0."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)  # population (ddof=0), like scaler.rs
    std = np.where(std < 1e-12, 1.0, std)
    return mean, std


def invert_target(y: np.ndarray, transform: str) -> np.ndarray:
    """Map transformed-space targets back to price space."""
    if transform == "log1p":
        return np.expm1(y)
    return y


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Price-space metrics, formula identical to pg-core::compute_metrics
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


def decision_function(x: np.ndarray, basis: np.ndarray, dual_coef: np.ndarray,
                      kernel: str, gamma: float, degree: int,
                      coef0: float) -> np.ndarray:
    """KRR decision function f(x) = K(x, X_train) . dual_coef with numpy
    kernels, matching sklearn's pairwise_kernels exactly (gamma must be the
    resolved numeric). sklearn's KernelRidge fits NO intercept, so the
    caller centers the target on the train mean and adds it back — without
    that, regularization shrinks predictions toward 0 in log space (MAE 81k
    at alpha 1 on the imputed dataset) instead of toward the mean."""
    if kernel == "linear":
        k = x @ basis.T
    elif kernel == "rbf":
        sq = (np.sum(x**2, axis=1)[:, None] + np.sum(basis**2, axis=1)[None, :]
              - 2.0 * (x @ basis.T))
        k = np.exp(-gamma * np.maximum(sq, 0.0))
    elif kernel == "poly":
        k = (gamma * (x @ basis.T) + coef0) ** degree
    elif kernel == "sigmoid":
        k = np.tanh(gamma * (x @ basis.T) + coef0)
    else:
        raise ValueError(f"unknown kernel {kernel!r}")
    return k @ dual_coef


def train(dataset: Path, output: Path, hp_path: Path) -> None:
    hp = {**DEFAULTS, **json.loads(hp_path.read_text())}
    kernel = str(hp["kernel"])
    alpha = float(hp["alpha"])
    degree, coef0 = int(hp["degree"]), float(hp["coef0"])

    manifest = json.loads((dataset / "manifest.json").read_text())
    n_rows, n_cols = manifest["n_rows"], manifest["n_cols"]
    features = read_features(dataset, n_rows, n_cols)
    target = np.fromfile(dataset / "target.f32", dtype="<f4").astype(np.float64)
    row_ids = np.fromfile(dataset / "row_ids.u64", dtype="<u8")
    train_idx = np.fromfile(dataset / "train_idx.u32", dtype="<u4")
    test_idx = np.fromfile(dataset / "test_idx.u32", dtype="<u4")

    emit({"event": "log", "msg": f"dataset {manifest['dataset_id']}: "
          f"{len(train_idx)} train / {len(test_idx)} test rows, "
          f"{n_cols} features"})

    # Standardize with train-split statistics only.
    mean, std = fit_scaler(features[train_idx])
    output.mkdir(parents=True, exist_ok=True)
    (output / "scaler.json").write_text(json.dumps(
        {"mean": mean.tolist(), "std": std.tolist()}))
    train_x = (features[train_idx] - mean) / std
    test_x = (features[test_idx] - mean) / std

    # 0 means the svm predictor's "scale" convention; resolve to the numeric
    # value so the saved model is self-contained for the numpy predict path.
    gamma = float(hp["gamma"])
    if gamma <= 0.0:
        gamma = 1.0 / (n_cols * train_x.var())

    emit({"event": "log", "msg": f"kernel ridge kernel={kernel} "
          f"alpha={alpha:g} gamma={gamma:.6g}"
          + (f" degree={degree} coef0={coef0:g}" if kernel == "poly"
             else f" coef0={coef0:g}" if kernel == "sigmoid" else "")})

    # Center the target on the train mean (KernelRidge has no intercept).
    y_offset = float(target[train_idx].mean())
    model = KernelRidge(kernel=kernel, alpha=alpha, gamma=gamma,
                        degree=degree, coef0=coef0)
    model.fit(train_x, target[train_idx] - y_offset)
    emit({"event": "log", "msg": f"dense dual solution over "
          f"{train_x.shape[0]} train rows (target centered at "
          f"{y_offset:.4f})"})

    transform = manifest["target"]["transform"]
    predicted = np.maximum(
        invert_target(model.predict(test_x) + y_offset, transform), 0.0)
    actual = invert_target(target[test_idx], transform)

    metrics = compute_metrics(actual, predicted)
    predictions = [
        {"row_id": int(row_ids[i]), "actual": float(a), "predicted": float(p)}
        for i, a, p in zip(test_idx, actual, predicted)
    ]
    (output / "metrics.json").write_text(json.dumps(metrics))
    (output / "predictions.json").write_text(json.dumps(predictions))

    # Persist the model as JSON + the raw f32 basis (the standardized train
    # matrix); verify the numpy decision function reproduces sklearn before
    # trusting it for predict.
    dual_coef = model.dual_coef_.ravel().astype(np.float64)
    check = decision_function(test_x[:256], train_x, dual_coef,
                              kernel, gamma, degree, coef0)
    drift = float(np.max(np.abs(check - model.predict(test_x[:256]))))
    assert drift < 1e-6, f"serialized decision function drifts {drift:g}"

    train_x.astype("<f4").tofile(output / "basis.f32")
    (output / "model.json").write_text(json.dumps({
        "kernel": kernel,
        "alpha": alpha,
        "gamma": gamma,
        "degree": degree,
        "coef0": coef0,
        "y_offset": y_offset,
        "dual_coef": dual_coef.tolist(),
        "n_basis": int(train_x.shape[0]),
        "n_features": n_cols,
    }))

    emit({"event": "log", "msg": f"MAE {metrics['mae']:,.0f}  "
          f"RMSE {metrics['rmse']:,.0f}  medAPE {metrics['medape']:.1%}  "
          f"R² {metrics['r2']:.3f}"})
    emit({"event": "done"})


def predict(model_dir: Path, input_dir: Path, output: Path) -> None:
    """Contract v2 predict: standardize the input mini-artifact with the
    trained scaler, evaluate the stored KRR decision function with numpy,
    write price-space predictions."""
    model = json.loads((model_dir / "model.json").read_text())
    scaler = json.loads((model_dir / "scaler.json").read_text())
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    std = np.asarray(scaler["std"], dtype=np.float64)
    n_basis, n_feat = model["n_basis"], model["n_features"]
    basis = np.fromfile(model_dir / "basis.f32", dtype="<f4")
    assert basis.size == n_basis * n_feat, (
        f"basis.f32 has {basis.size} values, model says {n_basis}x{n_feat}")
    basis = basis.reshape(n_basis, n_feat).astype(np.float64)
    dual_coef = np.asarray(model["dual_coef"], dtype=np.float64)

    manifest = json.loads((input_dir / "manifest.json").read_text())
    n_rows, n_cols = manifest["n_rows"], manifest["n_cols"]
    assert len(mean) == n_cols, (
        f"scaler was fit on {len(mean)} columns, input has {n_cols}")
    features = read_features(input_dir, n_rows, n_cols)
    row_ids = np.fromfile(input_dir / "row_ids.u64", dtype="<u8")
    emit({"event": "log", "msg": f"predicting {n_rows} rows, "
          f"{n_cols} features, kernel ridge {model['kernel']} "
          f"({n_basis} basis rows)"})

    x = (features - mean) / std
    y = decision_function(x, basis, dual_coef, model["kernel"],
                          model["gamma"], model["degree"], model["coef0"])
    y += model["y_offset"]
    transform = manifest["target"]["transform"]
    predicted = np.maximum(invert_target(y, transform), 0.0)

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
