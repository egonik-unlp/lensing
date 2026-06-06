#!/usr/bin/env python3
"""SVM (epsilon-SVR) predictor. Implements the predictor contract in
README.md with numpy + scikit-learn: standardize on the train split, fit
sklearn.svm.SVR in transformed (log) target space with a selectable
kernel (rbf/linear/poly/sigmoid), report target-space metrics. The model is
saved as plain JSON + a binary support-vector matrix, so predict needs no
sklearn version pinning — the decision function is recomputed with numpy.

Run from the repo's shared predictor venv (predictors/.venv, see
`zig build py-setup`)."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.svm import SVR

DEFAULTS = {
    "kernel": "rbf",
    "C": 1.0,
    "epsilon": 0.1,
    "gamma": 0.0,  # 0 = sklearn's "scale": 1 / (n_features * X.var())
    "degree": 3,
    "coef0": 0.0,
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


def decision_function(x: np.ndarray, sv: np.ndarray, dual_coef: np.ndarray,
                      intercept: float, kernel: str, gamma: float,
                      degree: int, coef0: float) -> np.ndarray:
    """SVR decision function f(x) = dual_coef . K(SV, x) + b with numpy
    kernels, matching sklearn exactly (gamma must be the resolved numeric)."""
    if kernel == "linear":
        k = x @ sv.T
    elif kernel == "rbf":
        sq = (np.sum(x**2, axis=1)[:, None] + np.sum(sv**2, axis=1)[None, :]
              - 2.0 * (x @ sv.T))
        k = np.exp(-gamma * np.maximum(sq, 0.0))
    elif kernel == "poly":
        k = (gamma * (x @ sv.T) + coef0) ** degree
    elif kernel == "sigmoid":
        k = np.tanh(gamma * (x @ sv.T) + coef0)
    else:
        raise ValueError(f"unknown kernel {kernel!r}")
    return k @ dual_coef + intercept


def train(dataset: Path, output: Path, hp_path: Path) -> None:
    hp = {**DEFAULTS, **json.loads(hp_path.read_text())}
    kernel = str(hp["kernel"])
    c, epsilon = float(hp["C"]), float(hp["epsilon"])
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

    # 0 means sklearn's "scale"; resolve to the numeric value so the saved
    # model is self-contained for the numpy predict path.
    gamma = float(hp["gamma"])
    if gamma <= 0.0:
        gamma = 1.0 / (n_cols * train_x.var())

    emit({"event": "log", "msg": f"svr kernel={kernel} C={c:g} "
          f"epsilon={epsilon:g} gamma={gamma:.6g}"
          + (f" degree={degree} coef0={coef0:g}" if kernel == "poly"
             else f" coef0={coef0:g}" if kernel == "sigmoid" else "")})

    model = SVR(kernel=kernel, C=c, epsilon=epsilon, gamma=gamma,
                degree=degree, coef0=coef0, cache_size=500)
    model.fit(train_x, target[train_idx])
    emit({"event": "log", "msg": f"{len(model.support_)} support vectors "
          f"({len(model.support_) / len(train_idx):.0%} of train rows)"})

    transform = manifest["target"]["transform"]
    predicted = np.maximum(invert_target(model.predict(test_x), transform), 0.0)
    actual = invert_target(target[test_idx], transform)

    metrics = compute_metrics(actual, predicted)
    predictions = [
        {"row_id": int(row_ids[i]), "actual": float(a), "predicted": float(p)}
        for i, a, p in zip(test_idx, actual, predicted)
    ]
    (output / "metrics.json").write_text(json.dumps(metrics))
    (output / "predictions.json").write_text(json.dumps(predictions))

    # Persist the model as JSON + raw f32 support vectors; verify the numpy
    # decision function reproduces sklearn before trusting it for predict.
    sv = model.support_vectors_.astype(np.float64)
    dual_coef = model.dual_coef_.ravel().astype(np.float64)
    intercept = float(model.intercept_[0])
    check = decision_function(test_x[:256], sv, dual_coef, intercept,
                              kernel, gamma, degree, coef0)
    drift = float(np.max(np.abs(check - model.predict(test_x[:256]))))
    assert drift < 1e-6, f"serialized decision function drifts {drift:g}"

    sv.astype("<f4").tofile(output / "support_vectors.f32")
    (output / "model.json").write_text(json.dumps({
        "kernel": kernel,
        "C": c,
        "epsilon": epsilon,
        "gamma": gamma,
        "degree": degree,
        "coef0": coef0,
        "dual_coef": dual_coef.tolist(),
        "intercept": intercept,
        "n_sv": int(sv.shape[0]),
        "n_features": n_cols,
    }))

    emit({"event": "log", "msg": f"MAE {metrics['mae']:,.0f}  "
          f"RMSE {metrics['rmse']:,.0f}  medAPE {metrics['medape']:.1%}  "
          f"R² {metrics['r2']:.3f}"})
    emit({"event": "done"})


def predict(model_dir: Path, input_dir: Path, output: Path) -> None:
    """Contract v2 predict: standardize the input mini-artifact with the
    trained scaler, evaluate the stored SVR decision function with numpy,
    write target-space predictions."""
    model = json.loads((model_dir / "model.json").read_text())
    scaler = json.loads((model_dir / "scaler.json").read_text())
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    std = np.asarray(scaler["std"], dtype=np.float64)
    n_sv, n_feat = model["n_sv"], model["n_features"]
    sv = np.fromfile(model_dir / "support_vectors.f32", dtype="<f4")
    assert sv.size == n_sv * n_feat, (
        f"support_vectors.f32 has {sv.size} values, model says {n_sv}x{n_feat}")
    sv = sv.reshape(n_sv, n_feat).astype(np.float64)
    dual_coef = np.asarray(model["dual_coef"], dtype=np.float64)

    manifest = json.loads((input_dir / "manifest.json").read_text())
    n_rows, n_cols = manifest["n_rows"], manifest["n_cols"]
    assert len(mean) == n_cols, (
        f"scaler was fit on {len(mean)} columns, input has {n_cols}")
    features = read_features(input_dir, n_rows, n_cols)
    row_ids = np.fromfile(input_dir / "row_ids.u64", dtype="<u8")
    emit({"event": "log", "msg": f"predicting {n_rows} rows, "
          f"{n_cols} features, svr kernel {model['kernel']} ({n_sv} SVs)"})

    x = (features - mean) / std
    y = decision_function(x, sv, dual_coef, model["intercept"],
                          model["kernel"], model["gamma"],
                          model["degree"], model["coef0"])
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
