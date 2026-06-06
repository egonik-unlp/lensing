#!/usr/bin/env python3
"""Ridge regression predictor, adapted to this dataset's limitations.
Implements the predictor contract in README.md with numpy + scikit-learn:
standardize on the train split, fit a linear model in transformed (log)
target space, report target-space metrics. Adaptations:

- auto alpha (default): the feature matrix is ~200 correlated columns (PCA
  dims capture only ~77% embedding variance), so a fixed alpha is arbitrary —
  RidgeCV picks one by efficient LOOCV over a log-spaced grid.
- weight_gamma: every model on this corpus underpredicts the high tail
  (~-34% medbias on the high tail); target-proportional sample weights counter that
  compression bias.
- loss=huber: the cheap/luxury tails are outlier-heavy even after quality
  filters; Huber loss resists them at the cost of an iterative solver.

The model file stays plain JSON (coefficients + intercept) regardless of
loss/weights, so predict needs no sklearn version pinning — it's just
X @ coef + intercept.

Run from the repo's shared predictor venv (predictors/.venv, see
`zig build py-setup`)."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import HuberRegressor, Ridge, RidgeCV
from sklearn.model_selection import KFold

DEFAULT_ALPHA = 1.0
# Ridge penalizes the sum of squared errors; HuberRegressor penalizes the
# mean loss, so its alpha lives ~n lower on the scale. Separate CV grids.
RIDGE_ALPHA_GRID = np.logspace(-2, 5, 29)
HUBER_ALPHA_GRID = np.logspace(-5, 3, 17)
MAX_WEIGHT_RATIO = 16.0  # cap so a single luxury row can't dominate the fit


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


def sample_weights(train_y: np.ndarray, transform: str, gamma: float) -> np.ndarray | None:
    """Target-proportional weights w = (target / median)^gamma, clipped so no
    row carries more than MAX_WEIGHT_RATIO x the median weight, normalized to
    mean 1. gamma=0 (default) means uniform weights (returns None)."""
    if gamma <= 0.0:
        return None
    target = np.maximum(invert_target(train_y, transform), 0.0)
    w = (target / np.median(target)) ** gamma
    w = np.minimum(w, np.median(w) * MAX_WEIGHT_RATIO)
    return w / w.mean()


def fit_huber_cv(train_x: np.ndarray, train_y: np.ndarray,
                 weights: np.ndarray | None, epsilon: float) -> tuple[HuberRegressor, float]:
    """5-fold CV over HUBER_ALPHA_GRID (no RidgeCV equivalent for Huber, and
    its alpha is on a different scale). Selects by weighted MSE in transformed
    space, then refits on the full train split."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kf.split(train_x))
    best_alpha, best_mse = None, np.inf
    for alpha in HUBER_ALPHA_GRID:
        mse_sum, n_sum = 0.0, 0.0
        for fit_i, val_i in folds:
            m = HuberRegressor(alpha=alpha, epsilon=epsilon, max_iter=1000)
            m.fit(train_x[fit_i], train_y[fit_i],
                  sample_weight=None if weights is None else weights[fit_i])
            err2 = (m.predict(train_x[val_i]) - train_y[val_i]) ** 2
            w = np.ones(len(val_i)) if weights is None else weights[val_i]
            mse_sum += float(np.sum(w * err2))
            n_sum += float(np.sum(w))
        mse = mse_sum / n_sum
        if mse < best_mse:
            best_alpha, best_mse = float(alpha), mse
    model = HuberRegressor(alpha=best_alpha, epsilon=epsilon, max_iter=1000)
    model.fit(train_x, train_y, sample_weight=weights)
    return model, best_alpha


def train(dataset: Path, output: Path, hp_path: Path) -> None:
    hp = json.loads(hp_path.read_text())
    auto_alpha = bool(hp.get("auto_alpha", True))
    alpha = float(hp.get("alpha", DEFAULT_ALPHA))
    weight_gamma = float(hp.get("weight_gamma", 0.0))
    loss = str(hp.get("loss", "squared"))
    huber_epsilon = float(hp.get("huber_epsilon", 1.35))

    manifest = json.loads((dataset / "manifest.json").read_text())
    n_rows, n_cols = manifest["n_rows"], manifest["n_cols"]
    features = read_features(dataset, n_rows, n_cols)
    target = np.fromfile(dataset / "target.f32", dtype="<f4").astype(np.float64)
    row_ids = np.fromfile(dataset / "row_ids.u64", dtype="<u8")
    train_idx = np.fromfile(dataset / "train_idx.u32", dtype="<u4")
    test_idx = np.fromfile(dataset / "test_idx.u32", dtype="<u4")

    emit({"event": "log", "msg": f"dataset {manifest['dataset_id']}: "
          f"{len(train_idx)} train / {len(test_idx)} test rows, "
          f"{n_cols} features, loss {loss}, "
          f"alpha {'auto (CV)' if auto_alpha else alpha}"})

    # Standardize with train-split statistics only.
    mean, std = fit_scaler(features[train_idx])
    output.mkdir(parents=True, exist_ok=True)
    (output / "scaler.json").write_text(json.dumps(
        {"mean": mean.tolist(), "std": std.tolist()}))
    train_x = (features[train_idx] - mean) / std
    test_x = (features[test_idx] - mean) / std

    transform = manifest["target"]["transform"]
    train_y = target[train_idx]
    weights = sample_weights(train_y, transform, weight_gamma)
    if weights is not None:
        emit({"event": "log", "msg": f"target weights: gamma {weight_gamma}, "
              f"max {weights.max():.1f}x mean (clipped at "
              f"{MAX_WEIGHT_RATIO:.0f}x median)"})

    if loss == "huber":
        if auto_alpha:
            model, alpha = fit_huber_cv(train_x, train_y, weights, huber_epsilon)
        else:
            model = HuberRegressor(alpha=alpha, epsilon=huber_epsilon,
                                   max_iter=1000)
            model.fit(train_x, train_y, sample_weight=weights)
    elif auto_alpha:
        model = RidgeCV(alphas=RIDGE_ALPHA_GRID, fit_intercept=True)
        model.fit(train_x, train_y, sample_weight=weights)
        alpha = float(model.alpha_)
    else:
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(train_x, train_y, sample_weight=weights)
    if auto_alpha:
        emit({"event": "log", "msg": f"CV selected alpha {alpha:g}"})

    predicted = np.maximum(invert_target(model.predict(test_x), transform), 0.0)
    actual = invert_target(target[test_idx], transform)

    metrics = compute_metrics(actual, predicted)
    predictions = [
        {"row_id": int(row_ids[i]), "actual": float(a), "predicted": float(p)}
        for i, a, p in zip(test_idx, actual, predicted)
    ]
    (output / "metrics.json").write_text(json.dumps(metrics))
    (output / "predictions.json").write_text(json.dumps(predictions))
    (output / "model.json").write_text(json.dumps({
        "coef": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "alpha": alpha,  # effective (CV-chosen or fixed)
        "auto_alpha": auto_alpha,
        "loss": loss,
        "weight_gamma": weight_gamma,
        "huber_epsilon": huber_epsilon,
        "n_features": n_cols,
    }))

    emit({"event": "log", "msg": f"MAE {metrics['mae']:,.0f}  "
          f"RMSE {metrics['rmse']:,.0f}  medAPE {metrics['medape']:.1%}  "
          f"R² {metrics['r2']:.3f}"})
    emit({"event": "done"})


def predict(model_dir: Path, input_dir: Path, output: Path) -> None:
    """Contract v2 predict: standardize the input mini-artifact with the
    trained scaler, apply the stored linear model, write target-space
    predictions."""
    model = json.loads((model_dir / "model.json").read_text())
    scaler = json.loads((model_dir / "scaler.json").read_text())
    coef = np.asarray(model["coef"], dtype=np.float64)
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    std = np.asarray(scaler["std"], dtype=np.float64)

    manifest = json.loads((input_dir / "manifest.json").read_text())
    n_rows, n_cols = manifest["n_rows"], manifest["n_cols"]
    assert len(mean) == n_cols, (
        f"scaler was fit on {len(mean)} columns, input has {n_cols}")
    features = read_features(input_dir, n_rows, n_cols)
    row_ids = np.fromfile(input_dir / "row_ids.u64", dtype="<u8")
    emit({"event": "log", "msg": f"predicting {n_rows} rows, "
          f"{n_cols} features, ridge alpha {model['alpha']}"})

    x = (features - mean) / std
    transform = manifest["target"]["transform"]
    predicted = np.maximum(
        invert_target(x @ coef + model["intercept"], transform), 0.0)

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
