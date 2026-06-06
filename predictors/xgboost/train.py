#!/usr/bin/env python3
"""XGBoost predictor. Implements the predictor contract in README.md with
numpy + xgboost: gradient-boosted trees fit in transformed (log) target
space, target-space metrics on the test split. Tree ensembles are
scale-invariant, so there is no standardization step and no scaler.json.

The model file is XGBoost's native UBJSON format (model.ubj), which is stable
across library versions — predict needs no pickle/version pinning.

Boosting rounds are reported as epoch events so the UI loss chart works, and
the server's STOP file is honored between rounds (supports_stop = true).

Early stopping (early_stopping_rounds > 0): a val_fraction slice of the TRAIN
rows is held out as the monitored set — the test split stays untouched for
final metrics — and the saved ensemble is truncated at best_iteration, so the
artifact keeps only the trees up to the validation optimum.

Run from the repo's shared predictor venv (predictors/.venv, see
`zig build py-setup`)."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

DEFAULTS = {
    "n_estimators": 500,
    "early_stopping_rounds": 0,  # 0 = off
    "val_fraction": 0.15,  # train-carved monitored split (early stopping only)
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,
    "seed": 42,
    "weight_gamma": 0.0,
}
MAX_WEIGHT_RATIO = 16.0  # cap so a single luxury row can't dominate the fit


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


def sample_weights(train_y: np.ndarray, transform: str, gamma: float) -> np.ndarray | None:
    """Target-proportional weights w = (target / median)^gamma, clipped so no
    row carries more than MAX_WEIGHT_RATIO x the median weight, normalized to
    mean 1 — counters the corpus-wide high-tail underprediction.
    gamma=0 (default) means uniform weights (returns None)."""
    if gamma <= 0.0:
        return None
    target = np.maximum(invert_target(train_y.astype(np.float64), transform), 0.0)
    w = (target / np.median(target)) ** gamma
    w = np.minimum(w, np.median(w) * MAX_WEIGHT_RATIO)
    return w / w.mean()


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


class ProgressCallback(xgb.callback.TrainingCallback):
    """Per-round epoch events + graceful stop on the server's STOP file."""

    def __init__(self, run_dir: Path, total: int):
        self.run_dir = run_dir
        self.total = total

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        emit({"event": "epoch", "epoch": epoch + 1, "total_epochs": self.total,
              "train_loss": evals_log["validation_0"]["rmse"][-1],
              "val_loss": evals_log["validation_1"]["rmse"][-1]})
        # Graceful stop (contract v2): the server drops a STOP file into the
        # run dir; stop boosting and run the normal end-of-training path.
        if (self.run_dir / "STOP").exists():
            emit({"event": "stopping"})
            emit({"event": "log", "msg": "stop requested; evaluating with "
                  f"the {epoch + 1} trees built so far"})
            return True
        return False


def train(dataset: Path, output: Path, hp_path: Path) -> None:
    hp = {**DEFAULTS, **json.loads(hp_path.read_text())}

    manifest = json.loads((dataset / "manifest.json").read_text())
    n_rows, n_cols = manifest["n_rows"], manifest["n_cols"]
    features = read_features(dataset, n_rows, n_cols)
    target = np.fromfile(dataset / "target.f32", dtype="<f4")
    row_ids = np.fromfile(dataset / "row_ids.u64", dtype="<u8")
    train_idx = np.fromfile(dataset / "train_idx.u32", dtype="<u4")
    test_idx = np.fromfile(dataset / "test_idx.u32", dtype="<u4")

    emit({"event": "log", "msg": f"dataset {manifest['dataset_id']}: "
          f"{len(train_idx)} train / {len(test_idx)} test rows, "
          f"{n_cols} features"})
    emit({"event": "log", "msg": f"xgboost {hp['n_estimators']} rounds, "
          f"depth {hp['max_depth']}, lr {hp['learning_rate']}, "
          f"subsample {hp['subsample']}, colsample {hp['colsample_bytree']}"})

    # No standardization: trees split on raw feature values.
    train_x, train_y = features[train_idx], target[train_idx]
    test_x, test_y = features[test_idx], target[test_idx]

    # Early stopping: carve a held-out validation set from the TRAIN rows to
    # monitor — never the test split, which stays untouched for final
    # metrics. xgboost monitors the last eval_set entry.
    es_rounds = int(hp["early_stopping_rounds"])
    if es_rounds > 0:
        n_val = max(1, round(len(train_idx) * float(hp["val_fraction"])))
        assert n_val < len(train_idx), "early-stop carve left no training rows"
        perm = np.random.default_rng(int(hp["seed"])).permutation(len(train_idx))
        val_pos, fit_pos = perm[:n_val], perm[n_val:]
        fit_x, fit_y = train_x[fit_pos], train_y[fit_pos]
        val_x, val_y = train_x[val_pos], train_y[val_pos]
        emit({"event": "log", "msg": f"early stopping: {es_rounds} rounds "
              f"patience, monitoring {n_val} held-out train rows "
              f"(val_fraction {hp['val_fraction']}); test split untouched"})
    else:
        fit_x, fit_y = train_x, train_y
        val_x, val_y = test_x, test_y

    transform = manifest["target"]["transform"]
    weights = sample_weights(fit_y, transform, float(hp["weight_gamma"]))
    if weights is not None:
        emit({"event": "log", "msg": f"target weights: gamma {hp['weight_gamma']}, "
              f"max {weights.max():.1f}x mean (clipped at "
              f"{MAX_WEIGHT_RATIO:.0f}x median)"})

    output.mkdir(parents=True, exist_ok=True)
    model = xgb.XGBRegressor(
        n_estimators=int(hp["n_estimators"]),
        max_depth=int(hp["max_depth"]),
        learning_rate=float(hp["learning_rate"]),
        subsample=float(hp["subsample"]),
        colsample_bytree=float(hp["colsample_bytree"]),
        min_child_weight=float(hp["min_child_weight"]),
        reg_lambda=float(hp["reg_lambda"]),
        random_state=int(hp["seed"]),
        eval_metric="rmse",
        early_stopping_rounds=es_rounds if es_rounds > 0 else None,
        callbacks=[ProgressCallback(output, int(hp["n_estimators"]))],
        n_jobs=-1,
    )
    model.fit(fit_x, fit_y, sample_weight=weights,
              eval_set=[(fit_x, fit_y), (val_x, val_y)], verbose=False)

    # Best-checkpoint restore: predict with and save only the trees up to the
    # best validation round (defensive fallback to all rounds in case a STOP
    # arrived before the early-stopping callback recorded a best iteration).
    best = getattr(model, "best_iteration", None) if es_rounds > 0 else None
    if best is not None:
        n_total = model.get_booster().num_boosted_rounds()
        emit({"event": "log", "msg": f"early stop: best iteration "
              f"{best + 1} of {n_total} rounds; saving truncated ensemble"})
        predicted_t = model.predict(test_x, iteration_range=(0, best + 1))
    else:
        predicted_t = model.predict(test_x)
    predicted = np.maximum(invert_target(predicted_t, transform), 0.0)
    actual = invert_target(test_y.astype(np.float64), transform)

    metrics = compute_metrics(actual, predicted)
    predictions = [
        {"row_id": int(row_ids[i]), "actual": float(a), "predicted": float(p)}
        for i, a, p in zip(test_idx, actual, predicted)
    ]
    (output / "metrics.json").write_text(json.dumps(metrics))
    (output / "predictions.json").write_text(json.dumps(predictions))
    if best is not None:
        model.get_booster()[: best + 1].save_model(output / "model.ubj")
    else:
        model.save_model(output / "model.ubj")

    emit({"event": "log", "msg": f"MAE {metrics['mae']:,.0f}  "
          f"RMSE {metrics['rmse']:,.0f}  medAPE {metrics['medape']:.1%}  "
          f"R² {metrics['r2']:.3f}"})
    emit({"event": "done"})


def predict(model_dir: Path, input_dir: Path, output: Path) -> None:
    """Contract v2 predict: load the native-format booster, predict on the
    server-featurized mini-artifact, write target-space predictions."""
    model = xgb.XGBRegressor()
    model.load_model(model_dir / "model.ubj")

    manifest = json.loads((input_dir / "manifest.json").read_text())
    n_rows, n_cols = manifest["n_rows"], manifest["n_cols"]
    assert model.n_features_in_ == n_cols, (
        f"model was fit on {model.n_features_in_} columns, input has {n_cols}")
    features = read_features(input_dir, n_rows, n_cols)
    row_ids = np.fromfile(input_dir / "row_ids.u64", dtype="<u8")
    emit({"event": "log", "msg": f"predicting {n_rows} rows, {n_cols} "
          f"features, xgboost {model.get_booster().num_boosted_rounds()} trees"})

    transform = manifest["target"]["transform"]
    predicted = np.maximum(invert_target(model.predict(features), transform), 0.0)

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
