#!/usr/bin/env python3
"""XGBoost classifier (binary + multiclass). Gradient-boosted trees optimizing
logistic / softmax loss; emits calibrated class probabilities and classification
metrics (the proper objective for a categorical target).

Model file is XGBoost's native UBJSON (model.ubj); meta.json records the class
vocabulary so predict can reconstruct canonical column order.

Run from the shared predictor venv (predictors/.venv)."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _clf_common as clf  # noqa: E402

DEFAULTS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,
    "seed": 42,
}


def build_model(hp: dict, k: int) -> xgb.XGBClassifier:
    objective = "binary:logistic" if k == 2 else "multi:softprob"
    return xgb.XGBClassifier(
        objective=objective,
        num_class=None if k == 2 else k,
        n_estimators=int(hp["n_estimators"]),
        max_depth=int(hp["max_depth"]),
        learning_rate=float(hp["learning_rate"]),
        subsample=float(hp["subsample"]),
        colsample_bytree=float(hp["colsample_bytree"]),
        min_child_weight=float(hp["min_child_weight"]),
        reg_lambda=float(hp["reg_lambda"]),
        random_state=int(hp["seed"]),
        eval_metric="logloss" if k == 2 else "mlogloss",
        n_jobs=-1,
    )


def train(dataset: Path, output: Path, hp_path: Path) -> None:
    hp = {**DEFAULTS, **json.loads(hp_path.read_text())}
    manifest, x, target, row_ids, train_idx, test_idx = clf.load_dataset(dataset)
    y, classes, k = clf.encode_labels(target)
    assert k >= 2, f"classification needs >=2 classes, target has {k}"
    clf.emit({"event": "log", "msg": f"dataset {manifest['dataset_id']}: "
              f"{len(train_idx)} train / {len(test_idx)} test rows, {manifest['n_cols']} cols; "
              f"{k} classes {classes}"})
    clf.emit({"event": "log", "msg": f"xgboost-classifier {hp['n_estimators']} rounds, "
              f"depth {hp['max_depth']}, lr {hp['learning_rate']}"})

    model = build_model(hp, k)
    model.fit(x[train_idx], y[train_idx])

    proba = clf.align_proba(model.predict_proba(x[test_idx]), model.classes_, k)
    y_test = y[test_idx]
    metrics = clf.classification_metrics(y_test, proba, k)
    predictions = clf.predictions_payload(row_ids, test_idx, y_test, proba, k)

    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics))
    (output / "predictions.json").write_text(json.dumps(predictions))
    model.get_booster().save_model(output / "model.ubj")
    (output / "meta.json").write_text(json.dumps({"classes": classes, "k": k}))

    auc = metrics.get("auc")
    clf.emit({"event": "log", "msg": f"AUC {auc:.4f}  logloss {metrics['logloss']:.4f}  "
              f"acc {metrics['accuracy']:.3f}"})
    clf.emit({"event": "done"})


def predict(model_dir: Path, input_dir: Path, output: Path) -> None:
    meta = json.loads((model_dir / "meta.json").read_text())
    k = int(meta["k"])
    model = xgb.XGBClassifier()
    model.load_model(model_dir / "model.ubj")

    manifest = json.loads((input_dir / "manifest.json").read_text())
    n_rows, n_cols = manifest["n_rows"], manifest["n_cols"]
    x = clf.read_features(input_dir, n_rows, n_cols)
    row_ids = np.fromfile(input_dir / "row_ids.u64", dtype="<u8")
    proba = clf.align_proba(model.predict_proba(x), model.classes_, k)
    clf.emit({"event": "log", "msg": f"predicting {n_rows} rows, {k} classes"})
    output.write_text(json.dumps(clf.inference_payload(row_ids, proba, k)))
    clf.emit({"event": "done"})


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
