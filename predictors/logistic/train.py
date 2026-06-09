#!/usr/bin/env python3
"""Logistic regression classifier (binary + multinomial softmax). Standardizes
on the train split, fits a calibrated linear classifier, emits class
probabilities + classification metrics. A clean linear baseline on the
categorical-identity features.

Model + scaler are persisted via joblib; meta.json records the class vocabulary.

Run from the shared predictor venv (predictors/.venv)."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _clf_common as clf  # noqa: E402

DEFAULTS = {
    "C": 1.0,
    "max_iter": 2000,
    "seed": 42,
}


def train(dataset: Path, output: Path, hp_path: Path) -> None:
    hp = {**DEFAULTS, **json.loads(hp_path.read_text())}
    manifest, x, target, row_ids, train_idx, test_idx = clf.load_dataset(dataset)
    y, classes, k = clf.encode_labels(target)
    assert k >= 2, f"classification needs >=2 classes, target has {k}"
    clf.emit({"event": "log", "msg": f"dataset {manifest['dataset_id']}: "
              f"{len(train_idx)} train / {len(test_idx)} test rows; {k} classes {classes}"})

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=float(hp["C"]), max_iter=int(hp["max_iter"]),
                           random_state=int(hp["seed"])),
    )
    model.fit(x[train_idx], y[train_idx])
    lr = model.steps[-1][1]

    proba = clf.align_proba(model.predict_proba(x[test_idx]), lr.classes_, k)
    y_test = y[test_idx]
    metrics = clf.classification_metrics(y_test, proba, k)
    predictions = clf.predictions_payload(row_ids, test_idx, y_test, proba, k)

    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics))
    (output / "predictions.json").write_text(json.dumps(predictions))
    joblib.dump(model, output / "model.joblib")
    (output / "meta.json").write_text(json.dumps({"classes": classes, "k": k}))

    clf.emit({"event": "log", "msg": f"AUC {metrics.get('auc'):.4f}  "
              f"logloss {metrics['logloss']:.4f}  acc {metrics['accuracy']:.3f}"})
    clf.emit({"event": "done"})


def predict(model_dir: Path, input_dir: Path, output: Path) -> None:
    meta = json.loads((model_dir / "meta.json").read_text())
    k = int(meta["k"])
    model = joblib.load(model_dir / "model.joblib")
    lr = model.steps[-1][1]

    manifest = json.loads((input_dir / "manifest.json").read_text())
    n_rows, n_cols = manifest["n_rows"], manifest["n_cols"]
    x = clf.read_features(input_dir, n_rows, n_cols)
    row_ids = np.fromfile(input_dir / "row_ids.u64", dtype="<u8")
    proba = clf.align_proba(model.predict_proba(x), lr.classes_, k)
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
