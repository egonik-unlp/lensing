#!/usr/bin/env python3
"""Median-by-category baseline. Implements the predictor contract in
README.md using only the Python stdlib, proving the dataset artifact and
contract are language-neutral. The target field comes from the dataset
manifest; rows are grouped by the manifest's first one-hot group (the
domain's leading categorical), falling back to a single global median."""

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from statistics import median


def read_u32(path: Path) -> list[int]:
    data = path.read_bytes()
    return list(struct.unpack(f"<{len(data) // 4}I", data))


def read_u64(path: Path) -> list[int]:
    data = path.read_bytes()
    return list(struct.unpack(f"<{len(data) // 8}Q", data))


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def group_of(item: dict, group_field: str | None) -> str:
    if group_field is None:
        return ""
    v = item.get(group_field)
    return v if isinstance(v, str) else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    train = sub.add_parser("train")
    train.add_argument("--dataset", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--hyperparams", required=True, type=Path)
    pred = sub.add_parser("predict")
    pred.add_argument("--model", required=True, type=Path)
    pred.add_argument("--input", required=True, type=Path)
    pred.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    if args.cmd == "predict":
        predict(args.model, args.input, args.output)
        return

    manifest = json.loads((args.dataset / "manifest.json").read_text())
    items = json.loads((args.dataset / "items.json").read_text())
    row_ids = read_u64(args.dataset / "row_ids.u64")
    train_idx = read_u32(args.dataset / "train_idx.u32")
    test_idx = read_u32(args.dataset / "test_idx.u32")

    # The manifest's target field is qualified by the payload root
    # (e.g. "metadata.price"); items.json keys are unqualified.
    target_field = manifest["target"]["field"].split(".")[-1]
    group_field = next(
        (c["kind"]["group"] for c in manifest["columns"] if c["kind"]["type"] == "onehot"),
        None,
    )

    emit({"event": "log", "msg": f"dataset {manifest['dataset_id']}: "
          f"{len(train_idx)} train / {len(test_idx)} test rows, "
          f"target {target_field}, grouped by {group_field or '(none: global median)'}"})

    # Median target per group over the train split, in target space.
    by_group: dict[str, list[float]] = {}
    train_targets: list[float] = []
    for i in train_idx:
        item = items[str(row_ids[i])]
        value = float(item[target_field])
        by_group.setdefault(group_of(item, group_field), []).append(value)
        train_targets.append(value)
    medians = {g: median(v) for g, v in by_group.items()}
    global_median = median(train_targets)
    emit({"event": "log", "msg": "train medians: " + ", ".join(
        f"{g}={m:,.0f}" for g, m in sorted(medians.items()))})

    predictions = []
    abs_err = sq_err = 0.0
    apes = []
    actuals = []
    for i in test_idx:
        item = items[str(row_ids[i])]
        actual = float(item[target_field])
        predicted = medians.get(group_of(item, group_field), global_median)
        predictions.append(
            {"row_id": row_ids[i], "actual": actual, "predicted": predicted})
        e = predicted - actual
        abs_err += abs(e)
        sq_err += e * e
        apes.append(abs(e / actual))
        actuals.append(actual)

    n = len(test_idx)
    mean_actual = sum(actuals) / n
    ss_tot = sum((a - mean_actual) ** 2 for a in actuals)
    metrics = {
        "mae": abs_err / n,
        "rmse": math.sqrt(sq_err / n),
        "r2": 1.0 - (sq_err / ss_tot),
        "mape": sum(apes) / n,
        "medape": median(apes),
        "n_test": n,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics))
    (args.output / "predictions.json").write_text(json.dumps(predictions))
    (args.output / "model.json").write_text(json.dumps(
        {"medians": medians, "global_median": global_median,
         "group_field": group_field}))

    emit({"event": "log", "msg": f"MAE {metrics['mae']:,.0f}  "
          f"medAPE {metrics['medape']:.1%}  R² {metrics['r2']:.3f}"})
    emit({"event": "done"})


def predict(model_dir: Path, input_dir: Path, output: Path) -> None:
    """Contract v2 predict: load the trained medians, read the input
    mini-artifact's items.json (this predictor is payload-based, the feature
    matrix is irrelevant to it), write target-space predictions."""
    model = json.loads((model_dir / "model.json").read_text())
    medians = model["medians"]
    global_median = model["global_median"]
    # Models frozen before the field went generic grouped by propertyType.
    group_field = model.get("group_field", "propertyType")

    items = json.loads((input_dir / "items.json").read_text())
    row_ids = read_u64(input_dir / "row_ids.u64")
    emit({"event": "log", "msg": f"predicting {len(row_ids)} rows "
          f"from {len(medians)} trained medians"})

    predictions = [
        {"row_id": rid,
         "predicted": medians.get(group_of(items[str(rid)], group_field), global_median)}
        for rid in row_ids
    ]
    output.write_text(json.dumps(predictions))
    emit({"event": "done"})


if __name__ == "__main__":
    main()
