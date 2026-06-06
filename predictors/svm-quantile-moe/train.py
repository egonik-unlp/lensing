#!/usr/bin/env python3
"""Target-stratified mixture-of-experts SVM predictor. Implements the
predictor contract in README.md: train targets are cut into quantile bands
(a supervised regime split on the *target*, unlike svm-moe's unsupervised
GMM over features), one epsilon-SVR (rbf) expert is trained per band, and a
multinomial logistic gate predicts band membership from the features.
Predictions are the gate-probability-weighted blend of the experts (in
log-target space). Motivation (experiments/2026-06-04-svm-moe-cluster-scan.md):
the high tail is >80% of the squared error and feature-space GMM
clustering cannot isolate it — quantile bins do, by construction.

The model is saved as plain JSON + a binary support-vector matrix; predict
recomputes the gate softmax and the SVR decision functions with numpy, so
it needs no sklearn version pinning (same philosophy as ridge/svm/svm-moe).

Run from the repo's shared predictor venv (predictors/.venv, see
`zig build py-setup`)."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR

DEFAULTS = {
    "n_bands": 4,
    "C": 7.0,
    "epsilon": 0.05,
    "gamma": 0.0,  # 0 = sklearn's "scale", resolved per expert
    "gate_c": 1.0,  # logistic gate inverse regularization
    "seed": 42,
}

# A starved band still gets a viable expert: pad with the nearest train rows
# by target distance. Quantile bins are ~equal-sized by construction, so this
# only fires on degenerate targets (many ties at a band edge).
MIN_EXPERT_ROWS = 50


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


def gate_probs(x: np.ndarray, coef: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    """Logistic gate band probabilities, matching sklearn's predict_proba.
    Multinomial softmax for k classes; sklearn stores the binary case as a
    single sigmoid row."""
    scores = x @ coef.T + intercept[None, :]
    if coef.shape[0] == 1:  # binary: p(class 1) = sigmoid(score)
        p1 = 1.0 / (1.0 + np.exp(-scores[:, 0]))
        return np.stack([1.0 - p1, p1], axis=1)
    scores -= scores.max(axis=1, keepdims=True)
    p = np.exp(scores)
    return p / p.sum(axis=1, keepdims=True)


def rbf_decision(x: np.ndarray, sv: np.ndarray, dual_coef: np.ndarray,
                 intercept: float, gamma: float) -> np.ndarray:
    """SVR rbf decision function f(x) = dual_coef . K(SV, x) + b."""
    sq = (np.sum(x**2, axis=1)[:, None] + np.sum(sv**2, axis=1)[None, :]
          - 2.0 * (x @ sv.T))
    return np.exp(-gamma * np.maximum(sq, 0.0)) @ dual_coef + intercept


def blend(x: np.ndarray, gate: dict, experts: list[dict],
          svs: list[np.ndarray]) -> np.ndarray:
    """Gate-probability-weighted mixture prediction in transformed space."""
    probs = gate_probs(x, np.asarray(gate["coef"]), np.asarray(gate["intercept"]))
    out = np.zeros(len(x))
    for k, (e, sv) in enumerate(zip(experts, svs)):
        out += probs[:, k] * rbf_decision(
            x, sv, np.asarray(e["dual_coef"]), e["intercept"], e["gamma"])
    return out


def band_assign(y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Band index per target: edges are the interior quantile cuts, so
    band k = (edges[k-1], edges[k]]."""
    return np.searchsorted(edges, y, side="left")


def train(dataset: Path, output: Path, hp_path: Path) -> None:
    hp = {**DEFAULTS, **json.loads(hp_path.read_text())}
    n_bands, seed = int(hp["n_bands"]), int(hp["seed"])
    c, epsilon, gate_c = float(hp["C"]), float(hp["epsilon"]), float(hp["gate_c"])

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
    train_y = target[train_idx]

    # Supervised regime split: quantile bands over the train targets
    # (transformed space — equivalently target bands, the transform is
    # monotone). Interior edges only; searchsorted assigns bands.
    edges = np.quantile(train_y, np.linspace(0, 1, n_bands + 1)[1:-1])
    bands = band_assign(train_y, edges)
    transform = manifest["target"]["transform"]
    pretty = [f"${invert_target(np.asarray([e]), transform)[0]:,.0f}" for e in edges]
    emit({"event": "log", "msg": f"{n_bands} quantile bands, "
          f"edges at {', '.join(pretty)}"})

    # The soft gate: which band does a listing's *features* say it is in?
    gate_model = LogisticRegression(C=gate_c, max_iter=2000, random_state=seed)
    gate_model.fit(train_x, bands)
    gate = {
        "coef": gate_model.coef_.tolist(),
        "intercept": gate_model.intercept_.tolist(),
    }
    np_probs = gate_probs(train_x, gate_model.coef_, gate_model.intercept_)
    drift = float(np.max(np.abs(np_probs - gate_model.predict_proba(train_x))))
    assert drift < 1e-8, f"numpy gate probabilities drift {drift:g} from sklearn"
    gate_acc = float((np_probs.argmax(axis=1) == bands).mean())
    emit({"event": "log", "msg": f"gate train accuracy {gate_acc:.1%} "
          f"(C {gate_c})"})

    # One rbf-SVR expert per target band.
    experts, svs, fitted = [], [], []
    for k in range(n_bands):
        rows = np.where(bands == k)[0]
        if len(rows) < MIN_EXPERT_ROWS:  # degenerate split; pad by proximity
            center = train_y[rows].mean() if len(rows) else float(np.median(train_y))
            rows = np.argsort(np.abs(train_y - center))[:MIN_EXPERT_ROWS]
        x_k, y_k = train_x[rows], train_y[rows]
        gamma = float(hp["gamma"])
        if gamma <= 0.0:  # sklearn "scale" on this expert's training set
            gamma = 1.0 / (n_cols * x_k.var())
        svr = SVR(kernel="rbf", C=c, epsilon=epsilon, gamma=gamma,
                  cache_size=500)
        svr.fit(x_k, y_k)
        experts.append({
            "dual_coef": svr.dual_coef_.ravel().tolist(),
            "intercept": float(svr.intercept_[0]),
            "gamma": gamma,
            "n_sv": int(svr.support_vectors_.shape[0]),
        })
        svs.append(svr.support_vectors_.astype(np.float64))
        fitted.append(svr)
        lo = pretty[k - 1] if k > 0 else "$0"
        hi = pretty[k] if k < n_bands - 1 else "∞"
        emit({"event": "log", "msg": f"expert {k} ({lo}–{hi}): "
              f"{len(rows)} rows, {experts[-1]['n_sv']} SVs, "
              f"gamma {gamma:.6g}"})

    # Blended test predictions; verify the pure-numpy path against sklearn
    # before trusting it for the predict subcommand.
    test_probs = gate_model.predict_proba(test_x)
    sk_pred = np.zeros(len(test_x))
    for k in range(n_bands):
        sk_pred += test_probs[:, k] * fitted[k].predict(test_x)
    np_pred = blend(test_x, gate, experts, svs)
    drift = float(np.max(np.abs(np_pred - sk_pred)))
    assert drift < 1e-6, f"serialized mixture drifts {drift:g}"

    predicted = np.maximum(invert_target(np_pred, transform), 0.0)
    actual = invert_target(target[test_idx], transform)

    metrics = compute_metrics(actual, predicted)
    predictions = [
        {"row_id": int(row_ids[i]), "actual": float(a), "predicted": float(p)}
        for i, a, p in zip(test_idx, actual, predicted)
    ]
    (output / "metrics.json").write_text(json.dumps(metrics))
    (output / "predictions.json").write_text(json.dumps(predictions))

    np.concatenate(svs).astype("<f4").tofile(output / "support_vectors.f32")
    (output / "model.json").write_text(json.dumps({
        "kernel": "rbf",
        "C": c,
        "epsilon": epsilon,
        "n_bands": n_bands,
        "band_edges": edges.tolist(),
        "gate": gate,
        "experts": experts,
        "n_features": n_cols,
    }))

    # Per-band test breakdown by *true* band (where does the error live?),
    # plus the gate's view — the direct check of the high-tail claim.
    test_bands = band_assign(target[test_idx], edges)
    gate_assign = test_probs.argmax(axis=1)
    bands_report = []
    for k in range(n_bands):
        sel = test_bands == k
        if not sel.any():
            bands_report.append({"cluster": k, "n_test": 0})
            continue
        m = compute_metrics(actual[sel], predicted[sel])
        bands_report.append({
            "cluster": k,
            "n_test": int(sel.sum()),
            "median_target": float(np.median(actual[sel])),
            "mean_target": float(actual[sel].mean()),
            "mae": m["mae"],
            "medape": m["medape"],
            "share_of_sq_err": float(
                np.sum((predicted[sel] - actual[sel]) ** 2)
                / np.sum((predicted - actual) ** 2)),
            "gate_recall": float((gate_assign[sel] == k).mean()),
        })
        emit({"event": "log", "msg": f"band {k}: {bands_report[-1]['n_test']} "
              f"test rows, median {bands_report[-1]['median_target']:,.0f}, "
              f"MAE {m['mae']:,.0f}, medAPE {m['medape']:.1%}, "
              f"{bands_report[-1]['share_of_sq_err']:.0%} of sq err, "
              f"gate recall {bands_report[-1]['gate_recall']:.0%}"})
    (output / "cluster_report.json").write_text(json.dumps(bands_report))

    emit({"event": "log", "msg": f"MAE {metrics['mae']:,.0f}  "
          f"RMSE {metrics['rmse']:,.0f}  medAPE {metrics['medape']:.1%}  "
          f"R² {metrics['r2']:.3f}"})
    emit({"event": "done"})


def predict(model_dir: Path, input_dir: Path, output: Path) -> None:
    """Contract v2 predict: standardize the input mini-artifact with the
    trained scaler, evaluate the stored gate + expert mixture with numpy,
    write target-space predictions."""
    model = json.loads((model_dir / "model.json").read_text())
    scaler = json.loads((model_dir / "scaler.json").read_text())
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    std = np.asarray(scaler["std"], dtype=np.float64)
    n_feat = model["n_features"]
    flat = np.fromfile(model_dir / "support_vectors.f32", dtype="<f4")
    total_sv = sum(e["n_sv"] for e in model["experts"])
    assert flat.size == total_sv * n_feat, (
        f"support_vectors.f32 has {flat.size} values, "
        f"model says {total_sv}x{n_feat}")
    flat = flat.reshape(total_sv, n_feat).astype(np.float64)
    svs, off = [], 0
    for e in model["experts"]:
        svs.append(flat[off:off + e["n_sv"]])
        off += e["n_sv"]

    manifest = json.loads((input_dir / "manifest.json").read_text())
    n_rows, n_cols = manifest["n_rows"], manifest["n_cols"]
    assert len(mean) == n_cols, (
        f"scaler was fit on {len(mean)} columns, input has {n_cols}")
    features = read_features(input_dir, n_rows, n_cols)
    row_ids = np.fromfile(input_dir / "row_ids.u64", dtype="<u8")
    emit({"event": "log", "msg": f"predicting {n_rows} rows, {n_cols} "
          f"features, {model['n_bands']}-band target mixture "
          f"({total_sv} SVs)"})

    x = (features - mean) / std
    y = blend(x, model["gate"], model["experts"], svs)
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
