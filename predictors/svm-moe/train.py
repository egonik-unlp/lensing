#!/usr/bin/env python3
"""Mixture-of-experts SVM predictor. Implements the predictor contract in
README.md: a Gaussian mixture model soft-clusters the standardized feature
space, one epsilon-SVR (rbf) expert is trained per component with
responsibility sample weights, and predictions are the responsibility-
weighted blend of the experts (in log-price space). Motivation: the worst
errors of the global models concentrate on atypical segments (very
expensive / very cheap commercial sites); per-regime experts may fit those
better than one global function. A per-cluster test breakdown is written to
cluster_report.json to evaluate exactly that.

The model is saved as plain JSON + a binary support-vector matrix; predict
recomputes GMM responsibilities and the SVR decision functions with numpy,
so it needs no sklearn version pinning (same philosophy as ridge/svm).

Run from the repo's shared predictor venv (predictors/.venv, see
`zig build py-setup`)."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVR

DEFAULTS = {
    "n_clusters": 4,
    "covariance": "diag",  # diag | spherical (kept numpy-invertible)
    "C": 7.0,
    "epsilon": 0.05,
    "gamma": 0.0,  # 0 = sklearn's "scale", resolved per expert
    "seed": 42,
}

# Rows with responsibility below this are left out of an expert's training
# set (they contribute ~nothing at weight ~0 and only slow the QP down).
RESP_FLOOR = 0.02
# A starved component still gets an expert: its top rows by responsibility.
MIN_EXPERT_ROWS = 50
# Regularization added to GMM covariances; one-hot columns can be constant
# within a component, and this keeps their variances (and the numpy
# log-pdf below) well behaved.
REG_COVAR = 1e-3


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


def responsibilities(x: np.ndarray, weights: np.ndarray, means: np.ndarray,
                     variances: np.ndarray) -> np.ndarray:
    """GMM posterior p(component | x) for diagonal covariances, matching
    sklearn's predict_proba. Expanded quadratic form avoids the (n, k, d)
    intermediate: sum_d (x-mu)^2/v = x^2 @ (1/v) - 2 x @ (mu/v) + sum mu^2/v."""
    inv_v = 1.0 / variances  # (k, d)
    sq = (x**2) @ inv_v.T - 2.0 * (x @ (means * inv_v).T) \
        + np.sum(means**2 * inv_v, axis=1)[None, :]
    log_prob = -0.5 * (x.shape[1] * np.log(2.0 * np.pi)
                       + np.sum(np.log(variances), axis=1)[None, :] + sq)
    log_resp = np.log(weights)[None, :] + log_prob
    log_resp -= log_resp.max(axis=1, keepdims=True)
    resp = np.exp(log_resp)
    return resp / resp.sum(axis=1, keepdims=True)


def rbf_decision(x: np.ndarray, sv: np.ndarray, dual_coef: np.ndarray,
                 intercept: float, gamma: float) -> np.ndarray:
    """SVR rbf decision function f(x) = dual_coef . K(SV, x) + b."""
    sq = (np.sum(x**2, axis=1)[:, None] + np.sum(sv**2, axis=1)[None, :]
          - 2.0 * (x @ sv.T))
    return np.exp(-gamma * np.maximum(sq, 0.0)) @ dual_coef + intercept


def blend(x: np.ndarray, gmm: dict, experts: list[dict],
          svs: list[np.ndarray]) -> np.ndarray:
    """Responsibility-weighted mixture prediction in transformed space."""
    resp = responsibilities(x, np.asarray(gmm["weights"]),
                            np.asarray(gmm["means"]),
                            np.asarray(gmm["variances"]))
    out = np.zeros(len(x))
    for k, (e, sv) in enumerate(zip(experts, svs)):
        out += resp[:, k] * rbf_decision(
            x, sv, np.asarray(e["dual_coef"]), e["intercept"], e["gamma"])
    return out


def train(dataset: Path, output: Path, hp_path: Path) -> None:
    hp = {**DEFAULTS, **json.loads(hp_path.read_text())}
    n_clusters, cov = int(hp["n_clusters"]), str(hp["covariance"])
    c, epsilon, seed = float(hp["C"]), float(hp["epsilon"]), int(hp["seed"])

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

    # Soft clustering of the standardized feature space.
    emit({"event": "log", "msg": f"fitting gmm: {n_clusters} components, "
          f"{cov} covariance, seed {seed}"})
    gmm_model = GaussianMixture(
        n_components=n_clusters, covariance_type=cov, reg_covar=REG_COVAR,
        n_init=3, max_iter=200, random_state=seed)
    gmm_model.fit(train_x)
    variances = gmm_model.covariances_
    if cov == "spherical":  # expand (k,) to (k, d) so predict has one path
        variances = np.repeat(variances[:, None], n_cols, axis=1)
    gmm = {
        "weights": gmm_model.weights_.tolist(),
        "means": gmm_model.means_.tolist(),
        "variances": variances.tolist(),
    }
    resp = responsibilities(train_x, np.asarray(gmm["weights"]),
                            np.asarray(gmm["means"]),
                            np.asarray(gmm["variances"]))
    drift = float(np.max(np.abs(resp - gmm_model.predict_proba(train_x))))
    assert drift < 1e-8, f"numpy responsibilities drift {drift:g} from sklearn"

    # One responsibility-weighted SVR expert per component.
    experts, svs, fitted = [], [], []
    for k in range(n_clusters):
        rows = np.where(resp[:, k] >= RESP_FLOOR)[0]
        if len(rows) < MIN_EXPERT_ROWS:
            rows = np.argsort(-resp[:, k])[:MIN_EXPERT_ROWS]
        x_k, y_k, w_k = train_x[rows], train_y[rows], resp[rows, k]
        gamma = float(hp["gamma"])
        if gamma <= 0.0:  # sklearn "scale" on this expert's training set
            gamma = 1.0 / (n_cols * x_k.var())
        svr = SVR(kernel="rbf", C=c, epsilon=epsilon, gamma=gamma,
                  cache_size=500)
        svr.fit(x_k, y_k, sample_weight=w_k)
        experts.append({
            "dual_coef": svr.dual_coef_.ravel().tolist(),
            "intercept": float(svr.intercept_[0]),
            "gamma": gamma,
            "n_sv": int(svr.support_vectors_.shape[0]),
        })
        svs.append(svr.support_vectors_.astype(np.float64))
        fitted.append(svr)
        emit({"event": "log", "msg": f"expert {k}: {len(rows)} rows "
              f"(weight mass {resp[:, k].sum():,.0f}), "
              f"{experts[-1]['n_sv']} SVs, gamma {gamma:.6g}"})

    # Blended test predictions; verify the pure-numpy path against sklearn
    # before trusting it for the predict subcommand.
    test_resp = gmm_model.predict_proba(test_x)
    sk_pred = np.zeros(len(test_x))
    for k in range(n_clusters):
        sk_pred += test_resp[:, k] * fitted[k].predict(test_x)
    np_pred = blend(test_x, gmm, experts, svs)
    drift = float(np.max(np.abs(np_pred - sk_pred)))
    assert drift < 1e-6, f"serialized mixture drifts {drift:g}"

    transform = manifest["target"]["transform"]
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
        "n_clusters": n_clusters,
        "covariance": cov,
        "gmm": gmm,
        "experts": experts,
        "n_features": n_cols,
    }))

    # Per-cluster test breakdown (hard argmax assignment): does an
    # expensive/cheap segment exist, and is it where the error lives?
    assign = test_resp.argmax(axis=1)
    clusters = []
    for k in range(n_clusters):
        sel = assign == k
        if not sel.any():
            clusters.append({"cluster": k, "n_test": 0})
            continue
        m = compute_metrics(actual[sel], predicted[sel])
        clusters.append({
            "cluster": k,
            "n_test": int(sel.sum()),
            "median_price": float(np.median(actual[sel])),
            "mean_price": float(actual[sel].mean()),
            "mae": m["mae"],
            "medape": m["medape"],
            "share_of_sq_err": float(
                np.sum((predicted[sel] - actual[sel]) ** 2)
                / np.sum((predicted - actual) ** 2)),
        })
        emit({"event": "log", "msg": f"cluster {k}: {clusters[-1]['n_test']} "
              f"test rows, median ${clusters[-1]['median_price']:,.0f}, "
              f"MAE {m['mae']:,.0f}, medAPE {m['medape']:.1%}, "
              f"{clusters[-1]['share_of_sq_err']:.0%} of sq err"})
    (output / "cluster_report.json").write_text(json.dumps(clusters))

    emit({"event": "log", "msg": f"MAE {metrics['mae']:,.0f}  "
          f"RMSE {metrics['rmse']:,.0f}  medAPE {metrics['medape']:.1%}  "
          f"R² {metrics['r2']:.3f}"})
    emit({"event": "done"})


def predict(model_dir: Path, input_dir: Path, output: Path) -> None:
    """Contract v2 predict: standardize the input mini-artifact with the
    trained scaler, evaluate the stored GMM + expert mixture with numpy,
    write price-space predictions."""
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
          f"features, {model['n_clusters']}-component svm mixture "
          f"({total_sv} SVs)"})

    x = (features - mean) / std
    y = blend(x, model["gmm"], model["experts"], svs)
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
