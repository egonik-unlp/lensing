#!/usr/bin/env python3
"""1D CNN price predictor (PyTorch, CPU). Implements the predictor contract
in README.md; mirrors crates/predictor-burn-cnn (burn) and
predictors/flux-cnn (Flux.jl) layer for layer.

The leading PCA columns of the feature vector are convolved as a 1-channel
signal; the metadata tail (numeric + one-hots) joins the dense head after
global pooling. Loss values are MSE in transformed (log-price) target space.

Run from the repo's shared predictor venv (predictors/.venv, see
`zig build py-setup`)."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

DEFAULTS = {
    "epochs": 50,
    "lr": 1e-3,
    "batch_size": 256,
    "channels": [32, 64],
    "kernel_size": 3,
    "dense_hidden": 64,
    "dropout": 0.1,
    "checkpoint_every": 0,
}
SEED = 1337


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def load_hp(path: Path) -> dict:
    hp = {**DEFAULTS, **json.loads(path.read_text())}
    assert hp["epochs"] >= 1, "epochs must be >= 1"
    assert hp["batch_size"] >= 1, "batch_size must be >= 1"
    assert hp["channels"], "channels must have at least one layer"
    assert hp["kernel_size"] >= 1, "kernel_size must be >= 1"
    return hp


def count_pca(columns: list[dict]) -> int:
    """Leading PCA columns = the conv stack's signal width. Re-derived from
    whichever manifest accompanies the features, so train and predict always
    agree with the data on disk."""
    n = 0
    for c in columns:
        if c["kind"]["type"] != "pca":
            break
        n += 1
    return n


def read_features(dir: Path, n_rows: int, n_cols: int) -> np.ndarray:
    x = np.fromfile(dir / "features.f32", dtype="<f4")
    assert x.size == n_rows * n_cols, (
        f"features.f32 has {x.size} values, manifest says {n_rows}x{n_cols}")
    return x.reshape(n_rows, n_cols)


def fit_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Train-split standardization matching the other predictors' scaler:
    population std, near-constant columns get std 1.0."""
    mean = x.astype(np.float64).mean(axis=0)
    std = x.astype(np.float64).std(axis=0)  # population (ddof=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean, std


def invert_target(y: np.ndarray, transform: str) -> np.ndarray:
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


class Cnn(nn.Module):
    """Conv1d stack over the PCA signal + dense head over (pooled ⊕ meta)."""

    def __init__(self, n_pca: int, n_meta: int, channels: list[int],
                 kernel_size: int, dense_hidden: int, dropout: float):
        super().__init__()
        self.n_pca = n_pca
        k = kernel_size
        convs = []
        prev_ch = 1
        for ch in channels:
            # Symmetric k//2 padding, same as burn-cnn and flux-cnn.
            convs.append(nn.Conv1d(prev_ch, ch, k, padding=k // 2))
            prev_ch = ch
        self.convs = nn.ModuleList(convs)
        self.pool = nn.MaxPool1d(2)  # stride defaults to kernel_size
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dense = nn.Linear(prev_ch + n_meta, dense_hidden)
        self.output = nn.Linear(dense_hidden, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pca, meta = x[:, :self.n_pca], x[:, self.n_pca:]
        # PyTorch Conv1d is NCL: (batch, channels, length).
        signal = pca.unsqueeze(1)
        for conv in self.convs:
            signal = self.relu(conv(signal))
            # Halve the length, but never pool a length-1 signal away.
            if signal.shape[2] >= 2:
                signal = self.pool(signal)
            signal = self.dropout(signal)
        pooled = self.global_pool(signal).flatten(1)
        h = torch.cat([pooled, meta], dim=1)
        h = self.dropout(self.relu(self.dense(h)))
        return self.output(h)


def build_model(hp: dict, n_pca: int, n_cols: int) -> Cnn:
    return Cnn(n_pca, n_cols - n_pca, hp["channels"], hp["kernel_size"],
               hp["dense_hidden"], hp["dropout"])


def save_atomic(model: Cnn, run_dir: Path) -> None:
    """Atomic checkpoint: a kill mid-write can never corrupt model.pt.
    state_dict only — predict rebuilds the module from hyperparams.json."""
    tmp = run_dir / "model-tmp.pt"
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, run_dir / "model.pt")


def write_viz(path: Path, n_pca: int, n_meta: int, channels: list[int],
              kernel_size: int, dense_hidden: int, dropout: float) -> None:
    """Architecture SVG, same box renderer as the burn/flux CNNs."""
    boxes = [("input", str(n_pca + n_meta),
              f"{n_pca} pca + {n_meta} meta", "conv sees pca only")]
    prev_ch, length = 1, n_pca
    for i, ch in enumerate(channels):
        if length >= 2:
            length //= 2
        boxes.append((f"conv {i + 1}", f"{ch}×{length}",
                      f"Conv1d {prev_ch}→{ch} k{kernel_size}",
                      f"ReLU · pool/2 · drop {dropout}"))
        prev_ch = ch
    boxes.append(("pool", str(prev_ch + n_meta),
                  f"global avg → {prev_ch}", f"⊕ {n_meta} meta"))
    boxes.append(("dense", str(dense_hidden),
                  f"Dense {prev_ch + n_meta}→{dense_hidden}",
                  f"ReLU · dropout {dropout}"))
    boxes.append(("output", "1", f"Dense {dense_hidden}→1", "linear"))

    box_w, box_h, gap, pad, box_y, height = 150, 86, 44, 24, 46, 168
    n = len(boxes)
    w = pad * 2 + n * box_w + (n - 1) * gap
    arch = " → ".join(str(c) for c in channels)
    s = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {height}" '
         f'font-family="ui-monospace, SFMono-Regular, Menlo, monospace">'
         f'<rect x="0" y="0" width="{w}" height="{height}" rx="6" fill="#fcfcfb"/>'
         f'<defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" '
         f'markerWidth="7" markerHeight="7" orient="auto">'
         f'<path d="M0,0L8,4L0,8z" fill="#8a8a8a"/></marker></defs>'
         f'<text x="{pad}" y="28" font-size="13" fill="#5a5a5a">'
         f'CNN · {n_pca} pca → [{arch}] k{kernel_size} → '
         f'{dense_hidden} → 1</text>')
    for i, (title, units, sub1, sub2) in enumerate(boxes):
        x = pad + i * (box_w + gap)
        cx = x + box_w // 2
        s += (f'<rect x="{x}" y="{box_y}" width="{box_w}" height="{box_h}" '
              f'rx="6" fill="#ffffff" stroke="#b9b6ae"/>'
              f'<text x="{cx}" y="{box_y + 18}" font-size="11" fill="#7a766c" '
              f'text-anchor="middle">{title}</text>'
              f'<text x="{cx}" y="{box_y + 44}" font-size="22" fill="#2b2a26" '
              f'text-anchor="middle">{units}</text>'
              f'<text x="{cx}" y="{box_y + 62}" font-size="10" fill="#7a766c" '
              f'text-anchor="middle">{sub1}</text>'
              f'<text x="{cx}" y="{box_y + 76}" font-size="10" fill="#7a766c" '
              f'text-anchor="middle">{sub2}</text>')
        if i + 1 < n:
            ym = box_y + box_h // 2
            s += (f'<line x1="{x + box_w + 4}" y1="{ym}" x2="{x + box_w + gap - 6}" '
                  f'y2="{ym}" stroke="#8a8a8a" stroke-width="1.25" '
                  f'marker-end="url(#arr)"/>')
    s += "</svg>"
    path.write_text(s)


def train(dataset: Path, run_dir: Path, hp_path: Path) -> None:
    hp = load_hp(hp_path)
    torch.manual_seed(SEED)

    manifest = json.loads((dataset / "manifest.json").read_text())
    n_rows, n_cols = manifest["n_rows"], manifest["n_cols"]
    n_pca = count_pca(manifest["columns"])
    assert n_pca > 0, "dataset has no pca columns; the CNN needs a signal"
    features = read_features(dataset, n_rows, n_cols)
    target = np.fromfile(dataset / "target.f32", dtype="<f4")
    row_ids = np.fromfile(dataset / "row_ids.u64", dtype="<u8")
    train_idx = np.fromfile(dataset / "train_idx.u32", dtype="<u4")
    test_idx = np.fromfile(dataset / "test_idx.u32", dtype="<u4")

    emit({"event": "log", "msg": f"dataset {manifest['dataset_id']}: "
          f"{len(train_idx)} train / {len(test_idx)} test rows, "
          f"{n_cols} features ({n_pca} pca signal + {n_cols - n_pca} meta)"})
    emit({"event": "log", "msg": f"cnn {hp['channels']} k{hp['kernel_size']}, "
          f"dense {hp['dense_hidden']}, dropout {hp['dropout']}, "
          f"lr {hp['lr']}, batch {hp['batch_size']}, {hp['epochs']} epochs"})

    # Architecture diagram (registry capability `visualization`): written
    # before the first epoch so the live run view can show it immediately.
    run_dir.mkdir(parents=True, exist_ok=True)
    write_viz(run_dir / "viz.svg", n_pca, n_cols - n_pca, hp["channels"],
              hp["kernel_size"], hp["dense_hidden"], hp["dropout"])

    # Standardize features with train-split statistics only.
    mean, std = fit_scaler(features[train_idx])
    (run_dir / "scaler.json").write_text(json.dumps(
        {"mean": mean.tolist(), "std": std.tolist()}))
    train_x = torch.from_numpy(
        ((features[train_idx] - mean) / std).astype(np.float32))
    test_x = torch.from_numpy(
        ((features[test_idx] - mean) / std).astype(np.float32))
    train_y = torch.from_numpy(target[train_idx]).unsqueeze(1)
    test_y_np = target[test_idx].astype(np.float64)

    model = build_model(hp, n_pca, n_cols)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    loss_fn = nn.MSELoss()
    shuffle_rng = torch.Generator().manual_seed(SEED)

    n_train = len(train_idx)
    epochs, batch_size = hp["epochs"], hp["batch_size"]
    for epoch in range(1, epochs + 1):
        # Graceful stop (contract v2): the server drops a STOP file into the
        # run dir; finish early and run the normal end-of-training path.
        if (run_dir / "STOP").exists():
            emit({"event": "stopping"})
            emit({"event": "log", "msg": "stop requested; evaluating with "
                  f"params as of epoch {epoch - 1}"})
            break

        model.train()
        order = torch.randperm(n_train, generator=shuffle_rng)
        loss_sum = 0.0
        for start in range(0, n_train, batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            pred = model(train_x[batch])
            loss = loss_fn(pred, train_y[batch])
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch)
        train_loss = loss_sum / n_train

        # Validation: dropout inactive.
        model.eval()
        with torch.no_grad():
            val_pred = predict_batched(model, test_x, batch_size)
        val_loss = float(np.mean((val_pred - test_y_np) ** 2))

        emit({"event": "epoch", "epoch": epoch, "total_epochs": epochs,
              "train_loss": train_loss, "val_loss": val_loss})

        # Periodic checkpoint: after this event the run is promotable even if
        # the process is later killed. The final epoch is skipped — we save
        # right after training anyway.
        ck = hp["checkpoint_every"]
        if ck > 0 and epoch % ck == 0 and epoch < epochs:
            save_atomic(model, run_dir)
            emit({"event": "checkpoint", "epoch": epoch})

    # Predictions in price space.
    model.eval()
    with torch.no_grad():
        predicted_t = predict_batched(model, test_x, batch_size)
    transform = manifest["target"]["transform"]
    predicted = np.maximum(invert_target(predicted_t, transform), 0.0)
    actual = invert_target(test_y_np, transform)

    metrics = compute_metrics(actual, predicted)
    predictions = [
        {"row_id": int(row_ids[i]), "actual": float(a), "predicted": float(p)}
        for i, a, p in zip(test_idx, actual, predicted)
    ]
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    (run_dir / "predictions.json").write_text(json.dumps(predictions))
    save_atomic(model, run_dir)

    emit({"event": "log", "msg": f"MAE {metrics['mae']:,.0f}  "
          f"RMSE {metrics['rmse']:,.0f}  medAPE {metrics['medape']:.1%}  "
          f"R² {metrics['r2']:.3f}"})
    emit({"event": "done"})


def predict_batched(model: Cnn, x: torch.Tensor, batch_size: int) -> np.ndarray:
    out = []
    for start in range(0, len(x), max(batch_size, 1)):
        out.append(model(x[start:start + batch_size]).squeeze(1).numpy())
    return np.concatenate(out).astype(np.float64)


def predict(model_dir: Path, input_dir: Path, output: Path) -> None:
    """Contract v2 predict: rebuild the module from the hyperparams snapshot,
    load the trained weights, standardize the input mini-artifact, write
    price-space predictions."""
    hp = load_hp(model_dir / "hyperparams.json")
    scaler = json.loads((model_dir / "scaler.json").read_text())
    mean = np.asarray(scaler["mean"])
    std = np.asarray(scaler["std"])

    manifest = json.loads((input_dir / "manifest.json").read_text())
    n_rows, n_cols = manifest["n_rows"], manifest["n_cols"]
    n_pca = count_pca(manifest["columns"])
    assert n_pca > 0, "input has no pca columns; the CNN needs a signal"
    assert len(mean) == n_cols, (
        f"scaler was fit on {len(mean)} columns, input has {n_cols}")
    features = read_features(input_dir, n_rows, n_cols)
    row_ids = np.fromfile(input_dir / "row_ids.u64", dtype="<u8")
    emit({"event": "log", "msg": f"predicting {n_rows} rows, {n_cols} features "
          f"({n_pca} pca), cnn {hp['channels']} k{hp['kernel_size']}"})

    model = build_model(hp, n_pca, n_cols)
    model.load_state_dict(torch.load(model_dir / "model.pt", weights_only=True))
    model.eval()

    x = torch.from_numpy(((features - mean) / std).astype(np.float32))
    with torch.no_grad():
        predicted_t = predict_batched(model, x, hp["batch_size"])
    transform = manifest["target"]["transform"]
    predicted = np.maximum(invert_target(predicted_t, transform), 0.0)

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

    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    if args.cmd == "train":
        train(args.dataset, args.output, args.hyperparams)
    else:
        predict(args.model, args.input, args.output)


if __name__ == "__main__":
    main()
