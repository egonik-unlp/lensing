---
name: model-export
description: Export a trained, promoted model as a portable, self-contained ONNX bundle (a .tar.gz with model.onnx + a featurize.json preprocessing spec) that runs outside lensing — in a Node.js app, the browser, or a Cloudflare Worker — via the lensing-server API. Use when the user wants to export a model, download a model as ONNX, ship a model elsewhere, run a model without the framework, or consume a model from JavaScript.
user-invocable: true
argument-hint: "[model-name]"
allowed-tools:
  - Read
  - Glob
  - Bash(curl *)
  - Bash(tar *)
  - Bash(python3 *)
---

Export a **promoted model** as a portable bundle that runs **outside lensing** —
no Rust, no Python, no Qdrant. The export is just an ONNX graph plus a
declarative preprocessing spec; the only thing a consumer brings is an ONNX
runtime and (optionally) the reference featurizer `@lensing/inference`
(`clients/js/`).

The point of the export is **uniformity**: an MLP, an XGBoost forest, a
LightGBM, an SVR and a blend ensemble all produce the *same* bundle layout —
only the bytes inside `model.onnx` differ — so one downstream consumer handles
every {{target_noun}}-prediction model this repo trains.

## Ground rules

- The server must be running (default `{{api_base_url}}`; `zig build serve`).
- Export operates on **promoted models** (`data/models/<name>/`), not runs.
  Promote a run first (the `model-definitions` skill / `POST /api/models`) if
  needed. Model names match `^[a-z0-9][a-z0-9-]{0,63}$`.
- Export is read-only: it never mutates the model. Re-export any time.

## Export a model

```bash
curl -fsS -OJ {{api_base_url}}/api/models/<name>/export    # -> <name>-export.tar.gz
tar xzf <name>-export.tar.gz                                 # -> <name>/
```

A `422` means the model's predictor family has no ONNX export yet (see the
support matrix below) — nothing is wrong with the model; pick an exportable
family or keep serving it through the API.

## Bundle layout (the consumer contract)

```
<name>/
  lensing-export.json   manifest: schema version, predictor family, target, file index
  model.onnx            input  `input`:  float32[N, n_cols]  (the assembled feature vector)
                        output `output`: float32[N, 1]       (target in TRANSFORMED space)
  featurize.json        ordered per-column build instructions + PCA + target + imputation
  pca_components.f32     PCA basis, row-major float32[dims, {{embedding_dim}}]
  input-schema.json     the fields a caller must supply (besides the embedding)
  README.md             this model's specifics, machine-readable
  # ensembles (blend) also contain:
  members/<i>/model.onnx
  combination.json      { rule, members:[{ index, predictor, dir, columns, weight }] }
```

Every `model.onnx` has the **same I/O signature**, regardless of family. Tree
models embed their ensemble; the neural nets and SVR/ridge bake their
standardizer (and any output clamp) into the graph — so `model.onnx`'s input is
always the *assembled feature vector*, and `featurize.json` is family-agnostic.

### Latent-encoder export (representations)

A **representation** (PCA / autoencoder from `lensing-compression`, the UI's
"Representations" section) exports its *encoder* to ONNX the same way — an
`encoder.onnx` written under the representation's artifact directory. The PCA
encoder is a mean-center + single `Gemm` (`z = (x − mean)·componentsᵀ`, the same
basis as `pca_components.f32`); the autoencoder encoder is a `Gemm`+activation
stack. Input is the source vector (or the assembled multimodal feature row), and
output is the latent — portable for encoding new items outside the framework,
just like the predictor bundles above.

## Consuming it

The embedding (`{{embedding_dim}}`-dim) is supplied by the caller — it comes
from the same embedding model the corpus used; the export can't generate it.
Other inputs are listed in `input-schema.json`.

- **JavaScript (Node / browser / Cloudflare Worker):** use `@lensing/inference`
  (`clients/js/`). It ports the featurizer exactly and is ONNX-runtime-agnostic
  (you wire `onnxruntime-node` or `onnxruntime-web`). See `clients/js/README.md`
  and `clients/js/examples/`.
- **Anything else:** `featurize.json` is fully declarative — `columns` is an
  ordered list of ops (`pca`, `numeric_log1p`, `onehot`, `coord_lat`, …). Build
  the `float32[n_cols]` vector, run `model.onnx`, then invert the target
  transform (`expm1` for `log1p`) and clamp at 0. The bundle README spells out
  every op.

## Verifying an export

Confirm the graph loads and the structure is intact:

```bash
python3 -c "import onnx; m=onnx.load('<name>/model.onnx'); onnx.checker.check_model(m); print([n.op_type for n in m.graph.node][:12])"
```

For correctness, compare against the live server on the same item:
`POST {{api_base_url}}/api/models/<name>/predict` (server featurizer + native
model) vs `@lensing/inference` (JS featurizer + ONNX) — they should agree to a
small float tolerance.

## Family support

| Family | Export |
|---|---|
| burn-mlp, burn-cnn, torch-cnn | ✅ (scaler + net baked into the graph) |
| xgboost, lightgbm, random-forest | ✅ (TreeEnsembleRegressor) |
| ridge, svm, kernel-ridge | ✅ (linear / kernel decision function) |
| blend | ✅ (each member exported under `members/<i>/` + `combination.json`) |
| svm-moe, svm-quantile-moe, flux-mlp, flux-cnn | ✗ — `422` (not yet supported) |

A blend is exportable only when **every** included member's family is
exportable; otherwise the export returns `422` naming the offending member.
