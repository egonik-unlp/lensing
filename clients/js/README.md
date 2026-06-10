# @lensing/inference

Run a **lensing model export** outside lensing — in Node, the browser, or a
Cloudflare Worker. An export (`GET /api/models/<name>/export` → `.tar.gz`) is
framework-agnostic: a `model.onnx` graph plus a declarative `featurize.json`
preprocessing spec. This package reproduces the featurization the model was
trained with and runs the graph; it's the only "framework" a consumer needs.

It is **runtime-agnostic** — it never imports an ONNX runtime. You provide a
small adapter that runs `model.onnx` with whichever runtime fits your target
(`onnxruntime-node` in Node, `onnxruntime-web` / WASM in browsers and Workers).

## Bundle layout

```
<name>/
  lensing-export.json   manifest (predictor family, target, file index)
  model.onnx            input: float32[N, n_cols]; output: float32[N, 1] (transformed space)
  featurize.json        ordered per-column build instructions + PCA + target + imputation
  pca_components.f32     PCA basis, row-major float32[dims, embedding_dim]
  input-schema.json     fields a caller must provide
  README.md             this model's specifics
  # ensembles add:
  members/<i>/model.onnx
  combination.json      { rule, members:[{ index, predictor, dir, columns, weight }] }
```

## Usage

```js
import { loadExport } from "@lensing/inference";
import * as ort from "onnxruntime-node"; // or onnxruntime-web

// Adapter: run a feature matrix through an ONNX graph → flat output vector.
async function runOnnx(modelBytes, features, nRows, nCols) {
  const session = await ort.InferenceSession.create(modelBytes);
  const input = new ort.Tensor("float32", features, [nRows, nCols]);
  const out = await session.run({ [session.inputNames[0]]: input });
  return out[session.outputNames[0]].data;
}

// `read(rel)` returns a bundle file as bytes (Uint8Array).
const model = await loadExport((rel) => readFileBytes(`${dir}/${rel}`));

const items = [{ embedding: [...1536 floats], totalArea: 80, propertyType: "apartment" }];
const predictions = await model.predict(items, runOnnx); // target-space numbers
```

The **embedding** comes from the same embedding model the corpus used; the
export cannot generate it. Other fields are listed in `input-schema.json`.

See `examples/node.mjs` and `examples/worker.js`.

## What it reproduces

`featurize.json.columns` is an ordered list of column instructions:

- `pca` — `(embedding − pca.mean) · componentsᵀ`
- `numeric_verbatim | numeric_log1p | numeric_present_raw | numeric_missing_flag`
- `coord_lat | coord_lon | coord_missing` (bounds-checked `{lat, lon}`)
- `onehot` (with an `__other__` catch-all for unseen categories)

If `featurize.json.imputation` is present, missing numerics are filled with the
frozen train-split medians before encoding. `model.onnx` outputs the target in
transformed space; `predict` applies the inverse (`expm1`) and clamps at 0.

Ensembles (`blend`) build the full feature vector once, slice each member's
column subset, run each member's `model.onnx`, invert per member, then combine
by `rule` (`mean` weighted, `median` unweighted).
