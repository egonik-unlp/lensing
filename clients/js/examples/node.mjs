// Run an unpacked lensing export from Node, with onnxruntime-node.
//
//   npm i onnxruntime-node
//   # unpack a bundle:  tar xzf my-model-export.tar.gz   (-> ./my-model/)
//   node examples/node.mjs ./my-model
//
// The bundle is the directory produced by `GET /api/models/<name>/export`.

import { readFile } from "node:fs/promises";
import { join } from "node:path";
import * as ort from "onnxruntime-node";
import { loadExport } from "../src/index.js";

const dir = process.argv[2];
if (!dir) {
  console.error("usage: node examples/node.mjs <unpacked-export-dir>");
  process.exit(1);
}

// Adapter: run model.onnx on a feature matrix, return the flat output vector.
async function runOnnx(modelBytes, features, nRows, nCols) {
  const session = await ort.InferenceSession.create(modelBytes);
  const input = new ort.Tensor("float32", features, [nRows, nCols]);
  const outputs = await session.run({ [session.inputNames[0]]: input });
  return outputs[session.outputNames[0]].data; // Float32Array, length nRows
}

const model = await loadExport((rel) => readFile(join(dir, rel)));
console.log("loaded:", model.envelope.producer, "· features:", model.spec.nCols,
  "· embedding dim:", model.spec.embeddingDim, model.ensemble ? "· ensemble" : "");

// One item: the fields named in input-schema.json (numeric/categorical) plus
// the embedding vector (from the same embedding model the corpus was built
// with). Here we send only the embedding (zeros) — add your fields alongside:
//   const item = { embedding: [...], totalArea: 80, propertyType: "apartment" };
const item = { embedding: new Array(model.spec.embeddingDim).fill(0) };

const preds = await model.predict([item], runOnnx);
const warnings = model.warnings([item]);
console.log("prediction:", preds[0]);
if (warnings.length) console.log("warnings:", warnings);
