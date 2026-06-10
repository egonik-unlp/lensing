// Cloudflare Worker: serve predictions from a lensing export.
//
// onnxruntime-web runs on the WASM backend in Workers. Bundle the export files
// with your Worker (e.g. as imported bytes / KV / R2). This example expects the
// five bundle files imported as Uint8Array (wrangler `rules` for `.onnx`/`.f32`
// as `Data`, JSON via text imports), and a POST body of `{ items: [...] }`.
//
//   npm i onnxruntime-web
//
// Note: ship `featurize.json` / `lensing-export.json` as text and the binary
// files as bytes; wire `read(rel)` to your asset source (KV/R2/embedded).

import * as ort from "onnxruntime-web";
import { loadExport } from "@lensing/inference";

// Replace with your asset source. Each call returns the bundle file as bytes.
async function readBundleFile(env, rel) {
  const obj = await env.MODEL_BUCKET.get(rel); // e.g. an R2 bucket
  if (!obj) throw new Error(`missing bundle file ${rel}`);
  return new Uint8Array(await obj.arrayBuffer());
}

async function runOnnx(modelBytes, features, nRows, nCols) {
  const session = await ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"] });
  const input = new ort.Tensor("float32", features, [nRows, nCols]);
  const outputs = await session.run({ [session.inputNames[0]]: input });
  return outputs[session.outputNames[0]].data;
}

let cached;

export default {
  async fetch(request, env) {
    if (request.method !== "POST") return new Response("POST { items: [...] }", { status: 405 });
    cached ??= await loadExport((rel) => readBundleFile(env, rel));
    const { items } = await request.json();
    const predictions = await cached.predict(items, runOnnx);
    return Response.json({ predictions, warnings: cached.warnings(items) });
  },
};
