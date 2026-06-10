// @lensing/inference — run a lensing model export anywhere.
//
// The export bundle is framework-agnostic (an ONNX graph + a declarative
// `featurize.json`); this package is the "framework you bring". It is itself
// runtime-agnostic: it never imports an ONNX runtime. You pass a small
// `runOnnx(modelBytes, features, nRows, nCols) => Promise<Float32Array>`
// adapter wired to onnxruntime-web (browser / Cloudflare Workers) or
// onnxruntime-node (Node) — see examples/.

import { buildFeatures, invertTarget, loadFeaturize, categoryWarnings } from "./featurize.js";

export { buildFeatures, invertTarget, loadFeaturize, categoryWarnings };

const dec = new TextDecoder();
const text = (bytes) => dec.decode(bytes);
const json = (bytes) => JSON.parse(text(bytes));

/**
 * Load an unpacked export bundle.
 * @param {(relPath: string) => Promise<Uint8Array>} read  reads a bundle file
 *   (relative to the bundle root, e.g. "model.onnx") as bytes.
 * @returns {Promise<LensingModel>}
 */
export async function loadExport(read) {
  const envelope = json(await read("lensing-export.json"));
  const spec = loadFeaturize(json(await read("featurize.json")), await read("pca_components.f32"));

  if (envelope.ensemble) {
    const combination = json(await read("combination.json"));
    const members = await Promise.all(
      combination.members.map(async (m) => ({
        ...m,
        // Indices of this member's columns within the full feature vector.
        colIdx: m.columns.map((name) => {
          const idx = spec.columnNames.indexOf(name);
          if (idx < 0) throw new Error(`member ${m.index}: column ${name} not in featurize spec`);
          return idx;
        }),
        modelBytes: await read(`${m.dir}/model.onnx`),
      })),
    );
    return new LensingModel(envelope, spec, { rule: combination.rule, members });
  }

  return new LensingModel(envelope, spec, { modelBytes: await read("model.onnx") });
}

export class LensingModel {
  constructor(envelope, spec, impl) {
    this.envelope = envelope;
    this.spec = spec;
    this.impl = impl;
    this.ensemble = !!envelope.ensemble;
  }

  /** Fields a caller must provide on each item (besides `embedding`). */
  get inputSchema() {
    return { ...this.envelope, featureCount: this.spec.nCols, embeddingDim: this.spec.embeddingDim };
  }

  warnings(items) {
    return categoryWarnings(items, this.spec);
  }

  /**
   * Predict target-space values for items `[{ embedding: number[], ...fields }]`.
   * @param {object[]} items
   * @param {(modelBytes: Uint8Array, features: Float32Array, nRows: number, nCols: number) => Promise<Float32Array>} runOnnx
   * @returns {Promise<number[]>}
   */
  async predict(items, runOnnx) {
    const full = buildFeatures(items, this.spec);
    if (!this.ensemble) {
      const out = await runOnnx(this.impl.modelBytes, full.data, full.nRows, full.nCols);
      return Array.from(out, (v) => invertTarget(v, this.spec.target));
    }
    return this.#predictEnsemble(full, runOnnx);
  }

  async #predictEnsemble(full, runOnnx) {
    const { rule, members } = this.impl;
    // Each member: select its column subset, run, invert to target space.
    const memberPreds = await Promise.all(
      members.map(async (m) => {
        const mCols = m.colIdx.length;
        const sub = new Float32Array(full.nRows * mCols);
        for (let r = 0; r < full.nRows; r++) {
          for (let j = 0; j < mCols; j++) sub[r * mCols + j] = full.data[r * full.nCols + m.colIdx[j]];
        }
        const out = await runOnnx(m.modelBytes, sub, full.nRows, mCols);
        return Array.from(out, (v) => invertTarget(v, this.spec.target));
      }),
    );
    const weights = members.map((m) => m.weight);
    const result = new Array(full.nRows);
    for (let r = 0; r < full.nRows; r++) {
      result[r] = combineRow(rule, memberPreds.map((p) => p[r]), weights);
    }
    return result;
  }
}

/** Combine one row's member predictions. `mean` is a weighted sum (weights are
 * the export's already-normalized weights); `median` is an unweighted vote. */
export function combineRow(rule, preds, weights) {
  if (rule === "median") {
    const v = [...preds].sort((a, b) => a - b);
    const mid = v.length >> 1;
    return v.length % 2 ? v[mid] : (v[mid - 1] + v[mid]) / 2;
  }
  let acc = 0;
  for (let i = 0; i < preds.length; i++) acc += weights[i] * preds[i];
  return acc;
}
