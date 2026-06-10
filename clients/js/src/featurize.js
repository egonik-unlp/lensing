// Reference featurizer for lensing model exports — a faithful, dependency-free
// port of the server's featurization (crates/lensing-pipeline). Turns a raw
// item (metadata fields + embedding) into the exact `float32[n_cols]` feature
// vector `model.onnx` expects. Runs anywhere: Node, browser, Cloudflare Workers.
//
// All arithmetic is done in JS doubles and rounded to float32 on store (the
// output is a Float32Array), matching the f32 feature matrix the trainer saw.

/** Numeric value of a field, or null. Mirrors Payload::num_of (JSON numbers
 * only — a numeric *string* is not coerced; send numbers as numbers). */
function numOf(item, field) {
  const v = item == null ? undefined : item[field];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** "Present" numeric: the value when > 0, else null. Mirrors the `val()`
 * closure in features.rs (0/absent = unspecified, the scraper convention). */
function presentNum(item, field) {
  const v = numOf(item, field);
  return v != null && v > 0 ? v : null;
}

/** String value for categorical encoding. Mirrors value_to_string: strings
 * as-is, integers integer-trimmed, bools stringified, everything else "". */
function strOf(item, field) {
  const v = item == null ? undefined : item[field];
  if (typeof v === "string") return v;
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : String(v);
  if (typeof v === "boolean") return String(v);
  return "";
}

/** Sanity-bounded coordinates `{lat, lon}`, or null. Mirrors coords_of:
 * both present, not (0,0), inside bounds = [[latMin,latMax],[lonMin,lonMax]]. */
function coordsOf(item, field, bounds) {
  const c = item == null ? undefined : item[field];
  if (c == null || typeof c.lat !== "number" || typeof c.lon !== "number") return null;
  const { lat, lon } = c;
  if (lat === 0 && lon === 0) return null;
  const [[latMin, latMax], [lonMin, lonMax]] = bounds;
  return lat >= latMin && lat <= latMax && lon >= lonMin && lon <= lonMax ? [lat, lon] : null;
}

/**
 * Parse a loaded export's `featurize.json` + `pca_components.f32` into a spec
 * the featurizer can execute.
 * @param {object} featurizeJson  parsed featurize.json
 * @param {ArrayBuffer|Uint8Array} pcaComponents  raw bytes of pca_components.f32
 */
export function loadFeaturize(featurizeJson, pcaComponents) {
  const buf = pcaComponents instanceof Uint8Array ? pcaComponents : new Uint8Array(pcaComponents);
  // Little-endian float32 view (copy to guarantee 4-byte alignment).
  const aligned = buf.byteOffset % 4 === 0 ? buf : new Uint8Array(buf);
  const components = new Float32Array(
    aligned.buffer.slice(aligned.byteOffset, aligned.byteOffset + aligned.byteLength),
  );
  const [k, d] = featurizeJson.pca.components_shape;
  if (components.length !== k * d) {
    throw new Error(`pca_components.f32 has ${components.length} floats, expected ${k}×${d}`);
  }
  // The `__other__` one-hot is a group-level catch-all (it fires when no
  // listed value matched), so precompute each group's listed value set.
  const onehotGroups = new Map();
  for (const col of featurizeJson.columns) {
    if (col.op === "onehot") {
      if (!onehotGroups.has(col.group)) onehotGroups.set(col.group, new Set());
      if (col.value !== "__other__") onehotGroups.get(col.group).add(col.value);
    }
  }
  return {
    nCols: featurizeJson.n_cols,
    embeddingDim: featurizeJson.embedding_dim,
    columns: featurizeJson.columns,
    columnNames: featurizeJson.column_names,
    pcaDims: k,
    pcaMean: Float32Array.from(featurizeJson.pca.mean),
    pcaComponents: components, // row-major [k, d]
    target: featurizeJson.target,
    imputation: featurizeJson.imputation || null,
    onehotGroups,
  };
}

/** Project one embedding to its PCA components: (x - mean) · componentsᵀ. */
function projectPca(embedding, spec) {
  const { pcaDims: k, embeddingDim: d, pcaMean, pcaComponents } = spec;
  if (embedding.length !== d) {
    throw new Error(`embedding has ${embedding.length} dims, model expects ${d}`);
  }
  const centered = new Float64Array(d);
  for (let j = 0; j < d; j++) centered[j] = Math.fround(embedding[j]) - pcaMean[j];
  const out = new Float64Array(k);
  for (let c = 0; c < k; c++) {
    let acc = 0;
    const base = c * d;
    for (let j = 0; j < d; j++) acc += centered[j] * pcaComponents[base + j];
    out[c] = acc;
  }
  return out;
}

/**
 * Apply frozen train-split imputation: fill missing numerics with the group's
 * median (falling back to the global median), exactly as training did. Returns
 * a shallow-augmented copy of the item.
 */
function applyImputation(item, imputation) {
  if (!imputation) return item;
  const group = imputation.group_field;
  const key = group ? strOf(item, group) : "";
  const medians =
    (imputation.by_property_type && imputation.by_property_type[key]) || imputation.global || {};
  const filled = { ...item };
  for (const [field, median] of Object.entries(medians)) {
    if (presentNum(filled, field) == null) filled[field] = median;
  }
  return filled;
}

/** Encode one feature column for one item per its plan instruction. */
function encodeColumn(col, item, spec) {
  switch (col.op) {
    case "numeric_verbatim": {
      const v = numOf(item, col.field);
      return v == null ? 0 : v;
    }
    case "numeric_log1p": {
      const v = presentNum(item, col.field);
      return v == null ? 0 : Math.log1p(v);
    }
    case "numeric_present_raw": {
      const v = presentNum(item, col.field);
      return v == null ? 0 : v;
    }
    case "numeric_missing_flag":
      return presentNum(item, col.field) == null ? 1 : 0;
    case "coord_lat": {
      const c = coordsOf(item, col.field, col.bounds);
      return c == null ? 0 : c[0];
    }
    case "coord_lon": {
      const c = coordsOf(item, col.field, col.bounds);
      return c == null ? 0 : c[1];
    }
    case "coord_missing":
      return coordsOf(item, col.field, col.bounds) == null ? 1 : 0;
    case "onehot": {
      const v = strOf(item, col.group);
      if (col.value === "__other__") {
        // Catch-all: fires when the value matched no listed category.
        return spec.onehotGroups.get(col.group).has(v) ? 0 : 1;
      }
      return v === col.value ? 1 : 0;
    }
    default:
      throw new Error(`unknown feature op ${col.op}`);
  }
}

/**
 * Build the feature matrix for a batch of items.
 * @param {object[]} items  each `{ embedding: number[], ...fields }`
 * @param {object} spec  from loadFeaturize
 * @returns {{ data: Float32Array, nRows: number, nCols: number }}
 */
export function buildFeatures(items, spec) {
  const { nCols, columns } = spec;
  const data = new Float32Array(items.length * nCols);
  for (let i = 0; i < items.length; i++) {
    let item = items[i];
    if (!item || !Array.isArray(item.embedding)) {
      throw new Error(`item ${i}: missing embedding array`);
    }
    item = applyImputation(item, spec.imputation);
    const pca = projectPca(item.embedding, spec);
    const row = i * nCols;
    for (let c = 0; c < nCols; c++) {
      const col = columns[c];
      data[row + c] = col.op === "pca" ? pca[col.component] : encodeColumn(col, item, spec);
    }
  }
  return { data, nRows: items.length, nCols };
}

/** One-hot semantics caveat: a value not seen at training lands in the
 * `__other__` bucket. Surfaced for callers that want to warn. */
export function categoryWarnings(items, spec) {
  const groups = new Map(); // group -> Set(values)
  for (const col of spec.columns) {
    if (col.op === "onehot") {
      if (!groups.has(col.group)) groups.set(col.group, new Set());
      if (col.value !== "__other__") groups.get(col.group).add(col.value);
    }
  }
  const out = [];
  items.forEach((item, i) => {
    for (const [group, values] of groups) {
      const v = strOf(item, group);
      if (v === "") out.push(`item ${i}: ${group} is empty → __other__ bucket`);
      else if (!values.has(v)) out.push(`item ${i}: ${group} ${JSON.stringify(v)} unseen → __other__ bucket`);
    }
  });
  return out;
}

/** Invert the target transform on a model output and clamp at 0 — what every
 * predictor does internally before returning a target-space prediction. */
export function invertTarget(transformed, target) {
  let v = transformed;
  if (target.transform === "log1p") v = Math.expm1(v);
  if (target.clamp_nonnegative) v = Math.max(v, 0);
  return v;
}
