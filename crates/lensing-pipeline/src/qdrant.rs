use anyhow::{bail, Context, Result};
use lensing_core::domain::{Domain, FieldRole};
use serde::Serialize;
use serde_json::{json, Map, Value};

/// One point fetched from Qdrant. The vector is empty for payload-only
/// fetches (quality preflight); every featurization path requires it.
#[derive(Debug, Clone)]
pub struct RawPoint {
    pub id: u64,
    pub vector: Vec<f32>,
    pub payload: Payload,
}

/// A point payload, flattened at ingest into one field map: the metadata
/// subobject's keys plus the payload-root keys (content, cluster, …).
/// Metadata wins key ties. Field names are exactly the domain's field names,
/// so every consumer reads by name without caring about `metadata_root`.
#[derive(Debug, Clone, Default)]
pub struct Payload {
    pub fields: Map<String, Value>,
}

impl Payload {
    /// Flatten a raw payload JSON object under the given metadata root.
    pub fn from_raw(mut raw: Value, metadata_root: &str) -> Payload {
        let mut fields = Map::new();
        if let Value::Object(root) = &mut raw {
            let metadata = if metadata_root.is_empty() {
                None
            } else {
                root.remove(metadata_root)
            };
            // Root keys first, metadata second: metadata wins ties.
            fields.append(root);
            if let Some(Value::Object(mut md)) = metadata {
                fields.append(&mut md);
            }
        }
        Payload { fields }
    }

    pub fn get(&self, name: &str) -> &Value {
        self.fields.get(name).unwrap_or(&Value::Null)
    }

    /// String value of a field; numbers render integer-trimmed (cluster tags
    /// are numeric). Null/absent is "".
    pub fn str_of(&self, name: &str) -> String {
        value_to_string(self.get(name))
    }

    /// Numeric value of a field; null/absent/non-numeric is None.
    pub fn num_of(&self, name: &str) -> Option<f64> {
        self.get(name).as_f64()
    }

    pub fn set(&mut self, name: &str, value: Value) {
        self.fields.insert(name.to_string(), value);
    }

    /// Document text under the domain's content field.
    pub fn content_of(&self, content_field: &str) -> &str {
        self.get(content_field).as_str().unwrap_or("")
    }

    /// Sanity-bounded coordinates of a geo field (`{"lat": …, "lon": …}`):
    /// both members present, non-zero, inside the plausible bounds. The
    /// single source of truth for "has a usable location".
    pub fn coords_of(&self, field: &str, bounds: &[[f64; 2]; 2]) -> Option<(f64, f64)> {
        let c = self.get(field);
        let (lat, lon) = (c.get("lat")?.as_f64()?, c.get("lon")?.as_f64()?);
        if lat == 0.0 && lon == 0.0 {
            return None;
        }
        let [[lat_min, lat_max], [lon_min, lon_max]] = *bounds;
        ((lat_min..=lat_max).contains(&lat) && (lon_min..=lon_max).contains(&lon))
            .then_some((lat, lon))
    }
}

/// Render a payload value for categorical encoding / display.
pub fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n
            .as_i64()
            .map(|i| i.to_string())
            .unwrap_or_else(|| n.to_string()),
        Value::Bool(b) => b.to_string(),
        _ => String::new(),
    }
}

fn parse_point(p: &Value, metadata_root: &str, require_vector: bool) -> Result<RawPoint> {
    let id = p["id"].as_u64().with_context(|| format!("point {} has non-u64 id", p["id"]))?;
    let vector: Vec<f32> = if p["vector"].is_null() {
        Vec::new()
    } else {
        serde_json::from_value(p["vector"].clone())
            .with_context(|| format!("point {id} has an invalid vector"))?
    };
    if require_vector && vector.is_empty() {
        bail!("point {id} has no vector");
    }
    Ok(RawPoint { id, vector, payload: Payload::from_raw(p["payload"].clone(), metadata_root) })
}

/// Count points matching the domain's corpus filter (exact).
pub fn count(base_url: &str, collection: &str, domain: &Domain) -> Result<usize> {
    let client = reqwest::blocking::Client::new();
    let resp: Value = client
        .post(format!("{base_url}/collections/{collection}/points/count"))
        .json(&json!({ "filter": domain.qdrant_filter(), "exact": true }))
        .send()
        .context("qdrant count request")?
        .error_for_status()?
        .json()?;
    resp["result"]["count"]
        .as_u64()
        .map(|n| n as usize)
        .context("malformed count response")
}

/// Fetch specific points by id, vectors included. Result is in request
/// order; missing ids are an error (inference must not silently drop rows).
pub fn fetch_points(
    base_url: &str,
    collection: &str,
    ids: &[u64],
    domain: &Domain,
) -> Result<Vec<RawPoint>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()?;
    let resp: Value = client
        .post(format!("{base_url}/collections/{collection}/points"))
        .json(&json!({ "ids": ids, "with_payload": true, "with_vector": true }))
        .send()
        .context("qdrant retrieve request")?
        .error_for_status()?
        .json()?;

    let mut by_id = std::collections::HashMap::with_capacity(ids.len());
    for p in resp["result"].as_array().context("malformed retrieve response")? {
        let point = parse_point(p, &domain.corpus.metadata_root, true)?;
        by_id.insert(point.id, point);
    }

    let missing: Vec<String> = ids
        .iter()
        .filter(|id| !by_id.contains_key(id))
        .map(|id| id.to_string())
        .collect();
    if !missing.is_empty() {
        bail!("points not found in {collection}: {}", missing.join(", "));
    }
    Ok(ids.iter().map(|id| by_id.remove(id).unwrap()).collect())
}

/// Fetch a projection of metadata fields per id from a companion collection,
/// joined by point id. Payload include-projection keeps this cheap against
/// the raw scrape collection (heavy payloads). Ids absent from the companion
/// are simply not in the map — the caller reports them as missing.
pub fn fetch_fields(
    base_url: &str,
    collection: &str,
    ids: &[u64],
    domain: &Domain,
    field_names: &[String],
) -> Result<std::collections::HashMap<u64, Map<String, Value>>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()?;
    let url = format!("{base_url}/collections/{collection}/points");
    let root = &domain.corpus.metadata_root;
    let include: Vec<String> = field_names
        .iter()
        .map(|f| if root.is_empty() { f.clone() } else { format!("{root}.{f}") })
        .collect();
    let mut out = std::collections::HashMap::with_capacity(ids.len());
    for batch in ids.chunks(1000) {
        let resp: Value = client
            .post(&url)
            .json(&json!({
                "ids": batch,
                "with_payload": { "include": include },
                "with_vector": false,
            }))
            .send()
            .context("qdrant projected retrieve request")?
            .error_for_status()?
            .json()?;
        for p in resp["result"].as_array().context("malformed retrieve response")? {
            let id = p["id"].as_u64().context("non-u64 id in projected retrieve")?;
            let payload = Payload::from_raw(p["payload"].clone(), root);
            out.insert(id, payload.fields);
        }
    }
    Ok(out)
}

/// Fetch all matching points via paginated scroll, vectors included.
pub fn scroll_all(
    base_url: &str,
    collection: &str,
    domain: &Domain,
    progress: &(dyn Fn(usize) + Sync),
) -> Result<Vec<RawPoint>> {
    scroll_impl(base_url, collection, domain, true, progress)
}

/// Payload-only scroll (no vectors): cheap corpus pass for quality preflight.
pub fn scroll_payloads(
    base_url: &str,
    collection: &str,
    domain: &Domain,
    progress: &(dyn Fn(usize) + Sync),
) -> Result<Vec<RawPoint>> {
    scroll_impl(base_url, collection, domain, false, progress)
}

fn scroll_impl(
    base_url: &str,
    collection: &str,
    domain: &Domain,
    with_vector: bool,
    progress: &(dyn Fn(usize) + Sync),
) -> Result<Vec<RawPoint>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;
    let url = format!("{base_url}/collections/{collection}/points/scroll");
    let mut points: Vec<RawPoint> = Vec::new();
    let mut offset: Value = Value::Null;

    loop {
        let mut body = json!({
            "filter": domain.qdrant_filter(),
            "limit": 1024,
            "with_payload": true,
            "with_vector": with_vector,
        });
        if !offset.is_null() {
            body["offset"] = offset.clone();
        }
        let resp: Value = client
            .post(&url)
            .json(&body)
            .send()
            .context("qdrant scroll request")?
            .error_for_status()?
            .json()?;

        let page = resp["result"]["points"]
            .as_array()
            .context("malformed scroll response")?;
        for p in page {
            points.push(parse_point(p, &domain.corpus.metadata_root, with_vector)?);
        }
        progress(points.len());

        offset = resp["result"]["next_page_offset"].clone();
        if offset.is_null() {
            break;
        }
    }
    Ok(points)
}

// ---------- collection shape validation ----------

/// How many points the validation sample inspects (one unfiltered scroll page).
const VALIDATE_SAMPLE: usize = 64;

/// Result of probing a collection's shape against what this product expects.
/// Shape problems are data (`errors`/`warnings`), not transport failures, so
/// the API can always render a structured verdict.
#[derive(Debug, Clone, Serialize)]
pub struct CollectionValidation {
    pub collection: String,
    pub exists: bool,
    /// `errors` is empty — safe to point a build at this collection.
    pub ok_to_build: bool,
    pub vector: Option<VectorParams>,
    /// Points matching the dataset filter; exact count.
    pub count_filtered: Option<usize>,
    pub sample_size: usize,
    /// Fraction of sampled points with each expected field present/usable,
    /// keyed by field name (plus "content" and the filter fields).
    pub coverage: Option<std::collections::BTreeMap<String, f64>>,
    /// Distinct vector lengths seen in the sample (excluding missing vectors).
    pub vector_dims_in_sample: Vec<usize>,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct VectorParams {
    pub size: u64,
    pub distance: String,
}

/// Fetch one unfiltered scroll page (payload + vectors) as raw JSON for shape
/// inspection. The caller parses each point leniently: an alien schema (e.g.
/// UUID ids) must show up as a verdict in the report, not a request failure.
pub fn sample_points_raw(base_url: &str, collection: &str, limit: usize) -> Result<Vec<Value>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()?;
    let resp: Value = client
        .post(format!("{base_url}/collections/{collection}/points/scroll"))
        .json(&json!({ "limit": limit, "with_payload": true, "with_vector": true }))
        .send()
        .context("qdrant sample scroll request")?
        .error_for_status()?
        .json()?;
    resp["result"]["points"]
        .as_array()
        .cloned()
        .context("malformed scroll response")
}

/// Probe `collection` and report whether its shape fits the domain: a single
/// unnamed vector, points matching the corpus filter, and payloads carrying
/// the expected metadata fields. Returns `Err` only on transport failure.
pub fn validate_collection(
    base_url: &str,
    collection: &str,
    domain: &Domain,
) -> Result<CollectionValidation> {
    let mut v = CollectionValidation {
        collection: collection.to_string(),
        exists: false,
        ok_to_build: false,
        vector: None,
        count_filtered: None,
        sample_size: 0,
        coverage: None,
        vector_dims_in_sample: Vec::new(),
        errors: Vec::new(),
        warnings: Vec::new(),
    };

    // Existence + vector config. get_collection_params errors both when the
    // collection is missing and when it uses named vectors; only the former
    // means "does not exist", so probe existence separately.
    if !list_collections(base_url)?.iter().any(|c| c == collection) {
        v.errors.push(format!("collection {collection} not found"));
        return Ok(v);
    }
    v.exists = true;
    match get_collection_params(base_url, collection) {
        Ok((size, distance)) => {
            if !(8..=4096).contains(&size) {
                v.warnings.push(format!("unusual vector size {size}"));
            }
            v.vector = Some(VectorParams { size, distance });
        }
        Err(e) => v.errors.push(format!("{e:#}")),
    }

    let count_filtered = count(base_url, collection, domain)?;
    v.count_filtered = Some(count_filtered);

    // Parse the sample leniently: points that don't fit the schema (UUID
    // ids, alien payloads) are a shape verdict, not a failure.
    let raw_sample = sample_points_raw(base_url, collection, VALIDATE_SAMPLE)?;
    v.sample_size = raw_sample.len();
    let mut sample: Vec<RawPoint> = Vec::with_capacity(raw_sample.len());
    let mut first_parse_error: Option<String> = None;
    for p in &raw_sample {
        match parse_point(p, &domain.corpus.metadata_root, false) {
            Ok(point) => sample.push(point),
            Err(e) => {
                first_parse_error.get_or_insert_with(|| format!("point {}: {e}", p["id"]));
            }
        }
    }
    let n_unparseable = raw_sample.len() - sample.len();
    if n_unparseable > 0 {
        v.errors.push(format!(
            "{n_unparseable} of {} sampled points do not match the expected schema \
             (u64 id + JSON payload); first: {}",
            raw_sample.len(),
            first_parse_error.unwrap_or_default()
        ));
    }

    if count_filtered == 0 {
        v.errors.push(if raw_sample.is_empty() {
            "collection is empty".to_string()
        } else {
            format!(
                "collection has points but none match the dataset filter ({})",
                domain.corpus.filter.desc
            )
        });
    }

    if !sample.is_empty() {
        let n = sample.len() as f64;
        let frac = |pred: &dyn Fn(&RawPoint) -> bool| {
            sample.iter().filter(|p| pred(p)).count() as f64 / n
        };
        let mut coverage = std::collections::BTreeMap::new();
        // Filter fields: fraction matching the must-clauses.
        for m in &domain.corpus.filter.must {
            let field = m.field.clone();
            let expect = m.equals.clone();
            coverage
                .insert(format!("{field}=={expect}"), frac(&|p| *p.payload.get(&field) == expect));
        }
        // Target present and positive.
        coverage.insert(
            format!("{}>0", domain.target.field),
            frac(&|p| p.payload.num_of(&domain.target.field).is_some_and(|x| x > 0.0)),
        );
        // Feature fields by role.
        for f in &domain.fields {
            let name = f.name.clone();
            let usable = match f.role {
                FieldRole::Categorical => {
                    frac(&|p| !p.payload.str_of(&name).trim().is_empty())
                }
                FieldRole::Numeric => {
                    frac(&|p| p.payload.num_of(&name).is_some_and(|x| x > 0.0))
                }
                FieldRole::Coordinates => {
                    let bounds = domain.coordinate_bounds_arr();
                    frac(&|p| p.payload.coords_of(&name, &bounds).is_some())
                }
                _ => continue,
            };
            coverage.insert(name, usable);
        }
        coverage.insert(
            "content".into(),
            frac(&|p| !p.payload.content_of(&domain.corpus.content_field).is_empty()),
        );
        if let Some(c) = &domain.currency {
            let field = c.currency_field.clone();
            coverage.insert(field.clone(), frac(&|p| p.payload.get(&field).is_string()));
        }
        let vector_present = frac(&|p| !p.vector.is_empty());
        coverage.insert("vector".into(), vector_present);

        if vector_present == 0.0 {
            v.errors.push("no sampled point carries a vector; embeddings are required".into());
        } else if vector_present < 1.0 {
            v.warnings.push(format!(
                "{:.0}% of sampled points are missing vectors",
                (1.0 - vector_present) * 100.0
            ));
        }
        for f in domain.critical_fields() {
            if let Some(value) = coverage.get(&f.name) {
                if *value < 0.5 {
                    v.warnings.push(format!(
                        "{} present in only {:.0}% of sampled points",
                        f.name,
                        value * 100.0
                    ));
                }
            }
        }
        if let Some(value) = coverage.get("content") {
            if *value < 0.5 {
                v.warnings.push(format!(
                    "content present in only {:.0}% of sampled points",
                    value * 100.0
                ));
            }
        }
        v.coverage = Some(coverage);

        let mut dims: Vec<usize> = sample
            .iter()
            .filter(|p| !p.vector.is_empty())
            .map(|p| p.vector.len())
            .collect();
        dims.sort_unstable();
        dims.dedup();
        if dims.len() > 1 {
            v.warnings.push(format!("inconsistent vector dimensions in sample: {dims:?}"));
        }
        v.vector_dims_in_sample = dims;
    }

    v.ok_to_build = v.errors.is_empty();
    Ok(v)
}

/// Cheap hard checks shared by build/analyze: the collection must have a
/// single unnamed vector and at least one point matching the corpus filter.
/// Fails fast with a clear message instead of erroring mid-scroll.
pub fn ensure_buildable(base_url: &str, collection: &str, domain: &Domain) -> Result<()> {
    get_collection_params(base_url, collection)
        .with_context(|| format!("collection {collection} is not usable as a dataset source"))?;
    let n = count(base_url, collection, domain)?;
    if n == 0 {
        bail!(
            "collection {collection} has no points matching the dataset filter ({})",
            domain.corpus.filter.desc
        );
    }
    Ok(())
}

// ---------- collection management + raw payload round-trip (export) ----------

/// A point with its payload kept as raw JSON, so export can write it back
/// verbatim. The flattened [`Payload`] restructures fields on ingest;
/// writing that back would alter the stored shape, so export never uses it.
#[derive(Debug, Clone)]
pub struct RawJsonPoint {
    pub id: u64,
    pub vector: Vec<f32>,
    pub payload: Value,
}

/// Vector size + distance of a collection's (single, unnamed) vector.
pub fn get_collection_params(base_url: &str, collection: &str) -> Result<(u64, String)> {
    let client = reqwest::blocking::Client::new();
    let resp: Value = client
        .get(format!("{base_url}/collections/{collection}"))
        .send()
        .context("qdrant get collection request")?
        .error_for_status()
        .with_context(|| format!("collection {collection} not found"))?
        .json()?;
    let vectors = &resp["result"]["config"]["params"]["vectors"];
    // Unnamed-vector form: { "size": N, "distance": "Cosine" }. A named-vector
    // map has no top-level "size" key; export does not support those.
    let size = vectors["size"]
        .as_u64()
        .context("collection uses named vectors (unsupported); expected a single unnamed vector")?;
    let distance = vectors["distance"]
        .as_str()
        .context("malformed collection vector params: missing distance")?
        .to_string();
    Ok((size, distance))
}

/// List all collection names.
pub fn list_collections(base_url: &str) -> Result<Vec<String>> {
    let client = reqwest::blocking::Client::new();
    let resp: Value = client
        .get(format!("{base_url}/collections"))
        .send()
        .context("qdrant list collections request")?
        .error_for_status()?
        .json()?;
    let names = resp["result"]["collections"]
        .as_array()
        .context("malformed list collections response")?
        .iter()
        .filter_map(|c| c["name"].as_str().map(String::from))
        .collect();
    Ok(names)
}

/// Create a collection with a single unnamed vector. Errors if it exists.
pub fn create_collection(base_url: &str, name: &str, size: u64, distance: &str) -> Result<()> {
    let client = reqwest::blocking::Client::new();
    let resp = client
        .put(format!("{base_url}/collections/{name}"))
        .json(&json!({ "vectors": { "size": size, "distance": distance } }))
        .send()
        .context("qdrant create collection request")?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        bail!("create collection {name} failed ({status}): {body}");
    }
    Ok(())
}

/// Delete a collection (best-effort cleanup on failed exports).
pub fn delete_collection(base_url: &str, name: &str) -> Result<()> {
    let client = reqwest::blocking::Client::new();
    client
        .delete(format!("{base_url}/collections/{name}"))
        .send()
        .context("qdrant delete collection request")?
        .error_for_status()?;
    Ok(())
}

/// Scroll all points matching the corpus filter, keeping each payload as raw
/// JSON (and the vector). For export only.
pub fn scroll_all_raw(
    base_url: &str,
    collection: &str,
    domain: &Domain,
    progress: &(dyn Fn(usize) + Sync),
) -> Result<Vec<RawJsonPoint>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;
    let url = format!("{base_url}/collections/{collection}/points/scroll");
    let mut points: Vec<RawJsonPoint> = Vec::new();
    let mut offset: Value = Value::Null;

    loop {
        let mut body = json!({
            "filter": domain.qdrant_filter(),
            "limit": 1024,
            "with_payload": true,
            "with_vector": true,
        });
        if !offset.is_null() {
            body["offset"] = offset.clone();
        }
        let resp: Value = client
            .post(&url)
            .json(&body)
            .send()
            .context("qdrant scroll request")?
            .error_for_status()?
            .json()?;

        let page = resp["result"]["points"]
            .as_array()
            .context("malformed scroll response")?;
        for p in page {
            let id = p["id"].as_u64().with_context(|| format!("point {} has non-u64 id", p["id"]))?;
            let vector: Vec<f32> = serde_json::from_value(p["vector"].clone())
                .with_context(|| format!("point {id} has no/invalid vector"))?;
            if vector.is_empty() {
                bail!("point {id} has no vector");
            }
            points.push(RawJsonPoint { id, vector, payload: p["payload"].clone() });
        }
        progress(points.len());

        offset = resp["result"]["next_page_offset"].clone();
        if offset.is_null() {
            break;
        }
    }
    Ok(points)
}

/// Scroll all points of a collection with NO filter, payloads as raw JSON,
/// vectors excluded. For manual-entry collections, where points need not
/// match the corpus filter (the target may be 0 — that's what prediction is
/// for).
pub fn scroll_collection_raw(base_url: &str, collection: &str) -> Result<Vec<RawJsonPoint>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;
    let url = format!("{base_url}/collections/{collection}/points/scroll");
    let mut points: Vec<RawJsonPoint> = Vec::new();
    let mut offset: Value = Value::Null;

    loop {
        let mut body = json!({
            "limit": 1024,
            "with_payload": true,
            "with_vector": false,
        });
        if !offset.is_null() {
            body["offset"] = offset.clone();
        }
        let resp: Value = client
            .post(&url)
            .json(&body)
            .send()
            .context("qdrant scroll request")?
            .error_for_status()?
            .json()?;

        let page = resp["result"]["points"]
            .as_array()
            .context("malformed scroll response")?;
        for p in page {
            let id = p["id"].as_u64().with_context(|| format!("point {} has non-u64 id", p["id"]))?;
            points.push(RawJsonPoint { id, vector: Vec::new(), payload: p["payload"].clone() });
        }

        offset = resp["result"]["next_page_offset"].clone();
        if offset.is_null() {
            break;
        }
    }
    Ok(points)
}

/// Retrieve one point by id with payload as raw JSON and its vector.
/// `Ok(None)` when the point (or the whole collection) does not exist, so the
/// caller can map it to a 404 instead of a generic failure.
pub fn get_point_raw(base_url: &str, collection: &str, id: u64) -> Result<Option<RawJsonPoint>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()?;
    let resp = client
        .post(format!("{base_url}/collections/{collection}/points"))
        .json(&json!({ "ids": [id], "with_payload": true, "with_vector": true }))
        .send()
        .context("qdrant retrieve request")?;
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    let resp: Value = resp.error_for_status()?.json()?;
    let Some(p) = resp["result"].as_array().and_then(|a| a.first()) else {
        return Ok(None);
    };
    let id = p["id"].as_u64().with_context(|| format!("point {} has non-u64 id", p["id"]))?;
    let vector: Vec<f32> = serde_json::from_value(p["vector"].clone())
        .with_context(|| format!("point {id} has no/invalid vector"))?;
    Ok(Some(RawJsonPoint { id, vector, payload: p["payload"].clone() }))
}

/// Delete one point by id, waiting so a subsequent scroll reflects the delete.
pub fn delete_point(base_url: &str, collection: &str, id: u64) -> Result<()> {
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(format!("{base_url}/collections/{collection}/points/delete?wait=true"))
        .json(&json!({ "points": [id] }))
        .send()
        .context("qdrant delete point request")?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        bail!("delete point {id} from {collection} failed ({status}): {body}");
    }
    Ok(())
}

/// Upsert raw points into a collection in batches, waiting for each batch so a
/// later count reflects the full write.
pub fn upsert_raw(base_url: &str, collection: &str, points: &[RawJsonPoint]) -> Result<()> {
    const BATCH: usize = 512;
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;
    let url = format!("{base_url}/collections/{collection}/points?wait=true");
    for chunk in points.chunks(BATCH) {
        let body: Vec<Value> = chunk
            .iter()
            .map(|p| json!({ "id": p.id, "vector": p.vector, "payload": p.payload }))
            .collect();
        let resp = client
            .put(&url)
            .json(&json!({ "points": body }))
            .send()
            .context("qdrant upsert request")?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            bail!("upsert into {collection} failed ({status}): {text}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_flattens_metadata_over_root() {
        let raw = json!({
            "content": "casa en venta",
            "cluster": 3,
            "metadata": { "propertyType": "house", "price": 100000.0, "bedrooms": 2 }
        });
        let p = Payload::from_raw(raw, "metadata");
        assert_eq!(p.str_of("propertyType"), "house");
        assert_eq!(p.num_of("price"), Some(100000.0));
        assert_eq!(p.str_of("cluster"), "3"); // numeric → integer-trimmed
        assert_eq!(p.content_of("content"), "casa en venta");
        assert_eq!(p.num_of("absent"), None);
        assert_eq!(p.str_of("absent"), "");
    }

    #[test]
    fn coords_bounds_check() {
        let bounds = lensing_core::manifest::legacy_coordinate_bounds();
        let mut p = Payload::default();
        p.set("coordinates", json!({"lat": -34.92, "lon": -57.95}));
        assert_eq!(p.coords_of("coordinates", &bounds), Some((-34.92, -57.95)));
        p.set("coordinates", json!({"lat": 37.4, "lon": 13.5})); // not in bounds
        assert_eq!(p.coords_of("coordinates", &bounds), None);
        p.set("coordinates", json!({"lat": 0.0, "lon": 0.0}));
        assert_eq!(p.coords_of("coordinates", &bounds), None);
        p.set("coordinates", json!({"lat": null, "lon": -57.9}));
        assert_eq!(p.coords_of("coordinates", &bounds), None);
    }
}
