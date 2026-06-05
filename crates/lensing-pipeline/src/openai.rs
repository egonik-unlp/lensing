//! OpenAI embeddings client for manually authored listings. Blocking reqwest,
//! called from `spawn_blocking` like the qdrant helpers.

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

pub const DEFAULT_MODEL: &str = "text-embedding-3-small";
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Embed one text via the OpenAI embeddings API.
pub fn embed(api_key: &str, model: &str, base_url: &str, input: &str) -> Result<Vec<f32>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;
    let resp = client
        .post(format!("{base_url}/embeddings"))
        .bearer_auth(api_key)
        .json(&json!({ "model": model, "input": input }))
        .send()
        .context("openai embeddings request")?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        bail!("openai embeddings failed ({status}): {body}");
    }
    let v: Value = resp.json().context("parse openai embeddings response")?;
    let arr = v["data"][0]["embedding"]
        .as_array()
        .context("malformed openai embeddings response")?;
    arr.iter()
        .map(|x| x.as_f64().map(|f| f as f32).context("non-numeric embedding element"))
        .collect()
}
