//! Automated interpretability (Bills et al. 2023, "Language models can explain
//! neurons in language models"): label an SAE atom by showing a chat model the
//! items that most strongly activate it and asking what they share. A minimal
//! blocking OpenAI chat-completions client — enabled only when `OPENAI_API_KEY`
//! is set (no key means labeling is skipped, atoms are returned unlabeled). The
//! prompt is templated from the domain's own nouns so it reads naturally for any
//! prediction problem.

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

use lensing_core::domain::Domain;

pub struct LlmConfig {
    pub api_key: String,
    pub model: String,
    pub base_url: String,
}

impl LlmConfig {
    /// From env: `OPENAI_API_KEY` (required — `None` if absent/empty),
    /// `LENSING_LLM_MODEL` (default `gpt-4o-mini` — cheap is plenty for labels),
    /// `OPENAI_BASE_URL` (default the public API).
    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("OPENAI_API_KEY").ok().filter(|k| !k.is_empty())?;
        let model = std::env::var("LENSING_LLM_MODEL")
            .ok()
            .filter(|m| !m.is_empty())
            .unwrap_or_else(|| "gpt-4o-mini".to_string());
        let base_url = std::env::var("OPENAI_BASE_URL")
            .ok()
            .filter(|u| !u.is_empty())
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
        Some(Self { api_key, model, base_url })
    }
}

/// Domain vocabulary that flavors the auto-interp prompt (from `domain.toml`).
pub struct Vocab {
    pub entity_noun: String,
    pub entity_noun_plural: String,
    pub target_noun: String,
}

impl Vocab {
    pub fn from_domain(d: &Domain) -> Self {
        Vocab {
            entity_noun: d.project.entity_noun.clone(),
            entity_noun_plural: d.project.entity_noun_plural.clone(),
            target_noun: d.project.target_noun.clone(),
        }
    }
}

fn cap_first(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

/// Ask the model for a short label naming what the top-activating `items` for
/// atom `atom` share. Returns the trimmed label (possibly "no clear pattern").
/// The prompt is built from the domain nouns in `vocab`.
pub fn label_atom(cfg: &LlmConfig, vocab: &Vocab, atom: usize, items: &[String]) -> Result<String> {
    let (en, ep, tn) = (&vocab.entity_noun, &vocab.entity_noun_plural, &vocab.target_noun);
    let mut user = format!(
        "{} that most strongly activate latent feature #{atom} of a {en} {tn} model:\n\n",
        cap_first(ep)
    );
    for (i, t) in items.iter().enumerate() {
        user.push_str(&format!("{}. {}\n", i + 1, t));
    }
    user.push_str(&format!(
        "\nIn 8 words or fewer, name the concrete {en} attribute these {ep} share. \
         If there is no coherent shared attribute, reply exactly: no clear pattern."
    ));

    let body = json!({
        "model": cfg.model,
        "messages": [
            {"role": "system", "content":
                format!("You label latent features of a {en} {tn} model. \
                 Reply with only the short label — no preamble, no quotes.")},
            {"role": "user", "content": user}
        ],
        "temperature": 0.2,
        "max_tokens": 32
    });

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()?;
    let resp = client
        .post(format!("{}/chat/completions", cfg.base_url))
        .bearer_auth(&cfg.api_key)
        .json(&body)
        .send()
        .context("openai chat request")?;
    let status = resp.status();
    let v: Value = resp.json().context("parse openai chat response")?;
    if !status.is_success() {
        bail!("openai chat failed ({status}): {v}");
    }
    Ok(v["choices"][0]["message"]["content"].as_str().unwrap_or("").trim().to_string())
}
