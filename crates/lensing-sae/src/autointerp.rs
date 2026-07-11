//! Automated-interpretability helpers (Bills et al. 2023) shared by the dataset
//! SAE (`lensing-sae analyze`) and any per-model SAE (e.g. `predictor-burn-mlp`'s
//! `model-sae` subcommand): find the items that most strongly activate an atom
//! and label the atom from them with an LLM. The labeling primitive itself lives
//! in [`crate::llm`]; this module supplies the row selection, item-text loading,
//! and the concurrent labeling loop around it, so both SAE callers share one copy.

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;

use crate::llm;

/// The rows that most strongly activate `atom`, highest first (active only).
/// `code` is the row-major `[n_rows, m]` per-atom activation matrix.
pub fn top_k_rows(code: &[f32], m: usize, atom: usize, k: usize) -> Vec<usize> {
    let n = code.len() / m;
    let mut v: Vec<(f32, usize)> =
        (0..n).map(|r| (code[r * m + atom], r)).filter(|(x, _)| *x > 1e-6).collect();
    v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    v.into_iter().take(k).map(|(_, r)| r).collect()
}

/// Item text per row: `items.json` is keyed by `str(point_id)` and
/// `row_ids[i]` is row `i`'s point id, so `content[i] = items[row_ids[i]].content`.
pub fn load_content(dataset: &Path, row_ids: &[u64], n_rows: usize) -> Vec<String> {
    let map: serde_json::Map<String, Value> = std::fs::read_to_string(dataset.join("items.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    (0..n_rows)
        .map(|i| {
            map.get(&row_ids[i].to_string())
                .and_then(|v| v.get("content"))
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string()
        })
        .collect()
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        return s.to_string();
    }
    let mut e = n;
    while e > 0 && !s.is_char_boundary(e) {
        e -= 1;
    }
    format!("{}…", &s[..e])
}

/// Label each atom from its top-activating items, a few requests at a time
/// (the LLM calls are independent and network-bound). Atoms that error or have
/// too few example items are simply left unlabeled.
pub fn label_atoms_concurrent(
    cfg: &llm::LlmConfig,
    vocab: &llm::Vocab,
    code: &[f32],
    m: usize,
    atoms: &[usize],
    content: &[String],
) -> HashMap<usize, String> {
    let labels = std::sync::Mutex::new(HashMap::new());
    for chunk in atoms.chunks(6) {
        std::thread::scope(|s| {
            for &atom in chunk {
                let labels = &labels;
                s.spawn(move || {
                    let items: Vec<String> = top_k_rows(code, m, atom, 12)
                        .iter()
                        .map(|&r| truncate(&content[r], 500))
                        .filter(|t| !t.is_empty())
                        .collect();
                    if items.len() >= 3 {
                        if let Ok(lbl) = llm::label_atom(cfg, vocab, atom, &items) {
                            if !lbl.is_empty() {
                                labels.lock().unwrap().insert(atom, lbl);
                            }
                        }
                    }
                });
            }
        });
    }
    labels.into_inner().unwrap()
}
