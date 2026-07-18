"""Serving-time ranking helpers for next-item (task = ranking) predictors.

Framework-neutral counterpart to the pointwise predict path: a promoted ranking
model ranks a caller-supplied session prefix over the training vocabulary. The
server (`predict_ranking` in lensing-server) writes the request to
`<input>/prefix.json` as `{"prefix": [...], "k": <int|null>,
"identity_field": <str|null>}` and reads back the standard `predictions.json`
shape (`[{row_id, predicted, top_k_ids}]`).

A predictor plugs in by supplying a `score_fn(prefix_idx) -> full-vocab score
vector` and an artifact object exposing:
  - `art.n_items`  : int vocabulary size
  - `art.items`    : dict `{ "<vocab_index>": { ...opaque item metadata... } }`
The item-identity field name (`identity_field`) is domain-configured
(`[sequence].identity_field`) and arrives in the request — NO token shape or
metadata field name is hardcoded here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def emit(obj: dict) -> None:
    """One JSON line on stdout — the predictor <-> server log/event channel."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _bare_id(tok: str) -> str:
    """Last path/scheme segment of a URI/URL-ish token: strip a query string,
    a trailing slash, then take the final '/'- and ':'-delimited component.
    Domain-agnostic — works for any colon/slash-delimited identifier."""
    tok = tok.split("?", 1)[0].rstrip("/")
    return tok.rsplit("/", 1)[-1].rsplit(":", 1)[-1]


def resolve_prefix(art, tokens: list, identity_field: str = "id") -> tuple[np.ndarray, list]:
    """Map serving prefix tokens to vocab item indices, preserving order.

    A token may be: the item's integer vocab index (int or a purely-numeric
    string); the item's canonical identity value (`art.items[i][identity_field]`,
    matched exactly); or the bare id of such a value (see `_bare_id`, for
    URI/URL forms). Unknown tokens (cold items outside the vocab) are dropped and
    returned separately so the caller can warn. Returns (indices int64, unknown).
    """
    n_items = art.n_items
    ident_to_idx: dict[str, int] = {}
    bareid_to_idx: dict[str, int] = {}
    for key, meta in art.items.items():
        i = int(key)
        val = meta.get(identity_field) if isinstance(meta, dict) else None
        if val:
            ident_to_idx[str(val)] = i
            bareid_to_idx.setdefault(_bare_id(str(val)), i)

    idxs: list[int] = []
    unknown: list = []
    for tok in tokens:
        # Bare integer vocab index (int, or a purely-numeric string).
        if isinstance(tok, int) or (isinstance(tok, str) and tok.strip().isdigit()):
            i = int(tok)
            (idxs.append(i) if 0 <= i < n_items else unknown.append(tok))
            continue
        t = str(tok).strip()
        idx = ident_to_idx.get(t)
        if idx is None:
            idx = bareid_to_idx.get(_bare_id(t))
        (idxs.append(idx) if idx is not None else unknown.append(tok))
    return np.asarray(idxs, dtype=np.int64), unknown


def rank_topk(art, score_fn, prefix_idx: np.ndarray, k: int = 10) -> list[int]:
    """Score the full vocab for one prefix, exclude the prefix items (next-
    distinct rule), and return the top-k item indices best-first."""
    scores = np.asarray(score_fn(prefix_idx), dtype=np.float64).copy()
    assert scores.shape[0] == art.n_items, "score_fn must cover the full vocab"
    if prefix_idx.size:
        scores[prefix_idx] = -np.inf
    order = np.argsort(-scores, kind="stable")
    return [int(i) for i in order[:k]]


def load_prefix_request(input_dir: Path) -> tuple[list, "int | None", str]:
    """Read the server-written ranking input (`prefix.json`): an ordered list of
    prefix tokens, an OPTIONAL requested k (None ⇒ use the model's trained k),
    and the domain's `identity_field` (defaults to "id" when unset). Also accepts
    a bare JSON list of tokens (⇒ no k override, default identity_field)."""
    req = json.loads((input_dir / "prefix.json").read_text())
    if isinstance(req, list):
        return req, None, "id"
    k = req.get("k")
    identity_field = req.get("identity_field") or "id"
    return list(req.get("prefix", [])), (int(k) if k is not None else None), identity_field


def predict_ranking(art, score_fn, input_dir: Path, output: Path,
                    k: "int | None" = None) -> None:
    """Full serving-time predict: read the prefix request, rank the vocab, and
    write the `predictions.json` shape ([{row_id, predicted, top_k_ids}]). One
    query (prefix) per call ⇒ a single-element list. Result-size precedence:
    caller-requested k (prefix.json) > the model's trained k (arg) > 10."""
    tokens, req_k, identity_field = load_prefix_request(input_dir)
    kk = int(req_k if req_k is not None else (k if k is not None else 10))
    prefix_idx, unknown = resolve_prefix(art, tokens, identity_field)
    if unknown:
        emit({"event": "log",
              "msg": f"predict: {len(unknown)} prefix token(s) not in vocab "
                     f"(cold, dropped): {unknown[:5]}"})
    if prefix_idx.size == 0:
        raise SystemExit("predict: no prefix token resolved to a known vocab "
                         "item — cannot rank a next item")
    top = rank_topk(art, score_fn, prefix_idx, k=kk)
    preds = [{
        "row_id": 0,                              # single query
        "predicted": float(top[0]) if top else -1.0,
        "top_k_ids": [int(i) for i in top],
    }]
    output.write_text(json.dumps(preds))
    emit({"event": "log",
          "msg": f"ranked top-{kk} over {art.n_items} items from a "
                 f"{prefix_idx.size}-item prefix"})
