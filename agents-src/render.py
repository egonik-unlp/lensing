#!/usr/bin/env python3
"""Render the agent/skill templates in agents-src/ from domain.toml.

Single source of truth for the agentic layer: the templates here carry the
process knowledge; domain.toml carries the domain parameters; this script
substitutes them and writes the (previously hand-triplicated) copies:

  agents-src/agents/<name>.md          -> .claude/agents/<name>.md
  agents-src/skills/<name>/SKILL.md    -> .claude/skills/<name>/SKILL.md
                                          .agents/skills/<name>/SKILL.md
                                          .gemini/skills/<name>/SKILL.md

Template syntax (deliberately tiny, stdlib-only):
  {{key}}            substitute a placeholder (see placeholders() below)
  {{> name}}         include agents-src/_partials/<name>.md (then substitute)
Unknown placeholders are a hard error — no silent half-rendered output.

The `impeccable` skill is NOT rendered: it legitimately diverges per harness
and stays vendored.

Usage:
  python3 agents-src/render.py            render in place
  python3 agents-src/render.py --check    exit 1 if any output differs (CI)
"""

import sys
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "agents-src"

SKILL_TARGETS = [".claude/skills", ".agents/skills", ".gemini/skills"]
AGENT_TARGETS = [".claude/agents"]


def die(msg: str) -> None:
    print(f"render.py: error: {msg}", file=sys.stderr)
    sys.exit(1)


def load_domain() -> dict:
    path = ROOT / "domain.toml"
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        die(f"{path} not found — the domain config is the render input")
    except tomllib.TOMLDecodeError as e:
        die(f"{path}: {e}")


def listing_schema_rows(d: dict) -> str:
    """Markdown table rows for the ingestion agent's field schema, generated
    from the domain's field descriptors (target/display/content rows are
    static template text — they carry domain-independent process notes)."""
    rows = []
    for f in d["fields"]:
        role = f["role"]
        if role in ("target", "display"):
            continue
        name = f["name"]
        notes = []
        if f.get("label"):
            notes.append(f["label"])
        if role == "categorical":
            if f.get("suggestions"):
                vocab = ", ".join(f"`{s}`" for s in f["suggestions"])
                notes.append(f"corpus vocabulary: {vocab}")
            else:
                notes.append("categorical, as on the page")
        elif role == "numeric":
            notes.append("plain number")
            if f.get("zero_is_missing"):
                notes.append("0 = unspecified (omit instead)")
        elif role == "coordinates":
            notes.append('`{"lat": <f64>, "lon": <f64>}` if the page has a map/geocode')
        elif role == "filter_only":
            notes.append("corpus filter field — match the corpus convention")
        elif role == "timestamp":
            notes.append("omit; the server stamps it")
        if f.get("required"):
            notes.append("**required**")
        rows.append(f"| `{name}` | {'; '.join(notes)}. |")
    return "\n".join(rows)


def dataset_feature_rows(d: dict) -> str:
    """Markdown table rows for the dataset-design skill's per-field toggles."""
    rows = []
    for f in d["fields"]:
        role = f["role"]
        if role not in ("categorical", "numeric", "coordinates"):
            continue
        default = "true" if f.get("default_on") else "false"
        bits = [f.get("label") or f["name"], role]
        if role == "categorical":
            vocab = f.get("vocab", "all")
            bits.append(
                f"top-{vocab['top_n']} one-hot + `__other__`"
                if isinstance(vocab, dict)
                else "one-hot over all values + `__other__`"
            )
        if f.get("reconcile"):
            bits.append("reconciled from the companion collection")
        if f.get("group"):
            bits.append(f"group `{f['group']}` toggles together")
        rows.append(f"| `{f['name']}` | {default} | {'; '.join(bits)} |")
    return "\n".join(rows)


def placeholders(d: dict) -> dict:
    a = d.get("agents", {})
    project = d["project"]
    corpus = d["corpus"]
    metrics = d["metrics"]
    currency = d.get("currency") or {}
    api_base_url = a.get("api_base_url", "http://localhost:8080")
    return {
        "project_name": project["name"],
        "project_title": project["title"],
        "entity_noun": project["entity_noun"],
        "entity_noun_plural": project["entity_noun_plural"],
        "target_noun": project["target_noun"],
        "target_field": d["target"]["field"],
        "api_base_url": api_base_url,
        "api_host": re.sub(r"^https?://", "", api_base_url),
        "report_dir": a.get("report_dir", "experiments"),
        "facts_file": a.get("facts_file", "experiments/PROJECT-FACTS.md"),
        "definition_naming": a.get("definition_naming", "<predictor>-<slug>"),
        "definition_naming_example": a.get("definition_naming_example", ""),
        "collection": corpus["collection"],
        "manual_collection": corpus["manual_collection"],
        "qdrant_url": corpus["qdrant_url"],
        "embedding_dim": str(corpus["embedding_dim"]),
        "content_field": corpus.get("content_field", "content"),
        "primary_metric": metrics["primary"],
        "metric_columns_md": " | ".join(metrics["columns"]),
        "metric_columns_slash": " / ".join(metrics["columns"]),
        "value_unit": metrics["value_unit"],
        "currency_keep": currency.get("keep", ""),
        "reconcile_collection": currency.get("reconcile_collection", "") or "",
        "listing_schema_rows": listing_schema_rows(d),
        "dataset_feature_rows": dataset_feature_rows(d),
    }


PARTIAL_RE = re.compile(r"\{\{>\s*([a-zA-Z0-9_-]+)\s*\}\}")
TOKEN_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


def render_text(text: str, ph: dict, src: Path) -> str:
    def include(m: re.Match) -> str:
        p = SRC / "_partials" / f"{m.group(1)}.md"
        if not p.is_file():
            die(f"{src}: partial {p} not found")
        return p.read_text()

    text = PARTIAL_RE.sub(include, text)

    leftovers = []

    def sub(m: re.Match) -> str:
        key = m.group(1)
        if key not in ph:
            leftovers.append(key)
            return m.group(0)
        return ph[key]

    out = TOKEN_RE.sub(sub, text)
    if leftovers:
        die(f"{src}: unknown placeholder(s): {sorted(set(leftovers))}")
    return out


def with_marker(rendered: str, rel: str) -> str:
    marker = (
        f"<!-- GENERATED from agents-src/{rel} by agents-src/render.py — "
        f"edit the template (and domain.toml), not this file; "
        f"then run `zig build render-agents`. -->\n"
    )
    # Frontmatter must stay first (the harness parses it); marker goes after.
    if rendered.startswith("---\n"):
        end = rendered.find("\n---\n", 4)
        if end != -1:
            head = rendered[: end + len("\n---\n")]
            return head + marker + rendered[end + len("\n---\n") :]
    return marker + rendered


def outputs(d: dict) -> list[tuple[Path, Path]]:
    """(template, output) pairs."""
    pairs = []
    a = d.get("agents", {})
    for tpl in sorted((SRC / "agents").glob("*.md")):
        if tpl.stem == "listing-generator" and not a.get("ingestion", True):
            continue
        for target in AGENT_TARGETS:
            pairs.append((tpl, ROOT / target / tpl.name))
    for tpl in sorted((SRC / "skills").glob("*/SKILL.md")):
        for target in SKILL_TARGETS:
            pairs.append((tpl, ROOT / target / tpl.parent.name / "SKILL.md"))
    # Repo-root documents (CLAUDE.md, …).
    for tpl in sorted((SRC / "root").glob("*.md")):
        pairs.append((tpl, ROOT / tpl.name))
    return pairs


def main() -> None:
    check = "--check" in sys.argv[1:]
    d = load_domain()
    ph = placeholders(d)
    drift = []
    for tpl, out in outputs(d):
        rel = tpl.relative_to(SRC)
        rendered = with_marker(render_text(tpl.read_text(), ph, tpl), str(rel))
        current = out.read_text() if out.is_file() else None
        if rendered == current:
            continue
        if check:
            drift.append(str(out.relative_to(ROOT)))
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(rendered)
            print(f"rendered {out.relative_to(ROOT)}")
    if check and drift:
        print("render.py --check: out-of-date outputs (run `zig build render-agents`):")
        for f in drift:
            print(f"  {f}")
        sys.exit(1)
    if check:
        print("render.py --check: all rendered outputs up to date")


if __name__ == "__main__":
    main()
