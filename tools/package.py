#!/usr/bin/env python3
"""Package this repository as a distributable TEMPLATE (the "library").

Copies the framework files into dist/<name>-template/ (+ a .tar.gz), and
SEEDS the instance files — the new project starts with an empty empirical
record instead of this project's history:

  models.toml                  -> empty definitions registry
  experiments/                 -> PROJECT-FACTS.md skeleton only
  docs/experiments.tex         -> minimal skeleton (title from domain.toml)
  docs/figures/make_figures.py -> style header only
  PRODUCT.md                   -> stub pointing at /bootstrap
  CLAUDE.md                    -> bootstrap-pending stub (NOT this repo's
                                  rendered copy — a fresh instance must never
                                  read the packaging domain as its own project
                                  truth; `zig build render-agents` regenerates
                                  it once domain.toml describes the new domain)

Excluded entirely: data/ (binary artifacts), target/, ui/node_modules,
ui/dist, .git, docker volumes, personal settings, and `.lensing-mother` (the
marker that identifies THIS checkout as the framework — an instance carrying
it would disable its own instantiation guard). The consumer unpacks,
runs `/bootstrap` (or follows BOOTSTRAP.md), and owns their own history.

The package also carries `.lensing-upstream.json`: a provenance manifest
(framework version, source commit, sha256 per framework file) that the
upstream-sync agent uses downstream to classify framework files 3-way
(changed upstream / changed locally / both) on later upgrades.

Usage: python3 tools/package.py [--out DIR]   (also: zig build package)
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Single definition of the stub sentinel lives next to the renderer that
# honors it (render.py --check accepts stub outputs; a plain render replaces
# them).
sys.path.insert(0, str(ROOT / "agents-src"))
from render import TEMPLATE_SENTINEL  # noqa: E402

# Framework files/dirs copied verbatim. (path, ignore-glob-patterns)
INCLUDE: list[tuple[str, tuple[str, ...]]] = [
    ("VERSION", ()),
    ("Cargo.toml", ()),
    ("build.zig", ()),
    ("docker-compose.yml", ()),
    ("domain.toml", ()),
    ("registry.toml", ()),
    (".gitignore", ()),
    ("BOOTSTRAP.md", ()),
    ("README.md", ()),
    ("DESIGN.md", ()),
    ("Cargo.lock", ()),
    ("assets", ()),
    ("docs/DEPLOY.md", ()),
    ("docs/REFERENCE.md", ()),
    # The server serves this from disk at /api/openapi.yaml (and Swagger UI
    # at /docs reads it) — instances need it shipped or both 404.
    ("docs/openapi.yaml", ()),
    ("crates", ("target",)),
    ("predictors", (".venv", "__pycache__", "Manifest.toml")),
    ("ui", ("node_modules", "dist")),
    ("agents-src", ("__pycache__",)),
    ("tools", ("__pycache__",)),
    (".claude", ("settings.local.json",)),
    (".agents", ()),
    (".gemini", ()),
    ("docs/build.sh", ()),
]

FACTS_SKELETON = """\
# PROJECT FACTS — {title}

Living, agent-maintained roll-up of empirical knowledge: the leaderboard,
noise bands, dataset lineage, per-family field guide and hard-won pitfalls.
The campaign reports in experiments/*.md are PRIMARY; this file is their
index. When they disagree, the newest report wins — and this file is stale
and must be updated. Writers: the experiment-runner agent and the
model-definitions experiment workflow reconcile this file as part of every
campaign report.

Last updated: never — no facts yet; the first campaign seeds this.

## Best on record (leaderboard)

| model | dataset | {metrics} | source |
{metrics_separator}
{metrics_empty_row}

## Noise bands & significance thresholds

- Unknown — run a same-split repeat and a seed study early; until then,
  treat small margins as noise.

## Dataset lineage

| dataset (recipe) | feeds | notes |
|---|---|---|
| _none yet_ | | |

## Known pitfalls per predictor family

Generic priors (replace with measured facts as campaigns land):

- Bounded-by-construction families (trees: leaf averages; rbf kernels:
  revert-to-intercept) cannot blow up through a log-target inverse;
  unbounded ones (linear models, MLPs) can — ship MLPs with
  `clamp_output: true` until proven safe.
- Features usually beat architecture: scan dataset-level axes before deep
  hyperparameter scans.
"""

TEX_SKELETON = r"""\documentclass[11pt]{article}
\usepackage[margin=2.6cm]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\newcommand{\runid}[1]{\texttt{\small #1}}
\newcommand{\dsid}[1]{\texttt{\small #1}}
\newcommand{\best}[1]{\textbf{#1}}

\title{%(title)s: Experiment Synthesis}
\author{}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
No campaigns yet. The report-curator agent grows this document from the
reports in \texttt{experiments/}.
\end{abstract}

\section{Introduction}
This document is the living synthesis of the experiment campaigns; the
markdown reports in \texttt{experiments/} are the primary record.

\appendix
\section{Source reports}
None yet.

\end{document}
"""

FIGURES_SKELETON = '''#!/usr/bin/env python3
"""Generate the vector-PDF figures for docs/experiments.tex.

All data is transcribed from the experiment reports in experiments/.
Each figure cites its source report in a comment; one fig_<name>()
function per figure. Run from anywhere: outputs land next to this script.
"""

from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT = Path(__file__).resolve().parent

# Consistent style
plt.rcParams.update(
    {{
        "figure.figsize": (6.0, 3.5),
        "figure.constrained_layout.use": True,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }}
)

ACCENT = "#2563eb"  # blue
GOOD = "#16a34a"  # green
BAD = "#dc2626"  # red
MUTED = "#9ca3af"  # gray

# Domain style header — keep axis units/labels going through these instead
# of hardcoding the unit per figure (values mirror [metrics] in domain.toml).
PRIMARY_METRIC = "{primary}"
VALUE_UNIT = "{unit}"
VALUE_AXIS = f"{{PRIMARY_METRIC}} ({{VALUE_UNIT}})"
kfmt = FuncFormatter(lambda v, _: f"{{v / 1000:.0f}}k")


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"wrote {{name}}")


# Figures are added by the report-curator agent, one fig_<name>() per
# figure, each citing its source report.

if __name__ == "__main__":
    pass
'''

PRODUCT_STUB = """\
# {title} — product one-pager

Rewrite me for your project. Run `/bootstrap` to initialize the
template around your dataset — it walks you through domain.toml, the
dataset-pane levers, ingestion, and the starter models.
"""

MODELS_TOML_STUB = """\
# Model definitions registry (server-managed; the database is authoritative
# and re-exports this file on every mutation). Empty on a fresh template —
# create definitions via the UI or the model-definitions skill.
"""

# Pre-bootstrap CLAUDE.md. This is what an agent reads as project
# instructions in a fresh (unbootstrapped) instance, BEFORE any domain
# exists — it must declare the template state, never describe any other
# project as if it were this one. The framework repo checks the same stub
# in at its root (it is itself a blank template).
CLAUDE_STUB = (
    TEMPLATE_SENTINEL
    + """
# Lensing template — bootstrap pending

This is an unconfigured lensing instance with NO domain of its own yet.
The FIRST action in this repository is `/bootstrap` (or follow
BOOTSTRAP.md). This instance is fully independent: it must never read
from or write to any other lensing project's corpus, datasets, server,
or experiment record.

Until bootstrap completes:

- `domain.toml` is a neutral placeholder (placeholder fields, no real
  corpus bindings); the generated agent layer (`.claude/`, `.agents/`,
  `.gemini/`) renders from it and is equally unconfigured. A complete
  worked example of domain.toml lives at
  `crates/lensing-core/src/example-domain.toml`.
- `zig build render-agents` regenerates the agent layer from
  `domain.toml` but leaves this stub in place. Once `domain.toml`
  describes the new domain, bootstrap finishes with
  `rm CLAUDE.md && zig build render-agents`, which writes the real
  project instructions here.
- The empirical record (models.toml, experiments/PROJECT-FACTS.md, docs
  skeletons) is seeded empty; the first campaign writes it.

## The instantiation rules (enforced, not advisory)

`tools/lensing_guard.py` decides whether this tree is a valid, finished
instance; the hooks in `.claude/settings.json` enforce its verdict. Run
`zig build bootstrap-check` (or `python3 tools/lensing_guard.py check`)
to see it at any time.

1. **This instance owns its folder.** It must be a full copy produced by
   `zig build package` and unpacked somewhere of its own — never the
   lensing framework checkout, never a subdirectory of it, never a
   partial tree. The provenance manifest `.lensing-upstream.json` is what
   proves it, and every framework file it lists must exist. A tree that
   fails this is `INVALID` and no work may proceed in it.
2. **Bootstrap always finishes.** While the instance is unbootstrapped,
   `/bootstrap` is the only work permitted: writes outside bootstrap's own
   files are denied, and so is every API mutation. Once bootstrap starts
   (`python3 tools/lensing_guard.py bootstrap start`) the session may not
   end until the invariants pass — or until the user's decision to stop is
   recorded with `bootstrap abort --reason "<why>"`. Do not improvise on
   the placeholder domain, and do not read another lensing project's
   corpus, datasets, server or experiment record to fill the gaps.
3. **Explorations go through the agents.** After bootstrap, run launches
   and dataset builds require a run ticket, so an experiment cannot be
   started outside the experiment-designer → experiment-runner path that
   writes the report and reconciles experiments/PROJECT-FACTS.md. A single
   ad-hoc run the user explicitly asked for is the one exception
   (`ticket issue --kind oneshot --reason "<what they asked for>"`).
"""
)


# Seeded (instance-owned) paths — recorded in the manifest so the
# upstream-sync agent knows they are never part of a framework sync.
SEEDED = [
    "models.toml",
    "experiments/PROJECT-FACTS.md",
    "docs/experiments.tex",
    "docs/figures/make_figures.py",
    "PRODUCT.md",
    "data/",
]

# Rendered agent-layer outputs: shipped for first-boot convenience, but
# downstream they are re-rendered from the instance's domain.toml and always
# diverge — the sync agent skips them and re-renders instead.
RENDERED_PREFIXES = (".claude/", ".agents/", ".gemini/")


def framework_version() -> str:
    try:
        return (ROOT / "VERSION").read_text().strip()
    except FileNotFoundError:
        print("package.py: warning: VERSION file missing; using 0.0.0-unversioned", file=sys.stderr)
        return "0.0.0-unversioned"


def git_provenance() -> tuple[str | None, bool, str | None]:
    """Best-effort (commit, dirty, origin_url) of the packaging checkout;
    (None, False, None) outside git. The origin url lets a downstream
    instance's upstream-contribute agent find the mother repo for PRs."""
    try:
        run = lambda *args: subprocess.run(  # noqa: E731
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        commit, dirty = run("rev-parse", "HEAD"), bool(run("status", "--porcelain"))
        try:
            origin = run("remote", "get-url", "origin") or None
        except Exception:
            origin = None
        return commit, dirty, origin
    except Exception:
        return None, False, None


def file_manifest(pkg: Path) -> dict[str, dict]:
    """sha256 per packaged file. Called after the INCLUDE copies and before
    seeding, so instance-owned files and the manifest itself are structurally
    excluded."""
    files = {}
    for p in sorted(pkg.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(pkg).as_posix()
        kind = "rendered" if rel.startswith(RENDERED_PREFIXES) else "framework"
        files[rel] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "kind": kind}
    return files


def copy_entry(src: Path, dst: Path, ignores: tuple[str, ...]) -> None:
    if src.is_dir():
        shutil.copytree(
            src,
            dst,
            ignore=shutil.ignore_patterns(*ignores) if ignores else None,
            symlinks=False,
        )
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "dist"))
    args = ap.parse_args()

    with open(ROOT / "domain.toml", "rb") as f:
        domain = tomllib.load(f)
    title = domain["project"]["title"]
    columns = domain["metrics"]["columns"]
    metrics = " | ".join(columns)
    # Leaderboard table geometry: model + dataset + metric columns + source.
    n_cols = 3 + len(columns)
    metrics_separator = "|" + "---|" * n_cols
    metrics_empty_row = "| _none yet_ " + "| " * (n_cols - 1) + "|"
    # The framework's own name, independent of the shipped placeholder domain.
    name = "lensing"

    out_root = Path(args.out)
    pkg = out_root / name
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True)

    missing = []
    for rel, ignores in INCLUDE:
        src = ROOT / rel
        if not src.exists():
            missing.append(rel)
            continue
        copy_entry(src, pkg / rel, ignores)
    if missing:
        print(f"package.py: warning: missing entries skipped: {missing}", file=sys.stderr)

    # Provenance manifest: hash the framework files (before seeding, so
    # instance-owned files never enter the map).
    version = framework_version()
    source_commit, source_dirty, source_repo = git_provenance()
    manifest = {
        "framework": name,
        "version": version,
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        # Where instances file upstream-contribute PRs; null outside git or
        # without an origin remote (the agent then asks for the repo).
        "source_repo": source_repo,
        "schema_version": domain.get("schema_version"),
        "files": file_manifest(pkg),
        "seeded": SEEDED,
        # CLAUDE.md is a rendered output too (agents-src/root/CLAUDE.md):
        # downstream it always diverges and is re-rendered, never synced.
        "rendered_outputs_excluded": [*RENDERED_PREFIXES, "CLAUDE.md"],
    }

    # Seeded instance files.
    (pkg / "models.toml").write_text(MODELS_TOML_STUB)
    (pkg / "experiments").mkdir(exist_ok=True)
    (pkg / "experiments/PROJECT-FACTS.md").write_text(
        FACTS_SKELETON.format(
            title=title,
            metrics=metrics,
            metrics_separator=metrics_separator,
            metrics_empty_row=metrics_empty_row,
        )
    )
    (pkg / "docs").mkdir(exist_ok=True)
    (pkg / "docs/experiments.tex").write_text(TEX_SKELETON % {"title": title})
    (pkg / "docs/figures").mkdir(exist_ok=True)
    (pkg / "docs/figures/make_figures.py").write_text(
        FIGURES_SKELETON.format(
            primary=domain["metrics"]["primary"], unit=domain["metrics"]["value_unit"]
        )
    )
    (pkg / "PRODUCT.md").write_text(PRODUCT_STUB.format(title=title))
    (pkg / "CLAUDE.md").write_text(CLAUDE_STUB)
    (pkg / "data").mkdir(exist_ok=True)
    (pkg / "data/.gitkeep").write_text("")
    (pkg / ".lensing-upstream.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # The instantiation guard reads these two paths to tell an instance from
    # the framework checkout. Ship neither wrong: .lensing-mother must stay
    # behind (it is not in INCLUDE, but a stray copy would make the instance
    # think it IS the framework and disable every gate), and the hook wiring
    # must travel (without it the invariants are documentation again).
    if (pkg / ".lensing-mother").exists():
        print(
            "package.py: error: .lensing-mother leaked into the package — an instance "
            "carrying it would disable the instantiation guard",
            file=sys.stderr,
        )
        sys.exit(1)
    settings = pkg / ".claude" / "settings.json"
    if not settings.exists() or "lensing_guard.py" not in settings.read_text():
        print(
            "package.py: error: .claude/settings.json is missing or does not wire "
            "tools/lensing_guard.py — the packaged instance would be unguarded",
            file=sys.stderr,
        )
        sys.exit(1)

    tarball = out_root / f"{name}.tar.gz"
    with tarfile.open(tarball, "w:gz") as tf:
        tf.add(pkg, arcname=pkg.name)

    n_files = sum(1 for p in pkg.rglob("*") if p.is_file())
    size_mb = tarball.stat().st_size / 1e6
    print(f"packaged {n_files} files (v{version}) -> {pkg}")
    print(f"tarball: {tarball} ({size_mb:.1f} MB)")
    print("consumer quick start: unpack, then run /bootstrap (or follow BOOTSTRAP.md)")


if __name__ == "__main__":
    main()
