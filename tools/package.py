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

Excluded entirely: data/ (binary artifacts), target/, ui/node_modules,
ui/dist, .git, docker volumes, personal settings. The consumer unpacks,
runs `/bootstrap` (or follows BOOTSTRAP.md), and owns their own history.

Usage: python3 tools/package.py [--out DIR]   (also: zig build package)
"""

import argparse
import shutil
import sys
import tarfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Framework files/dirs copied verbatim. (path, ignore-glob-patterns)
INCLUDE: list[tuple[str, tuple[str, ...]]] = [
    ("Cargo.toml", ()),
    ("build.zig", ()),
    ("docker-compose.yml", ()),
    ("domain.toml", ()),
    ("registry.toml", ()),
    (".gitignore", ()),
    ("BOOTSTRAP.md", ()),
    ("CLAUDE.md", ()),
    ("README.md", ()),
    ("DESIGN.md", ()),
    ("Cargo.lock", ()),
    ("assets", ()),
    ("docs/DEPLOY.md", ()),
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
and must be updated. Writers: the experiment-designer agent and the
model-definitions experiment workflow reconcile this file as part of every
campaign report.

Last updated: never — no facts yet; the first campaign seeds this.

## Best on record (leaderboard)

| model | dataset | {metrics} | source |
|---|---|---|---|
| _none yet_ | | | |

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

Rewrite me for your project (the original described an internal tool for
evaluating price-prediction models). Run `/bootstrap` to initialize the
template around your dataset — it walks you through domain.toml, the
dataset-pane levers, ingestion, and the starter models.
"""

MODELS_TOML_STUB = """\
# Model definitions registry (server-managed; the database is authoritative
# and re-exports this file on every mutation). Empty on a fresh template —
# create definitions via the UI or the model-definitions skill.
"""


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
    metrics = " | ".join(domain["metrics"]["columns"])
    # The framework's own name, independent of the shipped example domain.
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

    # Seeded instance files.
    (pkg / "models.toml").write_text(MODELS_TOML_STUB)
    (pkg / "experiments").mkdir(exist_ok=True)
    (pkg / "experiments/PROJECT-FACTS.md").write_text(
        FACTS_SKELETON.format(title=title, metrics=metrics)
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
    (pkg / "data").mkdir(exist_ok=True)
    (pkg / "data/.gitkeep").write_text("")

    tarball = out_root / f"{name}.tar.gz"
    with tarfile.open(tarball, "w:gz") as tf:
        tf.add(pkg, arcname=pkg.name)

    n_files = sum(1 for p in pkg.rglob("*") if p.is_file())
    size_mb = tarball.stat().st_size / 1e6
    print(f"packaged {n_files} files -> {pkg}")
    print(f"tarball: {tarball} ({size_mb:.1f} MB)")
    print("consumer quick start: unpack, then run /bootstrap (or follow BOOTSTRAP.md)")


if __name__ == "__main__":
    main()
