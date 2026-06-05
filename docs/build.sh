#!/usr/bin/env bash
# Build the docs/ PDFs: regenerate the matplotlib figures, then compile both
# LaTeX documents with Tectonic (https://tectonic-typesetting.github.io).
#
#   bash docs/build.sh
#
# Requires: python3 + matplotlib, tectonic on PATH (or in ~/.local/bin).
set -euo pipefail

cd "$(dirname "$0")"

TECTONIC=$(command -v tectonic || echo "$HOME/.local/bin/tectonic")
if [ ! -x "$TECTONIC" ]; then
    echo "error: tectonic not found (install: https://tectonic-typesetting.github.io/install.html)" >&2
    exit 1
fi

echo "== figures =="
python3 figures/make_figures.py

echo "== experiments.tex =="
"$TECTONIC" experiments.tex

echo "== api.tex =="
"$TECTONIC" api.tex

echo "== done =="
ls -la experiments.pdf api.pdf
