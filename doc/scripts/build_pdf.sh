#!/usr/bin/env bash
# Compile the monograph to PDF with pdflatex (via latexmk).
#
#   bash scripts/build_pdf.sh            # build/monograph.pdf
#   REGEN=0 bash scripts/build_pdf.sh    # skip figure regeneration
#
# Run from the doc/ directory or anywhere; paths are resolved relative to the
# script location.
set -euo pipefail

DOC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$DOC_DIR/build"
cd "$DOC_DIR"
# Mirror the chapters/ subdir so \include's .aux files have somewhere to go
# (pdflatex -output-directory does not create subdirectories itself).
mkdir -p "$BUILD_DIR/chapters"

# --- 1. (Re)generate figures from the package, unless REGEN=0 ----------------
if [[ "${REGEN:-1}" != "0" ]]; then
  echo ">> Generating figures from the package ..."
  PY="${PYTHON:-}"
  if [[ -z "$PY" ]]; then
    if [[ -x "$DOC_DIR/../.venv/bin/python" ]]; then PY="$DOC_DIR/../.venv/bin/python";
    else PY="python3"; fi
  fi
  "$PY" "$DOC_DIR/scripts/make_figures.py" || \
    echo "!! figure generation failed; continuing (figures may be stale/missing)"
  "$PY" "$DOC_DIR/scripts/make_foundations_figures.py" || \
    echo "!! foundations figure generation failed; continuing"
fi

# --- 2. Compile with latexmk (handles the multiple passes) -------------------
echo ">> Compiling PDF with latexmk/pdflatex ..."
if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error \
          -output-directory="$BUILD_DIR" main.tex
else
  echo ">> latexmk not found; falling back to three pdflatex passes"
  for _ in 1 2 3; do
    pdflatex -interaction=nonstopmode -halt-on-error \
             -output-directory="$BUILD_DIR" main.tex
  done
fi

cp -f "$BUILD_DIR/main.pdf" "$BUILD_DIR/monograph.pdf" 2>/dev/null || true
echo ">> Done: $BUILD_DIR/monograph.pdf"
