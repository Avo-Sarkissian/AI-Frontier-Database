#!/usr/bin/env bash
# Compile report.tex -> FinalReport_Sarkissian.pdf, reproducibly.
#
# This existed only as a comment saying "compile on Overleaf", so the tracked
# PDF drifted from its source: report.tex was corrected while the PDF beside it
# kept printing 255+ models, eleven tabs, a Trends view that was never built,
# and Render deployment. Nothing said so, because nobody could rebuild it.
#
# neurips_2024.sty is vendored next to report.tex for the same reason — a build
# that depends on a file you have to fetch by hand is a build that stops
# happening.
#
# Requires tectonic (brew install tectonic). It fetches its own TeX packages on
# first run, so there is no TeX distribution to install.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v tectonic >/dev/null 2>&1; then
  echo "tectonic not found. Install it with:  brew install tectonic" >&2
  exit 1
fi

OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

echo "compiling report.tex…"
tectonic -X compile report.tex --outdir "$OUT"

cp "$OUT/report.pdf" FinalReport_Sarkissian.pdf
echo "wrote FinalReport_Sarkissian.pdf ($(du -h FinalReport_Sarkissian.pdf | cut -f1))"
