#!/usr/bin/env bash
# OCR an image-only PDF into a text file (pdftoppm + tesseract).
# Usage: scripts/ocr-pdf.sh <input.pdf> <output.txt>
# Needed for scanned manuals (e.g. Panasonic microwaves) that pdf-parse can't read.
set -euo pipefail
IN="$1"; OUT="$2"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "OCR: $IN -> $OUT (pdftoppm + tesseract)"
pdftoppm -r 200 -png "$IN" "$TMP/page" 2>/dev/null
: > "$OUT"
for img in "$TMP"/page-*.png; do
  tesseract "$img" stdout 2>/dev/null >> "$OUT"
  # page separator for the chunker
  echo -e "\n\n=== PAGE BREAK ===\n" >> "$OUT"
done
echo "OCR done: $(wc -c < "$OUT") chars -> $OUT"
