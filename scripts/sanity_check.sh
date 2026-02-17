#!/usr/bin/env bash
set -euo pipefail

echo "== Agentic RAG Chatbot - Sanity Check =="

rm -rf artifacts
mkdir -p artifacts

echo "Running: python3 chatbot.py sanity"
python3 chatbot.py sanity

OUT="artifacts/sanity_output.json"
if [[ ! -f "$OUT" ]]; then
  echo "ERROR: Missing $OUT"
  echo "Your sanity check must generate: artifacts/sanity_output.json"
  exit 1
fi

echo "Verifying output format..."
python3 scripts/verify_output.py "$OUT"

echo "✓ Sanity check passed!"

echo "OK: sanity check passed"

After updating, run: chmod +x scripts/sanity_check.sh