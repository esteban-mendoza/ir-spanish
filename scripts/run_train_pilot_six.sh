#!/usr/bin/env bash
# Complete the six-model pilot: BM25 (needs JDK 21) and SPLADE.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-$HOME/miniconda3/envs/proyecto/bin/python}"
export MESSIRVE_PILOT="${MESSIRVE_PILOT:-$HOME/.cache/messirve_embeddings/pilot}"
JDK="$HOME/jdk21/jdk-21.0.13+9"

echo "===== $(date -Is) — bm25 ====="
JAVA_HOME="$JDK" PATH="$JDK/bin:$PATH" "$PYTHON" -m baselines.bm25

echo "===== $(date -Is) — splade-v3 ====="
"$PYTHON" -m baselines.splade_v3

echo "===== $(date -Is) — done ====="
