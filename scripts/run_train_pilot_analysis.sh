#!/usr/bin/env bash
# Analyze the training pilot with the four dense models.
#
# BM25 and SPLADE are left out of the pilot because the server's Java/Pyserini
# environment and the SPLADE run need fixes; the pilot's purpose is to validate
# the pipeline and the selection rule, not to replace the full-corpus run.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-$HOME/miniconda3/envs/proyecto/bin/python}"
export MESSIRVE_PILOT="${MESSIRVE_PILOT:-$HOME/.cache/messirve_embeddings/pilot}"
MODELS="qwen3-0.6b,e5-large,bge-m3,jina-v5-small"
TRIO="e5-large+bge-m3+jina-v5-small"

mkdir -p results/train_pilot

echo "===== $(date -Is) — algorithm check ====="
"$PYTHON" -m rerankers.fuse_sweep \
  --models "$MODELS" \
  --strategies combmnz,rbc,rrf,isr \
  --combos "$TRIO" \
  --max-size 4 \
  --out-dir results/train_pilot/algorithm_check \
  --no-save-runs

echo "===== $(date -Is) — 11-subset sweep (CombMNZ) ====="
"$PYTHON" -m rerankers.fuse_sweep \
  --models "$MODELS" \
  --strategies combmnz \
  --max-size 4 \
  --out-dir results/train_pilot/sweep \
  --no-save-runs

echo "===== $(date -Is) — selection report ====="
"$PYTHON" -m evaluation.train_selection \
  --models "$MODELS" \
  --out-dir results/train_pilot

echo "===== $(date -Is) — done ====="
