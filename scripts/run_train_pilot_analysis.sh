#!/usr/bin/env bash
# Analyze the full training pilot: 57 subsets, six algorithms, selection and
# stability. The pilot is the selection environment; the test partition is used
# only for the descriptive stability check.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-$HOME/miniconda3/envs/proyecto/bin/python}"
export MESSIRVE_PILOT="${MESSIRVE_PILOT:-$HOME/.cache/messirve_embeddings/pilot}"
PER_QUERY="${PER_QUERY:-$HOME/.cache/messirve_embeddings/train_selection/per_query}"
TRIO="e5-large+bge-m3+jina-v5-small"

mkdir -p "$PER_QUERY" results/train_pilot

echo "===== $(date -Is) — 57-subset sweep (CombMNZ) ====="
"$PYTHON" -m rerankers.fuse_sweep \
  --strategies combmnz \
  --max-size 6 \
  --out-dir results/train_pilot/sweep_all \
  --per-query-dir "$PER_QUERY" \
  --no-save-runs

echo "===== $(date -Is) — six algorithms on the trio ====="
"$PYTHON" -m rerankers.fuse_sweep \
  --models e5-large,bge-m3,jina-v5-small \
  --strategies combmnz,rbc,rrf,isr,bordafuse,condorcet \
  --combos "$TRIO" \
  --max-size 3 \
  --out-dir results/train_pilot/algorithm_check_six \
  --no-save-runs

echo "===== $(date -Is) — selection report ====="
"$PYTHON" -m evaluation.train_selection \
  --sweep results/train_pilot/sweep_all/results.csv \
  --algorithm-check results/train_pilot/algorithm_check_six/results.csv \
  --per-query-dir "$PER_QUERY" \
  --out-dir results/train_pilot

echo "===== $(date -Is) — stability ====="
"$PYTHON" -m evaluation.selection_stability --out-dir results/train_pilot

echo "===== $(date -Is) — done ====="
