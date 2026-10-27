#!/usr/bin/env bash
# Run the six first-stage models on the closed training pilot.
#
# The pilot directory is built by scripts/build_train_pilot.py. This script
# runs the models sequentially so they do not compete for the two GPUs.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-$HOME/miniconda3/envs/proyecto/bin/python}"
export MESSIRVE_PILOT="${MESSIRVE_PILOT:-$HOME/.cache/messirve_embeddings/pilot}"

if [[ ! -f "$MESSIRVE_PILOT/metadata.json" ]]; then
  echo "Pilot not found at $MESSIRVE_PILOT; run scripts/build_train_pilot.py first" >&2
  exit 1
fi

for model in jina_v5_small bge_m3 e5_large qwen bm25 splade_v3; do
  echo "===== $(date -Is) — $model ====="
  "$PYTHON" -m "baselines.$model"
done

echo "===== $(date -Is) — done ====="
