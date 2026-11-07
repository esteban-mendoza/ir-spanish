#!/usr/bin/env bash
# Mide el costo de codificar documentos e índices con 1 y con 3 modelos.
#
# Requiere el servidor libre (sin ColBERT ni otros benchmarks en la GPU).
# Uso:
#   bash scripts/run_encoding_cost.sh
#
# El piloto usa 1 000 documentos y una repetición; la corrida completa usa
# 20 000 documentos y tres repeticiones. Los resultados van a
# results/encoding_cost/ y el piloto a results/encoding_cost/pilot/.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-$HOME/miniconda3/envs/proyecto/bin/python}"
OUT="results/encoding_cost"
PILOT="$OUT/pilot"
LOG="logs/encoding_cost.log"
mkdir -p logs "$OUT" "$PILOT"

run() {
  echo "== $(date -Is) $*" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
}

run "$PYTHON" -m evaluation.encoding_cost --mode sample --n 1000 --out-dir "$PILOT"
for model in e5 bge jina; do
  run "$PYTHON" -m evaluation.encoding_cost --mode encode --model "$model" --reps 1 --out-dir "$PILOT"
done
run "$PYTHON" -m evaluation.encoding_cost --mode aggregate --out-dir "$PILOT"

run "$PYTHON" -m evaluation.encoding_cost --mode sample --n 20000 --out-dir "$OUT"
for model in e5 bge jina; do
  run "$PYTHON" -m evaluation.encoding_cost --mode encode --model "$model" --reps 3 --out-dir "$OUT"
done
run "$PYTHON" -m evaluation.encoding_cost --mode aggregate --out-dir "$OUT"

echo "Listo: $OUT/summary.md" | tee -a "$LOG"
