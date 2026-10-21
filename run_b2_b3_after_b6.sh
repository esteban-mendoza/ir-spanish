#!/bin/bash
set +e
while kill -0 1567728 2>/dev/null; do
  sleep 30
done
cd /home/jmendoza/ir-spanish || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1

run_one() {
  name="$1"
  combo="$2"
  log="$3"
  rss="$4"
  echo "=== $name start $(date -Is) ===" >> "$log"
  /home/jmendoza/miniconda3/envs/proyecto/bin/python -m rerankers.fuse_sweep \
    --strategies bordafuse --combos "$combo" --no-save-runs \
    --out-dir results/fuse_stage_b \
    --runs-dir /home/jmendoza/.cache/messirve_embeddings/fusions_stage_b \
    >> "$log" 2>&1 &
  pid=$!
  echo "PID=$pid" >> "$log"
  : > "$rss"
  while kill -0 "$pid" 2>/dev/null; do
    echo "$(date +%H:%M:%S) $(ps -o rss= -p "$pid" | tr -d ' ')" >> "$rss"
    sleep 15
  done
  wait "$pid"
  echo "EXIT=$?" >> "$log"
}

run_one B2 "bm25+splade-v3+qwen3-0.6b+e5-large+bge-m3+jina-v5-small" \
  logs/fuse_stage_b_b2_memory.log logs/fuse_stage_b_b2_rss.log
run_one B3 "splade-v3+qwen3-0.6b+e5-large+bge-m3+jina-v5-small" \
  logs/fuse_stage_b_b3_memory.log logs/fuse_stage_b_b3_rss.log
