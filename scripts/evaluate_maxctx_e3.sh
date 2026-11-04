#!/usr/bin/env bash
# Evaluate Experiment 3 with the maximum-context reranker runs.
#
# Usage (after scripts/run_maxctx_rerankers.sh):
#   cd ~/ir-spanish
#   bash scripts/evaluate_maxctx_e3.sh
#
# The output files are e3_maxctx.{csv,md} and e3_maxctx_recall.{csv,md}, so the
# previous evaluation in results/evaluation/ is preserved.

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

PYTHON="${PYTHON:-$HOME/miniconda3/envs/proyecto/bin/python}"
CACHE="${HOME}/.cache/messirve_embeddings"
FIRST_STAGE="over__jinaai__jina-embeddings-v5-text-small-retrieval"

export BGE_RERANKER_RUN="$CACHE/BAAI__bge-reranker-v2-m3/$FIRST_STAGE/full_v1.2_q8192_d8192_nofilter/retrieval_run.lz4"
export COLBERT_RERANKER_RUN="$CACHE/jinaai__jina-colbert-v2/$FIRST_STAGE/full_v1.2_q32_d8192_nofilter/retrieval_run.lz4"

"$PYTHON" -m evaluation.e3 --metric ndcg@10 --suffix _maxctx 2>&1 | tee logs/e3_maxctx.log
"$PYTHON" -m evaluation.e3 --metric recall@100 --suffix _maxctx_recall 2>&1 | tee logs/e3_maxctx_recall.log
