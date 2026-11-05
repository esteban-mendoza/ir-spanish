#!/usr/bin/env bash
# Rerun the neural rerankers with their native maximum context.
#
# bge-reranker-v2-m3: 512 -> 8192 tokens (query + document).
# jina-colbert-v2:    300 -> 8192 tokens per document (query budget stays 32).
# jina-reranker-v3:   no rerun.  Its 8192-token document cap is already the
#                     model maximum (paper Table 5), the 1024-token query cap
#                     exceeds the longest MessIRve query, and the model packs
#                     the 100 candidates into its shared 131k-token window.
#
# Run it on the IIMAS server, detached:
#   cd ~/ir-spanish
#   setsid nohup bash scripts/run_maxctx_rerankers.sh \
#     > logs/maxctx_rerankers.log 2>&1 < /dev/null &
#
# The script is resumable: each reranker writes per-query checkpoints and skips
# the whole run when its cached retrieval_run.lz4 already exists.

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

PYTHON="${PYTHON:-$HOME/miniconda3/envs/proyecto/bin/python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Group four queries per predict() call to fill the batches, but keep the
# quadratic budget at 2**27 so a query with several 8192-token documents does
# not exhaust the 24 GiB of an A5000.
export RERANKER_QUERY_BATCH="${RERANKER_QUERY_BATCH:-4}"
export RERANKER_QUADRATIC_TOKEN_BUDGET="${RERANKER_QUADRATIC_TOKEN_BUDGET:-134217728}"

log() {
	printf '%s | %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

log "Starting bge-reranker-v2-m3 at 8192 tokens"
"$PYTHON" -m rerankers.bge_reranker_v2_m3 2>&1 | tee logs/bge-reranker-v2-m3-maxctx.log

log "Starting jina-colbert-v2 at 8192 document tokens"
"$PYTHON" -m rerankers.jina_colbert_v2 2>&1 | tee logs/jina-colbert-v2-maxctx.log

log "All rerankers finished"
