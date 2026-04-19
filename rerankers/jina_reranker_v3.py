#!/usr/bin/env python3
"""
Rerank a first-stage retrieval run using jinaai/jina-reranker-v3
(listwise reranker) and evaluate against MessIRve qrels.

Usage:
    python -m rerankers.jina_reranker_v3
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path

# Reduce CUDA memory fragmentation over long-running inference
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

# utils.__init__ sets up logging and NUMA/threading env-vars
from utils import cache, data, retrieval
from utils.observability import Timer, log_gpu_memory, log_ram_usage

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FIRST_STAGE_MODEL = "jinaai/jina-embeddings-v5-text-small-retrieval"
RERANKER_MODEL = "jinaai/jina-reranker-v3"
FIRST_STAGE_QUERY_LENGTH = 512  # query length used by the first-stage embedder
FIRST_STAGE_DOC_LENGTH = 512    # doc length used by the first-stage embedder
MAX_QUERY_LENGTH = 512          # reranker query truncation (model default)
MAX_DOC_LENGTH = 8192           # reranker doc truncation — model's effective sequence length (paper Table 5)
MAX_DOCS_PER_RERANK = 25        # cap docs per rerank() call to limit internal block size
GPU_DEVICES = (
    [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if torch.cuda.is_available()
    else ["cpu"]
)
CACHE_DIR = Path.home() / ".cache" / "messirve_embeddings"
NUM_WORKERS = os.cpu_count() or 32
CHECKPOINT_EVERY = 200  # save a checkpoint file every N queries per shard


# ---------------------------------------------------------------------------
# Cache path helper
# ---------------------------------------------------------------------------
def reranker_cache_base() -> Path:
    """Build the cache directory for this reranker + first-stage combination.

    Layout:
        <cache_dir>/<reranker_slug>/over__<first_stage_slug>/
            <country>_v<version>_q<ql>_d<dl>_<filter>/
                retrieval_run.lz4
    """
    reranker_slug = cache.model_slug(RERANKER_MODEL)
    first_stage_slug = cache.model_slug(FIRST_STAGE_MODEL)
    filter_suffix = "nofilter" if data.MAX_WORD_COUNT is None else f"filt{data.MAX_WORD_COUNT}w"
    return (
        CACHE_DIR
        / reranker_slug
        / f"over__{first_stage_slug}"
        / f"{data.COUNTRY}_v{data.DATASET_VERSION}_q{MAX_QUERY_LENGTH}_d{MAX_DOC_LENGTH}_{filter_suffix}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_first_stage_run():
    """Load the cached first-stage retrieval run."""
    base = cache.cache_base(
        CACHE_DIR,
        FIRST_STAGE_MODEL,
        data.COUNTRY,
        data.DATASET_VERSION,
        FIRST_STAGE_QUERY_LENGTH,
        FIRST_STAGE_DOC_LENGTH,
        data.MAX_WORD_COUNT,
    )
    run_path = cache.run_cache_path(base)
    log.info("Loading first-stage run from %s", run_path)
    return retrieval.load_run(run_path)


def build_doc_lookup(num_workers: int) -> dict[str, str]:
    """Load the corpus and return a {doc_id: text} dict."""
    doc_ids, doc_texts, _ = data.load_corpus(num_workers)
    lookup = dict(zip(doc_ids, doc_texts))
    log.info("Doc lookup built: %d documents", len(lookup))
    return lookup


def _rerank_shard(
    model,
    device: str,
    chunk: list[tuple[str, str, list[str]]],
    shard_index: int,
    doc_lookup: dict[str, str],
    checkpoint_dir: Path,
) -> dict[str, dict[str, float]]:
    shard_result: dict[str, dict[str, float]] = {}
    shard_docs = 0
    t0 = time.perf_counter()
    cp_file = checkpoint_dir / f"shard_{shard_index:04d}.json"

    # Sort queries by max candidate doc word count (ascending) so that
    # short-document queries are processed first and CUDA memory stays
    # predictable for the heavier queries at the end.
    chunk = sorted(
        chunk,
        key=lambda item: max(len(doc_lookup[did].split()) for did in item[2]),
    )

    with torch.no_grad():
        for idx, (query_id, query_text, doc_ids) in enumerate(chunk, 1):
            query_scores: dict[str, float] = {}

            for batch_start in range(0, len(doc_ids), MAX_DOCS_PER_RERANK):
                batch_doc_ids = doc_ids[batch_start : batch_start + MAX_DOCS_PER_RERANK]
                batch_doc_texts = [doc_lookup[did] for did in batch_doc_ids]

                results = model.rerank(
                    query_text,
                    batch_doc_texts,
                    max_query_length=MAX_QUERY_LENGTH,
                    max_doc_length=MAX_DOC_LENGTH,
                )
                for r in results:
                    query_scores[batch_doc_ids[r["index"]]] = float(r["relevance_score"])

            shard_result[query_id] = query_scores
            shard_docs += len(doc_ids)

            if idx % CHECKPOINT_EVERY == 0:
                # Release fragmented CUDA memory to prevent OOM over long runs
                torch.cuda.empty_cache()

                with open(cp_file, "w") as f:
                    json.dump(shard_result, f)
                elapsed = time.perf_counter() - t0
                rate = shard_docs / elapsed if elapsed > 0 else 0
                log.info(
                    "[%s] %d/%d queries | %d docs scored | %.0f docs/s | %.1fs | checkpoint saved",
                    device, idx, len(chunk), shard_docs, rate, elapsed,
                )

    # Final checkpoint for this shard
    if shard_result:
        with open(cp_file, "w") as f:
            json.dump(shard_result, f)
    elapsed = time.perf_counter() - t0
    rate = shard_docs / elapsed if elapsed > 0 else 0
    log.info(
        "[%s] Shard done: %d queries, %d docs scored in %.1fs (%.0f docs/s)",
        device, len(chunk), shard_docs, elapsed, rate,
    )
    return shard_result


def rerank(
    first_stage_run,
    query_map: dict[str, str],
    doc_lookup: dict[str, str],
    models: list[tuple],
    checkpoint_dir: Path,
) -> dict[str, dict[str, float]]:
    """Score documents with jina-reranker-v3 across multiple GPUs."""
    run_dict = first_stage_run.to_dict()
    num_queries = len(run_dict)
    devices = [dev for _, dev in models]

    # Load existing checkpoints for resume
    done: dict[str, dict[str, float]] = {}
    for cp_path in sorted(checkpoint_dir.glob("shard_*.json")):
        with open(cp_path) as f:
            done.update(json.load(f))
    done_qids: set[str] = set(done)
    if done_qids:
        log.info("Resuming: loaded %d queries from checkpoints", len(done_qids))

    with Timer(f"Reranking {num_queries} queries on {len(devices)} GPU(s)"):
        # Collect all query work items, skipping already-done queries
        query_items: list[tuple[str, str, list[str]]] = []
        total_docs = 0
        for query_id, doc_scores in run_dict.items():
            if query_id in done_qids:
                continue
            query_text = query_map.get(query_id)
            if query_text is None:
                continue
            doc_ids = [did for did in doc_scores if did in doc_lookup]
            if doc_ids:
                total_docs += len(doc_ids)
                query_items.append((query_id, query_text, doc_ids))

        log.info(
            "Prepared %d queries (%d resumed), %d total docs across %d GPU(s)",
            len(query_items), len(done_qids), total_docs, len(devices),
        )

        # Split queries evenly across GPUs
        n = len(models)
        chunks = [query_items[i::n] for i in range(n)]

        # Run shards sequentially — Jina's custom modeling.py is not thread-safe
        # (mutates model state in forward()), causing cudaErrorIllegalAddress
        # under ThreadPoolExecutor.
        reranked: dict[str, dict[str, float]] = {}
        reranked.update(done)
        for shard_idx, ((model, dev), chunk) in enumerate(zip(models, chunks)):
            shard_result = _rerank_shard(
                model, dev, chunk, shard_idx, doc_lookup, checkpoint_dir,
            )
            reranked.update(shard_result)

        log.info("Reranking complete: %d queries, %d total docs scored", len(reranked), total_docs)
        log_gpu_memory()

    return reranked


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    reranker_base = reranker_cache_base()
    run_path = cache.run_cache_path(reranker_base)

    # Load qrels (needed for evaluation in all code paths)
    dataset_cache_dir = cache.dataset_cache_base(
        CACHE_DIR, data.COUNTRY, data.DATASET_VERSION, data.MAX_WORD_COUNT
    )
    qrels, query_map = data.get_pruned_qrels_and_queries(
        data.COUNTRY,
        data.DATASET_VERSION,
        kept_doc_ids=None,
        dataset_cache_dir=dataset_cache_dir,
        num_workers=NUM_WORKERS,
    )

    # Check cache
    if run_path.exists():
        log.info("Reranked run found in cache: %s", run_path)
        reranked_run = retrieval.load_run(run_path)
        retrieval.run_evaluation(qrels, reranked_run, RERANKER_MODEL)
        return

    # Full reranking pipeline
    first_stage_run = load_first_stage_run()
    doc_lookup = build_doc_lookup(num_workers=NUM_WORKERS)
    log_ram_usage()

    from transformers import AutoModel

    log.info("Loading %s on %d device(s): %s", RERANKER_MODEL, len(GPU_DEVICES), GPU_DEVICES)
    models = []
    for dev in GPU_DEVICES:
        m = AutoModel.from_pretrained(
            RERANKER_MODEL, torch_dtype="auto", trust_remote_code=True,
        )
        m.to(dev)
        m.eval()
        models.append((m, dev))
        log.info("  Loaded model on %s", dev)
    log_gpu_memory()

    checkpoint_dir = reranker_base / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    reranked_dict = rerank(first_stage_run, query_map, doc_lookup, models, checkpoint_dir)

    reranker_base.mkdir(parents=True, exist_ok=True)
    reranked_run = retrieval.save_run(reranked_dict, RERANKER_MODEL, run_path)
    log.info("Saved reranked run to %s", run_path)

    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        log.info("Cleaned up checkpoint directory")

    retrieval.run_evaluation(qrels, reranked_run, RERANKER_MODEL)


if __name__ == "__main__":
    main()
