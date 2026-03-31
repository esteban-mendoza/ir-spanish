#!/usr/bin/env python3
"""
Rerank a first-stage retrieval run using the BAAI/bge-reranker-v2-m3
cross-encoder and evaluate against MessIRve qrels.

Usage:
    python -m rerankers.bge_reranker_v2_m3
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

# utils.__init__ sets up logging and NUMA/threading env-vars
from utils import cache, data, retrieval
from utils.observability import Timer, log_gpu_memory, log_ram_usage

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FIRST_STAGE_MODEL = "jinaai/jina-embeddings-v5-text-small-retrieval"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
BATCH_SIZE = 512
MAX_QUERY_LENGTH = 512
MAX_DOC_LENGTH = 512
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
        MAX_QUERY_LENGTH,
        MAX_DOC_LENGTH,
        data.MAX_WORD_COUNT,
    )
    run_path = cache.run_cache_path(base)
    log.info("Loading first-stage run from %s", run_path)
    return retrieval.load_run(run_path)


def build_doc_lookup(num_workers: int) -> dict[str, str]:
    """Load the corpus and return a {doc_id: text} dict."""
    doc_ids, doc_texts = data.load_corpus(num_workers)
    lookup = dict(zip(doc_ids, doc_texts))
    log.info("Doc lookup built: %d documents", len(lookup))
    return lookup


def _rerank_shard(
    model,
    device: str,
    chunk: list[tuple[str, str, list[str]]],
    shard_index: int,
    doc_lookup: dict[str, str],
    batch_size: int,
    checkpoint_dir: Path,
) -> dict[str, dict[str, float]]:
    shard_result: dict[str, dict[str, float]] = {}
    shard_pairs = 0
    t0 = time.perf_counter()
    cp_file = checkpoint_dir / f"shard_{shard_index:04d}.json"
    for idx, (query_id, query_text, doc_ids) in enumerate(chunk, 1):
        pairs = [(query_text, doc_lookup[did]) for did in doc_ids]
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        shard_result[query_id] = {
            doc_id: float(score) for doc_id, score in zip(doc_ids, scores)
        }
        shard_pairs += len(pairs)
        if idx % CHECKPOINT_EVERY == 0:
            with open(cp_file, "w") as f:
                json.dump(shard_result, f)
            elapsed = time.perf_counter() - t0
            rate = shard_pairs / elapsed if elapsed > 0 else 0
            log.info(
                "[%s] %d/%d queries | %d pairs | %.0f pairs/s | %.1fs | checkpoint saved",
                device, idx, len(chunk), shard_pairs, rate, elapsed,
            )
    # Final checkpoint for this shard
    if shard_result:
        with open(cp_file, "w") as f:
            json.dump(shard_result, f)
    elapsed = time.perf_counter() - t0
    rate = shard_pairs / elapsed if elapsed > 0 else 0
    log.info(
        "[%s] Shard done: %d queries, %d pairs in %.1fs (%.0f pairs/s)",
        device, len(chunk), shard_pairs, elapsed, rate,
    )
    return shard_result


def rerank(
    first_stage_run,
    query_map: dict[str, str],
    doc_lookup: dict[str, str],
    models: list[tuple],
    batch_size: int,
    checkpoint_dir: Path,
) -> dict[str, dict[str, float]]:
    """Score (query, document) pairs with cross-encoders across multiple GPUs."""
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
        query_items: list[tuple[str, str, list[str]]] = []  # (qid, query_text, doc_ids)
        total_pairs = 0
        for query_id, doc_scores in run_dict.items():
            if query_id in done_qids:
                continue
            query_text = query_map.get(query_id)
            if query_text is None:
                continue
            doc_ids = [did for did in doc_scores if did in doc_lookup]
            if doc_ids:
                total_pairs += len(doc_ids)
                query_items.append((query_id, query_text, doc_ids))

        log.info(
            "Prepared %d queries (%d resumed), %d total pairs across %d GPU(s)",
            len(query_items), len(done_qids), total_pairs, len(devices),
        )

        # Split queries evenly across GPUs
        n = len(models)
        chunks = [query_items[i::n] for i in range(n)]

        # Run shards in parallel
        reranked: dict[str, dict[str, float]] = {}
        reranked.update(done)
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            futures = [
                executor.submit(
                    _rerank_shard, model, dev, chunk, shard_idx,
                    doc_lookup, batch_size, checkpoint_dir,
                )
                for shard_idx, ((model, dev), chunk) in enumerate(zip(models, chunks))
            ]
            for f in futures:
                reranked.update(f.result())

        log.info("Reranking complete: %d queries, %d total pairs scored", len(reranked), total_pairs)
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

    from sentence_transformers import CrossEncoder

    log.info("Loading cross-encoder %s on %d device(s): %s", RERANKER_MODEL, len(GPU_DEVICES), GPU_DEVICES)
    models = []
    for dev in GPU_DEVICES:
        m = CrossEncoder(RERANKER_MODEL, device=dev)
        models.append((m, dev))
        log.info("  Loaded model on %s", dev)
    log_gpu_memory()

    checkpoint_dir = reranker_base / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    reranked_dict = rerank(first_stage_run, query_map, doc_lookup, models, BATCH_SIZE, checkpoint_dir)

    reranker_base.mkdir(parents=True, exist_ok=True)
    reranked_run = retrieval.save_run(reranked_dict, RERANKER_MODEL, run_path)
    log.info("Saved reranked run to %s", run_path)

    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        log.info("Cleaned up checkpoint directory")

    retrieval.run_evaluation(qrels, reranked_run, RERANKER_MODEL)


if __name__ == "__main__":
    main()
