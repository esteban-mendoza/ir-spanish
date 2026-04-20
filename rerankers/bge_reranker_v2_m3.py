#!/usr/bin/env python3
"""
Rerank a first-stage retrieval run using BAAI/bge-reranker-v2-m3
(pointwise cross-encoder) and evaluate against MessIRve qrels.

bge-reranker-v2-m3 is a pointwise cross-encoder: it scores each
(query, document) pair independently. Scores are globally comparable,
so all ~100 candidates per query are batched into a single predict() call.

Multi-GPU parallelism uses separate OS processes (spawn) to match the
proven-stable pattern from jina_reranker_v3.py.

Usage:
    python -m rerankers.bge_reranker_v2_m3
"""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import time
from pathlib import Path

# Reduce CUDA memory fragmentation over long-running inference.
# Must be set before any CUDA initialization.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.multiprocessing as mp

# utils.__init__ sets up logging and NUMA/threading env-vars
from utils import cache, data, retrieval
from utils.observability import Timer, log_gpu_memory, log_ram_usage

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration — first-stage retriever
# ---------------------------------------------------------------------------
FIRST_STAGE_MODEL = "jinaai/jina-embeddings-v5-text-small-retrieval"
FIRST_STAGE_Q_LEN = 32768  # max_query_length used by the first-stage embedder
FIRST_STAGE_D_LEN = 32768  # max_doc_length used by the first-stage embedder

# ---------------------------------------------------------------------------
# Configuration — reranker
# ---------------------------------------------------------------------------
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
MAX_LENGTH = 512        # model's max sequence length (query + doc combined)
BATCH_SIZE = 256        # pairs per predict() call

# ---------------------------------------------------------------------------
# Configuration — infrastructure
# ---------------------------------------------------------------------------
GPU_DEVICES = (
    [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if torch.cuda.is_available()
    else ["cpu"]
)
CACHE_DIR = Path.home() / ".cache" / "messirve_embeddings"
NUM_WORKERS = os.cpu_count() or 32
CHECKPOINT_EVERY = 200  # save a checkpoint every N queries per worker


# ---------------------------------------------------------------------------
# Cache path helpers
# ---------------------------------------------------------------------------
def reranker_cache_base() -> Path:
    """Build the cache directory for this reranker + first-stage combination.

    Layout:
        <cache_dir>/<reranker_slug>/over__<first_stage_slug>/
            <country>_v<version>_q<ql>_d<dl>_<filter>/
                retrieval_run.lz4
                checkpoints/
    """
    reranker_slug = cache.model_slug(RERANKER_MODEL)
    first_stage_slug = cache.model_slug(FIRST_STAGE_MODEL)
    filter_suffix = "nofilter" if data.MAX_WORD_COUNT is None else f"filt{data.MAX_WORD_COUNT}w"
    return (
        CACHE_DIR
        / reranker_slug
        / f"over__{first_stage_slug}"
        / f"{data.COUNTRY}_v{data.DATASET_VERSION}_q{MAX_LENGTH}_d{MAX_LENGTH}_{filter_suffix}"
    )


def load_first_stage_run():
    """Load the cached first-stage retrieval run."""
    base = cache.cache_base(
        CACHE_DIR,
        FIRST_STAGE_MODEL,
        data.COUNTRY,
        data.DATASET_VERSION,
        FIRST_STAGE_Q_LEN,
        FIRST_STAGE_D_LEN,
        data.MAX_WORD_COUNT,
    )
    run_path = cache.run_cache_path(base)
    log.info("Loading first-stage run from %s", run_path)
    return retrieval.load_run(run_path)


# ---------------------------------------------------------------------------
# Doc lookup
# ---------------------------------------------------------------------------
def build_doc_lookup(num_workers: int) -> dict[str, str]:
    """Load the corpus and return a {doc_id: text} dict."""
    doc_ids, doc_texts, _ = data.load_corpus(num_workers)
    lookup = dict(zip(doc_ids, doc_texts))
    del doc_ids, doc_texts
    gc.collect()
    log.info("Doc lookup built: %d documents", len(lookup))
    return lookup


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def _save_checkpoint(path: Path, results: dict[str, dict[str, float]]) -> None:
    """Atomically save a checkpoint to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f)
    os.replace(tmp, path)


def _load_checkpoint(path: Path) -> dict[str, dict[str, float]]:
    """Load a checkpoint from disk, returning an empty dict if it doesn't exist."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _cleanup_checkpoints(reranker_base: Path) -> None:
    """Remove the checkpoint directory after a successful run."""
    checkpoint_dir = reranker_base / "checkpoints"
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        log.info("Cleaned up checkpoint directory")


# ---------------------------------------------------------------------------
# Worker — runs in a child process, one per GPU
# ---------------------------------------------------------------------------
def _worker(
    device: str,
    query_items: list[tuple[str, str, list[str]]],
    checkpoint_path: Path,
    corpus_workers: int,
) -> None:
    """Rerank a shard of queries on one GPU.

    Each worker loads its own copy of the corpus and model. This avoids
    pickling the multi-GB doc_lookup through mp.Process args (required
    because 'spawn' start method does not share parent memory).

    Args:
        device:          CUDA device string (e.g. "cuda:0").
        query_items:     List of (query_id, query_text, [doc_ids]) tuples.
        checkpoint_path: Path for this worker's checkpoint JSON file.
        corpus_workers:  Number of parallel workers for corpus loading.
    """
    from sentence_transformers import CrossEncoder

    # Each worker builds its own doc lookup from the HF-cached corpus.
    log.info("[%s] Building doc lookup ...", device)
    doc_lookup = build_doc_lookup(corpus_workers)

    log.info("[%s] Loading %s ...", device, RERANKER_MODEL)
    model = CrossEncoder(RERANKER_MODEL, max_length=MAX_LENGTH, device=device)
    log.info("[%s] Model loaded.", device)

    # Resume from checkpoint
    results = _load_checkpoint(checkpoint_path)
    done_qids: set[str] = set(results)
    if done_qids:
        log.info("[%s] Resuming: %d queries already scored", device, len(done_qids))

    # Filter out completed queries and sort by max doc word count (ascending)
    # so that short-document queries are processed first and CUDA memory
    # stays predictable for the heavier queries at the end.
    pending = [item for item in query_items if item[0] not in done_qids]
    pending.sort(
        key=lambda item: max(len(doc_lookup.get(did, "").split()) for did in item[2]),
    )

    total = len(pending)
    scored_pairs = 0
    t0 = time.perf_counter()

    with torch.no_grad():
        for idx, (query_id, query_text, doc_ids) in enumerate(pending, 1):
            pairs = [(query_text, doc_lookup.get(did, "")) for did in doc_ids]

            scores = model.predict(pairs, batch_size=BATCH_SIZE, show_progress_bar=False)

            results[query_id] = {
                did: float(score) for did, score in zip(doc_ids, scores)
            }
            scored_pairs += len(doc_ids)

            if idx % CHECKPOINT_EVERY == 0:
                torch.cuda.empty_cache()
                _save_checkpoint(checkpoint_path, results)
                elapsed = time.perf_counter() - t0
                rate = scored_pairs / elapsed if elapsed > 0 else 0
                log.info(
                    "[%s] %d/%d queries | %d pairs scored | %.0f pairs/s | %.1fs",
                    device, idx, total, scored_pairs, rate, elapsed,
                )

    # Final checkpoint
    _save_checkpoint(checkpoint_path, results)
    elapsed = time.perf_counter() - t0
    rate = scored_pairs / elapsed if elapsed > 0 else 0
    log.info(
        "[%s] Done: %d queries, %d pairs in %.1fs (%.0f pairs/s)",
        device, total, scored_pairs, elapsed, rate,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def rerank(
    first_stage_run,
    query_map: dict[str, str],
    reranker_base: Path,
) -> dict[str, dict[str, float]]:
    """Score documents with bge-reranker-v2-m3 across multiple GPUs using
    separate OS processes for stability."""
    run_dict = first_stage_run.to_dict()

    # Build work items: (query_id, query_text, [doc_ids])
    query_items: list[tuple[str, str, list[str]]] = []
    total_pairs = 0
    for query_id, doc_scores in run_dict.items():
        query_text = query_map.get(query_id)
        if query_text is None:
            continue
        doc_ids = list(doc_scores.keys())
        if doc_ids:
            total_pairs += len(doc_ids)
            query_items.append((query_id, query_text, doc_ids))

    log.info(
        "Prepared %d queries, %d total pairs across %d GPU(s)",
        len(query_items), total_pairs, len(GPU_DEVICES),
    )

    # Split queries across GPUs (round-robin)
    shards: list[list[tuple[str, str, list[str]]]] = [[] for _ in GPU_DEVICES]
    for i, item in enumerate(query_items):
        shards[i % len(GPU_DEVICES)].append(item)

    checkpoint_dir = reranker_base / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Give each worker a fair share of CPU cores for corpus loading
    corpus_workers = max(1, NUM_WORKERS // len(GPU_DEVICES))

    with Timer(f"Reranking {len(query_items)} queries on {len(GPU_DEVICES)} GPU(s)"):
        processes: list[mp.Process] = []
        for gpu_idx, device in enumerate(GPU_DEVICES):
            ckpt_path = checkpoint_dir / f"ckpt_gpu{gpu_idx}.json"
            p = mp.Process(
                target=_worker,
                args=(device, shards[gpu_idx], ckpt_path, corpus_workers),
            )
            p.start()
            processes.append(p)
            log.info("Started worker on %s (%d queries)", device, len(shards[gpu_idx]))

        for p in processes:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Worker exited with code {p.exitcode}")

    # Merge results from all workers
    merged: dict[str, dict[str, float]] = {}
    for gpu_idx in range(len(GPU_DEVICES)):
        ckpt_path = checkpoint_dir / f"ckpt_gpu{gpu_idx}.json"
        merged.update(_load_checkpoint(ckpt_path))

    log.info("Reranking complete: %d queries merged", len(merged))
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(42)

    base = reranker_cache_base()
    run_path = cache.run_cache_path(base)

    # Load qrels (needed for evaluation in all code paths)
    dataset_cache_dir = cache.dataset_cache_base(
        CACHE_DIR, data.COUNTRY, data.DATASET_VERSION, data.MAX_WORD_COUNT,
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

    reranked_dict = rerank(first_stage_run, query_map, base)

    del first_stage_run
    gc.collect()

    base.mkdir(parents=True, exist_ok=True)
    reranked_run = retrieval.save_run(reranked_dict, RERANKER_MODEL, run_path)
    log.info("Saved reranked run to %s", run_path)

    _cleanup_checkpoints(base)

    retrieval.run_evaluation(qrels, reranked_run, RERANKER_MODEL)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
