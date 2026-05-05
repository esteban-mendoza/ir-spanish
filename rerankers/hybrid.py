#!/usr/bin/env python3
"""
Hybrid reranker: fuse multiple first-stage retrieval runs, then rerank
the fused result with jinaai/jina-reranker-v3.

Pipeline:
    1. Load cached first-stage runs (e5-large, bge-m3, jina-v5-small)
    2. Fuse with a configurable rank fusion strategy (default: CombMNZ)
    3. Rerank the fused run using jina-reranker-v3 (listwise cross-encoder)
    4. Save and evaluate against MessIRve qrels

Multi-GPU parallelism uses separate OS processes (spawn) to match the
proven-stable pattern from jina_reranker_v3.py.

Usage:
    python -m rerankers.hybrid
"""

from __future__ import annotations

import gc
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.multiprocessing as mp
from ranx import fuse

from rerankers.fuse import ModelConfig, load_runs
from rerankers.jina_reranker_v3 import (
    build_doc_lookup,
    _cleanup_checkpoints,
    _load_checkpoint,
    _save_checkpoint,
)
from utils import cache, data, retrieval
from utils.observability import Timer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration — first-stage models to fuse
# ---------------------------------------------------------------------------
FIRST_STAGE_MODELS = [
    ModelConfig("intfloat/multilingual-e5-large-instruct", "e5large", 512, 512),
    ModelConfig("BAAI/bge-m3", "bgem3", 8192, 8192),
    ModelConfig("jinaai/jina-embeddings-v5-text-small-retrieval", "jinav5small", 32768, 32768),
]

# ---------------------------------------------------------------------------
# Configuration — fusion strategy
# ---------------------------------------------------------------------------
FUSION_METHOD = "mnz"      # ranx method name: rrf, rbc, mnz, isr, bordafuse, condorcet
FUSION_PARAMS: dict = {}   # e.g. {"k": 60} for rrf, {"phi": 0.95} for rbc
FUSION_LABEL = "combmnz"   # human-readable label for cache paths and logging

# ---------------------------------------------------------------------------
# Configuration — second-stage reranker
# ---------------------------------------------------------------------------
RERANKER_MODEL = "jinaai/jina-reranker-v3"
MAX_QUERY_LENGTH = 1024
MAX_DOC_LENGTH = 8192

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
CHECKPOINT_EVERY = 200


# ---------------------------------------------------------------------------
# Cache path helpers
# ---------------------------------------------------------------------------
def _fusion_slug() -> str:
    """Build a slug like 'combmnz_e5large+bgem3+jinav5small'."""
    models_part = "+".join(m.alias for m in FIRST_STAGE_MODELS)
    return f"{FUSION_LABEL}_{models_part}"


def hybrid_cache_base() -> Path:
    """Build the cache directory for this hybrid pipeline.

    Layout:
        <cache_dir>/<reranker_slug>/over__<fusion_slug>/
            <country>_v<version>_q<ql>_d<dl>_<filter>/
                retrieval_run.lz4
                checkpoints/
    """
    reranker_slug = cache.model_slug(RERANKER_MODEL)
    filter_suffix = "nofilter" if data.MAX_WORD_COUNT is None else f"filt{data.MAX_WORD_COUNT}w"
    return (
        CACHE_DIR
        / reranker_slug
        / f"over__{_fusion_slug()}"
        / f"{data.COUNTRY}_v{data.DATASET_VERSION}_q{MAX_QUERY_LENGTH}_d{MAX_DOC_LENGTH}_{filter_suffix}"
    )


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
def fuse_runs(runs: list):
    """Fuse multiple retrieval runs using the configured strategy."""
    with Timer(f"Fusing {len(runs)} runs with {FUSION_LABEL}"):
        fused_run = fuse(
            runs=runs,
            method=FUSION_METHOD,
            params=FUSION_PARAMS,
        )
    fused_run.name = _fusion_slug()
    run_dict = fused_run.to_dict()
    total_docs = sum(len(docs) for docs in run_dict.values())
    log.info("Fused run: %d queries, %d total docs", len(run_dict), total_docs)
    return fused_run


# ---------------------------------------------------------------------------
# Worker — runs in a child process, one per GPU
# ---------------------------------------------------------------------------
def _worker(
    device: str,
    query_items: list[tuple[str, str, list[str]]],
    checkpoint_path: Path,
    corpus_workers: int,
) -> None:
    """Rerank a shard of queries on one GPU."""
    from transformers import AutoModel

    log.info("[%s] Building doc lookup ...", device)
    doc_lookup = build_doc_lookup(corpus_workers)

    log.info("[%s] Loading %s ...", device, RERANKER_MODEL)
    model = AutoModel.from_pretrained(
        RERANKER_MODEL, torch_dtype="auto", trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    log.info("[%s] Model loaded.", device)

    results = _load_checkpoint(checkpoint_path)
    done_qids: set[str] = set(results)
    if done_qids:
        log.info("[%s] Resuming: %d queries already scored", device, len(done_qids))

    pending = [item for item in query_items if item[0] not in done_qids]
    pending.sort(
        key=lambda item: max(len(doc_lookup.get(did, "").split()) for did in item[2]),
    )

    total = len(pending)
    scored_docs = 0
    t0 = time.perf_counter()

    with torch.no_grad():
        for idx, (query_id, query_text, doc_ids) in enumerate(pending, 1):
            doc_texts = [doc_lookup.get(did, "") for did in doc_ids]

            reranked = model.rerank(
                query_text,
                doc_texts,
                max_query_length=MAX_QUERY_LENGTH,
                max_doc_length=MAX_DOC_LENGTH,
            )

            results[query_id] = {
                doc_ids[r["index"]]: float(r["relevance_score"])
                for r in reranked
            }
            scored_docs += len(doc_ids)

            if idx % CHECKPOINT_EVERY == 0:
                torch.cuda.empty_cache()
                _save_checkpoint(checkpoint_path, results)
                elapsed = time.perf_counter() - t0
                rate = scored_docs / elapsed if elapsed > 0 else 0
                log.info(
                    "[%s] %d/%d queries | %d docs scored | %.0f docs/s | %.1fs",
                    device, idx, total, scored_docs, rate, elapsed,
                )

    _save_checkpoint(checkpoint_path, results)
    elapsed = time.perf_counter() - t0
    rate = scored_docs / elapsed if elapsed > 0 else 0
    log.info(
        "[%s] Done: %d queries, %d docs in %.1fs (%.0f docs/s)",
        device, total, scored_docs, elapsed, rate,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def rerank(
    fused_run,
    query_map: dict[str, str],
    reranker_base: Path,
) -> dict[str, dict[str, float]]:
    """Score documents with jina-reranker-v3 across multiple GPUs."""
    run_dict = fused_run.to_dict()

    query_items: list[tuple[str, str, list[str]]] = []
    total_docs = 0
    for query_id, doc_scores in run_dict.items():
        query_text = query_map.get(query_id)
        if query_text is None:
            continue
        doc_ids = list(doc_scores.keys())
        if doc_ids:
            total_docs += len(doc_ids)
            query_items.append((query_id, query_text, doc_ids))

    log.info(
        "Prepared %d queries, %d total docs across %d GPU(s)",
        len(query_items), total_docs, len(GPU_DEVICES),
    )

    shards: list[list[tuple[str, str, list[str]]]] = [[] for _ in GPU_DEVICES]
    for i, item in enumerate(query_items):
        shards[i % len(GPU_DEVICES)].append(item)

    checkpoint_dir = reranker_base / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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

    base = hybrid_cache_base()
    run_path = cache.run_cache_path(base)

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

    if run_path.exists():
        log.info("Hybrid reranked run found in cache: %s", run_path)
        reranked_run = retrieval.load_run(run_path)
        retrieval.run_evaluation(qrels, reranked_run, RERANKER_MODEL)
        return

    runs = load_first_stage_runs()
    fused_run = fuse_runs(runs)
    del runs
    gc.collect()

    reranked_dict = rerank(fused_run, query_map, base)
    del fused_run
    gc.collect()

    base.mkdir(parents=True, exist_ok=True)
    reranked_run = retrieval.save_run(reranked_dict, RERANKER_MODEL, run_path)
    log.info("Saved hybrid reranked run to %s", run_path)

    _cleanup_checkpoints(base)

    retrieval.run_evaluation(qrels, reranked_run, RERANKER_MODEL)


def load_first_stage_runs() -> list:
    """Load cached retrieval runs for all first-stage models."""
    return load_runs(FIRST_STAGE_MODELS)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
