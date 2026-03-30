#!/usr/bin/env python3
"""
Rerank a first-stage retrieval run using the BAAI/bge-reranker-v2-m3
cross-encoder and evaluate against MessIRve qrels.

Usage:
    python -m rerankers.bge_reranker_v2_m3
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# utils.__init__ sets up logging and NUMA/threading env-vars
from utils import cache, data, retrieval

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FIRST_STAGE_MODEL = "jinaai/jina-embeddings-v5-text-small-retrieval"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
BATCH_SIZE = 256
MAX_QUERY_LENGTH = 512
MAX_DOC_LENGTH = 512
CACHE_DIR = Path.home() / ".cache" / "messirve_embeddings"
NUM_WORKERS = os.cpu_count() or 32


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


def rerank(
    first_stage_run,
    query_map: dict[str, str],
    doc_lookup: dict[str, str],
    model,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    """Score (query, document) pairs with the cross-encoder and return reranked results."""
    run_dict = first_stage_run.to_dict()
    reranked: dict[str, dict[str, float]] = {}
    num_queries = len(run_dict)

    for i, (query_id, doc_scores) in enumerate(run_dict.items()):
        query_text = query_map.get(query_id)
        if query_text is None:
            continue

        doc_ids = list(doc_scores.keys())
        pairs = []
        valid_doc_ids = []
        for doc_id in doc_ids:
            doc_text = doc_lookup.get(doc_id)
            if doc_text is not None:
                pairs.append((query_text, doc_text))
                valid_doc_ids.append(doc_id)

        if not pairs:
            continue

        scores = model.predict(pairs, batch_size=batch_size)
        reranked[query_id] = {
            doc_id: float(score) for doc_id, score in zip(valid_doc_ids, scores)
        }

        if (i + 1) % 500 == 0:
            log.info("Reranked %d / %d queries", i + 1, num_queries)

    log.info("Reranking complete: %d queries scored", len(reranked))
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

    from sentence_transformers import CrossEncoder
    log.info("Loading cross-encoder: %s", RERANKER_MODEL)
    model = CrossEncoder(RERANKER_MODEL, device="cuda")

    reranked_dict = rerank(first_stage_run, query_map, doc_lookup, model, BATCH_SIZE)

    reranker_base.mkdir(parents=True, exist_ok=True)
    reranked_run = retrieval.save_run(reranked_dict, RERANKER_MODEL, run_path)
    log.info("Saved reranked run to %s", run_path)

    retrieval.run_evaluation(qrels, reranked_run, RERANKER_MODEL)


if __name__ == "__main__":
    main()
