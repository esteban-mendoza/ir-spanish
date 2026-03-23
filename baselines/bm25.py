#!/usr/bin/env python3
"""
BM25 baseline for MessIRve (Spanish IR) using Pyserini.
Metrics: nDCG@10, Recall@100 via ranx.

Uses Lucene's BM25 with Spanish language analysis (stemming + stopwords).
No neural model or GPU required.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from pyserini.index.lucene import LuceneIndexer
from pyserini.search.lucene import LuceneSearcher

from utils import cache, data, retrieval
from utils.observability import Timer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_SLUG = "bm25_pyserini"
NUM_WORKERS = os.cpu_count() or 32
TOP_K = 100

CACHE_DIR = Path.home() / ".cache" / "messirve_embeddings"

# BM25 has no seq length, so path only reflects dataset config
MODEL_CACHE_BASE = (
    CACHE_DIR / MODEL_SLUG / f"{data.COUNTRY}_v{data.DATASET_VERSION}_filt{data.MAX_WORD_COUNT}w"
)
INDEX_DIR = MODEL_CACHE_BASE / "lucene_index"
RUN_PATH = cache.run_cache_path(MODEL_CACHE_BASE)
DATASET_CACHE_DIR = cache.dataset_cache_base(
    CACHE_DIR, data.COUNTRY, data.DATASET_VERSION, data.MAX_WORD_COUNT
)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
def build_index(doc_ids: list[str], doc_texts: list[str]) -> None:
    """Build a Lucene index from the filtered corpus."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with Timer(f"Building Lucene index ({len(doc_ids)} docs)"):
        indexer = LuceneIndexer(str(INDEX_DIR), language="es")
        for doc_id, doc_text in zip(doc_ids, doc_texts):
            indexer.add_doc_dict({"id": doc_id, "contents": doc_text})
        indexer.close()

    log.info("Lucene index saved to %s", INDEX_DIR)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def search(query_ids: list[str], query_texts: list[str]) -> dict[str, dict[str, float]]:
    """Run BM25 search over all queries and return a retrieval run dict."""
    searcher = LuceneSearcher(str(INDEX_DIR))
    searcher.set_language("es")

    with Timer(f"BM25 search ({len(query_ids)} queries x top-{TOP_K})"):
        hits = searcher.batch_search(
            queries=query_texts,
            qids=query_ids,
            k=TOP_K,
            threads=NUM_WORKERS,
        )

    retrieval_run: dict[str, dict[str, float]] = {}
    for qid, hit_list in hits.items():
        retrieval_run[qid] = {hit.docid: hit.score for hit in hit_list}

    return retrieval_run


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    need_retrieval_run = not RUN_PATH.exists()
    need_index = not INDEX_DIR.exists()

    if need_retrieval_run:
        qrels_cached = (DATASET_CACHE_DIR / "pruned_qrels.json").exists()

        if need_index:
            # Must load corpus to build index; also needed for qrels if not cached
            doc_ids, doc_texts = data.load_corpus(NUM_WORKERS)
            qrels, query_id_to_text = data.get_pruned_qrels_and_queries(
                data.COUNTRY, data.DATASET_VERSION,
                kept_doc_ids=None if qrels_cached else doc_ids,
                dataset_cache_dir=DATASET_CACHE_DIR,
                num_workers=NUM_WORKERS,
            )
            build_index(doc_ids, doc_texts)
            del doc_ids, doc_texts
        else:
            # Index exists, just need qrels (must already be cached by a prior run)
            qrels, query_id_to_text = data.get_pruned_qrels_and_queries(
                data.COUNTRY, data.DATASET_VERSION,
                kept_doc_ids=None,
                dataset_cache_dir=DATASET_CACHE_DIR,
                num_workers=NUM_WORKERS,
            )

        query_ids = list(query_id_to_text.keys())
        query_texts = list(query_id_to_text.values())
        retrieval_run = search(query_ids, query_texts)

        RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
        ranx_run = retrieval.save_run(retrieval_run, MODEL_SLUG, RUN_PATH)
        log.info("Saved retrieval run to %s", RUN_PATH)
    else:
        log.info("Loading cached retrieval run from %s", RUN_PATH)
        ranx_run = retrieval.load_run(RUN_PATH)
        qrels, _ = data.get_pruned_qrels_and_queries(
            data.COUNTRY, data.DATASET_VERSION,
            kept_doc_ids=None,
            dataset_cache_dir=DATASET_CACHE_DIR,
            num_workers=NUM_WORKERS,
        )

    retrieval.run_evaluation(qrels, ranx_run, MODEL_SLUG)
