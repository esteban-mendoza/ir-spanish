"""
FAISS-based retrieval and ranx evaluation.
"""

from __future__ import annotations

import logging

import faiss
import numpy as np
from ranx import Qrels, Run, evaluate
from tqdm import tqdm

log = logging.getLogger(__name__)

TOP_K = 100

# Number of query vectors to search in a single FAISS batch.
# Batching avoids materializing a huge score matrix all at once.
_FAISS_SEARCH_BATCH_SIZE = 1024


def retrieve(
    query_ids: list[str],
    query_embeddings: np.ndarray,
    doc_ids: list[str],
    doc_embeddings: np.ndarray,
    top_k: int,
    num_workers: int,
) -> dict[str, dict[str, float]]:
    """Search for the top-k most similar documents for each query using FAISS inner-product search.

    Embeddings must already be L2-normalized (so inner product equals cosine similarity).

    Returns:
        A run dict mapping each query ID to a {doc_id: score} dict of its top-k results.
    """
    num_queries = len(query_ids)
    embedding_dimension = doc_embeddings.shape[1]

    faiss.omp_set_num_threads(num_workers)

    # Build a flat (exact) inner-product index and add all document vectors
    index = faiss.IndexFlatIP(embedding_dimension)
    index.add(np.ascontiguousarray(doc_embeddings, dtype=np.float32))

    queries_as_float32 = np.ascontiguousarray(query_embeddings, dtype=np.float32)

    # Pre-allocate output arrays for scores and document indices
    top_k_scores = np.empty((num_queries, top_k), dtype=np.float32)
    top_k_doc_indices = np.empty((num_queries, top_k), dtype=np.int64)

    # Search in batches to keep peak memory usage bounded
    for batch_start in tqdm(range(0, num_queries, _FAISS_SEARCH_BATCH_SIZE), desc="FAISS", unit="batch"):
        batch_end = min(batch_start + _FAISS_SEARCH_BATCH_SIZE, num_queries)
        top_k_scores[batch_start:batch_end], top_k_doc_indices[batch_start:batch_end] = (
            index.search(queries_as_float32[batch_start:batch_end], top_k)
        )

    # Convert numpy index positions to document ID strings.
    # FAISS returns -1 for padded slots when fewer than top_k results exist.
    doc_ids_array = np.array(doc_ids, dtype=object)
    retrieval_run: dict[str, dict[str, float]] = {}

    for query_idx in range(num_queries):
        valid_indices_mask = top_k_doc_indices[query_idx] != -1
        retrieval_run[query_ids[query_idx]] = {
            str(doc_ids_array[doc_idx]): float(score)
            for doc_idx, score in zip(
                top_k_doc_indices[query_idx][valid_indices_mask],
                top_k_scores[query_idx][valid_indices_mask],
            )
        }

    return retrieval_run


def run_evaluation(
    qrels: Qrels,
    retrieval_run: dict[str, dict[str, float]],
    model_name: str,
) -> dict:
    """Evaluate the retrieval run against ground-truth qrels and log the results.

    Computes nDCG@10 and Recall@100 using ranx.

    Returns:
        A dict mapping metric names to their computed float values.
    """
    run = Run(retrieval_run)
    results = evaluate(qrels, run, metrics=["ndcg@10", "recall@100"])

    model_short_name = model_name.split("/")[-1]
    log.info("")
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  RESULTS — %-38s║", model_short_name + " ")
    log.info("╠══════════════════════════════════════════════════╣")
    for metric_name, metric_value in results.items():
        log.info("║  %-15s  %.4f                         ║", metric_name, metric_value)
    log.info("╚══════════════════════════════════════════════════╝")
    return results
