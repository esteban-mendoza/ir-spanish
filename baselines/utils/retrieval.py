"""
FAISS-based retrieval and ranx evaluation.
"""

from __future__ import annotations

import logging

import faiss
import numpy as np
from ranx import Qrels, Run, evaluate

log = logging.getLogger(__name__)

TOP_K = 100


def _build_search_index(doc_embeddings: np.ndarray, num_workers: int) -> faiss.Index:
    """Build an IndexFlatIP, then move it to all available GPUs."""
    embedding_dimension = doc_embeddings.shape[1]
    faiss.omp_set_num_threads(num_workers)

    cpu_index = faiss.IndexFlatIP(embedding_dimension)
    cpu_index.add(np.ascontiguousarray(doc_embeddings, dtype=np.float32))

    num_gpus = faiss.get_num_gpus()
    if num_gpus > 0:
        log.info("Transferring FAISS index to %d GPU(s) ...", num_gpus)
        try:
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True      # split vectors across GPUs instead of replicating
            co.useFloat16 = True  # halve VRAM: ~13.7 GB/GPU instead of ~27.4 GB/GPU
            gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co=co)
            log.info("FAISS index placed on GPU(s) (sharded, float16).")
            return gpu_index
        except Exception as gpu_error:
            log.warning(
                "GPU transfer failed (%s); falling back to CPU search.", gpu_error
            )
    else:
        log.info("No GPUs detected; using CPU search.")
    return cpu_index


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

    index = _build_search_index(doc_embeddings, num_workers)

    queries_as_float32 = np.ascontiguousarray(query_embeddings, dtype=np.float32)

    # Single search call — entire query matrix (0.64 GB) fits in VRAM
    log.info("Running FAISS search (%d queries x top-%d) ...", num_queries, top_k)
    top_k_scores, top_k_doc_indices = index.search(queries_as_float32, top_k)

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
