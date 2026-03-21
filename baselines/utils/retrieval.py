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


def retrieve(
    query_ids: list[str],
    query_emb: np.ndarray,
    doc_ids: list[str],
    doc_emb: np.ndarray,
    top_k: int,
    num_workers: int,
) -> dict[str, dict[str, float]]:
    nq, dim = len(query_ids), doc_emb.shape[1]
    faiss.omp_set_num_threads(num_workers)

    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(doc_emb, dtype=np.float32))

    q32 = np.ascontiguousarray(query_emb, dtype=np.float32)
    all_scores = np.empty((nq, top_k), dtype=np.float32)
    all_idx = np.empty((nq, top_k), dtype=np.int64)

    for lo in tqdm(range(0, nq, 1024), desc="FAISS", unit="batch"):
        hi = min(lo + 1024, nq)
        all_scores[lo:hi], all_idx[lo:hi] = index.search(q32[lo:hi], top_k)

    doc_arr = np.array(doc_ids, dtype=object)
    run = {
        query_ids[i]: {str(doc_arr[j]): float(sc) for j, sc in zip(all_idx[i][mask], all_scores[i][mask])}
        for i in range(nq)
        if (mask := all_idx[i] != -1) is not None
    }
    return run


def run_evaluation(qrels: Qrels, run_dict: dict[str, dict[str, float]], model_name: str) -> dict:
    run = Run(run_dict)
    results = evaluate(qrels, run, metrics=["ndcg@10", "recall@100"])

    label = model_name.split("/")[-1]
    log.info("")
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  RESULTS — %-38s║", label + " ")
    log.info("╠══════════════════════════════════════════════════╣")
    for m, v in results.items():
        log.info("║  %-15s  %.4f                         ║", m, v)
    log.info("╚══════════════════════════════════════════════════╝")
    return results
