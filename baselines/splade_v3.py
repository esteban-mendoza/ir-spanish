#!/usr/bin/env python3
"""
SPLADE-v3 sparse retrieval baseline for MessIRve (Spanish IR).
Metrics: nDCG@10, Recall@100 via ranx.

Uses naver/splade-v3 to produce learned sparse term-weight vectors,
then retrieves via scipy sparse dot product.
"""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sentence_transformers import SparseEncoder

# utils.__init__ configures logging and NUMA env-vars
from utils import cache, data, retrieval
from utils.observability import Timer

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "naver/splade-v3"
DOC_BATCH_SIZE = 256
QUERY_BATCH_SIZE = 512
MAX_QUERY_LENGTH = 64
MAX_DOC_LENGTH = 256
TOP_K = 100

GPU_DEVICES = ["cuda:0", "cuda:1"]
SEED = 42
NUM_WORKERS = os.cpu_count() or 32
DOC_CHUNK_SIZE = 50_000
QUERY_SEARCH_BATCH = 5000

CACHE_DIR = Path.home() / ".cache" / "messirve_embeddings"

MODEL_CACHE_BASE = cache.cache_base(
    CACHE_DIR, MODEL_NAME, data.COUNTRY, data.DATASET_VERSION,
    MAX_QUERY_LENGTH, MAX_DOC_LENGTH, data.MAX_WORD_COUNT,
)
DOC_EMB_DIR = cache.emb_dir(MODEL_CACHE_BASE, "doc")
QUERY_EMB_DIR = cache.emb_dir(MODEL_CACHE_BASE, "query")
RUN_PATH = cache.run_cache_path(MODEL_CACHE_BASE)
DATASET_CACHE_DIR = cache.dataset_cache_base(
    CACHE_DIR, data.COUNTRY, data.DATASET_VERSION, data.MAX_WORD_COUNT,
)


def _to_scipy_csr(tensor: torch.Tensor) -> sp.csr_matrix:
    """Convert a PyTorch sparse COO tensor to scipy CSR."""
    t = tensor.coalesce()
    indices = t.indices().numpy()
    values = t.values().numpy()
    return sp.csr_matrix((values, (indices[0], indices[1])), shape=t.shape)


# ---------------------------------------------------------------------------
# Sparse cache helpers
# ---------------------------------------------------------------------------
def is_sparse_complete(embedding_dir: Path) -> bool:
    return (embedding_dir / "ids.json").exists() and (embedding_dir / "merged.npz").exists()


def load_sparse_cached(embedding_dir: Path):
    ids = cache.load_ids(embedding_dir / "ids.json")
    matrix = sp.load_npz(embedding_dir / "merged.npz")
    return ids, matrix


def save_sparse_chunk(matrix: sp.csr_matrix, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(path, matrix)
    log.info(
        "  Saved %s (%d×%d, %d nnz, %.1f MB)",
        path.name, *matrix.shape, matrix.nnz,
        (matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes) / (1024**2),
    )


def merge_sparse_chunks(embedding_dir: Path) -> sp.csr_matrix:
    chunk_paths = sorted(embedding_dir.glob("chunk_*.npz"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk files found in {embedding_dir}")

    log.info("Merging %d sparse chunks …", len(chunk_paths))
    chunks = [sp.load_npz(p) for p in chunk_paths]
    merged = sp.vstack(chunks, format="csr")

    sp.save_npz(embedding_dir / "merged.npz", merged)
    for p in chunk_paths:
        p.unlink()

    return merged


# ---------------------------------------------------------------------------
# Sparse encoding
# ---------------------------------------------------------------------------
def encode_documents_chunked(
    doc_ids: list[str],
    doc_texts: list[str],
    embedding_dir: Path,
    model: SparseEncoder,
    pool,
) -> tuple[list[str], sp.csr_matrix]:
    """Encode documents in chunks with crash-safe resumption."""
    if is_sparse_complete(embedding_dir):
        log.info("Document embeddings already complete — loading from cache.")
        return load_sparse_cached(embedding_dir)

    total = len(doc_ids)
    total_chunks = (total + DOC_CHUNK_SIZE - 1) // DOC_CHUNK_SIZE
    num_existing = len(list(embedding_dir.glob("chunk_*.npz"))) if embedding_dir.exists() else 0

    if num_existing > 0:
        log.info("Resuming from chunk %d / %d", num_existing, total_chunks)

    for ci in range(num_existing, total_chunks):
        start = ci * DOC_CHUNK_SIZE
        end = min(start + DOC_CHUNK_SIZE, total)
        log.info("── Chunk %d/%d [%d–%d] ──", ci + 1, total_chunks, start, end - 1)

        chunk_texts = list(doc_texts[start:end])
        chunk_sparse = model.encode_multi_process(chunk_texts, pool=pool, batch_size=DOC_BATCH_SIZE)
        del chunk_texts

        if isinstance(chunk_sparse, torch.Tensor):
            chunk_sparse = _to_scipy_csr(chunk_sparse) if chunk_sparse.is_sparse else sp.csr_matrix(chunk_sparse.numpy())
        elif not sp.issparse(chunk_sparse):
            chunk_sparse = sp.csr_matrix(chunk_sparse)

        save_sparse_chunk(chunk_sparse, embedding_dir / f"chunk_{ci:04d}.npz")
        del chunk_sparse
        gc.collect()

    cache.save_ids(doc_ids, embedding_dir / "ids.json")
    return doc_ids, merge_sparse_chunks(embedding_dir)


def encode_queries(
    query_id_to_text: dict[str, str],
    embedding_dir: Path,
    model: SparseEncoder,
    pool,
) -> tuple[list[str], sp.csr_matrix]:
    """Encode all queries in one shot."""
    if is_sparse_complete(embedding_dir):
        log.info("Query embeddings already complete — loading from cache.")
        return load_sparse_cached(embedding_dir)

    query_ids = list(query_id_to_text.keys())
    query_texts = list(query_id_to_text.values())

    query_sparse = model.encode_multi_process(query_texts, pool=pool, batch_size=QUERY_BATCH_SIZE)

    if isinstance(query_sparse, torch.Tensor):
        query_sparse = _to_scipy_csr(query_sparse) if query_sparse.is_sparse else sp.csr_matrix(query_sparse.numpy())
    elif not sp.issparse(query_sparse):
        query_sparse = sp.csr_matrix(query_sparse)

    embedding_dir.mkdir(parents=True, exist_ok=True)
    cache.save_ids(query_ids, embedding_dir / "ids.json")
    sp.save_npz(embedding_dir / "merged.npz", query_sparse)
    return query_ids, query_sparse


# ---------------------------------------------------------------------------
# Sparse retrieval
# ---------------------------------------------------------------------------
def sparse_retrieve(
    query_ids: list[str],
    query_sparse: sp.csr_matrix,
    doc_ids: list[str],
    doc_sparse: sp.csr_matrix,
    top_k: int,
) -> dict[str, dict[str, float]]:
    """Retrieve top-k documents per query via sparse dot product."""
    n_queries = len(query_ids)
    doc_ids_array = np.array(doc_ids, dtype=object)
    doc_sparse_t = doc_sparse.T.tocsc()

    retrieval_run: dict[str, dict[str, float]] = {}

    with Timer(f"Sparse retrieval ({n_queries} queries × top-{top_k})"):
        for batch_start in range(0, n_queries, QUERY_SEARCH_BATCH):
            batch_end = min(batch_start + QUERY_SEARCH_BATCH, n_queries)
            batch_queries = query_sparse[batch_start:batch_end]

            scores = batch_queries.dot(doc_sparse_t)  # (batch, n_docs) sparse

            for j in range(batch_queries.shape[0]):
                row = scores.getrow(j)
                row_data = row.data
                row_indices = row.indices

                if len(row_data) == 0:
                    retrieval_run[query_ids[batch_start + j]] = {}
                    continue

                if len(row_data) > top_k:
                    top_k_pos = np.argpartition(row_data, -top_k)[-top_k:]
                else:
                    top_k_pos = np.arange(len(row_data))

                qid = query_ids[batch_start + j]
                retrieval_run[qid] = {
                    str(doc_ids_array[row_indices[k]]): float(row_data[k])
                    for k in top_k_pos
                }

            if batch_end < n_queries:
                log.info("  Searched %d / %d queries", batch_end, n_queries)

    return retrieval_run


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(SEED)

    need_retrieval_run = not RUN_PATH.exists()
    need_doc_emb = not is_sparse_complete(DOC_EMB_DIR)
    need_query_emb = not is_sparse_complete(QUERY_EMB_DIR)
    need_pruned_qrels = not (
        (DATASET_CACHE_DIR / "pruned_qrels.json").exists()
        and (DATASET_CACHE_DIR / "pruned_q_map.json").exists()
    )

    if need_retrieval_run:
        # --- Load data ---
        if need_doc_emb:
            doc_ids, doc_texts = data.load_corpus(NUM_WORKERS)
        else:
            doc_ids = cache.load_ids(DOC_EMB_DIR / "ids.json")
            doc_texts = None

        qrels, query_id_to_text = data.get_pruned_qrels_and_queries(
            data.COUNTRY, data.DATASET_VERSION,
            kept_doc_ids=doc_ids if need_pruned_qrels else None,
            dataset_cache_dir=DATASET_CACHE_DIR,
            num_workers=NUM_WORKERS,
        )

        # --- Encode ---
        doc_sparse = query_sparse = query_ids = None

        if need_doc_emb or need_query_emb:
            with Timer("Loading SparseEncoder"):
                model = SparseEncoder(MODEL_NAME, device="cpu")
                model.max_seq_length = MAX_DOC_LENGTH

            with Timer(f"Starting pool on {GPU_DEVICES}"):
                pool = model.start_multi_process_pool(target_devices=GPU_DEVICES)

            if need_doc_emb:
                doc_ids, doc_sparse = encode_documents_chunked(
                    doc_ids, doc_texts, DOC_EMB_DIR, model, pool,
                )
                del doc_texts
                gc.collect()

            if need_query_emb:
                if model.max_seq_length != MAX_QUERY_LENGTH:
                    log.info("Switching max_seq_length %d → %d — restarting pool", model.max_seq_length, MAX_QUERY_LENGTH)
                    model.stop_multi_process_pool(pool)
                    model.max_seq_length = MAX_QUERY_LENGTH
                    pool = model.start_multi_process_pool(target_devices=GPU_DEVICES)

                query_ids, query_sparse = encode_queries(
                    query_id_to_text, QUERY_EMB_DIR, model, pool,
                )

            model.stop_multi_process_pool(pool)
            del model
            gc.collect()
            torch.cuda.empty_cache()

        if doc_sparse is None:
            doc_ids, doc_sparse = load_sparse_cached(DOC_EMB_DIR)
        if query_sparse is None:
            query_ids, query_sparse = load_sparse_cached(QUERY_EMB_DIR)

        # --- Retrieve ---
        retrieval_run = sparse_retrieve(query_ids, query_sparse, doc_ids, doc_sparse, TOP_K)

        RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
        ranx_run = retrieval.save_run(retrieval_run, MODEL_NAME, RUN_PATH)
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

    retrieval.run_evaluation(qrels, ranx_run, MODEL_NAME)
