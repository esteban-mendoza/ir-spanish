#!/usr/bin/env python3
"""
Prototipo: SPLADE-v3 adaptado al español sobre MessIRve.

Usa el mismo checkpoint, los mismos lotes y la misma evaluación que
``baselines/splade_v3.py``, pero recorre la matriz de documentos por bloques
alineados a la caché de 50 000 documentos.

Motivo del cambio: este checkpoint activa ~400 términos por documento (frente a
~233 de ``naver/splade-v3``), de modo que los 14 M de documentos del corpus no
caben a la vez en las dos GPU (un CSR en GPU usa 12 bytes por nnz: 67 GB en
total). La puntuación es el mismo producto interno disperso y el top-100 se
acumula bloque a bloque, así que el run es idéntico al de una sola pasada; solo
cambia el orden en que se recorren los documentos.

Uso (desde la raíz del repositorio)::

    python -m baselines.splade_v3_spanish
"""

from __future__ import annotations

import gc
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sentence_transformers import SparseEncoder

# utils.__init__ configura logging y las variables de NUMA
from utils import cache, data, retrieval
from utils.observability import Timer

# Reutiliza la configuración y los helpers del baseline de SPLADE-v3
from baselines import splade_v3 as base

log = logging.getLogger(__name__)

MODEL_NAME = "alrolo3/splade-v3-spanish-vllm"
MODEL_ALIAS = "splade-v3-es"

MODEL_MAX_LENGTH = base.MODEL_MAX_LENGTH
QUERY_BATCH_SIZE = base.QUERY_BATCH_SIZE
DOC_BATCH_SIZE = base.DOC_BATCH_SIZE
DOC_CHUNK_SIZE = base.DOC_CHUNK_SIZE
TOP_K = base.TOP_K
GPU_DEVICES = base.GPU_DEVICES
SEED = base.SEED
NUM_WORKERS = base.NUM_WORKERS
QUERY_SEARCH_BATCH = base.QUERY_SEARCH_BATCH
MATMUL_SUB_BATCH = base.MATMUL_SUB_BATCH
CACHE_DIR = base.CACHE_DIR

# Presupuesto de VRAM por bloque de documentos: 12 bytes por nnz (valor float32
# e índice int64) y ~256 MB por producto parcial (sub-lote × documentos).
BLOCK_BYTES = 8 * 1024**3
BYTES_PER_NNZ = 12
SUB_BATCH_BYTES = 256 * 1024**2

MODEL_CACHE_BASE = cache.cache_base(
    CACHE_DIR, MODEL_NAME, data.COUNTRY, data.DATASET_VERSION,
    MODEL_MAX_LENGTH, MODEL_MAX_LENGTH, data.MAX_WORD_COUNT,
)
DOC_EMB_DIR = cache.emb_dir(MODEL_CACHE_BASE, "doc")
QUERY_EMB_DIR = cache.emb_dir(MODEL_CACHE_BASE, "query")
RUN_PATH = cache.run_cache_path(MODEL_CACHE_BASE)
DATASET_CACHE_DIR = cache.dataset_cache_base(
    CACHE_DIR, data.COUNTRY, data.DATASET_VERSION, data.MAX_WORD_COUNT,
)


# ---------------------------------------------------------------------------
# Codificación de documentos (bloques en disco, sin fusión en RAM)
# ---------------------------------------------------------------------------
def doc_chunk_paths() -> list[Path]:
    return sorted(DOC_EMB_DIR.glob("chunk_*.npz"))


def expected_chunks(num_docs: int) -> int:
    return (num_docs + DOC_CHUNK_SIZE - 1) // DOC_CHUNK_SIZE


def encode_documents_chunked(
    doc_ids: list[str],
    doc_texts: list[str],
    model: SparseEncoder,
    pool,
) -> None:
    """Encode documents into one .npz chunk per 50 000 documents.

    The chunks are the retrieval unit, so no raster copy of the full matrix is
    ever built in RAM. ``ids.json`` is written before encoding so an interrupted
    run can resume without reloading the corpus to learn the document order.
    """
    total = len(doc_ids)
    total_chunks = expected_chunks(total)
    cache.save_ids(doc_ids, DOC_EMB_DIR / "ids.json")

    existing = len(doc_chunk_paths())
    if existing > total_chunks:
        raise RuntimeError(
            f"{existing} chunks on disk but only {total_chunks} expected for {total} documents"
        )
    if existing:
        log.info("Resuming document encoding from chunk %d / %d", existing, total_chunks)

    for ci in range(existing, total_chunks):
        start = ci * DOC_CHUNK_SIZE
        end = min(start + DOC_CHUNK_SIZE, total)
        log.info("── Chunk %d/%d [%d–%d] ──", ci + 1, total_chunks, start, end - 1)

        chunk_sparse = model.encode(doc_texts[start:end], pool=pool, batch_size=DOC_BATCH_SIZE)
        if isinstance(chunk_sparse, torch.Tensor):
            chunk_sparse = (
                base._to_scipy_csr(chunk_sparse) if chunk_sparse.is_sparse
                else sp.csr_matrix(chunk_sparse.numpy())
            )
        elif not sp.issparse(chunk_sparse):
            chunk_sparse = sp.csr_matrix(chunk_sparse)

        base.save_sparse_chunk(chunk_sparse, DOC_EMB_DIR / f"chunk_{ci:04d}.npz")
        del chunk_sparse
        gc.collect()


# ---------------------------------------------------------------------------
# Recuperación por bloques
# ---------------------------------------------------------------------------
def nnz_per_doc(paths: list[Path]) -> float:
    sample = sp.load_npz(paths[0])
    return sample.nnz / sample.shape[0]


def _search_block(
    paths: list[Path],
    first_chunk: int,
    device: str,
    query_sparse: sp.csr_matrix,
    n_queries: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score every query against one block of documents on one GPU."""
    dev = torch.device(device)
    block = sp.vstack([sp.load_npz(p) for p in paths], format="csr")
    n_block_docs = block.shape[0]
    row_offset = first_chunk * DOC_CHUNK_SIZE
    log.info(
        "  %s: %d docs [%d–%d), %d nnz, %.1f GiB en GPU",
        device, n_block_docs, row_offset, row_offset + n_block_docs - 1,
        block.nnz, (block.nnz * BYTES_PER_NNZ) / (1024**3),
    )

    shard = base._scipy_csr_to_torch(block, dev)
    del block, paths

    sub_batch = max(1, min(MATMUL_SUB_BATCH, int(SUB_BATCH_BYTES / (4 * n_block_docs))))
    block_k = min(top_k, n_block_docs)
    best_scores = torch.empty((n_queries, block_k), dtype=torch.float32, device=dev)
    best_indices = torch.empty((n_queries, block_k), dtype=torch.int64, device=dev)

    for qs in range(0, n_queries, QUERY_SEARCH_BATCH):
        qe = min(qs + QUERY_SEARCH_BATCH, n_queries)
        q_dense = base._scipy_csr_to_torch(query_sparse[qs:qe], dev).to_dense()
        for i in range(0, qe - qs, sub_batch):
            q_sub = q_dense[i:i + sub_batch]
            scores = torch.sparse.mm(shard, q_sub.t()).t()
            values, positions = torch.topk(scores, k=block_k, dim=1)
            row_end = qs + i + values.shape[0]
            best_scores[qs + i:row_end] = values
            best_indices[qs + i:row_end] = positions
            del scores, values, positions
        del q_dense
        if qe < n_queries:
            log.info("    %s: %d / %d queries", device, qe, n_queries)

    del shard
    torch.cuda.empty_cache()
    best_indices += row_offset
    return best_scores.cpu(), best_indices.cpu()


def sparse_retrieve_blocked(
    query_ids: list[str],
    query_sparse: sp.csr_matrix,
    doc_ids: list[str],
    top_k: int,
) -> dict[str, dict[str, float]]:
    """Retrieve top-k documents per query, one document block at a time."""
    paths = doc_chunk_paths()
    if not paths:
        raise FileNotFoundError(f"No document chunks found in {DOC_EMB_DIR}")

    n_queries = len(query_ids)
    density = nnz_per_doc(paths)
    chunks_per_block = max(
        1, int(BLOCK_BYTES / (BYTES_PER_NNZ * density * DOC_CHUNK_SIZE)),
    )
    log.info(
        "Matriz de documentos: %.1f nnz/doc, bloques de %d chunks (%d bloques)",
        density, chunks_per_block, (len(paths) + chunks_per_block - 1) // chunks_per_block,
    )

    doc_ids_array = np.array(doc_ids, dtype=object)
    running_scores: torch.Tensor | None = None
    running_indices: torch.Tensor | None = None

    with Timer(f"Recuperación por bloques en {len(GPU_DEVICES)} GPUs ({n_queries} consultas × top-{top_k})"):
        with ThreadPoolExecutor(max_workers=len(GPU_DEVICES)) as executor:
            for group_start in range(0, len(paths), chunks_per_block * len(GPU_DEVICES)):
                jobs = []
                for device in GPU_DEVICES:
                    first = group_start + len(jobs) * chunks_per_block
                    if first >= len(paths):
                        break
                    last = min(first + chunks_per_block, len(paths))
                    jobs.append(
                        (device, first, paths[first:last])
                    )
                futures = [
                    executor.submit(_search_block, block_paths, first, device,
                                    query_sparse, n_queries, top_k)
                    for device, first, block_paths in jobs
                ]
                for future in futures:
                    scores, indices = future.result()
                    if running_scores is None:
                        running_scores, running_indices = scores, indices
                        continue
                    merged_scores = torch.cat([running_scores, scores], dim=1)
                    merged_indices = torch.cat([running_indices, indices], dim=1)
                    keep = min(top_k, merged_scores.shape[1])
                    running_scores, positions = torch.topk(merged_scores, k=keep, dim=1)
                    running_indices = torch.gather(merged_indices, 1, positions)
                    del merged_scores, merged_indices, scores, indices

                if running_scores is not None and running_scores.shape[1] > top_k:
                    running_scores = running_scores[:, :top_k]
                    running_indices = running_indices[:, :top_k]
                log.info(
                    "  Bloques procesados: %d / %d",
                    min(group_start + len(jobs) * chunks_per_block, len(paths)),
                    len(paths),
                )

    scores_np = running_scores.numpy()
    indices_np = running_indices.numpy()
    retrieval_run: dict[str, dict[str, float]] = {}
    for j, qid in enumerate(query_ids):
        mask = scores_np[j] > 0
        retrieval_run[qid] = {
            str(doc_ids_array[idx]): float(score)
            for idx, score in zip(indices_np[j][mask], scores_np[j][mask])
        }
    return retrieval_run


# ---------------------------------------------------------------------------
# Flujo principal
# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(SEED)

    if RUN_PATH.exists():
        log.info("Loading cached retrieval run from %s", RUN_PATH)
        ranx_run = retrieval.load_run(RUN_PATH)
        qrels, _ = data.get_pruned_qrels_and_queries(
            data.COUNTRY, data.DATASET_VERSION,
            kept_doc_ids=None,
            dataset_cache_dir=DATASET_CACHE_DIR,
            num_workers=NUM_WORKERS,
        )
        retrieval.run_evaluation(qrels, ranx_run, MODEL_ALIAS)
        return

    # --- Documentos ---
    ids_path = DOC_EMB_DIR / "ids.json"
    doc_texts = None
    if ids_path.exists():
        doc_ids = cache.load_ids(ids_path)
    else:
        doc_ids, doc_texts, _ = data.load_corpus(NUM_WORKERS)

    need_doc_emb = len(doc_chunk_paths()) < expected_chunks(len(doc_ids))
    if need_doc_emb:
        log.info("Document chunks: %d / %d", len(doc_chunk_paths()), expected_chunks(len(doc_ids)))
        if doc_texts is None:
            doc_ids, doc_texts, _ = data.load_corpus(NUM_WORKERS)
    elif not ids_path.exists():
        cache.save_ids(doc_ids, ids_path)

    need_query_emb = not base.is_sparse_complete(QUERY_EMB_DIR)

    qrels_cached = (
        (DATASET_CACHE_DIR / "pruned_qrels.json").exists()
        and (DATASET_CACHE_DIR / "pruned_q_map.json").exists()
    )
    qrels, query_id_to_text = data.get_pruned_qrels_and_queries(
        data.COUNTRY, data.DATASET_VERSION,
        kept_doc_ids=None if qrels_cached else doc_ids,
        dataset_cache_dir=DATASET_CACHE_DIR,
        num_workers=NUM_WORKERS,
    )

    # --- Codificación ---
    if need_doc_emb or need_query_emb:
        with Timer("Loading SparseEncoder"):
            model = SparseEncoder(MODEL_NAME, device="cpu")
            model.max_seq_length = MODEL_MAX_LENGTH
            model.tokenizer.model_max_length = MODEL_MAX_LENGTH

        with Timer(f"Starting pool on {GPU_DEVICES}"):
            pool = model.start_multi_process_pool(target_devices=GPU_DEVICES)

        if need_doc_emb:
            encode_documents_chunked(doc_ids, doc_texts, model, pool)
        if need_query_emb:
            base.encode_queries(query_id_to_text, QUERY_EMB_DIR, model, pool)

        model.stop_multi_process_pool(pool)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    if doc_texts is not None:
        del doc_texts
        gc.collect()

    # --- Recuperación ---
    doc_ids = cache.load_ids(DOC_EMB_DIR / "ids.json")
    query_ids, query_sparse = base.load_sparse_cached(QUERY_EMB_DIR)
    retrieval_run = sparse_retrieve_blocked(query_ids, query_sparse, doc_ids, TOP_K)

    RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
    ranx_run = retrieval.save_run(retrieval_run, MODEL_ALIAS, RUN_PATH)
    log.info("Saved retrieval run to %s", RUN_PATH)
    retrieval.run_evaluation(qrels, ranx_run, MODEL_ALIAS)


if __name__ == "__main__":
    main()
