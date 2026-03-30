"""
Chunked document encoding and query encoding with cache-aware resumption.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np

from .cache import is_complete, load_cached, merge_chunks, save_chunk, save_ids
from .observability import Timer

log = logging.getLogger(__name__)


def encode_documents_chunked(
    doc_ids: list[str],
    doc_texts: list[str],
    embedding_dir: Path,
    model,
    doc_chunk_size: int,
):
    """Encode documents in chunks, resuming from existing chunks on disk if interrupted.

    Each chunk is saved to disk immediately after encoding. If the process is interrupted
    and re-run, already-encoded chunks are skipped, so only the remaining chunks need
    to be computed.

    The model object must expose:
      - model.encode(texts, batch_size, is_query) -> np.ndarray
      - model.doc_batch_size: int

    Returns:
        doc_ids:         The full list of document IDs (unchanged).
        doc_embeddings:  A float32 numpy array of shape (num_docs, embedding_dim).
    """
    if is_complete(embedding_dir):
        log.info("Document embeddings already complete — loading from cache.")
        return load_cached(embedding_dir)

    total_documents = len(doc_ids)
    chunk_size = doc_chunk_size
    total_chunks = (total_documents + chunk_size - 1) // chunk_size
    num_existing_chunks = len(list(embedding_dir.glob("chunk_*.npy"))) if embedding_dir.exists() else 0

    if num_existing_chunks > 0:
        log.info("Resuming from chunk %d / %d", num_existing_chunks, total_chunks)

    for chunk_index in range(num_existing_chunks, total_chunks):
        chunk_start = chunk_index * chunk_size
        chunk_end = min((chunk_index + 1) * chunk_size, total_documents)
        log.info("── Chunk %d/%d [%d–%d] ──", chunk_index + 1, total_chunks, chunk_start, chunk_end - 1)

        chunk_texts = list(doc_texts[chunk_start:chunk_end])
        chunk_embeddings = model.encode(texts=chunk_texts, batch_size=model.doc_batch_size, is_query=False)
        del chunk_texts

        save_chunk(chunk_embeddings, embedding_dir / f"chunk_{chunk_index:04d}.npy")
        del chunk_embeddings
        gc.collect()

    save_ids(doc_ids, embedding_dir / "ids.json")
    return doc_ids, merge_chunks(embedding_dir)


def encode_queries(query_id_to_text: dict[str, str], embedding_dir: Path, model):
    """Encode all queries, using the cache if a completed encoding already exists.

    The model object must expose:
      - model.encode(texts, batch_size, is_query) -> np.ndarray
      - model.query_batch_size: int

    Returns:
        query_ids:        Ordered list of query ID strings.
        query_embeddings: A float32 numpy array of shape (num_queries, embedding_dim).
    """
    if is_complete(embedding_dir):
        log.info("Query embeddings already complete — loading from cache.")
        return load_cached(embedding_dir)

    query_ids = list(query_id_to_text.keys())
    query_texts = list(query_id_to_text.values())

    query_embeddings = model.encode(
        texts=query_texts,
        batch_size=model.query_batch_size,
        is_query=True,
    )

    embedding_dir.mkdir(parents=True, exist_ok=True)
    save_ids(query_ids, embedding_dir / "ids.json")
    np.save(embedding_dir / "merged.npy", query_embeddings)
    return query_ids, query_embeddings
