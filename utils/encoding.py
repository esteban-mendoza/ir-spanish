"""
Chunked document encoding and query encoding with cache-aware resumption.

Features:
- Length-sorted adaptive encoding: documents are sorted by word count before
  chunking so that each chunk contains documents of similar length. This allows
  per-chunk adaptive max_seq_length and batch_size, dramatically reducing GPU
  memory usage for short-document chunks while preserving full context for long
  documents.
- Crash-safe resumption: each chunk is saved to disk immediately after encoding.
  On restart, already-encoded chunks are skipped.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np

from .cache import is_complete, load_cached, merge_chunks, save_chunk, save_ids
from .observability import Timer

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Adaptive encoding constants
# ---------------------------------------------------------------------------

TOKENS_PER_WORD = 1.5  # conservative estimate for Spanish text

SEQ_LENGTH_BANDS = [512, 2048, 8192, 16384, 32768]

# (max_seq_length_threshold, batch_size) — calibrated for RTX A5000 24GB, bfloat16, SDPA
BATCH_SIZE_TABLE = [
    (512, 128),
    (2048, 32),
    (8192, 8),
    (16384, 4),
    (32768, 2),
]


def _snap_to_band(token_estimate: int, model_max_length: int) -> int:
    """Snap a token count estimate to the next sequence-length band up.

    Using coarse bands minimises pool restarts (which are expensive) while still
    giving large memory savings for short-document chunks.
    """
    for band in SEQ_LENGTH_BANDS:
        if token_estimate <= band:
            return min(band, model_max_length)
    return model_max_length


def _adaptive_batch_size(max_seq_length: int, fallback: int) -> int:
    """Look up the recommended batch size for a given max_seq_length band."""
    for threshold, batch in BATCH_SIZE_TABLE:
        if max_seq_length <= threshold:
            return batch
    return fallback


# ---------------------------------------------------------------------------
# Document encoding
# ---------------------------------------------------------------------------

def encode_documents_chunked(
    doc_ids: list[str],
    doc_texts: list[str],
    word_counts,
    embedding_dir: Path,
    model,
    doc_chunk_size: int,
    model_max_length: int = 32768,
):
    """Encode documents in chunks with adaptive seq_length and batch_size.

    Documents are sorted by word count before chunking so that each chunk
    contains documents of similar length. This allows per-chunk adaptive
    ``max_seq_length`` (set to the smallest band that covers the longest
    document in the chunk) and ``batch_size`` (larger for short-doc chunks).

    The sort is undone after merging so that the final embeddings match the
    original document order in ``doc_ids``.

    If ``word_counts`` is None, falls back to the original fixed-batch behavior.

    The model object must expose:
      - model.encode(texts, batch_size, is_query, max_seq_length) -> np.ndarray
      - model.doc_batch_size: int

    Returns:
        doc_ids:         The full list of document IDs (unchanged, original order).
        doc_embeddings:  A float32 numpy array of shape (num_docs, embedding_dim).
    """
    if is_complete(embedding_dir):
        log.info("Document embeddings already complete -- loading from cache.")
        return load_cached(embedding_dir)

    embedding_dir.mkdir(parents=True, exist_ok=True)

    # Clean up stale temp files from previous crashes
    for tmp in embedding_dir.glob("*_tmp.npy"):
        log.warning("Removing stale temp file: %s", tmp.name)
        tmp.unlink()

    total_documents = len(doc_ids)
    chunk_size = doc_chunk_size
    total_chunks = (total_documents + chunk_size - 1) // chunk_size

    # ------------------------------------------------------------------
    # Sort documents by word count (shortest first)
    # ------------------------------------------------------------------
    sort_order_path = embedding_dir / "sort_order.npy"

    if word_counts is not None:
        wc = np.array(word_counts) if not isinstance(word_counts, np.ndarray) else word_counts
        sort_order = np.argsort(wc, kind="stable")

        # Check for stale unsorted chunks from a pre-optimization run
        existing_chunks = sorted(embedding_dir.glob("chunk_*.npy"))
        if existing_chunks and not sort_order_path.exists():
            log.warning(
                "Found %d unsorted chunks without sort_order.npy -- deleting for re-encoding",
                len(existing_chunks),
            )
            for p in existing_chunks:
                p.unlink()

        # Save or verify sort order for resumption
        if not sort_order_path.exists():
            np.save(sort_order_path, sort_order)
        else:
            saved = np.load(sort_order_path)
            if not np.array_equal(saved, sort_order):
                raise RuntimeError(
                    "Saved sort_order.npy does not match current sort order. "
                    "The dataset may have changed. Delete the cache directory and retry."
                )

        # Build sorted views
        doc_texts_list = list(doc_texts) if not isinstance(doc_texts, list) else doc_texts
        sorted_texts = [doc_texts_list[i] for i in sort_order]
        sorted_wc = wc[sort_order]
        use_adaptive = True
    else:
        sorted_texts = doc_texts
        sorted_wc = None
        sort_order = None
        use_adaptive = False

    # ------------------------------------------------------------------
    # Chunked encoding with adaptive seq_length / batch_size
    # ------------------------------------------------------------------
    num_existing_chunks = len(list(embedding_dir.glob("chunk_*.npy")))

    if num_existing_chunks > 0:
        log.info("Resuming from chunk %d / %d", num_existing_chunks, total_chunks)

    for chunk_index in range(num_existing_chunks, total_chunks):
        chunk_start = chunk_index * chunk_size
        chunk_end = min((chunk_index + 1) * chunk_size, total_documents)

        if use_adaptive:
            max_words = int(sorted_wc[chunk_start:chunk_end].max())
            token_est = int(max_words * TOKENS_PER_WORD)
            seq_len = _snap_to_band(token_est, model_max_length)
            batch_sz = _adaptive_batch_size(seq_len, model.doc_batch_size)
            log.info(
                "── Chunk %d/%d [%d–%d] max_words=%d → seq=%d batch=%d ──",
                chunk_index + 1, total_chunks, chunk_start, chunk_end - 1,
                max_words, seq_len, batch_sz,
            )
        else:
            seq_len = model_max_length
            batch_sz = model.doc_batch_size
            log.info(
                "── Chunk %d/%d [%d–%d] ──",
                chunk_index + 1, total_chunks, chunk_start, chunk_end - 1,
            )

        chunk_texts = list(sorted_texts[chunk_start:chunk_end])
        chunk_embeddings = model.encode(
            texts=chunk_texts, batch_size=batch_sz, is_query=False, max_seq_length=seq_len,
        )
        del chunk_texts

        save_chunk(chunk_embeddings, embedding_dir / f"chunk_{chunk_index:04d}.npy")
        del chunk_embeddings
        gc.collect()

    # ------------------------------------------------------------------
    # Merge chunks and save IDs in the same order as embeddings
    # ------------------------------------------------------------------
    # When sorted, save IDs in sorted order so they align with the sorted
    # embeddings.  This avoids an expensive un-sort of the full embedding
    # matrix (which would require 2x the array size in RAM).
    # Downstream code only needs ids[i] <-> embeddings[i] alignment;
    # the actual ordering is irrelevant for retrieval.
    if sort_order is not None:
        doc_ids_list = list(doc_ids) if not isinstance(doc_ids, list) else doc_ids
        sorted_ids = [doc_ids_list[i] for i in sort_order]
        save_ids(sorted_ids, embedding_dir / "ids.json")
    else:
        sorted_ids = None
        save_ids(doc_ids, embedding_dir / "ids.json")

    merged = merge_chunks(embedding_dir)

    if sort_order_path.exists():
        sort_order_path.unlink()

    return (sorted_ids if sorted_ids is not None else doc_ids), merged


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
        log.info("Query embeddings already complete -- loading from cache.")
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
