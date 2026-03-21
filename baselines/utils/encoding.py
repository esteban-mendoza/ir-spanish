"""
Chunked document encoding and query encoding with cache-aware resumption.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np

from .cache import is_complete, load_cached, merge_chunks, save_chunk, save_ids

log = logging.getLogger(__name__)


def encode_documents_chunked(doc_ids: list[str], doc_texts: list[str], emb_dir: Path, model, doc_chunk_size: int):
    """Encode documents in chunks, resuming from existing chunks on disk.

    The model object must expose:
      - model.encode(texts, batch_size, is_query) -> np.ndarray
      - model.doc_batch_size: int
    """
    if is_complete(emb_dir):
        return load_cached(emb_dir)

    n, chunk_sz = len(doc_ids), doc_chunk_size
    total_chunks = (n + chunk_sz - 1) // chunk_sz
    existing = len(list(emb_dir.glob("chunk_*.npy"))) if emb_dir.exists() else 0

    for ci in range(existing, total_chunks):
        lo, hi = ci * chunk_sz, min((ci + 1) * chunk_sz, n)
        log.info("── Chunk %d/%d [%d–%d] ──", ci + 1, total_chunks, lo, hi - 1)

        chunk_emb = model.encode(texts=doc_texts[lo:hi], batch_size=model.doc_batch_size, is_query=False)
        save_chunk(chunk_emb, emb_dir / f"chunk_{ci:04d}.npy")
        del chunk_emb
        gc.collect()

    save_ids(doc_ids, emb_dir / "ids.json")
    return doc_ids, merge_chunks(emb_dir)


def encode_queries(q_map: dict[str, str], emb_dir: Path, model):
    """Encode all queries, using cache if available.

    The model object must expose:
      - model.encode(texts, batch_size, is_query) -> np.ndarray
      - model.query_batch_size: int
    """
    if is_complete(emb_dir):
        return load_cached(emb_dir)

    qids, qtexts = list(q_map.keys()), list(q_map.values())
    emb = model.encode(texts=qtexts, batch_size=model.query_batch_size, is_query=True)

    emb_dir.mkdir(parents=True, exist_ok=True)
    save_ids(qids, emb_dir / "ids.json")
    np.save(emb_dir / "merged.npy", emb)
    return qids, emb
