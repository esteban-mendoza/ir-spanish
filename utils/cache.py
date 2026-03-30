"""
Cache path helpers and embedding chunk I/O.

Directory layout produced by these helpers:
  <cache_dir>/
    shared_datasets/<country>_v<version>_filt<max_words>w/
      pruned_qrels.json
      pruned_q_map.json
    <model_slug>/<country>_v<version>_seq<max_seq>_filt<max_words>w/
      doc_embeddings/
        chunk_0000.npy
        ...
        merged.npy
        ids.json
      query_embeddings/
        merged.npy
        ids.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache path builders
# ---------------------------------------------------------------------------

def model_slug(model_name: str) -> str:
    """Convert a HuggingFace model name (e.g. 'org/model') into a filesystem-safe slug."""
    return model_name.replace("/", "__")


def _filter_suffix(max_words: int | None) -> str:
    """Return the cache directory suffix for a word-count filter setting."""
    return "nofilter" if max_words is None else f"filt{max_words}w"


def dataset_cache_base(cache_dir: Path, country: str, version: str, max_words: int | None) -> Path:
    """Return the model-agnostic cache directory for filtered qrels and query maps.

    This directory is shared across all models that use the same dataset configuration,
    so the expensive qrels-pruning step only runs once.
    """
    return cache_dir / "shared_datasets" / f"{country}_v{version}_{_filter_suffix(max_words)}"


def cache_base(
    cache_dir: Path,
    model_name: str,
    country: str,
    version: str,
    max_query_length: int,
    max_doc_length: int,
    max_word_count: int | None,
) -> Path:
    """Return the model-specific cache root directory.

    Each combination of (model, dataset, sequence lengths, word-count filter) gets
    its own directory so cached embeddings are never silently reused across configs.
    """
    return (
        cache_dir
        / model_slug(model_name)
        / f"{country}_v{version}_q{max_query_length}_d{max_doc_length}_{_filter_suffix(max_word_count)}"
    )


def emb_dir(base_cache_dir: Path, embedding_type: str) -> Path:
    """Return the embedding subdirectory for the given type ('doc' or 'query')."""
    return base_cache_dir / f"{embedding_type}_embeddings"


def run_cache_path(model_cache_base: Path) -> Path:
    """Return the path for the saved ranx Run file inside a model's cache directory."""
    return model_cache_base / "retrieval_run.lz4"


# ---------------------------------------------------------------------------
# Cache completeness & status
# ---------------------------------------------------------------------------

def is_complete(embedding_dir: Path) -> bool:
    """Return True if the embedding directory contains both ids.json and merged.npy."""
    return (embedding_dir / "ids.json").exists() and (embedding_dir / "merged.npy").exists()


def are_qrels_cached(dataset_cache_dir: Path) -> bool:
    """Return True if the pruned qrels and query map JSON files both exist."""
    return (
        (dataset_cache_dir / "pruned_qrels.json").exists()
        and (dataset_cache_dir / "pruned_q_map.json").exists()
    )


def log_cache_status(model_cache_base: Path, dataset_cache_dir: Path):
    """Log whether each expected cache artifact (qrels, doc embeddings, query embeddings) exists."""
    qrels_are_cached = are_qrels_cached(dataset_cache_dir)
    log.info("  %-6s: %s", "qrels", "✓ cached" if qrels_are_cached else "✗ not cached")

    for embedding_type in ("doc", "query"):
        embedding_directory = emb_dir(model_cache_base, embedding_type)
        embeddings_are_complete = is_complete(embedding_directory)

        size_description = ""
        if embeddings_are_complete:
            size_in_mb = (embedding_directory / "merged.npy").stat().st_size / (1024**2)
            size_description = f" ({size_in_mb:.1f} MB)"

        num_partial_chunks = (
            len(list(embedding_directory.glob("chunk_*.npy")))
            if embedding_directory.exists()
            else 0
        )

        log.info(
            "  %-6s: %s%s  [%d chunks on disk]",
            embedding_type,
            "✓ complete" if embeddings_are_complete else "✗ incomplete",
            size_description,
            num_partial_chunks,
        )


# ---------------------------------------------------------------------------
# Embedding I/O
# ---------------------------------------------------------------------------

def save_ids(ids, path: Path):
    """Serialize a list of IDs to a JSON file, handling Arrow-backed columns efficiently."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(ids, list):
        # Arrow-backed columns (HuggingFace datasets) expose a fast C++ path via to_pylist()
        ids = ids.to_pylist() if hasattr(ids, "to_pylist") else list(ids)
    with open(path, "w") as f:
        json.dump(ids, f)


def load_ids(path: Path) -> list[str]:
    """Load a list of IDs from a JSON file."""
    with open(path) as f:
        return json.load(f)


def save_chunk(embeddings: np.ndarray, path: Path):
    """Save a single embedding chunk to disk and log its shape and file size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    log.info(
        "  Saved %s (%d×%d, %.1f MB)",
        path.name,
        *embeddings.shape,
        path.stat().st_size / (1024**2),
    )


def merge_chunks(embedding_dir: Path) -> np.ndarray:
    """Concatenate all chunk_*.npy files in a directory into a single merged.npy array.

    The individual chunk files are deleted after a successful merge to free disk space.
    """
    chunk_paths = sorted(embedding_dir.glob("chunk_*.npy"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk files found in {embedding_dir}")

    log.info("Merging %d chunks …", len(chunk_paths))
    merged_embeddings = np.concatenate([np.load(chunk_path) for chunk_path in chunk_paths], axis=0)

    merged_output_path = embedding_dir / "merged.npy"
    np.save(merged_output_path, merged_embeddings)

    for chunk_path in chunk_paths:
        chunk_path.unlink()

    return merged_embeddings


def load_cached(embedding_dir: Path) -> tuple[list[str], np.ndarray]:
    """Load IDs and the merged embedding matrix from a completed cache directory."""
    ids = load_ids(embedding_dir / "ids.json")
    embeddings = np.load(embedding_dir / "merged.npy")
    return ids, embeddings
