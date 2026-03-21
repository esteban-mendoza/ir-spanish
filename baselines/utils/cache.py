"""
Cache path helpers, timing utilities, and embedding chunk I/O.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------
class Timer:
    def __init__(self, description: str):
        self.description = description
        self.elapsed: float | None = None

    def __enter__(self):
        log.info("⏱  START: %s", self.description)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._t0
        log.info("⏱  DONE:  %s (%.1fs)", self.description, self.elapsed)


# ---------------------------------------------------------------------------
# Memory logging
# ---------------------------------------------------------------------------
def log_gpu_memory():
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        a = torch.cuda.memory_allocated(i) / (1024**3)
        r = torch.cuda.memory_reserved(i) / (1024**3)
        t = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        log.info("GPU %d (%s): %.1f/%.1f/%.1f GB (alloc/reserved/total)", i, torch.cuda.get_device_name(i), a, r, t)


def log_ram_usage():
    try:
        import resource

        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        log.info("RAM (peak RSS): %.1f GB", kb / (1024**2))
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Cache path builders
# ---------------------------------------------------------------------------
def model_slug(name: str) -> str:
    return name.replace("/", "__")


def dataset_cache_base(cache_dir: Path, country: str, version: str, max_words: int) -> Path:
    """Model-agnostic cache directory for the filtered qrels."""
    return cache_dir / "shared_datasets" / f"{country}_v{version}_filt{max_words}w"


def cache_base(cache_dir: Path, model: str, country: str, version: str, max_seq_length: int, max_word_count: int) -> Path:
    return cache_dir / model_slug(model) / f"{country}_v{version}_seq{max_seq_length}_filt{max_word_count}w"


def emb_dir(base: Path, kind: str) -> Path:
    return base / f"{kind}_embeddings"


# ---------------------------------------------------------------------------
# Cache completeness & status
# ---------------------------------------------------------------------------
def is_complete(d: Path) -> bool:
    return (d / "ids.json").exists() and (d / "merged.npy").exists()


def log_cache_status(base: Path, dataset_dir: Path):
    qrels_ok = (dataset_dir / "pruned_qrels.json").exists()
    log.info("  %-6s: %s", "qrels", "✓ cached" if qrels_ok else "✗ not cached")
    for kind in ("doc", "query"):
        d = emb_dir(base, kind)
        ok = is_complete(d)
        sz = ""
        if ok:
            mb = (d / "merged.npy").stat().st_size / (1024**2)
            sz = f" ({mb:.1f} MB)"
        nc = len(list(d.glob("chunk_*.npy"))) if d.exists() else 0
        log.info("  %-6s: %s%s  [%d chunks on disk]", kind, "✓ complete" if ok else "✗ incomplete", sz, nc)


# ---------------------------------------------------------------------------
# Embedding I/O
# ---------------------------------------------------------------------------
def save_ids(ids: list[str], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(ids, f)


def load_ids(path: Path) -> list[str]:
    with open(path) as f:
        return json.load(f)


def save_chunk(emb: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    log.info("  Saved %s (%d×%d, %.1f MB)", path.name, *emb.shape, path.stat().st_size / (1024**2))


def merge_chunks(d: Path) -> np.ndarray:
    chunks = sorted(d.glob("chunk_*.npy"))
    if not chunks:
        raise FileNotFoundError(f"No chunks in {d}")
    log.info("Merging %d chunks …", len(chunks))
    merged = np.concatenate([np.load(p) for p in chunks], axis=0)
    out = d / "merged.npy"
    np.save(out, merged)
    for p in chunks:
        p.unlink()
    return merged


def load_cached(d: Path) -> tuple[list[str], np.ndarray]:
    ids = load_ids(d / "ids.json")
    emb = np.load(d / "merged.npy")
    return ids, emb
