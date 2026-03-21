#!/usr/bin/env python3
"""
Baseline evaluation of multilingual-e5-large-instruct on MessIRve (Spanish IR).
Metrics: nDCG@10, Recall@100 via ranx.

Features:
- Filters out documents > 400 words to prevent truncation & OOM.
- Prunes qrels/queries missing from the filtered corpus to maintain metric validity (Cached).
- Chunked document encoding with crash-safe resumption.
- float32 `.npy` caching to avoid Parquet bottleneck.
- NUMA-aware multi-processing.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# NUMA / threading env-vars — MUST be set BEFORE imports
# ---------------------------------------------------------------------------
_NCPU = str(os.cpu_count() or 32)
os.environ.setdefault("OMP_NUM_THREADS", _NCPU)
os.environ.setdefault("MKL_NUM_THREADS", _NCPU)
os.environ.setdefault("OMP_PROC_BIND", "spread")
os.environ.setdefault("OMP_PLACES", "threads")
os.environ.setdefault("GOMP_CPU_AFFINITY", f"0-{int(_NCPU) - 1}")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import datasets
import faiss
import numpy as np
import torch
from ranx import Qrels, Run, evaluate
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COUNTRY = "full"
DATASET_VERSION = "1.2"
MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
CORPUS_NAME = "spanish-ir/eswiki_20240401_corpus"

# Memory & Sequence parameters
QUERY_BATCH_SIZE = 128  # Increased for smaller model
DOC_BATCH_SIZE = 128  # Increased for smaller model
MAX_SEQ_LENGTH = 512  # e5 max token limit
MAX_WORD_COUNT = 400  # Filters corpus to better match the 512 token limit

DOC_CHUNK_SIZE = 50_000
TOP_K = 100

_LOCAL_CACHE = Path.home() / ".cache" / "messirve_embeddings"
_EXTERNAL_CACHE = Path("/media/discoexterno/jmendoza/embeddings_cache")
CACHE_DIR = _LOCAL_CACHE if _LOCAL_CACHE.parent.exists() else _EXTERNAL_CACHE

GPU_DEVICES = ["cuda:0", "cuda:1"]
SEED = 42
NUM_WORKERS = os.cpu_count() or 32

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
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


def _model_slug(name: str) -> str:
    return name.replace("/", "__")


def _dataset_cache_base(country: str, version: str, max_words: int) -> Path:
    """Model-agnostic cache directory for the filtered qrels."""
    return CACHE_DIR / "shared_datasets" / f"{country}_v{version}_filt{max_words}w"


def _cache_base(model: str, country: str, version: str) -> Path:
    return CACHE_DIR / _model_slug(model) / f"{country}_v{version}_seq{MAX_SEQ_LENGTH}_filt{MAX_WORD_COUNT}w"


def _emb_dir(base: Path, kind: str) -> Path:
    return base / f"{kind}_embeddings"


def _is_complete(d: Path) -> bool:
    return (d / "ids.json").exists() and (d / "merged.npy").exists()


def _log_cache_status(base: Path, dataset_dir: Path):
    qrels_ok = (dataset_dir / "pruned_qrels.json").exists()
    log.info("  %-6s: %s", "qrels", "✓ cached" if qrels_ok else "✗ not cached")
    for kind in ("doc", "query"):
        d = _emb_dir(base, kind)
        ok = _is_complete(d)
        sz = ""
        if ok:
            mb = (d / "merged.npy").stat().st_size / (1024**2)
            sz = f" ({mb:.1f} MB)"
        nc = len(list(d.glob("chunk_*.npy"))) if d.exists() else 0
        log.info("  %-6s: %s%s  [%d chunks on disk]", kind, "✓ complete" if ok else "✗ incomplete", sz, nc)


def _save_ids(ids: list[str], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(ids, f)


def _load_ids(path: Path) -> list[str]:
    with open(path) as f:
        return json.load(f)


def _save_chunk(emb: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    log.info("  Saved %s (%d×%d, %.1f MB)", path.name, *emb.shape, path.stat().st_size / (1024**2))


def _merge_chunks(d: Path) -> np.ndarray:
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


def _load_cached(d: Path) -> tuple[list[str], np.ndarray]:
    ids = _load_ids(d / "ids.json")
    emb = np.load(d / "merged.npy")
    return ids, emb


# ---------------------------------------------------------------------------
# Data Loading & Filtering (With Dataset Caching)
# ---------------------------------------------------------------------------
def load_messirve_qrels(country: str, version: str):
    with Timer(f"Loading raw MessIRve qrels ({country}, v{version})"):
        ds = datasets.load_dataset("spanish-ir/messirve", country, revision=version, num_proc=NUM_WORKERS)
        test = ds["test"]

    raw_ids, raw_docids, raw_queries = test["id"], test["docid"], test["query"]
    qrels_dict, q_map = defaultdict(dict), {}

    for qid, did, qtxt in zip(raw_ids, raw_docids, raw_queries):
        qid_s = str(qid)
        qrels_dict[qid_s][str(did)] = 1
        q_map[qid_s] = qtxt

    return Qrels(qrels_dict), q_map


def get_pruned_qrels_and_queries(country: str, version: str, kept_doc_ids: list[str] | None = None) -> tuple[Qrels, dict[str, str]]:
    cache_dir = _dataset_cache_base(country, version, MAX_WORD_COUNT)
    qrels_path = cache_dir / "pruned_qrels.json"
    qmap_path = cache_dir / "pruned_q_map.json"

    # 1. Quick load from cache if available
    if qrels_path.exists() and qmap_path.exists():
        with Timer(f"Loading cached pruned qrels (Max Words: {MAX_WORD_COUNT})"):
            with open(qrels_path, "r") as f:
                pruned_dict = json.load(f)
            with open(qmap_path, "r") as f:
                pruned_q_map = json.load(f)
        return Qrels(pruned_dict), pruned_q_map

    # 2. Compute it if cache is missing
    if kept_doc_ids is None:
        raise RuntimeError("Cache miss for pruned qrels, but kept_doc_ids were not provided to build it.")

    original_qrels, original_q_map = load_messirve_qrels(country, version)

    with Timer("Pruning qrels against filtered corpus"):
        kept_set = set(kept_doc_ids)
        qrels_dict = original_qrels.to_dict()
        pruned_dict = defaultdict(dict)

        for qid, judgments in qrels_dict.items():
            for did, score in judgments.items():
                if did in kept_set:
                    pruned_dict[qid][did] = score

        pruned_dict = {qid: judgs for qid, judgs in pruned_dict.items() if judgs}
        pruned_q_map = {qid: original_q_map[qid] for qid in pruned_dict.keys()}

        log.info("Filtered queries: %d / %d", len(pruned_dict), len(qrels_dict))

    # 3. Save to cache for the next run / other models
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(qrels_path, "w") as f:
        json.dump(pruned_dict, f)
    with open(qmap_path, "w") as f:
        json.dump(pruned_q_map, f)

    return Qrels(pruned_dict), pruned_q_map


def _batch_format_and_count(batch: dict) -> dict:
    titles = batch.get("title", [""] * len(batch["docid"]))
    texts = batch.get("text", [""] * len(batch["docid"]))
    ft, si, wc = [], [], []
    for did, title, text in zip(batch["docid"], titles, texts):
        title, text = title or "", text or ""
        full = f"{title}. {text}".strip() if title else text
        ft.append(full)
        si.append(str(did))
        wc.append(len(full.split()))
    batch["full_text"] = ft
    batch["str_docid"] = si
    batch["word_count"] = wc
    return batch


def load_corpus():
    with Timer("Loading eswiki corpus"):
        cds = datasets.load_dataset(CORPUS_NAME, num_proc=NUM_WORKERS)
        split = cds["train"] if "train" in cds else cds[list(cds.keys())[0]]

    n_original = len(split)
    log.info("Original corpus: %d documents", n_original)

    with Timer(f"Formatting and counting words ({NUM_WORKERS} workers)"):
        split = split.map(
            _batch_format_and_count, batched=True, batch_size=10_000, num_proc=NUM_WORKERS, desc="Formatting"
        )

    with Timer(f"Filtering docs <= {MAX_WORD_COUNT} words"):
        split = split.filter(
            lambda x: [wc <= MAX_WORD_COUNT for wc in x["word_count"]],
            batched=True,
            batch_size=10_000,
            num_proc=NUM_WORKERS,
            desc="Filtering",
        )

    n_filtered = len(split)
    log.info(
        "Filtered corpus: %d / %d documents kept (%.1f%%)",
        n_filtered,
        n_original,
        (n_filtered / n_original) * 100 if n_original else 0,
    )

    doc_ids, doc_texts = split["str_docid"], split["full_text"]
    del split, cds
    gc.collect()
    return doc_ids, doc_texts


# ---------------------------------------------------------------------------
# Model & Inference
# ---------------------------------------------------------------------------
class EmbeddingModel:
    def __init__(self, model_name: str, devices: list[str]):
        self.model_name = model_name
        self.devices = devices
        self.model = None
        self.pool = None
        # Strict e5 instruction requirement
        self.task_description = "Given a web search query, retrieve relevant passages that answer the query"
        self.query_prompt = f"Instruct: {self.task_description}\nQuery: "

    def start(self):
        with Timer("Loading SentenceTransformer"):
            self.model = SentenceTransformer(
                self.model_name,
                device="cpu",
                model_kwargs={
                    "torch_dtype": torch.float16,
                },
            )
            self.model.max_seq_length = MAX_SEQ_LENGTH

        with Timer(f"Starting pool on {self.devices}"):
            self.pool = self.model.start_multi_process_pool(target_devices=self.devices)

    def stop(self):
        if self.pool is not None:
            self.model.stop_multi_process_pool(self.pool)
            self.pool = None
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()

    def encode(self, texts: list[str], batch_size: int, is_query: bool = False) -> np.ndarray:
        # e5 requires instructions strictly on the query side
        if is_query:
            texts = [self.query_prompt + t for t in texts]

        emb = self.model.encode(
            sentences=texts,
            pool=self.pool,
            batch_size=batch_size,
            normalize_embeddings=True,  # Handles the L2 norm + contrastive setup
            prompt_name=None,
        )
        return emb.astype(np.float32, copy=False)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def encode_documents_chunked(doc_ids: list[str], doc_texts: list[str], emb_dir: Path, model: EmbeddingModel):
    if _is_complete(emb_dir):
        return _load_cached(emb_dir)

    n, chunk_sz = len(doc_ids), DOC_CHUNK_SIZE
    total_chunks = (n + chunk_sz - 1) // chunk_sz
    existing = len(list(emb_dir.glob("chunk_*.npy"))) if emb_dir.exists() else 0

    for ci in range(existing, total_chunks):
        lo, hi = ci * chunk_sz, min((ci + 1) * chunk_sz, n)
        log.info("── Chunk %d/%d [%d–%d] ──", ci + 1, total_chunks, lo, hi - 1)

        chunk_emb = model.encode(texts=doc_texts[lo:hi], batch_size=DOC_BATCH_SIZE, is_query=False)
        _save_chunk(chunk_emb, emb_dir / f"chunk_{ci:04d}.npy")
        del chunk_emb
        gc.collect()

    _save_ids(doc_ids, emb_dir / "ids.json")
    return doc_ids, _merge_chunks(emb_dir)


def encode_queries(q_map: dict[str, str], emb_dir: Path, model: EmbeddingModel):
    if _is_complete(emb_dir):
        return _load_cached(emb_dir)
    qids, qtexts = list(q_map.keys()), list(q_map.values())
    emb = model.encode(texts=qtexts, batch_size=QUERY_BATCH_SIZE, is_query=True)

    emb_dir.mkdir(parents=True, exist_ok=True)
    _save_ids(qids, emb_dir / "ids.json")
    np.save(emb_dir / "merged.npy", emb)
    return qids, emb


# ---------------------------------------------------------------------------
# Retrieval & Evaluation
# ---------------------------------------------------------------------------
def retrieve(query_ids: list[str], query_emb: np.ndarray, doc_ids: list[str], doc_emb: np.ndarray, top_k: int):
    nq, dim = len(query_ids), doc_emb.shape[1]
    faiss.omp_set_num_threads(NUM_WORKERS)

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


def run_evaluation(qrels: Qrels, run_dict: dict[str, dict[str, float]]):
    run = Run(run_dict)
    results = evaluate(qrels, run, metrics=["ndcg@10", "recall@100"])

    log.info("")
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  RESULTS — multilingual-e5-large-instruct        ║")
    log.info("╠══════════════════════════════════════════════════╣")
    for m, v in results.items():
        log.info("║  %-15s  %.4f                         ║", m, v)
    log.info("╚══════════════════════════════════════════════════╝")
    return results


def main():
    datasets.config.NUM_PROC = NUM_WORKERS
    torch.manual_seed(SEED)

    base = _cache_base(MODEL_NAME, COUNTRY, DATASET_VERSION)
    doc_dir, query_dir = _emb_dir(base, "doc"), _emb_dir(base, "query")
    dataset_cache_dir = _dataset_cache_base(COUNTRY, DATASET_VERSION, MAX_WORD_COUNT)

    _log_cache_status(base, dataset_cache_dir)

    need_docs = not _is_complete(doc_dir)
    need_queries = not _is_complete(query_dir)
    need_pruned_qrels = not (
        (dataset_cache_dir / "pruned_qrels.json").exists() and
        (dataset_cache_dir / "pruned_q_map.json").exists()
    )

    doc_ids, doc_emb, query_ids, query_emb = None, None, None, None
    doc_texts = None

    # 1. Handle Document IDs & Texts
    if need_docs or need_pruned_qrels:
        if need_docs:
            doc_ids, doc_texts = load_corpus()
        else:
            doc_ids = _load_ids(doc_dir / "ids.json")
    else:
        doc_ids = _load_ids(doc_dir / "ids.json")

    # 2. Get the pruned Qrels & Queries
    qrels, q_map = get_pruned_qrels_and_queries(
        COUNTRY, DATASET_VERSION, doc_ids if need_pruned_qrels else None
    )

    # 3. Model Encoding Phase
    if need_docs or need_queries:
        with EmbeddingModel(MODEL_NAME, GPU_DEVICES) as model:
            if need_docs:
                doc_ids, doc_emb = encode_documents_chunked(doc_ids, doc_texts, doc_dir, model)
                del doc_texts
                gc.collect()

            if need_queries:
                query_ids, query_emb = encode_queries(q_map, query_dir, model)

    # 4. Load anything that was cached and skipped
    if doc_emb is None:
        doc_ids, doc_emb = _load_cached(doc_dir)
    if query_emb is None:
        query_ids, query_emb = _load_cached(query_dir)

    # 5. Retrieve and Evaluate
    run_dict = retrieve(query_ids, query_emb, doc_ids, doc_emb, TOP_K)
    run_evaluation(qrels, run_dict)


if __name__ == "__main__":
    main()