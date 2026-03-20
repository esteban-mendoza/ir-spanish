#!/usr/bin/env python3
"""
Baseline evaluation of Qwen3-Embedding-8B on MessIRve (Spanish IR).
Metrics: nDCG@10, Recall@100 via ranx.

Embeddings are cached to disk as .npy + .json so subsequent runs skip encoding.
Uses both GPUs via SentenceTransformer's multi-process encode pool.
Uses FAISS (CPU) for fast retrieval.

Design decisions
----------------
- float32 throughout — no fp16 round-trip quantisation noise.
- Cache: .npy (contiguous binary) + ids.json.  Avoids the pathological
  4096-column Parquet problem entirely.
- Chunked document encoding with periodic saves — crash-safe resumption.
- Qwen3-Embedding-8B uses asymmetric prompting:
    query → "Instruct: …\\nQuery: {text}"   (stored in model.prompts["query"])
    doc   → "{text}"                         (no prefix)
  We read the prompt from the model config at runtime and prepend it
  ourselves so multi-GPU child processes don't need prompt_name forwarding.
- flash_attention_2 enabled (Ampere A5000 supports it).
- NUMA-aware OMP settings for the dual-socket Xeon system.
- Cache defaults to fast local storage (~/.cache/…).
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
# NUMA / threading env-vars — MUST be set BEFORE numpy / faiss / torch import
# so that OpenMP, MKL, and GOMP pick them up during library initialisation.
# ---------------------------------------------------------------------------
_NCPU = str(os.cpu_count() or 32)
os.environ.setdefault("OMP_NUM_THREADS", _NCPU)
os.environ.setdefault("MKL_NUM_THREADS", _NCPU)
os.environ.setdefault("OMP_PROC_BIND", "spread")
os.environ.setdefault("OMP_PLACES", "threads")
os.environ.setdefault("GOMP_CPU_AFFINITY", f"0-{int(_NCPU) - 1}")
# sentence-transformers spawns child processes that also use tokenizers;
# the Rust-based HF tokenizer is already thread-safe, so this is fine.
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
MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
CORPUS_NAME = "spanish-ir/eswiki_20240401_corpus"

# Batch sizes PER GPU
QUERY_BATCH_SIZE = 32
DOC_BATCH_SIZE = 16

# Checkpoint interval: save every N documents so crashes lose at most one chunk
DOC_CHUNK_SIZE = 50_000

# Retrieval
TOP_K = 100

# Cache — prefer fast local storage; fall back to external disk
_LOCAL_CACHE = Path.home() / ".cache" / "messirve_embeddings"
_EXTERNAL_CACHE = Path("/media/discoexterno/jmendoza/embeddings_cache")
CACHE_DIR = _LOCAL_CACHE if _LOCAL_CACHE.parent.exists() else _EXTERNAL_CACHE

# GPUs
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
# Helpers: timing / monitoring
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


# ---------------------------------------------------------------------------
# Helpers: cache layout (.npy + ids.json)
# ---------------------------------------------------------------------------
def _model_slug(name: str) -> str:
    return name.replace("/", "__")


def _cache_base(model: str, country: str, version: str) -> Path:
    return CACHE_DIR / _model_slug(model) / f"{country}_v{version}"


def _emb_dir(base: Path, kind: str) -> Path:
    return base / f"{kind}_embeddings"


def _is_complete(d: Path) -> bool:
    return (d / "ids.json").exists() and (d / "merged.npy").exists()


def _log_cache_status(base: Path):
    for kind in ("doc", "query"):
        d = _emb_dir(base, kind)
        ok = _is_complete(d)
        sz = ""
        if ok:
            mb = (d / "merged.npy").stat().st_size / (1024**2)
            sz = f" ({mb:.1f} MB)"
        nc = len(list(d.glob("chunk_*.npy"))) if d.exists() else 0
        log.info("  %-6s: %s%s  [%d chunks on disk]", kind, "✓ complete" if ok else "✗ incomplete", sz, nc)


# ---------------------------------------------------------------------------
# Helpers: .npy / json I/O
# ---------------------------------------------------------------------------
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
    log.info("Merged: %d×%d, %.1f MB", *merged.shape, out.stat().st_size / (1024**2))
    for p in chunks:
        p.unlink()
    log.info("Cleaned up %d chunk files.", len(chunks))
    return merged


def _load_cached(d: Path) -> tuple[list[str], np.ndarray]:
    ids = _load_ids(d / "ids.json")
    p = d / "merged.npy"
    log.info("Loading %s (%.1f MB) …", p, p.stat().st_size / (1024**2))
    emb = np.load(p)
    log.info("Loaded %d embeddings, dim=%d, dtype=%s", *emb.shape, emb.dtype)
    assert len(ids) == emb.shape[0], f"ID/embedding count mismatch: {len(ids)} vs {emb.shape[0]}"
    log_ram_usage()
    return ids, emb


# ---------------------------------------------------------------------------
# 1. Load MessIRve qrels
# ---------------------------------------------------------------------------
def load_messirve_qrels(country: str, version: str):
    with Timer(f"Loading MessIRve qrels ({country}, v{version})"):
        ds = datasets.load_dataset(
            "spanish-ir/messirve",
            country,
            revision=version,
            num_proc=NUM_WORKERS,
        )
        test = ds["test"]

    log.info("Test split: %d pairs", len(test))

    with Timer("Building qrels + query map (columnar)"):
        raw_ids = test["id"]
        raw_docids = test["docid"]
        raw_queries = test["query"]

        qrels_dict: dict[str, dict[str, int]] = defaultdict(dict)
        q_map: dict[str, str] = {}
        for qid, did, qtxt in zip(raw_ids, raw_docids, raw_queries):
            qid_s = str(qid)
            qrels_dict[qid_s][str(did)] = 1
            q_map[qid_s] = qtxt

    nq = len(qrels_dict)
    nr = sum(len(v) for v in qrels_dict.values())
    log.info("Queries: %d | Judgments: %d | Avg rels/q: %.2f", nq, nr, nr / nq if nq else 0)
    return Qrels(qrels_dict), q_map


# ---------------------------------------------------------------------------
# 2. Load corpus
# ---------------------------------------------------------------------------
def _batch_format(batch: dict) -> dict:
    titles = batch.get("title", [""] * len(batch["docid"]))
    texts = batch.get("text", [""] * len(batch["docid"]))
    ft, si = [], []
    for did, title, text in zip(batch["docid"], titles, texts):
        title = title or ""
        text = text or ""
        ft.append(f"{title}. {text}".strip() if title else text)
        si.append(str(did))
    batch["full_text"] = ft
    batch["str_docid"] = si
    return batch


def load_corpus():
    with Timer("Loading eswiki corpus"):
        cds = datasets.load_dataset(CORPUS_NAME, num_proc=NUM_WORKERS)
        split = cds["train"] if "train" in cds else cds[list(cds.keys())[0]]

    n = len(split)
    log.info("Corpus: %d documents", n)

    with Timer(f"Formatting {n} docs ({NUM_WORKERS} workers)"):
        split = split.map(
            _batch_format,
            batched=True,
            batch_size=10_000,
            num_proc=NUM_WORKERS,
            desc="Formatting",
        )

    with Timer("Extracting columns"):
        doc_ids: list[str] = split["str_docid"]
        doc_texts: list[str] = split["full_text"]

    del split, cds
    gc.collect()

    sample = [len(t) for t in doc_texts[:10_000]]
    log.info(
        "Text lengths (10k sample): min=%d med=%d mean=%d max=%d",
        min(sample),
        int(np.median(sample)),
        int(np.mean(sample)),
        max(sample),
    )
    log_ram_usage()
    return doc_ids, doc_texts


# ---------------------------------------------------------------------------
# 3. Embedding model wrapper
# ---------------------------------------------------------------------------
class EmbeddingModel:
    """
    Wraps SentenceTransformer with a persistent multi-GPU pool.

    Prompt handling strategy
    -----------------------
    sentence-transformers ≥ 5.x stores the model's prompt templates in
    ``model.prompts``.  For Qwen3-Embedding-8B this contains a ``"query"``
    key whose value is the instruction prefix string.  Documents receive
    no prefix.

    The multi-process pool in sentence-transformers distributes sentences
    to child processes.  Whether ``prompt_name`` is forwarded correctly
    depends on the exact library version.  To guarantee correctness we:

      1. Load the model and read ``model.prompts["query"]`` to get the
         *exact* prefix the model was trained with.
      2. Prepend it to query texts ourselves.
      3. Always pass ``prompt_name=None`` to ``encode()``.

    This way child processes see already-prefixed strings and need no
    prompt awareness at all.
    """

    def __init__(self, model_name: str, devices: list[str]):
        self.model_name = model_name
        self.devices = devices
        self.model: SentenceTransformer | None = None
        self.pool = None
        self.query_prompt: str = ""

    def start(self):
        with Timer("Loading SentenceTransformer"):
            self.model = SentenceTransformer(
                self.model_name,
                device="cpu",
                model_kwargs={
                    "torch_dtype": torch.float16,
                    "attn_implementation": "sdpa",
                },
                tokenizer_kwargs={"padding_side": "left"},
            )

        # Read the query prompt from the model config
        if hasattr(self.model, "prompts") and "query" in self.model.prompts:
            self.query_prompt = self.model.prompts["query"]
            log.info("Query prompt from model config: %r", self.query_prompt)
        else:
            # Fallback based on Qwen3-Embedding model card
            self.query_prompt = (
                "Instruct: Given a web search query, retrieve relevant " "passages that answer the query\nQuery: "
            )
            log.warning(
                "Could not read prompt from model.prompts; using fallback: %r",
                self.query_prompt,
            )

        with Timer(f"Starting pool on {self.devices}"):
            self.pool = self.model.start_multi_process_pool(
                target_devices=self.devices,
            )
        log_gpu_memory()

    def stop(self):
        if self.pool is not None:
            with Timer("Stopping pool"):
                self.model.stop_multi_process_pool(self.pool)
            self.pool = None
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
        log_gpu_memory()

    def encode(
        self,
        texts: list[str],
        batch_size: int,
        is_query: bool = False,
    ) -> np.ndarray:
        """
        Encode texts → float32 ndarray.

        If ``is_query=True`` the model's query prompt is prepended to
        every text before tokenisation.  Documents are encoded as-is
        (no instruction prefix), matching the Qwen3-Embedding spec.
        """
        if self.model is None or self.pool is None:
            raise RuntimeError("Call .start() first")

        prefix = self.query_prompt if is_query else ""
        if prefix:
            texts = [prefix + t for t in texts]

        n = len(texts)
        label = "query" if is_query else "document"
        log.info("Encoding %d %s texts (batch=%d, prefix_len=%d) …", n, label, batch_size, len(prefix))

        t0 = time.perf_counter()
        emb = self.model.encode(
            sentences=texts,
            pool=self.pool,
            batch_size=batch_size,
            normalize_embeddings=True,
            prompt_name=None,  # we prepended the prompt ourselves
        )
        dt = time.perf_counter() - t0
        log.info("  → %d×%d %s  %.1f texts/sec", emb.shape[0], emb.shape[1], emb.dtype, n / dt if dt > 0 else 0)

        return emb.astype(np.float32, copy=False)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# ---------------------------------------------------------------------------
# 4. Encode with chunked checkpointing
# ---------------------------------------------------------------------------
def encode_documents_chunked(
    doc_ids: list[str],
    doc_texts: list[str],
    emb_dir: Path,
    model: EmbeddingModel,
) -> tuple[list[str], np.ndarray]:
    if _is_complete(emb_dir):
        return _load_cached(emb_dir)

    n = len(doc_ids)
    chunk_sz = DOC_CHUNK_SIZE
    total_chunks = (n + chunk_sz - 1) // chunk_sz
    existing = len(list(emb_dir.glob("chunk_*.npy"))) if emb_dir.exists() else 0

    if existing > 0:
        log.info("Resuming: %d / %d chunks found on disk.", existing, total_chunks)

    for ci in range(existing, total_chunks):
        lo = ci * chunk_sz
        hi = min(lo + chunk_sz, n)
        log.info("── Chunk %d/%d [%d–%d] ──", ci + 1, total_chunks, lo, hi - 1)

        chunk_emb = model.encode(
            texts=doc_texts[lo:hi],
            batch_size=DOC_BATCH_SIZE,
            is_query=False,
        )
        _save_chunk(chunk_emb, emb_dir / f"chunk_{ci:04d}.npy")
        del chunk_emb
        gc.collect()

    _save_ids(doc_ids, emb_dir / "ids.json")

    with Timer("Merging document chunks"):
        merged = _merge_chunks(emb_dir)

    log_ram_usage()
    return doc_ids, merged


def encode_queries(
    q_map: dict[str, str],
    emb_dir: Path,
    model: EmbeddingModel,
) -> tuple[list[str], np.ndarray]:
    if _is_complete(emb_dir):
        return _load_cached(emb_dir)

    qids = list(q_map.keys())
    qtexts = [q_map[q] for q in qids]

    emb = model.encode(texts=qtexts, batch_size=QUERY_BATCH_SIZE, is_query=True)

    emb_dir.mkdir(parents=True, exist_ok=True)
    _save_ids(qids, emb_dir / "ids.json")
    np.save(emb_dir / "merged.npy", emb)
    log.info("Saved query embeddings: %d×%d", *emb.shape)
    log_ram_usage()
    return qids, emb


# ---------------------------------------------------------------------------
# 5. FAISS retrieval
# ---------------------------------------------------------------------------
def retrieve(
    query_ids: list[str],
    query_emb: np.ndarray,
    doc_ids: list[str],
    doc_emb: np.ndarray,
    top_k: int,
) -> dict[str, dict[str, float]]:
    nq, nd, dim = len(query_ids), len(doc_ids), doc_emb.shape[1]
    faiss.omp_set_num_threads(NUM_WORKERS)

    log.info("=" * 60)
    log.info("FAISS: %d queries × %d docs, dim=%d, top-%d, %d threads", nq, nd, dim, top_k, NUM_WORKERS)
    log.info("=" * 60)

    with Timer("Building IndexFlatIP"):
        index = faiss.IndexFlatIP(dim)
        d32 = np.ascontiguousarray(doc_emb, dtype=np.float32)
        index.add(d32)
        del d32

    log.info("Index: %d vectors, ~%.1f GB", index.ntotal, (nd * dim * 4) / (1024**3))
    log_ram_usage()

    BATCH = 1024
    with Timer(f"Searching ({nq} queries)"):
        q32 = np.ascontiguousarray(query_emb, dtype=np.float32)
        all_scores = np.empty((nq, top_k), dtype=np.float32)
        all_idx = np.empty((nq, top_k), dtype=np.int64)

        for lo in tqdm(range(0, nq, BATCH), desc="FAISS", unit="batch"):
            hi = min(lo + BATCH, nq)
            s, ix = index.search(q32[lo:hi], top_k)
            all_scores[lo:hi] = s
            all_idx[lo:hi] = ix

    with Timer("Building run dict"):
        doc_arr = np.array(doc_ids, dtype=object)
        run: dict[str, dict[str, float]] = {}
        for i in range(nq):
            mask = all_idx[i] != -1
            run[query_ids[i]] = {str(doc_arr[j]): float(sc) for j, sc in zip(all_idx[i][mask], all_scores[i][mask])}

    log.info("Done: %d queries", len(run))
    log_ram_usage()
    return run


# ---------------------------------------------------------------------------
# 6. Evaluate
# ---------------------------------------------------------------------------
def run_evaluation(qrels: Qrels, run_dict: dict[str, dict[str, float]]):
    with Timer("ranx evaluate"):
        run = Run(run_dict)
        results = evaluate(qrels, run, metrics=["ndcg@10", "recall@100"])

    log.info("")
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  RESULTS — Qwen3-Embedding-8B on MessIRve       ║")
    log.info("║  Dataset: %s (test), v%s                       ║", COUNTRY, DATASET_VERSION)
    log.info("╠══════════════════════════════════════════════════╣")
    for m, v in results.items():
        log.info("║  %-15s  %.4f                         ║", m, v)
    log.info("╚══════════════════════════════════════════════════╝")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.perf_counter()

    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  Qwen3-Embedding-8B — MessIRve Baseline         ║")
    log.info("╚══════════════════════════════════════════════════╝")
    log.info("")
    log.info("Config:")
    log.info("  Model:       %s", MODEL_NAME)
    log.info("  Dataset:     messirve (%s, v%s)", COUNTRY, DATASET_VERSION)
    log.info("  Corpus:      %s", CORPUS_NAME)
    log.info("  GPUs:        %s", GPU_DEVICES)
    log.info("  CPUs:        %d", NUM_WORKERS)
    log.info("  Batch:       doc=%d  query=%d  (per GPU)", DOC_BATCH_SIZE, QUERY_BATCH_SIZE)
    log.info("  Chunk size:  %d docs (checkpoint interval)", DOC_CHUNK_SIZE)
    log.info("  Top-k:       %d", TOP_K)
    log.info("  Cache:       %s", CACHE_DIR)
    log.info("  Attention:   flash_attention_2")
    log.info("")

    datasets.config.NUM_PROC = NUM_WORKERS
    torch.manual_seed(SEED)
    log_gpu_memory()

    base = _cache_base(MODEL_NAME, COUNTRY, DATASET_VERSION)
    _log_cache_status(base)

    doc_dir = _emb_dir(base, "doc")
    query_dir = _emb_dir(base, "query")

    # ── 1. Qrels ─────────────────────────────────────────────────────────
    log.info("\n" + "─" * 60)
    log.info("STEP 1/5: Relevance judgments")
    log.info("─" * 60)
    qrels, q_map = load_messirve_qrels(COUNTRY, DATASET_VERSION)

    # ── 2 & 3. Encode (model loaded ONCE, pool shared) ──────────────────
    need_docs = not _is_complete(doc_dir)
    need_queries = not _is_complete(query_dir)

    doc_ids: list[str] | None = None
    doc_texts: list[str] | None = None
    doc_emb: np.ndarray | None = None
    query_ids: list[str] | None = None
    query_emb: np.ndarray | None = None

    if need_docs or need_queries:
        if need_docs:
            log.info("\n" + "─" * 60)
            log.info("STEP 2/5: Load corpus + encode documents")
            log.info("─" * 60)
            doc_ids, doc_texts = load_corpus()

        # Single model load, single pool — used for both doc and query encoding
        with EmbeddingModel(MODEL_NAME, GPU_DEVICES) as model:
            if need_docs:
                doc_ids, doc_emb = encode_documents_chunked(
                    doc_ids,
                    doc_texts,
                    doc_dir,
                    model,
                )
                del doc_texts
                doc_texts = None
                gc.collect()
                log.info("Freed doc_texts.")

            if need_queries:
                log.info("\n" + "─" * 60)
                log.info("STEP 3/5: Encode queries")
                log.info("─" * 60)
                query_ids, query_emb = encode_queries(q_map, query_dir, model)
        # pool + model freed by __exit__
    else:
        log.info("\n" + "─" * 60)
        log.info("STEP 2/5: Document embeddings — cached ✓")
        log.info("─" * 60)

    # Load from cache anything not just encoded
    if doc_emb is None:
        doc_ids, doc_emb = _load_cached(doc_dir)
    if query_emb is None:
        log.info("\n" + "─" * 60)
        log.info("STEP 3/5: Query embeddings — cached ✓")
        log.info("─" * 60)
        query_ids, query_emb = _load_cached(query_dir)

    # ── 4. Retrieve ──────────────────────────────────────────────────────
    log.info("\n" + "─" * 60)
    log.info("STEP 4/5: Retrieval")
    log.info("─" * 60)
    log_gpu_memory()
    run_dict = retrieve(query_ids, query_emb, doc_ids, doc_emb, TOP_K)

    # ── 5. Evaluate ──────────────────────────────────────────────────────
    log.info("\n" + "─" * 60)
    log.info("STEP 5/5: Evaluation")
    log.info("─" * 60)
    results = run_evaluation(qrels, run_dict)

    elapsed = time.perf_counter() - t_start
    log.info("Wall-clock: %.1fs (%.1f min)", elapsed, elapsed / 60)
    log_ram_usage()
    return results


if __name__ == "__main__":
    main()
