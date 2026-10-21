#!/usr/bin/env python3
"""Controlled second-stage efficiency benchmark (Experiment 3).

The benchmark measures the cost of the second stage only: fusion over three
top-100 runs, or reranking of the top-100 jina-v5 candidates.  First-stage
retrieval, model loading, corpus loading and metric evaluation are outside the
timed region.

Each system runs in its own process so that GPU memory is not shared between
models.  The driver interleaves systems across repetitions.  Raw per-query
latencies and one metadata record per system/repetition are written under
``results/efficiency``.

Examples:
    python -m evaluation.efficiency --mode pilot
    python -m evaluation.efficiency --mode run --reps 3 --batch 1,32
    python -m evaluation.efficiency --mode aggregate
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import os
import random
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Keep CPU fusion reproducible.  ``utils`` uses setdefault, so these survive.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from ranx import Run, fuse

from utils import cache, data, retrieval

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = Path.home() / ".cache" / "messirve_embeddings"
OUT_DIR = REPO_ROOT / "results" / "efficiency"
RAW_DIR = OUT_DIR / "raw"
META_DIR = OUT_DIR / "metadata"
QUERIES_PATH = OUT_DIR / "queries.csv"

N_MEASURE = 500
N_WARMUP = 50
N_REPS = 3
SEED = 42
TOP_K = 100

SYSTEMS = ("fusion", "bge", "jina-v3", "colbert")
LABELS = {
    "fusion": "Fusion B1 (CombMNZ)",
    "bge": "bge-reranker-v2-m3",
    "jina-v3": "jina-reranker-v3",
    "colbert": "jina-colbert-v2",
}

FUSION_MODELS = (
    ("e5-large", "intfloat/multilingual-e5-large-instruct", 512, 512),
    ("bge-m3", "BAAI/bge-m3", 8192, 8192),
    ("jina-v5-small", "jinaai/jina-embeddings-v5-text-small-retrieval", 32768, 32768),
)
JINA_MODEL = "jinaai/jina-embeddings-v5-text-small-retrieval"
JINA_Q_LEN = 32768
JINA_D_LEN = 32768

BGE_MODEL = "BAAI/bge-reranker-v2-m3"
BGE_MAX_LENGTH = 512
BGE_BATCH_SIZE = 256
JINA_V3_MODEL = "jinaai/jina-reranker-v3"
JINA_V3_MAX_QUERY = 1024
JINA_V3_MAX_DOC = 8192
COLBERT_MODEL = "jinaai/jina-colbert-v2"
COLBERT_MAX_QUERY = 32
COLBERT_MAX_DOC = 300
COLBERT_DOC_BATCH = 32

PRODUCTION_RUNS = {
    "fusion": CACHE_DIR / "fusions_stage_b" / "combmnz__e5-large+bge-m3+jina-v5-small.lz4",
    "bge": CACHE_DIR
    / "BAAI__bge-reranker-v2-m3"
    / "over__jinaai__jina-embeddings-v5-text-small-retrieval"
    / "full_v1.2_q512_d512_nofilter"
    / "retrieval_run.lz4",
    "jina-v3": CACHE_DIR
    / "jinaai__jina-reranker-v3"
    / "over__jinaai__jina-embeddings-v5-text-small-retrieval"
    / "full_v1.2_q1024_d8192_nofilter"
    / "retrieval_run.lz4",
    "colbert": CACHE_DIR
    / "jinaai__jina-colbert-v2"
    / "over__jinaai__jina-embeddings-v5-text-small-retrieval"
    / "full_v1.2_q32_d300_nofilter"
    / "retrieval_run.lz4",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _rss_mib() -> float:
    """Peak resident set size of this process in MiB."""
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _cuda_index(device: str) -> int:
    if not device.startswith("cuda"):
        return 0
    _, _, index = device.partition(":")
    return int(index) if index else 0


def _reset_gpu_peak(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.reset_peak_memory_stats(_cuda_index(device))


def _gpu_peak_mib(device: str) -> tuple[float, float]:
    if not device.startswith("cuda"):
        return 0.0, 0.0
    import torch

    index = _cuda_index(device)
    allocated = torch.cuda.max_memory_allocated(index) / (1024**2)
    reserved = torch.cuda.max_memory_reserved(index) / (1024**2)
    return allocated, reserved


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize(_cuda_index(device))


def _top_docs(run_dict: dict[str, dict[str, float]], query_id: str, k: int = TOP_K) -> list[str]:
    docs = run_dict.get(query_id, {})
    ordered = sorted(docs.items(), key=lambda item: (-item[1], item[0]))
    return [doc_id for doc_id, _ in ordered[:k]]


def _load_subset(path: Path, query_ids: list[str]) -> dict[str, dict[str, float]]:
    """Load a ranx run and keep only the requested queries."""
    run = retrieval.load_run(path)
    full = run.to_dict()
    subset = {query_id: full[query_id] for query_id in query_ids if query_id in full}
    del run, full
    gc.collect()
    log.info("Loaded %s (%d/%d queries)", path.name, len(subset), len(query_ids))
    return subset


def _fusion_run_path(model: tuple[str, str, int, int]) -> Path:
    _, name, q_len, d_len = model
    base = cache.cache_base(
        CACHE_DIR, name, data.COUNTRY, data.DATASET_VERSION, q_len, d_len, data.MAX_WORD_COUNT
    )
    return cache.run_cache_path(base)


def _jina_run_path() -> Path:
    base = cache.cache_base(
        CACHE_DIR,
        JINA_MODEL,
        data.COUNTRY,
        data.DATASET_VERSION,
        JINA_Q_LEN,
        JINA_D_LEN,
        data.MAX_WORD_COUNT,
    )
    return cache.run_cache_path(base)


# ---------------------------------------------------------------------------
# Query sample and inputs
# ---------------------------------------------------------------------------
def _load_query_map() -> dict[str, str]:
    dataset_cache_dir = cache.dataset_cache_base(
        CACHE_DIR, data.COUNTRY, data.DATASET_VERSION, data.MAX_WORD_COUNT
    )
    _qrels, query_map = data.get_pruned_qrels_and_queries(
        data.COUNTRY,
        data.DATASET_VERSION,
        kept_doc_ids=None,
        dataset_cache_dir=dataset_cache_dir,
        num_workers=1,
    )
    return query_map


def ensure_queries(seed: int = SEED) -> None:
    """Create the fixed 500 + 50 query sample once.

    The pilot uses a prefix of the same sample, so running the pilot first does
    not change the queries used by the full benchmark.
    """
    n_measure, n_warmup = N_MEASURE, N_WARMUP
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if QUERIES_PATH.exists():
        with QUERIES_PATH.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        measured = sum(1 for row in rows if row["split"] == "measure")
        warmup = sum(1 for row in rows if row["split"] == "warmup")
        if measured >= n_measure and warmup >= n_warmup:
            log.info("Reusing %s (%d measure, %d warmup)", QUERIES_PATH, measured, warmup)
            return

    query_map = _load_query_map()
    rng = random.Random(seed)
    all_ids = sorted(query_map)
    needed = n_measure + n_warmup
    if needed > len(all_ids):
        raise SystemExit(f"Not enough queries: {len(all_ids)} < {needed}")
    sample = rng.sample(all_ids, needed)
    warmup_ids, measure_ids = sample[:n_warmup], sample[n_warmup:]

    with QUERIES_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "split", "query", "n_chars"])
        writer.writeheader()
        for split, ids in (("warmup", warmup_ids), ("measure", measure_ids)):
            for query_id in ids:
                text = query_map[query_id]
                writer.writerow(
                    {
                        "query_id": query_id,
                        "split": split,
                        "query": text,
                        "n_chars": len(text),
                    }
                )
    log.info("Wrote %s (%d measure, %d warmup)", QUERIES_PATH, len(measure_ids), len(warmup_ids))


def _read_queries() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not QUERIES_PATH.exists():
        raise SystemExit(f"Missing {QUERIES_PATH}; run --mode pilot or --mode run first")
    with QUERIES_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    warmup = [row for row in rows if row["split"] == "warmup"]
    measure = [row for row in rows if row["split"] == "measure"]
    return warmup, measure


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------
class FusionSystem:
    name = "fusion"

    def __init__(self, query_ids: list[str], query_map: dict[str, str]) -> None:
        del query_map
        self.query_ids = query_ids
        run_dicts = []
        for model in FUSION_MODELS:
            path = _fusion_run_path(model)
            run_dicts.append(_load_subset(path, query_ids))

        common = set(query_ids)
        for run_dict in run_dicts:
            common &= set(run_dict)
        self.query_ids = [query_id for query_id in query_ids if query_id in common]
        if not self.query_ids:
            raise SystemExit("No query has all three fusion runs")

        self.per_query: dict[str, list[Run]] = {}
        for query_id in self.query_ids:
            self.per_query[query_id] = [
                Run({query_id: run_dict[query_id]}, name=model[0])
                for model, run_dict in zip(FUSION_MODELS, run_dicts)
            ]
        self.batch_runs = [
            Run({query_id: run_dict[query_id] for query_id in self.query_ids}, name=model[0])
            for model, run_dict in zip(FUSION_MODELS, run_dicts)
        ]
        log.info("Fusion prepared for %d queries", len(self.query_ids))

    def score_one(self, query_id: str) -> tuple[list[str], float]:
        started = time.perf_counter_ns()
        fused = fuse(self.per_query[query_id], norm="min-max", method="mnz", params={})
        scores = fused.to_dict()[query_id]
        order = [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        return order, elapsed_ms

    def score_batch(self, query_ids: list[str]) -> tuple[dict[str, list[str]], float]:
        started = time.perf_counter_ns()
        fused = fuse(self.batch_runs, norm="min-max", method="mnz", params={})
        scores = fused.to_dict()
        order = {
            query_id: [
                doc_id
                for doc_id, _ in sorted(scores[query_id].items(), key=lambda item: (-item[1], item[0]))
            ]
            for query_id in query_ids
        }
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        return order, elapsed_ms

    def close(self) -> None:
        self.per_query.clear()
        self.batch_runs.clear()


class BgeSystem:
    name = "bge"

    def __init__(
        self,
        query_ids: list[str],
        query_map: dict[str, str],
        candidates: dict[str, list[str]],
        doc_lookup: dict[str, str],
        device: str,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.query_ids = [query_id for query_id in query_ids if query_id in candidates]
        self.query_map = query_map
        self.candidates = candidates
        self.doc_lookup = doc_lookup
        self.device = device
        started = time.perf_counter()
        self.model = CrossEncoder(BGE_MODEL, max_length=BGE_MAX_LENGTH, device=device)
        self.load_seconds = time.perf_counter() - started
        log.info("Loaded %s in %.1fs", BGE_MODEL, self.load_seconds)

    def _pairs(self, query_ids: list[str]) -> tuple[list[tuple[str, str]], list[tuple[str, int, int]]]:
        pairs: list[tuple[str, str]] = []
        spans: list[tuple[str, int, int]] = []
        for query_id in query_ids:
            start = len(pairs)
            query_text = self.query_map[query_id]
            for doc_id in self.candidates[query_id]:
                pairs.append((query_text, self.doc_lookup.get(doc_id, "")))
            spans.append((query_id, start, len(pairs)))
        return pairs, spans

    def _orders(self, scores, spans: list[tuple[str, int, int]]) -> dict[str, list[str]]:
        orders: dict[str, list[str]] = {}
        for query_id, start, end in spans:
            local_scores = [float(value) for value in scores[start:end]]
            doc_ids = self.candidates[query_id]
            order = sorted(zip(doc_ids, local_scores), key=lambda item: (-item[1], item[0]))
            orders[query_id] = [doc_id for doc_id, _ in order]
        return orders

    def score_one(self, query_id: str) -> tuple[list[str], float]:
        pairs, spans = self._pairs([query_id])
        _sync(self.device)
        started = time.perf_counter_ns()
        scores = self.model.predict(pairs, batch_size=BGE_BATCH_SIZE, show_progress_bar=False)
        _sync(self.device)
        order = self._orders(scores, spans)[query_id]
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        return order, elapsed_ms

    def score_batch(self, query_ids: list[str]) -> tuple[dict[str, list[str]], float]:
        pairs, spans = self._pairs(query_ids)
        _sync(self.device)
        started = time.perf_counter_ns()
        scores = self.model.predict(pairs, batch_size=BGE_BATCH_SIZE, show_progress_bar=False)
        _sync(self.device)
        orders = self._orders(scores, spans)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        return orders, elapsed_ms

    def close(self) -> None:
        del self.model


class JinaV3System:
    name = "jina-v3"

    def __init__(
        self,
        query_ids: list[str],
        query_map: dict[str, str],
        candidates: dict[str, list[str]],
        doc_lookup: dict[str, str],
        device: str,
    ) -> None:
        from transformers import AutoModel

        self.query_ids = [query_id for query_id in query_ids if query_id in candidates]
        self.query_map = query_map
        self.candidates = candidates
        self.doc_lookup = doc_lookup
        self.device = device
        started = time.perf_counter()
        self.model = AutoModel.from_pretrained(
            JINA_V3_MODEL, torch_dtype="auto", trust_remote_code=True
        )
        self.model.to(device)
        self.model.eval()
        self.load_seconds = time.perf_counter() - started
        log.info("Loaded %s in %.1fs", JINA_V3_MODEL, self.load_seconds)

    def _docs(self, query_id: str) -> list[str]:
        return [self.doc_lookup.get(doc_id, "") for doc_id in self.candidates[query_id]]

    def score_one(self, query_id: str) -> tuple[list[str], float]:
        docs = self._docs(query_id)
        _sync(self.device)
        started = time.perf_counter_ns()
        reranked = self.model.rerank(
            self.query_map[query_id],
            docs,
            max_query_length=JINA_V3_MAX_QUERY,
            max_doc_length=JINA_V3_MAX_DOC,
        )
        _sync(self.device)
        order = [self.candidates[query_id][item["index"]] for item in reranked]
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        return order, elapsed_ms

    def score_batch(self, query_ids: list[str]) -> tuple[dict[str, list[str]], float]:
        _sync(self.device)
        started = time.perf_counter_ns()
        orders = {query_id: self.score_one(query_id)[0] for query_id in query_ids}
        _sync(self.device)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        return orders, elapsed_ms

    def close(self) -> None:
        del self.model


class ColbertSystem:
    name = "colbert"

    def __init__(
        self,
        query_ids: list[str],
        query_map: dict[str, str],
        candidates: dict[str, list[str]],
        doc_lookup: dict[str, str],
        device: str,
    ) -> None:
        _patch_pylate_dense()
        from pylate import models

        self.query_ids = [query_id for query_id in query_ids if query_id in candidates]
        self.query_map = query_map
        self.candidates = candidates
        self.doc_lookup = doc_lookup
        self.device = device
        started = time.perf_counter()
        self.model = models.ColBERT(
            model_name_or_path=COLBERT_MODEL,
            query_prefix="[QueryMarker]",
            document_prefix="[DocumentMarker]",
            attend_to_expansion_tokens=True,
            trust_remote_code=True,
            device=device,
            query_length=COLBERT_MAX_QUERY,
            document_length=COLBERT_MAX_DOC,
        )
        self.model._text_length = self.model._input_length
        self.load_seconds = time.perf_counter() - started
        log.info("Loaded %s in %.1fs", COLBERT_MODEL, self.load_seconds)

    def _docs(self, query_id: str) -> list[str]:
        return [self.doc_lookup.get(doc_id, "") for doc_id in self.candidates[query_id]]

    def score_one(self, query_id: str) -> tuple[list[str], float]:
        from pylate.scores import colbert_scores_pairwise

        docs = self._docs(query_id)
        _sync(self.device)
        started = time.perf_counter_ns()
        query_emb = self.model.encode(
            [self.query_map[query_id]], is_query=True, batch_size=1, show_progress_bar=False
        )
        doc_embs = self.model.encode(
            docs, is_query=False, batch_size=COLBERT_DOC_BATCH, show_progress_bar=False
        )
        scores = colbert_scores_pairwise([query_emb[0]] * len(doc_embs), doc_embs)
        _sync(self.device)
        order = sorted(
            zip(self.candidates[query_id], [float(score) for score in scores]),
            key=lambda item: (-item[1], item[0]),
        )
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        return [doc_id for doc_id, _ in order], elapsed_ms

    def score_batch(self, query_ids: list[str]) -> tuple[dict[str, list[str]], float]:
        _sync(self.device)
        started = time.perf_counter_ns()
        orders = {query_id: self.score_one(query_id)[0] for query_id in query_ids}
        _sync(self.device)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        return orders, elapsed_ms

    def close(self) -> None:
        del self.model


def _patch_pylate_dense() -> None:
    """Make pylate load jina-colbert-v2's projection layer.

    Copied from rerankers/jina_colbert_v2.py: the checkpoint has no modules.json,
    so SentenceTransformer creates [Transformer, Pooling] and pylate expects
    [Transformer, Dense].  The projection weights are loaded directly.
    """
    from pylate.models.Dense import Dense as PylateDense
    from sentence_transformers.models import Dense as SentenceTransformerDense
    from sentence_transformers.util import import_from_string

    if getattr(PylateDense, "_efficiency_patched", False):
        return

    def _fixed_from_st(dense):
        if not isinstance(dense, SentenceTransformerDense):
            from safetensors import safe_open
            from transformers.utils import cached_file

            path = cached_file(COLBERT_MODEL, "model.safetensors")
            with safe_open(path, framework="pt", device="cpu") as handle:
                weight = handle.get_tensor("linear.weight")
            model = PylateDense(
                in_features=weight.shape[1],
                out_features=weight.shape[0],
                bias=False,
            )
            model.load_state_dict({"linear.weight": weight})
            return model

        dense_keys = {
            "in_features",
            "out_features",
            "bias",
            "activation_function",
            "init_weight",
            "init_bias",
            "use_residual",
        }
        config = dense.get_config_dict()
        config["activation_function"] = import_from_string(
            config.get("activation_function", "torch.nn.modules.linear.Identity")
        )()
        filtered = {key: value for key, value in config.items() if key in dense_keys}
        model = PylateDense(**filtered)
        model.load_state_dict(dense.state_dict())
        return model

    PylateDense.from_sentence_transformers = staticmethod(_fixed_from_st)
    PylateDense._efficiency_patched = True


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _load_inputs(system: str, query_ids: list[str]) -> tuple[dict[str, str], dict[str, list[str]], dict[str, str]]:
    query_map_all = _load_query_map()
    query_map = {query_id: query_map_all[query_id] for query_id in query_ids}
    del query_map_all
    gc.collect()

    if system == "fusion":
        return query_map, {}, {}

    jina_run = _load_subset(_jina_run_path(), query_ids)
    candidates = {query_id: _top_docs(jina_run, query_id) for query_id in query_ids if query_id in jina_run}
    del jina_run
    gc.collect()

    needed = {doc_id for docs in candidates.values() for doc_id in docs}
    doc_ids, doc_texts, _ = data.load_corpus(1)
    doc_lookup = {
        doc_id: text
        for doc_id, text in zip(doc_ids, doc_texts)
        if doc_id in needed
    }
    del doc_ids, doc_texts
    gc.collect()
    log.info("Document lookup: %d/%d candidates", len(doc_lookup), len(needed))
    return query_map, candidates, doc_lookup


def _build_system(system: str, query_ids: list[str], query_map: dict[str, str],
                  candidates: dict[str, list[str]], doc_lookup: dict[str, str], device: str):
    if system == "fusion":
        return FusionSystem(query_ids, query_map)
    if system == "bge":
        return BgeSystem(query_ids, query_map, candidates, doc_lookup, device)
    if system == "jina-v3":
        return JinaV3System(query_ids, query_map, candidates, doc_lookup, device)
    if system == "colbert":
        return ColbertSystem(query_ids, query_map, candidates, doc_lookup, device)
    raise ValueError(system)


def _write_raw(system: str, rep: int, rows: list[dict]) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{system}_rep{rep}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["system", "rep", "query_id", "latency_ms"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def worker(args: argparse.Namespace) -> None:
    warmup_rows, measure_rows = _read_queries()
    warmup_ids = [row["query_id"] for row in warmup_rows[: args.warmup]]
    measure_ids = [row["query_id"] for row in measure_rows[: args.n]]
    if args.warmup and not warmup_ids:
        raise SystemExit("No warmup queries available")
    all_ids = warmup_ids + measure_ids
    query_map, candidates, doc_lookup = _load_inputs(args.system, all_ids)

    if args.system == "fusion":
        # Fusion needs the three runs; load them inside FusionSystem.
        system = FusionSystem(all_ids, query_map)
    else:
        system = _build_system(
            args.system, all_ids, query_map, candidates, doc_lookup, args.device
        )

    valid = set(system.query_ids)
    warmup_ids = [query_id for query_id in warmup_ids if query_id in valid]
    measure_ids = [query_id for query_id in measure_ids if query_id in valid]
    if not measure_ids:
        raise SystemExit(f"No valid measure queries for {args.system}")

    # Warmup is deliberately outside the timed region.
    for query_id in warmup_ids:
        system.score_one(query_id)

    _reset_gpu_peak(args.device)
    rows: list[dict] = []
    for query_id in measure_ids:
        _order, latency_ms = system.score_one(query_id)
        rows.append(
            {
                "system": args.system,
                "rep": args.rep,
                "query_id": query_id,
                "latency_ms": f"{latency_ms:.6f}",
            }
        )

    raw_path = _write_raw(args.system, args.rep, rows)
    latencies = [float(row["latency_ms"]) for row in rows]
    meta = {
        "system": args.system,
        "label": LABELS[args.system],
        "rep": args.rep,
        "n": len(latencies),
        "device": args.device,
        "load_seconds": getattr(system, "load_seconds", None),
        "median_ms": statistics.median(latencies) if latencies else None,
        "p95_ms": float(sorted(latencies)[max(0, math.ceil(0.95 * len(latencies)) - 1)])
        if latencies
        else None,
        "peak_rss_mib": _rss_mib(),
        "peak_vram_allocated_mib": _gpu_peak_mib(args.device)[0],
        "peak_vram_reserved_mib": _gpu_peak_mib(args.device)[1],
        "batch_qps": None,
        "batch_docs_per_s": None,
        "batch_size": None,
    }

    for batch_size in args.batch:
        if batch_size <= 1:
            qps = len(latencies) / (sum(latencies) / 1000.0) if latencies else 0.0
        else:
            _sync(args.device)
            started = time.perf_counter_ns()
            for start in range(0, len(measure_ids), batch_size):
                system.score_batch(measure_ids[start : start + batch_size])
            _sync(args.device)
            elapsed_s = (time.perf_counter_ns() - started) / 1e9
            qps = len(measure_ids) / elapsed_s if elapsed_s > 0 else 0.0
        docs_per_s = qps * TOP_K if args.system != "fusion" else None
        meta[f"batch_{batch_size}_qps"] = qps
        meta[f"batch_{batch_size}_docs_per_s"] = docs_per_s
        # Keep the last probed batch as the nominal batch in the summary.
        meta["batch_qps"] = qps
        meta["batch_docs_per_s"] = docs_per_s
        meta["batch_size"] = batch_size
        log.info("%s rep=%d batch=%d qps=%.3f", args.system, args.rep, batch_size, qps)

    META_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = META_DIR / f"{args.system}_rep{args.rep}.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)
    log.info("Wrote %s and %s", raw_path, meta_path)
    system.close()


# ---------------------------------------------------------------------------
# Driver and aggregation
# ---------------------------------------------------------------------------
def pilot(args: argparse.Namespace) -> None:
    ensure_queries(args.seed)
    for system in args.systems:
        cmd = [
            sys.executable,
            "-m",
            "evaluation.efficiency",
            "--mode",
            "worker",
            "--system",
            system,
            "--rep",
            "0",
            "--n",
            str(args.n),
            "--warmup",
            str(args.warmup),
            "--batch",
            ",".join(str(value) for value in args.batch),
            "--device",
            args.device,
        ]
        log.info("Pilot: %s", " ".join(cmd))
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def run(args: argparse.Namespace) -> None:
    ensure_queries(args.seed)
    for rep in range(1, args.reps + 1):
        order = list(args.systems)
        random.Random(args.seed + rep).shuffle(order)
        for system in order:
            cmd = [
                sys.executable,
                "-m",
                "evaluation.efficiency",
                "--mode",
                "worker",
                "--system",
                system,
                "--rep",
                str(rep),
                "--n",
                str(args.n),
                "--warmup",
                str(args.warmup),
                "--batch",
                ",".join(str(value) for value in args.batch),
                "--device",
                args.device,
            ]
            log.info("Run rep=%d: %s", rep, " ".join(cmd))
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    aggregate(args)


def _read_meta() -> dict[tuple[str, int], dict]:
    metas: dict[tuple[str, int], dict] = {}
    for path in sorted(META_DIR.glob("*_rep*.json")):
        with path.open(encoding="utf-8") as handle:
            meta = json.load(handle)
        metas[(meta["system"], int(meta["rep"]))] = meta
    return metas


def _read_raw(system: str, rep: int) -> dict[str, float]:
    path = RAW_DIR / f"{system}_rep{rep}.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["query_id"]: float(row["latency_ms"])
            for row in csv.DictReader(handle)
        }


def _bootstrap_ratio(
    lat_system: list[float], lat_fusion: list[float], rng: random.Random, n_boot: int = 10_000
) -> tuple[float, float, float]:
    ratios = []
    for _ in range(n_boot):
        indices = [rng.randrange(len(lat_system)) for _ in range(len(lat_system))]
        sample_system = [lat_system[index] for index in indices]
        sample_fusion = [lat_fusion[index] for index in indices]
        ratios.append(statistics.median(sample_system) / statistics.median(sample_fusion))
    ratios.sort()
    return (
        statistics.median(ratios),
        ratios[int(0.025 * len(ratios))],
        ratios[int(0.975 * len(ratios))],
    )


def aggregate(args: argparse.Namespace) -> None:
    metas = _read_meta()
    systems = [system for system in SYSTEMS if any(key[0] == system for key in metas)]
    summary: list[dict] = []
    rng = random.Random(args.seed)
    for system in systems:
        reps = sorted(rep for (name, rep) in metas if name == system)
        if not reps:
            continue
        meta = metas[(system, reps[-1])]
        summary_row = {
            "system": system,
            "label": LABELS[system],
            "n": meta["n"],
            "median_ms": meta["median_ms"],
            "p95_ms": meta["p95_ms"],
            "batch_size": meta["batch_size"],
            "batch_qps": meta["batch_qps"],
            "batch_docs_per_s": meta["batch_docs_per_s"],
            "peak_rss_mib": meta["peak_rss_mib"],
            "peak_vram_allocated_mib": meta["peak_vram_allocated_mib"],
        }
        if system != "fusion":
            fusion_rep = reps[0]
            lat_system = _read_raw(system, fusion_rep)
            lat_fusion = _read_raw("fusion", fusion_rep)
            paired = sorted(set(lat_system) & set(lat_fusion))
            if paired:
                ratio, low, high = _bootstrap_ratio(
                    [lat_system[qid] for qid in paired],
                    [lat_fusion[qid] for qid in paired],
                    rng,
                )
                summary_row.update(
                    {
                        "latency_ratio": ratio,
                        "latency_ratio_ci_low": low,
                        "latency_ratio_ci_high": high,
                    }
                )
        summary.append(summary_row)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "summary.csv"
    if summary:
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    log.info("Wrote %s and summary.json", summary_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_batch(value: str) -> list[int]:
    values = [int(part) for part in value.split(",") if part.strip()]
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("batch sizes must be positive integers")
    return sorted(set(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["pilot", "run", "worker", "aggregate"], default="pilot")
    parser.add_argument("--system", choices=SYSTEMS)
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--reps", type=int, default=N_REPS)
    parser.add_argument("--n", type=int, default=N_MEASURE)
    parser.add_argument("--warmup", type=int, default=N_WARMUP)
    parser.add_argument("--batch", type=_parse_batch, default=[1])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "pilot":
        pilot(args)
    elif args.mode == "run":
        run(args)
    elif args.mode == "worker":
        if not args.system:
            raise SystemExit("--system is required with --mode worker")
        worker(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
