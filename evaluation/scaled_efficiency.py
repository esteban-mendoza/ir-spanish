#!/usr/bin/env python3
"""Scaled end-to-end efficiency comparison with synthetic flat indexes.

At full scale the three flat indexes of the fusion do not fit in the server.
This script builds three *proxy* indexes over the first N documents of the
corpus and compares:

- F3: search in three indexes + CombMNZ fusion;
- R-bge / R-jina / R-colbert: search in the jina proxy index + reranker.

The vectors are random, so the result measures memory, search and fusion /
reranking cost, not effectiveness.  The same N, the same documents, the same
query sample and the same top-100 are used for all routes.

Usage:
    python -m evaluation.scaled_efficiency --n 2000000
    python -m evaluation.scaled_efficiency --n 4000000 --rerank
    python -m evaluation.scaled_efficiency --aggregate
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import statistics
import time
from pathlib import Path

import numpy as np

os_environ = __import__("os").environ
os_environ.setdefault("OMP_NUM_THREADS", "1")
os_environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os_environ.setdefault("MKL_NUM_THREADS", "1")
os_environ.setdefault("NUMBA_NUM_THREADS", "1")

import torch
from ranx import Run, fuse

from utils import data
from evaluation.efficiency import BgeSystem, ColbertSystem, JinaV3System
from evaluation.mem_trace import MemoryTrace, summarize as summarize_trace

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
QUERIES_PATH = REPO_ROOT / "results" / "efficiency" / "queries.csv"
OUT_DIR = REPO_ROOT / "results" / "scaled_efficiency"
TOP_K = 100
N_QUERIES = 100
N_REPS = 3
SEED = 42
DIM = 1024
ALIASES = ("e5", "bge", "jina")


def rss_mib() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def p95(values: list[float]) -> float:
    return sorted(values)[max(0, math.ceil(0.95 * len(values)) - 1)]


def load_queries() -> list[dict[str, str]]:
    with QUERIES_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row["split"] == "measure"][:N_QUERIES]


def load_doc_ids(n_docs: int) -> list[str]:
    doc_ids, doc_texts, _ = data.load_corpus(1)
    if len(doc_ids) < n_docs:
        raise SystemExit(f"Corpus has {len(doc_ids)} docs, requested {n_docs}")
    ids = doc_ids[:n_docs]
    # Keep texts only for the documents that can appear as candidates.
    global _DOC_TEXTS
    _DOC_TEXTS = dict(zip(doc_ids, doc_texts))
    del doc_ids, doc_texts
    gc.collect()
    log.info("Loaded %d document ids", len(ids))
    return ids


_DOC_TEXTS: dict[str, str] = {}


def build_proxy_indexes(n_docs: int, doc_ids: list[str], seed: int, trace: MemoryTrace | None = None) -> tuple[list, float, list[int]]:
    import faiss

    faiss.omp_set_num_threads(1)
    rng = np.random.default_rng(seed)
    gpu_before = gpu_used_mib()
    started = time.perf_counter()
    indexes = []
    chunk = 100_000
    for alias in ALIASES:
        if trace is not None:
            trace.set_phase(f"build_{alias}")
        log.info("Building %s proxy with %d docs ...", alias, n_docs)
        cpu_index = faiss.IndexFlatIP(DIM)
        added = 0
        while added < n_docs:
            size = min(chunk, n_docs - added)
            vectors = rng.standard_normal((size, DIM), dtype=np.float32)
            cpu_index.add(vectors)
            added += size
            del vectors
        options = faiss.GpuMultipleClonerOptions()
        options.shard = True
        options.useFloat16 = True
        gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co=options)
        if trace is not None:
            trace.set_phase(f"build_{alias}_gpu")
        indexes.append(gpu_index)
        del cpu_index
        gc.collect()
        log.info("  %s on GPU; RSS %.1f GiB", alias, rss_mib() / 1024)
    torch.cuda.synchronize()
    build_s = time.perf_counter() - started
    gpu_after = gpu_used_mib()
    return indexes, build_s, [a - b for a, b in zip(gpu_after, gpu_before)]


def gpu_used_mib() -> list[int]:
    import subprocess

    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return [int(line.strip()) for line in output.splitlines() if line.strip()]


def search_and_fuse(indexes: list, doc_ids: list[str], qids: list[str], queries: np.ndarray,
                    trace: MemoryTrace | None = None) -> dict:
    search_times: list[float] = []
    fusion_times: list[float] = []
    candidate_docs: dict[str, list[str]] = {}
    for qid, query in zip(qids, queries):
        if trace is not None:
            trace.set_phase("search_fusion")
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        results = [index.search(query.reshape(1, -1), TOP_K) for index in indexes]
        torch.cuda.synchronize()
        search_times.append((time.perf_counter_ns() - t0) / 1e6)

        runs = []
        for scores, indices in results:
            local = {
                doc_ids[int(idx)]: float(score)
                for score, idx in zip(scores[0], indices[0])
                if idx >= 0
            }
            runs.append(Run({qid: local}, name=qid))
        if qid not in candidate_docs:
            jina_scores, jina_indices = results[2]
            candidate_docs[qid] = [
                doc_ids[int(idx)] for idx in jina_indices[0] if idx >= 0
            ]
        t0 = time.perf_counter_ns()
        fused = fuse(runs, norm="min-max", method="mnz", params={})
        _ = fused.to_dict()[qid]
        fusion_times.append((time.perf_counter_ns() - t0) / 1e6)
    return {
        "search_median_ms": statistics.median(search_times),
        "search_p95_ms": p95(search_times),
        "search_qps": len(qids) / (sum(search_times) / 1000.0),
        "fusion_median_ms": statistics.median(fusion_times),
        "fusion_p95_ms": p95(fusion_times),
        "fusion_qps": len(qids) / (sum(fusion_times) / 1000.0),
        "total_median_ms": statistics.median([a + b for a, b in zip(search_times, fusion_times)]),
        "total_p95_ms": p95([a + b for a, b in zip(search_times, fusion_times)]),
        "candidate_docs": candidate_docs,
    }


def run_rerankers(qids: list[str], query_map: dict[str, str], candidates: dict[str, list[str]],
                  trace: MemoryTrace | None = None) -> dict:
    needed = {doc_id for docs in candidates.values() for doc_id in docs}
    doc_lookup = {doc_id: _DOC_TEXTS.get(doc_id, "") for doc_id in needed}
    results = {}
    for name, cls in (("bge", BgeSystem), ("jina-v3", JinaV3System), ("colbert", ColbertSystem)):
        log.info("Loading reranker %s ...", name)
        if trace is not None:
            trace.set_phase(f"rerank_{name}_load")
        system = cls(qids, query_map, candidates, doc_lookup, "cuda:0")
        for qid in qids[:5]:
            system.score_one(qid)
        per_rep = []
        for rep in range(1, N_REPS + 1):
            if trace is not None:
                trace.set_phase(f"rerank_{name}")
            lats = []
            for qid in qids:
                _, latency = system.score_one(qid)
                lats.append(latency)
            per_rep.append(
                {
                    "rep": rep,
                    "median_ms": statistics.median(lats),
                    "p95_ms": p95(lats),
                    "qps": len(lats) / (sum(lats) / 1000.0),
                }
            )
            log.info("%s rep %d: median %.2f ms", name, rep, per_rep[-1]["median_ms"])
        results[name] = {
            "load_seconds": getattr(system, "load_seconds", None),
            "reps": per_rep,
            "peak_vram_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        }
        system.close()
        del system
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return results


def run_scale(n_docs: int, do_rerank: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_queries()
    qids = [row["query_id"] for row in rows]
    query_map = {row["query_id"]: row["query"] for row in rows}
    doc_ids = load_doc_ids(n_docs)
    trace = MemoryTrace(OUT_DIR / f"memtrace_{n_docs}.csv", interval_s=1.0)
    trace.start()
    indexes, build_s, gpu_delta = build_proxy_indexes(n_docs, doc_ids, SEED, trace)
    rss = rss_mib()

    rng = np.random.default_rng(SEED + 1)
    queries = rng.standard_normal((len(qids), DIM), dtype=np.float32)
    reps = []
    candidate_docs = None
    for rep in range(1, N_REPS + 1):
        result = search_and_fuse(indexes, doc_ids, qids, queries, trace)
        if candidate_docs is None:
            candidate_docs = result.pop("candidate_docs")
        else:
            result.pop("candidate_docs")
        reps.append(result)
        log.info("N=%d rep %d search+fusion %.2f ms", n_docs, rep, result["total_median_ms"])

    payload = {
        "n_docs": n_docs,
        "n_queries": len(qids),
        "n_reps": N_REPS,
        "build_seconds": build_s,
        "rss_mib": rss,
        "gpu_delta_mib": gpu_delta,
        "search_fusion_reps": reps,
    }
    if candidate_docs is not None:
        (OUT_DIR / f"candidates_{n_docs}.json").write_text(
            json.dumps(candidate_docs, ensure_ascii=False), encoding="utf-8"
        )
    (OUT_DIR / f"scale_{n_docs}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if do_rerank:
        if trace is not None:
            trace.set_phase("rerank_release_indexes")
        del indexes
        gc.collect()
        torch.cuda.empty_cache()
        payload["rerankers"] = run_rerankers(qids, query_map, candidate_docs or {}, trace)
    trace.stop()
    payload["memory_trace"] = summarize_trace(OUT_DIR / f"memtrace_{n_docs}.csv")
    (OUT_DIR / f"scale_{n_docs}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log.info("Wrote %s", OUT_DIR / f"scale_{n_docs}.json")


def aggregate() -> None:
    results = {}
    for path in sorted(OUT_DIR.glob("scale_*.json")):
        data_json = json.loads(path.read_text(encoding="utf-8"))
        results[data_json["n_docs"]] = data_json
    if not results:
        raise SystemExit(f"No scale results in {OUT_DIR}")
    (OUT_DIR / "summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log.info("Wrote %s", OUT_DIR / "summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, choices=[2_000_000, 4_000_000])
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate()
        return
    if not args.n:
        raise SystemExit("--n is required")
    run_scale(args.n, args.rerank)


if __name__ == "__main__":
    main()
