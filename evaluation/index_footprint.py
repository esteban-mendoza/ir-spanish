#!/usr/bin/env python3
"""Measure the memory and search cost of one flat dense index.

The document embeddings and the FAISS indexes were not retained in the cache,
only the final runs.  This script builds a *synthetic* ``IndexFlatIP`` with the
same number of vectors and dimensions as the real corpus (14 047 759 x 1024)
and measures:

- host RSS before and after building the index;
- time to build it;
- CPU search latency for top-100;
- GPU memory and the time to copy it to the two GPUs with float16 sharding;
- GPU search latency for top-100.

The vectors are random, so this is a proxy for the real index.  Flat inner-product
search performs the same number of operations regardless of the vector values;
what changes is retrieval quality, which is out of scope here.  The result is a
measurement of one index at a time, not of the three indexes together.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import statistics
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

N_DOCS = 14_047_759
DIM = 1024
TOP_K = 100
N_QUERIES = 500
SEED = 42
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "index_footprint"


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


def gpu_used_mib() -> list[int]:
    import subprocess

    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [int(line.strip()) for line in output.splitlines() if line.strip()]


def build_and_measure(n_docs: int, dim: int, n_queries: int, cpu_queries: int, chunk: int, run_gpu: bool) -> dict:
    import faiss  # noqa: F401  (imported here to keep the CLI light)

    faiss.omp_set_num_threads(1)
    rng = np.random.default_rng(SEED)

    rss_before = rss_mib()
    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(dim)
    added = 0
    while added < n_docs:
        size = min(chunk, n_docs - added)
        vectors = rng.standard_normal((size, dim), dtype=np.float32)
        index.add(vectors)
        added += size
        del vectors
        if added % (chunk * 10) == 0:
            log.info("Added %d/%d vectors (RSS %.1f GiB)", added, n_docs, rss_mib() / 1024)
    build_s = time.perf_counter() - t0
    rss_after = rss_mib()
    log.info("Built index: %d x %d in %.1fs (RSS %.1f GiB)", index.ntotal, index.d, build_s, rss_after / 1024)

    queries = rng.standard_normal((n_queries, dim), dtype=np.float32)
    cpu_subset = queries[: min(cpu_queries, n_queries)]
    cpu_latencies: list[float] = []
    t0 = time.perf_counter()
    for query in cpu_subset:
        s = time.perf_counter_ns()
        index.search(query.reshape(1, -1), TOP_K)
        cpu_latencies.append((time.perf_counter_ns() - s) / 1e6)
    cpu_batch1_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for start in range(0, len(cpu_subset), 32):
        index.search(cpu_subset[start : start + 32], TOP_K)
    cpu_batch32_s = time.perf_counter() - t0
    log.info("CPU search: median %.2f ms, p95 %.2f ms", statistics.median(cpu_latencies), p95(cpu_latencies))

    result = {
        "n_docs": int(index.ntotal),
        "dim": int(index.d),
        "bytes_per_vector": int(index.d * 4),
        "theoretical_bytes": int(index.ntotal * index.d * 4),
        "rss_before_mib": rss_before,
        "rss_after_build_mib": rss_after,
        "rss_delta_mib": rss_after - rss_before,
        "build_seconds": build_s,
        "cpu_median_ms": statistics.median(cpu_latencies),
        "cpu_p95_ms": p95(cpu_latencies),
        "cpu_batch1_qps": len(cpu_subset) / cpu_batch1_s,
        "cpu_batch32_qps": len(cpu_subset) / cpu_batch32_s,
        "gpu": None,
    }

    if run_gpu:
        try:
            import torch

            before = gpu_used_mib()
            t0 = time.perf_counter()
            options = faiss.GpuMultipleClonerOptions()
            options.shard = True
            options.useFloat16 = True
            gpu_index = faiss.index_cpu_to_all_gpus(index, co=options)
            torch.cuda.synchronize()
            gpu_build_s = time.perf_counter() - t0
            after = gpu_used_mib()
            gpu_latencies: list[float] = []
            for query in queries:
                torch.cuda.synchronize()
                s = time.perf_counter_ns()
                gpu_index.search(query.reshape(1, -1), TOP_K)
                torch.cuda.synchronize()
                gpu_latencies.append((time.perf_counter_ns() - s) / 1e6)
            t0 = time.perf_counter()
            for start in range(0, n_queries, 32):
                gpu_index.search(queries[start : start + 32], TOP_K)
            torch.cuda.synchronize()
            gpu_batch32_s = time.perf_counter() - t0
            result["gpu"] = {
                "used_before_mib": before,
                "used_after_mib": after,
                "delta_mib_per_gpu": [a - b for a, b in zip(after, before)],
                "build_seconds": gpu_build_s,
                "median_ms": statistics.median(gpu_latencies),
                "p95_ms": p95(gpu_latencies),
                "batch1_qps": n_queries / (sum(gpu_latencies) / 1000.0),
                "batch32_qps": n_queries / gpu_batch32_s,
            }
            log.info(
                "GPU search: median %.2f ms, p95 %.2f ms; memory delta %s MiB",
                result["gpu"]["median_ms"],
                result["gpu"]["p95_ms"],
                result["gpu"]["delta_mib_per_gpu"],
            )
            del gpu_index
            gc.collect()
        except Exception as exc:  # GPU is optional; report the failure
            result["gpu"] = {"error": repr(exc)}
            log.exception("GPU measurement failed")

    del index
    gc.collect()
    return result


def write_summary(result: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    gib = result["theoretical_bytes"] / (1024**3)
    lines = [
        "# Huella de un índice plano",
        "",
        f"- Documentos: {result['n_docs']}",
        f"- Dimensión: {result['dim']}",
        f"- Tamaño teórico float32: {result['theoretical_bytes'] / 1e9:.2f} GB ({gib:.2f} GiB)",
        f"- RSS antes / después de construir: {result['rss_before_mib']:.0f} / {result['rss_after_build_mib']:.0f} MiB",
        f"- Construcción: {result['build_seconds']:.1f} s",
        f"- Búsqueda CPU: mediana {result['cpu_median_ms']:.2f} ms, p95 {result['cpu_p95_ms']:.2f} ms",
        f"- CPU qps lote 1 / 32: {result['cpu_batch1_qps']:.1f} / {result['cpu_batch32_qps']:.1f}",
    ]
    if result.get("gpu"):
        gpu = result["gpu"]
        if "error" in gpu:
            lines.append(f"- GPU: error ({gpu['error']})")
        else:
            lines += [
                f"- GPU memoria antes / después: {gpu['used_before_mib']} / {gpu['used_after_mib']} MiB",
                f"- GPU delta por tarjeta: {gpu['delta_mib_per_gpu']} MiB",
                f"- Copia a GPU: {gpu['build_seconds']:.1f} s",
                f"- Búsqueda GPU: mediana {gpu['median_ms']:.2f} ms, p95 {gpu['p95_ms']:.2f} ms",
                f"- GPU qps lote 1 / 32: {gpu['batch1_qps']:.1f} / {gpu['batch32_qps']:.1f}",
            ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Wrote %s", OUT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=N_DOCS)
    parser.add_argument("--dim", type=int, default=DIM)
    parser.add_argument("--queries", type=int, default=N_QUERIES)
    parser.add_argument("--cpu-queries", type=int, default=10)
    parser.add_argument("--chunk", type=int, default=100_000)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(vars(args), indent=2))
        return
    result = build_and_measure(args.n, args.dim, args.queries, args.cpu_queries, args.chunk, not args.no_gpu)
    write_summary(result)


if __name__ == "__main__":
    main()
