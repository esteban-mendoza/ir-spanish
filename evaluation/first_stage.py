#!/usr/bin/env python3
"""Measure the query-encoding cost of the extra first-stage retrievers.

The fusion pipeline uses three first-stage retrievers (e5-large, bge-m3 and
jina-v5-small); the rerankers use only jina-v5-small.  The jina-v5 query encoder
is common and cancels in the difference, so the extra first-stage cost of the
fusion is the query encoding of e5-large plus bge-m3, plus their index searches.

This script measures the query-encoding part on the same 500-query sample used
by the efficiency benchmark.  It does not rebuild the 14M-document indexes: the
document embeddings were not retained in the cache, only the final runs.  The
result is therefore a lower bound on the extra first-stage cost, not a full
end-to-end measurement.

Usage:
    python -m evaluation.first_stage --model e5 --batch 1,32
    python -m evaluation.first_stage --model bge --batch 1,32
    python -m evaluation.first_stage --aggregate
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import time
from pathlib import Path

import torch

from utils.observability import log_ram_usage

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
QUERIES_PATH = REPO_ROOT / "results" / "efficiency" / "queries.csv"
OUT_DIR = REPO_ROOT / "results" / "first_stage"

TASK = "Given a web search query, retrieve relevant passages that answer the query"
QUERY_PREFIX = f"Instruct: {TASK}\nQuery: "

MODELS = {
    "e5": {
        "name": "intfloat/multilingual-e5-large-instruct",
        "max_length": 512,
        "prefix": QUERY_PREFIX,
    },
    "bge": {
        "name": "BAAI/bge-m3",
        "max_length": 8192,
        "prefix": "",
    },
}


def _load_queries() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with QUERIES_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return (
        [row for row in rows if row["split"] == "warmup"],
        [row for row in rows if row["split"] == "measure"],
    )


def _rss_mib() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def _p95(values: list[float]) -> float:
    return sorted(values)[max(0, math.ceil(0.95 * len(values)) - 1)]


def measure(model_key: str, batch_sizes: list[int], limit: int, warmup: int) -> None:
    from sentence_transformers import SentenceTransformer

    spec = MODELS[model_key]
    warmup_rows, measure_rows = _load_queries()
    warmup_rows = warmup_rows[:warmup]
    measure_rows = measure_rows[:limit]
    if not measure_rows:
        raise SystemExit("No measure queries")

    log.info("Loading %s ...", spec["name"])
    started = time.perf_counter()
    model = SentenceTransformer(
        spec["name"],
        device="cuda:0",
        model_kwargs={"torch_dtype": torch.float16},
    )
    model.max_seq_length = spec["max_length"]
    load_seconds = time.perf_counter() - started
    log.info("Loaded %s in %.1fs", spec["name"], load_seconds)

    def encode(texts: list[str], batch_size: int):
        prefixed = [spec["prefix"] + text for text in texts]
        return model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    # Warmup outside the timed region.
    if warmup_rows:
        encode([row["query"] for row in warmup_rows], 8)
    torch.cuda.synchronize()

    rows: list[dict] = []
    latencies: list[float] = []
    for row in measure_rows:
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        encode([row["query"]], 1)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        latencies.append(elapsed_ms)
        rows.append(
            {
                "model": model_key,
                "query_id": row["query_id"],
                "latency_ms": f"{elapsed_ms:.6f}",
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / f"{model_key}_batch1.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "query_id", "latency_ms"])
        writer.writeheader()
        writer.writerows(rows)

    batch_qps: dict[str, float] = {}
    for batch_size in batch_sizes:
        if batch_size <= 1:
            batch_qps[str(batch_size)] = len(latencies) / (sum(latencies) / 1000.0)
            continue
        texts = [row["query"] for row in measure_rows]
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        for start in range(0, len(texts), batch_size):
            encode(texts[start : start + batch_size], batch_size)
        torch.cuda.synchronize()
        elapsed_s = (time.perf_counter_ns() - t0) / 1e9
        batch_qps[str(batch_size)] = len(texts) / elapsed_s

    meta = {
        "model": model_key,
        "name": spec["name"],
        "max_length": spec["max_length"],
        "n": len(latencies),
        "load_seconds": load_seconds,
        "median_ms": statistics.median(latencies),
        "p95_ms": _p95(latencies),
        "batch_qps": batch_qps,
        "peak_rss_mib": _rss_mib(),
        "peak_vram_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "peak_vram_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
    }
    meta_path = OUT_DIR / f"{model_key}.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)
    log.info("Wrote %s and %s", raw_path, meta_path)
    log_ram_usage()


def aggregate() -> None:
    metas = {}
    for key in MODELS:
        path = OUT_DIR / f"{key}.json"
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                metas[key] = json.load(handle)
    if not metas:
        raise SystemExit(f"No first-stage measurements in {OUT_DIR}")

    rows = []
    for key, meta in metas.items():
        rows.append(
            {
                "model": key,
                "median_ms": meta["median_ms"],
                "p95_ms": meta["p95_ms"],
                "batch_1_qps": meta["batch_qps"].get("1"),
                "batch_32_qps": meta["batch_qps"].get("32"),
                "peak_rss_mib": meta["peak_rss_mib"],
                "peak_vram_allocated_mib": meta["peak_vram_allocated_mib"],
            }
        )

    extra_ms = None
    if "e5" in metas and "bge" in metas:
        e5_qps = metas["e5"]["batch_qps"].get("32")
        bge_qps = metas["bge"]["batch_qps"].get("32")
        if e5_qps and bge_qps:
            extra_ms = 1000.0 / e5_qps + 1000.0 / bge_qps

    summary = {
        "models": rows,
        "extra_encoding_ms_per_query_batch32": extra_ms,
        "note": (
            "Cota inferior del costo adicional de la primera etapa de la fusión: "
            "codificación de e5-large y bge-m3. No incluye la búsqueda en los índices "
            "ni el costo de mantener tres índices; jina-v5 es común a ambas rutas y se cancela."
        ),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "| Modelo | Mediana (ms) | P95 (ms) | QPS b1 | QPS b32 | RSS pico (MiB) | VRAM pico (MiB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {median_ms:.3f} | {p95_ms:.3f} | {batch_1_qps:.2f} | {batch_32_qps:.2f} | {peak_rss_mib:.0f} | {peak_vram_allocated_mib:.0f} |".format(**row)
        )
    if extra_ms is not None:
        lines.append("")
        lines.append(f"Costo adicional de codificación (e5 + bge, lote 32): **{extra_ms:.2f} ms/consulta**.")
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Wrote first-stage summary (%s)", OUT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--batch", default="1,32")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate()
        return
    if not args.model:
        raise SystemExit("--model is required unless --aggregate is used")
    batches = sorted({int(value) for value in args.batch.split(",") if value.strip()})
    measure(args.model, batches, args.n, args.warmup)


if __name__ == "__main__":
    main()
