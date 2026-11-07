#!/usr/bin/env python3
"""Costo de codificar documentos e índices: un modelo frente a tres.

El banco de eficiencia mide la segunda etapa y la codificación de consultas;
la estimación de índices usa vectores sintéticos.  Este script cierra el hueco
que queda: cuánto tarda codificar documentos reales y cuánto ocupa un índice
construido con esos vectores, para la ruta de un modelo (`jina-v5-small`) y la
de tres (`e5-large` + `bge-m3` + `jina-v5-small`).

La muestra es estratificada por deciles de longitud y se usa la configuración
de producción de cada modelo (máximo nativo, lote de documentos y prompts).
Las horas a corpus completo son una extrapolación lineal; la búsqueda en los
índices no se vuelve a medir.

Uso:
    python -m evaluation.encoding_cost --mode sample
    python -m evaluation.encoding_cost --mode encode --model e5 --reps 3
    python -m evaluation.encoding_cost --mode aggregate
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from evaluation.first_stage import MODELS as QUERY_MODELS
from utils import data
from utils.encoding import TOKENS_PER_WORD, _adaptive_batch_size, _snap_to_band
from utils.observability import log_ram_usage

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "encoding_cost"
QUERIES_PATH = REPO_ROOT / "results" / "efficiency" / "queries.csv"
SAMPLE_CACHE_DIR = Path.home() / ".cache" / "messirve_embeddings" / "efficiency"

CORPUS_DOCS = 14_047_759
DIM = 1024
N_DOCS = 20_000
N_STRATA = 10
PER_STRATUM = N_DOCS // N_STRATA
SEED = 42
N_REPS = 3
N_WARMUP_DOCS = 1_000
N_QUERY_WARMUP = 50
N_QUERY_MEASURE = 500

# Lotes de documentos de producción (baselines/*.py).
DOC_BATCH = {"e5": 128, "bge": 8, "jina": 8}

# Prefijo o prompt de documento.  e5 y bge codifican el texto tal cual; jina
# distingue consulta y documento con sus prompts integrados.
DOC_PROMPT = {"e5": None, "bge": None, "jina": "document"}
DOC_PREFIX = {"e5": "", "bge": "", "jina": ""}

ALIASES = ("e5", "bge", "jina")
LABELS = {
    "e5": "e5-large-instruct",
    "bge": "bge-m3",
    "jina": "jina-v5-small",
}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def p95(values: list[float]) -> float:
    if not values:
        raise ValueError("empty values")
    return sorted(values)[max(0, math.ceil(0.95 * len(values)) - 1)]


def rss_mib() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(_cuda_index(device))


def _cuda_index(device: str) -> int:
    if not device.startswith("cuda"):
        return 0
    _, _, index = device.partition(":")
    return int(index) if index else 0


def _spec(key: str) -> dict:
    spec = dict(QUERY_MODELS[key])
    spec["doc_batch"] = DOC_BATCH[key]
    spec["doc_prompt"] = DOC_PROMPT[key]
    spec["doc_prefix"] = DOC_PREFIX[key]
    return spec


def _model_kwargs(spec: dict) -> dict:
    dtype = torch.bfloat16 if spec.get("torch_dtype") == "bfloat16" else torch.float16
    kwargs = {"torch_dtype": dtype}
    kwargs.update(spec.get("model_kwargs", {}))
    return kwargs


def _load_model(key: str, device: str) -> SentenceTransformer:
    spec = _spec(key)
    tokenizer_kwargs = {}
    if key == "jina":
        tokenizer_kwargs["padding_side"] = "left"
    extra = {"tokenizer_kwargs": tokenizer_kwargs} if tokenizer_kwargs else {}
    model = SentenceTransformer(
        spec["name"],
        device=device,
        model_kwargs=_model_kwargs(spec),
        **extra,
    )
    model.max_seq_length = spec["max_length"]
    return model


def _encode_kwargs(spec: dict, is_query: bool) -> dict:
    prompt = spec.get("prompt_name") if is_query else spec["doc_prompt"]
    return {"prompt_name": prompt} if prompt else {}


# ---------------------------------------------------------------------------
# Muestra
# ---------------------------------------------------------------------------
def _sample_path(out_dir: Path) -> Path:
    return out_dir / "sample.json"


def _text_cache_path(n_docs: int, seed: int) -> Path:
    return SAMPLE_CACHE_DIR / f"encoding_sample_{n_docs}_{seed}.jsonl"


def build_sample(out_dir: Path, n_docs: int, seed: int) -> None:
    if n_docs % N_STRATA:
        raise SystemExit("n_docs debe ser divisible entre 10")
    per_stratum = n_docs // N_STRATA
    log.info("Cargando corpus para muestrear %d documentos ...", n_docs)
    doc_ids, doc_texts, word_counts = data.load_corpus(16)

    wc = np.asarray(word_counts, dtype=np.int64)
    nonempty = np.asarray([bool(text and text.strip()) for text in doc_texts])
    valid = np.flatnonzero(nonempty)
    if len(valid) < n_docs:
        raise SystemExit(f"No hay suficientes documentos no vacíos: {len(valid)}")

    edges = np.quantile(wc[valid], np.linspace(0.0, 1.0, N_STRATA + 1))
    strata = np.clip(np.searchsorted(edges[1:-1], wc, side="right"), 0, N_STRATA - 1)

    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for stratum in range(N_STRATA):
        pool = valid[strata[valid] == stratum]
        take = min(per_stratum, len(pool))
        if take:
            chosen.extend(int(value) for value in rng.choice(pool, size=take, replace=False))
        if take < per_stratum:
            log.warning("Estrato %d solo tiene %d documentos", stratum, take)

    chosen_set = set(chosen)
    if len(chosen) < n_docs:
        remaining = np.asarray([value for value in valid if int(value) not in chosen_set])
        extra = rng.choice(remaining, size=n_docs - len(chosen), replace=False)
        chosen.extend(int(value) for value in extra)

    chosen = np.asarray(chosen, dtype=np.int64)
    chosen = chosen[rng.permutation(len(chosen))]

    cache_path = _text_cache_path(n_docs, seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        for index in chosen:
            handle.write(
                json.dumps(
                    {
                        "doc_id": str(doc_ids[index]),
                        "stratum": int(strata[index]),
                        "word_count": int(wc[index]),
                        "text": doc_texts[index],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    id_hash = hashlib.sha256(
        "\n".join(sorted(str(doc_ids[index]) for index in chosen)).encode()
    ).hexdigest()
    meta = {
        "n_docs": int(len(chosen)),
        "n_requested": int(n_docs),
        "seed": int(seed),
        "corpus_docs": CORPUS_DOCS,
        "n_strata": N_STRATA,
        "per_stratum_requested": int(per_stratum),
        "stratum_counts": {
            str(stratum): int(sum(1 for index in chosen if int(strata[index]) == stratum))
            for stratum in range(N_STRATA)
        },
        "decile_edges": [float(value) for value in edges],
        "id_sha256": id_hash,
        "text_cache": str(cache_path),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _sample_path(out_dir).write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info("Muestra guardada: %d documentos; hash %s", len(chosen), id_hash[:12])
    del doc_ids, doc_texts, word_counts
    log_ram_usage()


def read_sample(out_dir: Path) -> tuple[dict, list[dict]]:
    meta_path = _sample_path(out_dir)
    if not meta_path.exists():
        raise SystemExit(f"Falta {meta_path}; corre --mode sample")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cache = Path(meta["text_cache"])
    if not cache.exists():
        raise SystemExit(f"Falta la caché de textos {cache}; corre --mode sample")
    rows = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines()]
    if len(rows) != meta["n_docs"]:
        raise SystemExit(f"La caché tiene {len(rows)} documentos, se esperaban {meta['n_docs']}")
    return meta, rows


# ---------------------------------------------------------------------------
# Medición
# ---------------------------------------------------------------------------
def _token_lengths(model, texts: list[str], max_length: int, extra: int = 0) -> list[tuple[int, int]]:
    """Longitud sin truncar y longitud efectiva (tras el tope del modelo)."""
    tokenizer = model.tokenizer
    lengths: list[tuple[int, int]] = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        raw = len(ids) + extra
        lengths.append((raw, min(max_length, raw)))
    return lengths


def _encode_batches(
    model,
    texts: list[str],
    batch_size: int,
    spec: dict,
    *,
    is_query: bool,
    device: str,
) -> tuple[list[dict], np.ndarray]:
    """Encode in explicit batches, returning timings and the embeddings."""
    kwargs = _encode_kwargs(spec, is_query)
    rows: list[dict] = []
    embeddings: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        sync(device)
        started = time.perf_counter_ns()
        vectors = model.encode(
            batch,
            batch_size=len(batch),
            normalize_embeddings=True,
            show_progress_bar=False,
            **kwargs,
        )
        sync(device)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        embeddings.append(np.asarray(vectors, dtype=np.float32))
        rows.append(
            {
                "batch_index": len(rows),
                "n_docs": len(batch),
                "elapsed_ms": elapsed_ms,
            }
        )
    return rows, np.vstack(embeddings)


def _bootstrap_rate_ci(batches_per_rep: list[list[dict]], n_boot: int = 10_000) -> tuple[float, float]:
    """IC 95 % de documentos/s remuestreando lotes dentro de cada repetición."""
    if not batches_per_rep or not batches_per_rep[0]:
        return 0.0, 0.0
    rng = np.random.default_rng(SEED)
    lows, highs = [], []
    for batches in batches_per_rep:
        docs = np.asarray([row["n_docs"] for row in batches], dtype=np.float64)
        seconds = np.asarray([row["elapsed_ms"] for row in batches], dtype=np.float64) / 1000.0
        indices = rng.integers(0, len(docs), size=(n_boot, len(docs)))
        rates = docs[indices].sum(axis=1) / seconds[indices].sum(axis=1)
        low, high = np.percentile(rates, [2.5, 97.5])
        lows.append(float(low))
        highs.append(float(high))
    return float(statistics.median(lows)), float(statistics.median(highs))


def _measure_documents(
    model,
    rows: list[dict],
    spec: dict,
    device: str,
    reps: int,
    out_dir: Path,
    key: str,
) -> dict:
    # La producción ordena por longitud y ajusta la longitud de secuencia y el
    # lote por bandas (utils/encoding.py). Replicamos esa decisión para medir
    # el costo real de construir el índice, no el de un lote fijo pesimista.
    rows = sorted(rows, key=lambda row: row["word_count"])
    texts = [row["text"] for row in rows]
    max_words = max(row["word_count"] for row in rows)
    token_estimate = int(max_words * TOKENS_PER_WORD)
    seq_length = _snap_to_band(token_estimate, spec["max_length"])
    batch_size = _adaptive_batch_size(seq_length, spec["doc_batch"])
    model.max_seq_length = seq_length
    token_lengths = _token_lengths(model, texts, seq_length, extra=1 if key == "jina" else 0)
    corpus_tokens = [effective for _raw, effective in token_lengths]
    truncated = sum(1 for raw, _effective in token_lengths if raw > seq_length)

    warmup = min(N_WARMUP_DOCS, len(texts))
    log.info("[%s] Calentando %d documentos ...", key, warmup)
    _encode_batches(model, texts[:warmup], batch_size, spec, is_query=False, device=device)

    torch.cuda.reset_peak_memory_stats(_cuda_index(device))
    repetition_rows: list[dict] = []
    repetition_batches: list[list[dict]] = []
    last_embeddings = None
    batches_dir = out_dir / "raw"
    batches_dir.mkdir(parents=True, exist_ok=True)

    for rep in range(1, reps + 1):
        log.info("[%s] Repetición %d/%d: %d documentos", key, rep, reps, len(texts))
        batch_rows, last_embeddings = _encode_batches(
            model, texts, batch_size, spec, is_query=False, device=device
        )
        repetition_batches.append(batch_rows)
        seconds = sum(row["elapsed_ms"] for row in batch_rows) / 1000.0
        tokens = sum(
            sum(corpus_tokens[row["batch_index"] * batch_size : (row["batch_index"] + 1) * batch_size])
            for row in batch_rows
        )
        repetition_rows.append(
            {
                "rep": rep,
                "seconds": seconds,
                "docs": len(texts),
                "tokens": int(tokens),
                "docs_per_s": len(texts) / seconds if seconds else 0.0,
                "tokens_per_s": tokens / seconds if seconds else 0.0,
                "ms_per_doc": 1000.0 * seconds / len(texts),
            }
        )
        with (batches_dir / f"{key}_rep{rep}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["batch_index", "n_docs", "elapsed_ms"])
            writer.writeheader()
            writer.writerows(batch_rows)

    allocated = torch.cuda.max_memory_allocated(_cuda_index(device)) / (1024**2)
    reserved = torch.cuda.max_memory_reserved(_cuda_index(device)) / (1024**2)
    med_docs_per_s = statistics.median(row["docs_per_s"] for row in repetition_rows)
    med_seconds = statistics.median(row["seconds"] for row in repetition_rows)
    ci_low, ci_high = _bootstrap_rate_ci(repetition_batches)
    return {
        "model": key,
        "label": LABELS[key],
        "name": spec["name"],
        "max_length": int(spec["max_length"]),
        "seq_length": int(seq_length),
        "max_words": int(max_words),
        "token_estimate": int(token_estimate),
        "batch_size": int(batch_size),
        "n_docs": len(texts),
        "tokens": int(sum(corpus_tokens)),
        "truncated_docs": int(truncated),
        "truncated_fraction": truncated / len(texts) if texts else 0.0,
        "reps": repetition_rows,
        "median_seconds": med_seconds,
        "median_docs_per_s": med_docs_per_s,
        "docs_per_s_ci_low": ci_low,
        "docs_per_s_ci_high": ci_high,
        "median_tokens_per_s": statistics.median(row["tokens_per_s"] for row in repetition_rows),
        "median_ms_per_doc": 1000.0 / med_docs_per_s if med_docs_per_s else None,
        "peak_vram_allocated_mib": allocated,
        "peak_vram_reserved_mib": reserved,
        "peak_rss_mib": rss_mib(),
        "embeddings": last_embeddings,
    }


def _measure_index(embeddings: np.ndarray, key: str, out_dir: Path) -> dict:
    import faiss

    embeddings = np.asarray(embeddings, dtype=np.float32)
    n_docs, dim = embeddings.shape
    raw_bytes = int(embeddings.nbytes)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    handle = tempfile.NamedTemporaryFile(suffix=".faiss", delete=False)
    handle.close()
    path = Path(handle.name)
    try:
        faiss.write_index(index, str(path))
        file_bytes = int(path.stat().st_size)
    finally:
        path.unlink(missing_ok=True)
    del index
    index_dir = out_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "model": key,
        "n_docs": int(n_docs),
        "dim": int(dim),
        "raw_bytes": raw_bytes,
        "bytes_per_vector": raw_bytes / n_docs,
        "file_bytes": file_bytes,
        "overhead_bytes": file_bytes - raw_bytes,
        "overhead_per_vector": (file_bytes - raw_bytes) / n_docs,
    }
    (index_dir / f"{key}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log.info("[%s] Índice de muestra: %.2f MiB; sobrecarga %.1f KiB", key, file_bytes / 2**20, (file_bytes - raw_bytes) / 1024)
    return result


def _measure_queries(model, spec: dict, device: str, out_dir: Path, key: str) -> dict:
    with QUERIES_PATH.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    warmup = [row for row in all_rows if row["split"] == "warmup"][:N_QUERY_WARMUP]
    measure = [row for row in all_rows if row["split"] == "measure"][:N_QUERY_MEASURE]
    if not measure:
        raise SystemExit(f"No hay consultas de medida en {QUERIES_PATH}")

    def texts(rows):
        prefix = spec["prefix"]
        return [prefix + row["query"] for row in rows]

    model.max_seq_length = spec["max_length"]
    _encode_batches(model, texts(warmup), 8, spec, is_query=True, device=device)

    kwargs = _encode_kwargs(spec, True)
    latencies = []
    for row in measure:
        sync(device)
        started = time.perf_counter_ns()
        model.encode(
            [spec["prefix"] + row["query"]],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
            **kwargs,
        )
        sync(device)
        latencies.append((time.perf_counter_ns() - started) / 1e6)

    batch_texts = texts(measure)
    sync(device)
    started = time.perf_counter_ns()
    model.encode(
        batch_texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
        **kwargs,
    )
    sync(device)
    batch32_seconds = (time.perf_counter_ns() - started) / 1e9

    result = {
        "model": key,
        "label": LABELS[key],
        "max_length": int(spec["max_length"]),
        "n": len(latencies),
        "median_ms": statistics.median(latencies),
        "p95_ms": p95(latencies),
        "batch_1_qps": len(latencies) / (sum(latencies) / 1000.0),
        "batch_32_qps": len(measure) / batch32_seconds,
    }
    query_dir = out_dir / "query"
    query_dir.mkdir(parents=True, exist_ok=True)
    (query_dir / f"{key}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log.info("[%s] Consultas: mediana %.2f ms, P95 %.2f ms, b32 %.1f qps", key, result["median_ms"], result["p95_ms"], result["batch_32_qps"])
    return result


def run_model(out_dir: Path, key: str, reps: int, device: str) -> None:
    spec = _spec(key)
    _meta, rows = read_sample(out_dir)
    model = _load_model(key, device)
    try:
        result = _measure_documents(model, rows, spec, device, reps, out_dir, key)
        embeddings = result.pop("embeddings")
        index = _measure_index(embeddings, key, out_dir)
        result["index"] = index
        query = _measure_queries(model, spec, device, out_dir, key)
        result["query"] = query
        docs_dir = out_dir / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / f"{key}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    finally:
        del model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Agregación
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _full_corpus_hours(docs_per_s: float) -> float:
    return CORPUS_DOCS / docs_per_s / 3600.0 if docs_per_s else float("inf")


def aggregate(out_dir: Path) -> None:
    meta = _read_json(_sample_path(out_dir))
    if meta is None:
        raise SystemExit(f"Falta {_sample_path(out_dir)}")
    models = {key: _read_json(out_dir / "docs" / f"{key}.json") for key in ALIASES}
    missing = [key for key, value in models.items() if value is None]
    if missing:
        raise SystemExit(f"Faltan mediciones de documentos: {missing}")

    rows = []
    for key in ALIASES:
        model = models[key]
        index = model["index"]
        bytes_per_vector = index["bytes_per_vector"]
        rows.append(
            {
                "model": key,
                "label": model["label"],
                "max_length": model["max_length"],
                "seq_length": model["seq_length"],
                "batch_size": model["batch_size"],
                "docs_per_s": model["median_docs_per_s"],
                "docs_per_s_ci_low": model.get("docs_per_s_ci_low", model["median_docs_per_s"]),
                "docs_per_s_ci_high": model.get("docs_per_s_ci_high", model["median_docs_per_s"]),
                "tokens_per_s": model["median_tokens_per_s"],
                "ms_per_doc": model["median_ms_per_doc"],
                "full_hours": _full_corpus_hours(model["median_docs_per_s"]),
                "truncated_fraction": model["truncated_fraction"],
                "peak_vram_mib": model["peak_vram_allocated_mib"],
                "bytes_per_vector": bytes_per_vector,
                "one_index_gib": CORPUS_DOCS * bytes_per_vector / 2**30,
            }
        )

    by_key = {row["model"]: row for row in rows}
    one = by_key["jina"]
    three = {
        "docs_per_s": None,
        "ms_per_doc": sum(row["ms_per_doc"] for row in rows),
        "full_hours": sum(row["full_hours"] for row in rows),
        "one_index_gib": sum(row["one_index_gib"] for row in rows),
        "peak_vram_mib": sum(row["peak_vram_mib"] for row in rows),
    }
    three["docs_per_s"] = 1000.0 / three["ms_per_doc"]

    query = {}
    for key in ALIASES:
        value = _read_json(out_dir / "query" / f"{key}.json")
        if value is None:
            raise SystemExit(f"Falta la medición de consultas para {key}")
        query[key] = value

    q_one = query["jina"]
    q_three = {
        "median_ms": sum(query[key]["median_ms"] for key in ALIASES),
        "p95_ms": sum(query[key]["p95_ms"] for key in ALIASES),
        "batch_32_qps": 1.0 / sum(1.0 / query[key]["batch_32_qps"] for key in ALIASES),
    }
    q_three["batch_32_ms_per_query"] = 1000.0 / q_three["batch_32_qps"]
    q_one["batch_32_ms_per_query"] = 1000.0 / q_one["batch_32_qps"]

    summary = {
        "sample": meta,
        "models": rows,
        "route_one": {
            "label": "1 modelo: jina-v5-small",
            "full_hours": one["full_hours"],
            "one_index_gib": one["one_index_gib"],
            "query_median_ms": q_one["median_ms"],
            "query_p95_ms": q_one["p95_ms"],
            "query_batch32_ms": q_one["batch_32_ms_per_query"],
        },
        "route_three": {
            "label": "3 modelos: e5-large + bge-m3 + jina-v5-small",
            "full_hours": three["full_hours"],
            "three_index_gib": three["one_index_gib"],
            "query_median_ms": q_three["median_ms"],
            "query_p95_ms": q_three["p95_ms"],
            "query_batch32_ms": q_three["batch_32_ms_per_query"],
        },
        "delta": {
            "extra_hours": three["full_hours"] - one["full_hours"],
            "extra_index_gib": three["one_index_gib"] - one["one_index_gib"],
            "extra_query_median_ms": q_three["median_ms"] - q_one["median_ms"],
            "extra_query_batch32_ms": q_three["batch_32_ms_per_query"] - q_one["batch_32_ms_per_query"],
        },
        "query": query,
        "notes": [
            "Codificación de documentos medida en una sola GPU; el corpus completo se extrapola linealmente.",
            "El costo en línea (consultas) y el costo fuera de línea (documentos e índices) se reportan por separado.",
            "El espacio no incluye IDs, textos ni pesos de modelos; son comunes a las dos rutas.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "label",
                "max_length",
                "seq_length",
                "batch_size",
                "docs_per_s",
                "docs_per_s_ci_low",
                "docs_per_s_ci_high",
                "tokens_per_s",
                "ms_per_doc",
                "full_hours",
                "truncated_fraction",
                "peak_vram_mib",
                "bytes_per_vector",
                "one_index_gib",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Costo de codificar documentos e índices",
        "",
        f"Muestra: {meta['n_docs']} documentos, {meta['n_strata']} deciles de longitud, semilla {meta['seed']}.",
        "",
        "| Modelo | Tope nativo | Seq. usada | Lote | docs/s | IC 95 % | ms/doc | Horas a 14 M | VRAM (MiB) | Bytes/vector | Un índice (GiB) |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {max_length} | {seq_length} | {batch_size} | {docs_per_s:.1f} | [{docs_per_s_ci_low:.1f}, {docs_per_s_ci_high:.1f}] | {ms_per_doc:.2f} | {full_hours:.2f} | {peak_vram_mib:.0f} | {bytes_per_vector:.0f} | {one_index_gib:.2f} |".format(**row)
        )
    lines += [
        "",
        f"**Ruta de 1 modelo (`jina-v5-small`):** {one['full_hours']:.2f} h y {one['one_index_gib']:.2f} GiB.",
        f"**Ruta de 3 modelos:** {three['full_hours']:.2f} h y {three['one_index_gib']:.2f} GiB (tres índices).",
        f"**Costo adicional:** {summary['delta']['extra_hours']:.2f} h y {summary['delta']['extra_index_gib']:.2f} GiB.",
        "",
        "## Codificación de consultas",
        "",
        "| Modelo | Mediana b1 (ms) | P95 b1 (ms) | QPS b32 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in ALIASES:
        value = query[key]
        lines.append(f"| {LABELS[key]} | {value['median_ms']:.2f} | {value['p95_ms']:.2f} | {value['batch_32_qps']:.1f} |")
    lines += [
        "",
        f"**1 modelo:** {q_one['median_ms']:.2f} ms/consulta (b1), {q_one['batch_32_ms_per_query']:.2f} ms (b32).",
        f"**3 modelos:** {q_three['median_ms']:.2f} ms/consulta (b1), {q_three['batch_32_ms_per_query']:.2f} ms (b32).",
        "",
        "El costo de documentos e índices es fuera de línea; el de consultas es en línea. No se suman.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Resumen escrito en %s", out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["sample", "encode", "aggregate"], required=True)
    parser.add_argument("--model", choices=ALIASES)
    parser.add_argument("--reps", type=int, default=N_REPS)
    parser.add_argument("--n", type=int, default=N_DOCS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    out_dir = args.out_dir
    if args.mode == "sample":
        build_sample(out_dir, args.n, args.seed)
        return
    if args.mode == "encode":
        if not args.model:
            raise SystemExit("--model es obligatorio en --mode encode")
        run_model(out_dir, args.model, args.reps, args.device)
        return
    aggregate(out_dir)


if __name__ == "__main__":
    main()
