#!/usr/bin/env python3
"""
Baseline evaluation of Qwen3-Embedding-8B on MessIRve (Spanish IR).
Metrics: nDCG@10, Recall@100 via ranx.

Embeddings are cached to disk as Parquet (zstd) so subsequent runs skip encoding.
Uses both GPUs via SentenceTransformer's multi-process encode pool.
"""

import logging
import time
from collections import defaultdict
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
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

# Encoding — batch size PER GPU (keep conservative: ~8.5 GB headroom each)
QUERY_BATCH_SIZE = 32
DOC_BATCH_SIZE = 16

# Retrieval
TOP_K = 100

# Cache directory on external disk
CACHE_DIR = Path("/media/discoexterno/jmendoza/embeddings_cache")

# Parquet compression
PARQUET_COMPRESSION = "zstd"

# GPUs to use
GPU_DEVICES = ["cuda:0", "cuda:1"]

# Reproducibility
SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers: timing
# ---------------------------------------------------------------------------
class Timer:
    """Simple context manager for timing blocks of code."""

    def __init__(self, description: str):
        self.description = description
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        log.info("⏱  START: %s", self.description)
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        log.info("⏱  DONE:  %s (%.1fs)", self.description, self.elapsed)


def log_gpu_memory():
    """Log current GPU memory usage for all visible devices."""
    if not torch.cuda.is_available():
        log.info("GPU: CUDA not available")
        return
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        total = torch.cuda.get_device_properties(i).total_mem / (1024**3)
        log.info(
            "GPU %d (%s): %.1f GB allocated, %.1f GB reserved, %.1f GB total",
            i,
            torch.cuda.get_device_name(i),
            allocated,
            reserved,
            total,
        )


def log_ram_usage():
    """Log current process RAM usage."""
    try:
        import resource

        usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        log.info("RAM (peak RSS): %.1f GB", usage_kb / (1024**2))
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Helpers: cache paths
# ---------------------------------------------------------------------------
def _model_slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def _cache_paths(model_name: str, country: str, version: str) -> dict[str, Path]:
    slug = _model_slug(model_name)
    base = CACHE_DIR / slug / f"{country}_v{version}"
    paths = {
        "dir": base,
        "doc_embeddings": base / "doc_embeddings.parquet",
        "query_embeddings": base / "query_embeddings.parquet",
    }
    log.info("Cache directory: %s", base)
    for key in ["doc_embeddings", "query_embeddings"]:
        exists = paths[key].exists()
        size_str = ""
        if exists:
            size_mb = paths[key].stat().st_size / (1024 * 1024)
            size_str = f" ({size_mb:.1f} MB)"
        log.info(
            "  %-20s: %s%s",
            key,
            "✓ cached" if exists else "✗ not cached",
            size_str,
        )
    return paths


# ---------------------------------------------------------------------------
# Helpers: Parquet I/O for embeddings
# ---------------------------------------------------------------------------
def save_embeddings_parquet(
    ids: list[str],
    embeddings: np.ndarray,
    path: Path,
) -> None:
    n, d = embeddings.shape
    log.info(
        "Saving %d embeddings (dim=%d) to %s [%s] …",
        n,
        d,
        path,
        PARQUET_COMPRESSION,
    )

    with Timer(f"Building DataFrame ({n} rows × {d + 1} cols)"):
        data: dict[str, np.ndarray | list[str]] = {"id": ids}
        for col_idx in range(d):
            data[f"emb_{col_idx}"] = embeddings[:, col_idx]
        df = pd.DataFrame(data)

    with Timer(f"Writing Parquet to {path.name}"):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(
            path,
            engine="fastparquet",
            compression=PARQUET_COMPRESSION,
        )

    size_mb = path.stat().st_size / (1024 * 1024)
    raw_mb = (n * d * 2) / (1024 * 1024)
    log.info(
        "Saved: %.1f MB on disk (raw float16 would be %.1f MB, ratio %.2f×)",
        size_mb,
        raw_mb,
        raw_mb / size_mb if size_mb > 0 else 0,
    )


def load_embeddings_parquet(path: Path) -> tuple[list[str], np.ndarray]:
    size_mb = path.stat().st_size / (1024 * 1024)
    log.info("Loading cached embeddings from %s (%.1f MB on disk) …", path, size_mb)

    with Timer(f"Reading Parquet {path.name}"):
        df = pd.read_parquet(path, engine="fastparquet")

    with Timer("Extracting IDs and embedding matrix"):
        ids = df["id"].tolist()
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        emb_cols.sort(key=lambda c: int(c.split("_")[1]))
        embeddings = df[emb_cols].to_numpy(dtype=np.float16)

    ram_mb = (embeddings.shape[0] * embeddings.shape[1] * 2) / (1024 * 1024)
    log.info(
        "Loaded %d embeddings (dim=%d, %.1f MB in RAM)",
        embeddings.shape[0],
        embeddings.shape[1],
        ram_mb,
    )
    log_ram_usage()
    return ids, embeddings


# ---------------------------------------------------------------------------
# 1. Load MessIRve qrels (test split only)
# ---------------------------------------------------------------------------
def load_messirve_qrels(country: str, version: str):
    with Timer(f"Loading MessIRve qrels (country={country}, version={version})"):
        ds = datasets.load_dataset("spanish-ir/messirve", country, revision=version)
        test_ds = ds["test"]

    log.info("Test split: %d query-document pairs", len(test_ds))

    qrels_dict: dict[str, dict[str, int]] = defaultdict(dict)
    query_id_to_text: dict[str, str] = {}

    for row in test_ds:
        qid = str(row["id"])
        docid = str(row["docid"])
        qrels_dict[qid][docid] = 1
        query_id_to_text[qid] = row["query"]

    num_queries = len(qrels_dict)
    num_rels = sum(len(v) for v in qrels_dict.values())
    avg_rels = num_rels / num_queries if num_queries > 0 else 0

    log.info("Unique test queries:     %d", num_queries)
    log.info("Total relevance judgments: %d", num_rels)
    log.info("Avg relevant docs/query:  %.2f", avg_rels)

    qrels = Qrels(qrels_dict)
    return qrels, query_id_to_text


# ---------------------------------------------------------------------------
# 2. Load the Wikipedia corpus
# ---------------------------------------------------------------------------
def load_corpus():
    with Timer("Loading eswiki corpus from HuggingFace"):
        corpus_ds = datasets.load_dataset(CORPUS_NAME)
        if "train" in corpus_ds:
            corpus_split = corpus_ds["train"]
        else:
            split_name = list(corpus_ds.keys())[0]
            corpus_split = corpus_ds[split_name]

    num_docs = len(corpus_split)
    log.info("Corpus size: %d documents", num_docs)

    doc_ids: list[str] = []
    doc_texts: list[str] = []

    with Timer(f"Extracting {num_docs} document texts"):
        for row in corpus_split:
            did = str(row["docid"])
            title = row.get("title", "")
            text = row.get("text", "")
            full_text = f"{title}. {text}".strip() if title else text
            doc_ids.append(did)
            doc_texts.append(full_text)

    # Log some corpus statistics
    text_lengths = [len(t) for t in doc_texts[:10000]]  # sample for speed
    log.info(
        "Document text lengths (sampled first 10k): "
        "min=%d, median=%d, mean=%d, max=%d chars",
        min(text_lengths),
        int(np.median(text_lengths)),
        int(np.mean(text_lengths)),
        max(text_lengths),
    )
    log_ram_usage()

    return doc_ids, doc_texts


# ---------------------------------------------------------------------------
# 3. Multi-GPU encode via process pool
# ---------------------------------------------------------------------------
def encode_multi_gpu(
    texts: list[str],
    model_name: str,
    batch_size: int,
    prompt_name: str | None = None,
) -> np.ndarray:
    """
    Encode texts using both GPUs via SentenceTransformer's multi-process pool.
    """
    num_texts = len(texts)
    num_gpus = len(GPU_DEVICES)
    texts_per_gpu = num_texts // num_gpus

    log.info("=" * 60)
    log.info("Multi-GPU encoding configuration:")
    log.info("  Model:          %s", model_name)
    log.info("  Devices:        %s", GPU_DEVICES)
    log.info("  Total texts:    %d", num_texts)
    log.info("  Texts per GPU:  ~%d", texts_per_gpu)
    log.info("  Batch size:     %d per GPU", batch_size)
    log.info("  Prompt:         %s", prompt_name or "(none — document mode)")
    log.info("  Est. batches:   ~%d per GPU", texts_per_gpu // batch_size)
    log.info("=" * 60)

    log_gpu_memory()

    with Timer("Loading model on CPU"):
        model = SentenceTransformer(
            model_name,
            device="cpu",
            model_kwargs={"torch_dtype": torch.float16},
            tokenizer_kwargs={"padding_side": "left"},
        )

    with Timer("Starting multi-process pool"):
        pool = model.start_multi_process_pool(target_devices=GPU_DEVICES)

    log_gpu_memory()

    with Timer(f"Encoding {num_texts} texts across {num_gpus} GPUs"):
        embeddings = model.encode(
            sentences=texts,
            pool=pool,
            batch_size=batch_size,
            normalize_embeddings=True,
            prompt_name=prompt_name,
        )

    with Timer("Stopping multi-process pool"):
        model.stop_multi_process_pool(pool)

    throughput = num_texts / (time.perf_counter() - time.perf_counter())  # placeholder
    log.info("Encoding complete: shape=%s, dtype=%s", embeddings.shape, embeddings.dtype)
    log_gpu_memory()
    log_ram_usage()

    return embeddings.astype(np.float16)


# ---------------------------------------------------------------------------
# 4. Encode with disk caching
# ---------------------------------------------------------------------------
def encode_documents(
    doc_ids: list[str],
    doc_texts: list[str],
    paths: dict[str, Path],
) -> tuple[list[str], np.ndarray]:
    cache_path = paths["doc_embeddings"]

    if cache_path.exists():
        return load_embeddings_parquet(cache_path)

    num_docs = len(doc_ids)
    log.info("Encoding %d corpus documents (no cache found) …", num_docs)

    with Timer(f"Full document encoding pipeline ({num_docs} docs)"):
        doc_embeddings = encode_multi_gpu(
            texts=doc_texts,
            model_name=MODEL_NAME,
            batch_size=DOC_BATCH_SIZE,
            prompt_name=None,
        )

    save_embeddings_parquet(doc_ids, doc_embeddings, cache_path)
    return doc_ids, doc_embeddings


def encode_queries(
    query_id_to_text: dict[str, str],
    paths: dict[str, Path],
) -> tuple[list[str], np.ndarray]:
    cache_path = paths["query_embeddings"]

    if cache_path.exists():
        return load_embeddings_parquet(cache_path)

    query_ids = list(query_id_to_text.keys())
    query_texts = [query_id_to_text[qid] for qid in query_ids]
    num_queries = len(query_ids)
    log.info("Encoding %d queries (no cache found) …", num_queries)

    with Timer(f"Full query encoding pipeline ({num_queries} queries)"):
        query_embeddings = encode_multi_gpu(
            texts=query_texts,
            model_name=MODEL_NAME,
            batch_size=QUERY_BATCH_SIZE,
            prompt_name="query",
        )

    save_embeddings_parquet(query_ids, query_embeddings, cache_path)
    return query_ids, query_embeddings


# ---------------------------------------------------------------------------
# 5. Retrieve top-k
# ---------------------------------------------------------------------------
def retrieve(
    query_ids: list[str],
    query_embeddings: np.ndarray,
    doc_ids: list[str],
    doc_embeddings: np.ndarray,
    top_k: int,
) -> dict[str, dict[str, float]]:
    num_queries = len(query_ids)
    num_docs = len(doc_ids)

    log.info("=" * 60)
    log.info("Retrieval configuration:")
    log.info("  Queries:        %d", num_queries)
    log.info("  Documents:      %d", num_docs)
    log.info("  Top-k:          %d", top_k)
    log.info("  Query chunks:   %d queries per chunk", 256)
    log.info(
        "  Score matrix:   %d × %d per chunk (%.1f MB float32)",
        256,
        num_docs,
        (256 * num_docs * 4) / (1024 * 1024),
    )
    log.info("=" * 60)

    with Timer("Converting doc embeddings to float32 tensor"):
        doc_emb_tensor = torch.from_numpy(doc_embeddings.astype(np.float32))
        log.info(
            "Doc embedding tensor: shape=%s, dtype=%s, %.1f GB",
            doc_emb_tensor.shape,
            doc_emb_tensor.dtype,
            doc_emb_tensor.nelement() * 4 / (1024**3),
        )

    QUERY_CHUNK = 256
    run_dict: dict[str, dict[str, float]] = {}
    total_comparisons = 0

    with Timer(f"Retrieving top-{top_k} for {num_queries} queries"):
        for q_start in tqdm(
            range(0, num_queries, QUERY_CHUNK), desc="Retrieving", unit="chunk"
        ):
            q_end = min(q_start + QUERY_CHUNK, num_queries)
            chunk_size = q_end - q_start

            q_emb = torch.from_numpy(
                query_embeddings[q_start:q_end].astype(np.float32)
            )
            scores = q_emb @ doc_emb_tensor.T
            topk_scores, topk_indices = torch.topk(
                scores, k=min(top_k, num_docs), dim=1
            )

            for i in range(chunk_size):
                qid = query_ids[q_start + i]
                run_dict[qid] = {
                    doc_ids[topk_indices[i, j].item()]: float(
                        topk_scores[i, j].item()
                    )
                    for j in range(topk_scores.shape[1])
                }

            total_comparisons += chunk_size * num_docs

    log.info(
        "Retrieval complete: %d queries, %.1f billion similarity comparisons",
        len(run_dict),
        total_comparisons / 1e9,
    )
    log_ram_usage()
    return run_dict


# ---------------------------------------------------------------------------
# 6. Evaluate
# ---------------------------------------------------------------------------
def run_evaluation(qrels: Qrels, run_dict: dict[str, dict[str, float]]):
    log.info("Building ranx Run object from %d queries …", len(run_dict))

    with Timer("Creating ranx Run"):
        run = Run(run_dict)

    with Timer("Computing metrics (nDCG@10, Recall@100)"):
        results = evaluate(qrels, run, metrics=["ndcg@10", "recall@100"])

    log.info("")
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  RESULTS — Qwen3-Embedding-8B on MessIRve       ║")
    log.info("║  Dataset: full (test split), version %s         ║", DATASET_VERSION)
    log.info("╠══════════════════════════════════════════════════╣")
    for metric, value in results.items():
        log.info("║  %-15s  %.4f                         ║", metric, value)
    log.info("╚══════════════════════════════════════════════════╝")
    log.info("")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    overall_start = time.perf_counter()

    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  Qwen3-Embedding-8B — MessIRve Baseline          ║")
    log.info("╚══════════════════════════════════════════════════╝")
    log.info("")
    log.info("Configuration:")
    log.info("  Model:       %s", MODEL_NAME)
    log.info("  Dataset:     spanish-ir/messirve (%s, v%s)", COUNTRY, DATASET_VERSION)
    log.info("  Corpus:      %s", CORPUS_NAME)
    log.info("  GPUs:        %s", GPU_DEVICES)
    log.info("  Doc batch:   %d per GPU", DOC_BATCH_SIZE)
    log.info("  Query batch: %d per GPU", QUERY_BATCH_SIZE)
    log.info("  Top-k:       %d", TOP_K)
    log.info("  Cache dir:   %s", CACHE_DIR)
    log.info("")

    torch.manual_seed(SEED)
    log_gpu_memory()

    paths = _cache_paths(MODEL_NAME, COUNTRY, DATASET_VERSION)

    # 1. Qrels
    log.info("─" * 60)
    log.info("STEP 1/5: Loading relevance judgments (qrels)")
    log.info("─" * 60)
    qrels, query_id_to_text = load_messirve_qrels(COUNTRY, DATASET_VERSION)

    # 2. Corpus / document embeddings
    log.info("")
    log.info("─" * 60)
    log.info("STEP 2/5: Document embeddings")
    log.info("─" * 60)
    if paths["doc_embeddings"].exists():
        doc_ids, doc_embeddings = load_embeddings_parquet(paths["doc_embeddings"])
    else:
        doc_ids, doc_texts = load_corpus()
        doc_ids, doc_embeddings = encode_documents(doc_ids, doc_texts, paths)
        del doc_texts
        log.info("Freed doc_texts from memory.")

    # 3. Query embeddings
    log.info("")
    log.info("─" * 60)
    log.info("STEP 3/5: Query embeddings")
    log.info("─" * 60)
    if paths["query_embeddings"].exists():
        query_ids, query_embeddings = load_embeddings_parquet(
            paths["query_embeddings"]
        )
    else:
        query_ids, query_embeddings = encode_queries(query_id_to_text, paths)

    # 4. Retrieve
    log.info("")
    log.info("─" * 60)
    log.info("STEP 4/5: Retrieval (brute-force cosine similarity)")
    log.info("─" * 60)
    log_gpu_memory()
    run_dict = retrieve(query_ids, query_embeddings, doc_ids, doc_embeddings, TOP_K)

    # 5. Evaluate
    log.info("")
    log.info("─" * 60)
    log.info("STEP 5/5: Evaluation")
    log.info("─" * 60)
    results = run_evaluation(qrels, run_dict)

    # Final summary
    overall_elapsed = time.perf_counter() - overall_start
    log.info("Total wall-clock time: %.1f seconds (%.1f minutes)", overall_elapsed, overall_elapsed / 60)
    log_ram_usage()

    return results


if __name__ == "__main__":
    main()
