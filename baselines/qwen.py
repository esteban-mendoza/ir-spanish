#!/usr/bin/env python3
"""
Baseline evaluation of Qwen3-Embedding-8B on MessIRve (Spanish IR).
Metrics: nDCG@10, Recall@100 via ranx.

Embeddings are cached to disk so subsequent runs skip the encoding step.
"""

import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import datasets
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

# Encoding
QUERY_BATCH_SIZE = 64
DOC_BATCH_SIZE = 32

# Retrieval
TOP_K = 100

# Cache directory — all embeddings and ID mappings are stored here
CACHE_DIR = Path("/media/discoexterno/jmendoza/embeddings_cache")

# Reproducibility
SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers: cache paths
# ---------------------------------------------------------------------------
def _model_slug(model_name: str) -> str:
    """Turn 'Qwen/Qwen3-Embedding-8B' into 'Qwen__Qwen3-Embedding-8B'."""
    return model_name.replace("/", "__")


def _cache_paths(model_name: str, country: str, version: str) -> dict[str, Path]:
    """Return all cache file paths for a given model + dataset combo."""
    slug = _model_slug(model_name)
    base = CACHE_DIR / slug / f"{country}_v{version}"
    return {
        "dir": base,
        "doc_embeddings": base / "doc_embeddings.npy",
        "doc_ids": base / "doc_ids.json",
        "query_embeddings": base / "query_embeddings.npy",
        "query_ids": base / "query_ids.json",
    }


# ---------------------------------------------------------------------------
# 1. Load MessIRve qrels (test split only)
# ---------------------------------------------------------------------------
def load_messirve_qrels(country: str, version: str):
    """Return test split of MessIRve and build ranx Qrels."""
    log.info("Loading MessIRve dataset (country=%s, version=%s) …", country, version)
    ds = datasets.load_dataset(
        "spanish-ir/messirve", country, revision=version
    )
    test_ds = ds["test"]
    log.info("Test split: %d query-document pairs", len(test_ds))

    qrels_dict: dict[str, dict[str, int]] = defaultdict(dict)
    query_id_to_text: dict[str, str] = {}

    for row in test_ds:
        qid = str(row["id"])
        docid = str(row["docid"])
        qrels_dict[qid][docid] = 1
        query_id_to_text[qid] = row["query"]

    log.info("Unique test queries: %d", len(qrels_dict))
    qrels = Qrels(qrels_dict)
    return qrels, query_id_to_text


# ---------------------------------------------------------------------------
# 2. Load the Wikipedia corpus
# ---------------------------------------------------------------------------
def load_corpus():
    """Load the eswiki corpus and return ordered id and text lists."""
    log.info("Loading eswiki corpus …")
    corpus_ds = datasets.load_dataset(CORPUS_NAME)
    if "train" in corpus_ds:
        corpus_split = corpus_ds["train"]
    else:
        split_name = list(corpus_ds.keys())[0]
        corpus_split = corpus_ds[split_name]

    log.info("Corpus size: %d documents", len(corpus_split))

    doc_ids: list[str] = []
    doc_texts: list[str] = []

    for row in corpus_split:
        did = str(row["docid"])
        title = row.get("title", "")
        text = row.get("text", "")
        full_text = f"{title}. {text}".strip() if title else text
        doc_ids.append(did)
        doc_texts.append(full_text)

    return doc_ids, doc_texts


# ---------------------------------------------------------------------------
# 3. Load model
# ---------------------------------------------------------------------------
def load_model():
    """Load Qwen3-Embedding-8B across available GPUs."""
    log.info("Loading model %s …", MODEL_NAME)
    model = SentenceTransformer(
        MODEL_NAME,
        model_kwargs={
            "attn_implementation": "flash_attention_2",
            "device_map": "auto",
            "torch_dtype": torch.float16,
        },
        tokenizer_kwargs={"padding_side": "left"},
    )
    log.info("Model loaded successfully.")
    return model


# ---------------------------------------------------------------------------
# 4. Encode with disk caching
# ---------------------------------------------------------------------------
def encode_documents(
    model: SentenceTransformer,
    doc_ids: list[str],
    doc_texts: list[str],
    paths: dict[str, Path],
) -> tuple[list[str], np.ndarray]:
    """
    Encode corpus documents. If a cache exists on disk, load it instead.
    Returns (doc_ids, doc_embeddings) where embeddings is float16 numpy.
    """
    if paths["doc_embeddings"].exists() and paths["doc_ids"].exists():
        log.info("Loading cached document embeddings from %s", paths["dir"])
        doc_embeddings = np.load(paths["doc_embeddings"])
        with open(paths["doc_ids"], "r") as f:
            cached_doc_ids = json.load(f)
        log.info(
            "Loaded %d document embeddings (dim=%d)",
            doc_embeddings.shape[0],
            doc_embeddings.shape[1],
        )
        return cached_doc_ids, doc_embeddings

    num_docs = len(doc_ids)
    log.info("Encoding %d corpus documents (no cache found) …", num_docs)

    # Probe for embedding dimension
    probe = model.encode(["probe"], batch_size=1, normalize_embeddings=True)
    emb_dim = probe.shape[1]
    log.info("Embedding dimension: %d", emb_dim)

    doc_embeddings = np.zeros((num_docs, emb_dim), dtype=np.float16)

    for start in tqdm(
        range(0, num_docs, DOC_BATCH_SIZE), desc="Encoding docs", unit="batch"
    ):
        end = min(start + DOC_BATCH_SIZE, num_docs)
        embs = model.encode(
            doc_texts[start:end],
            batch_size=DOC_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        doc_embeddings[start:end] = embs.astype(np.float16)

    # Save to disk
    paths["dir"].mkdir(parents=True, exist_ok=True)
    np.save(paths["doc_embeddings"], doc_embeddings)
    with open(paths["doc_ids"], "w") as f:
        json.dump(doc_ids, f)
    log.info("Document embeddings cached to %s", paths["dir"])

    return doc_ids, doc_embeddings


def encode_queries(
    model: SentenceTransformer,
    query_id_to_text: dict[str, str],
    paths: dict[str, Path],
) -> tuple[list[str], np.ndarray]:
    """
    Encode queries. If a cache exists on disk, load it instead.
    Returns (query_ids, query_embeddings) where embeddings is float16 numpy.
    """
    if paths["query_embeddings"].exists() and paths["query_ids"].exists():
        log.info("Loading cached query embeddings from %s", paths["dir"])
        query_embeddings = np.load(paths["query_embeddings"])
        with open(paths["query_ids"], "r") as f:
            cached_query_ids = json.load(f)
        log.info(
            "Loaded %d query embeddings (dim=%d)",
            query_embeddings.shape[0],
            query_embeddings.shape[1],
        )
        return cached_query_ids, query_embeddings

    query_ids = list(query_id_to_text.keys())
    query_texts = [query_id_to_text[qid] for qid in query_ids]
    num_queries = len(query_ids)
    log.info("Encoding %d queries (no cache found) …", num_queries)

    # Probe for embedding dimension
    probe = model.encode(
        ["probe"], batch_size=1, prompt_name="query", normalize_embeddings=True
    )
    emb_dim = probe.shape[1]

    query_embeddings = np.zeros((num_queries, emb_dim), dtype=np.float16)

    for start in tqdm(
        range(0, num_queries, QUERY_BATCH_SIZE), desc="Encoding queries", unit="batch"
    ):
        end = min(start + QUERY_BATCH_SIZE, num_queries)
        embs = model.encode(
            query_texts[start:end],
            batch_size=QUERY_BATCH_SIZE,
            prompt_name="query",
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        query_embeddings[start:end] = embs.astype(np.float16)

    # Save to disk
    paths["dir"].mkdir(parents=True, exist_ok=True)
    np.save(paths["query_embeddings"], query_embeddings)
    with open(paths["query_ids"], "w") as f:
        json.dump(query_ids, f)
    log.info("Query embeddings cached to %s", paths["dir"])

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
    """Brute-force cosine retrieval via batched matrix multiplication."""
    num_queries = len(query_ids)
    doc_emb_tensor = torch.from_numpy(doc_embeddings.astype(np.float32))

    QUERY_CHUNK = 256
    run_dict: dict[str, dict[str, float]] = {}

    log.info("Retrieving top-%d documents for %d queries …", top_k, num_queries)

    for q_start in tqdm(
        range(0, num_queries, QUERY_CHUNK), desc="Retrieving", unit="chunk"
    ):
        q_end = min(q_start + QUERY_CHUNK, num_queries)
        q_emb = torch.from_numpy(
            query_embeddings[q_start:q_end].astype(np.float32)
        )
        scores = q_emb @ doc_emb_tensor.T
        topk_scores, topk_indices = torch.topk(
            scores, k=min(top_k, len(doc_ids)), dim=1
        )

        for i in range(q_end - q_start):
            qid = query_ids[q_start + i]
            run_dict[qid] = {
                doc_ids[topk_indices[i, j].item()]: float(topk_scores[i, j].item())
                for j in range(topk_scores.shape[1])
            }

    log.info("Retrieval complete.")
    return run_dict


# ---------------------------------------------------------------------------
# 6. Evaluate
# ---------------------------------------------------------------------------
def run_evaluation(qrels: Qrels, run_dict: dict[str, dict[str, float]]):
    """Compute nDCG@10 and Recall@100 with ranx."""
    run = Run(run_dict)
    results = evaluate(qrels, run, metrics=["ndcg@10", "recall@100"])

    log.info("=" * 50)
    log.info("RESULTS — Qwen3-Embedding-8B on MessIRve (full, test)")
    log.info("=" * 50)
    for metric, value in results.items():
        log.info("  %-15s  %.4f", metric, value)
    log.info("=" * 50)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)

    paths = _cache_paths(MODEL_NAME, COUNTRY, DATASET_VERSION)

    # 1. Qrels
    qrels, query_id_to_text = load_messirve_qrels(COUNTRY, DATASET_VERSION)

    # 2. Check if we can skip model loading entirely
    all_cached = all(
        paths[k].exists()
        for k in ["doc_embeddings", "doc_ids", "query_embeddings", "query_ids"]
    )

    if all_cached:
        log.info("All embeddings are cached — skipping model loading.")
        model = None
    else:
        model = load_model()

    # 3. Corpus (only load texts if we need to encode)
    if paths["doc_embeddings"].exists() and paths["doc_ids"].exists():
        doc_ids, doc_embeddings = encode_documents(model, [], [], paths)
    else:
        doc_ids_raw, doc_texts = load_corpus()
        doc_ids, doc_embeddings = encode_documents(model, doc_ids_raw, doc_texts, paths)
        del doc_texts  # free memory

    # 4. Queries
    query_ids, query_embeddings = encode_queries(model, query_id_to_text, paths)

    # Free model from GPU memory before retrieval
    if model is not None:
        del model
        torch.cuda.empty_cache()

    # 5. Retrieve
    run_dict = retrieve(query_ids, query_embeddings, doc_ids, doc_embeddings, TOP_K)

    # 6. Evaluate
    results = run_evaluation(qrels, run_dict)
    return results


if __name__ == "__main__":
    main()
