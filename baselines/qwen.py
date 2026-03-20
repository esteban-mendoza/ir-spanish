#!/usr/bin/env python3
"""
Baseline evaluation of Qwen3-Embedding-8B on MessIRve (Spanish IR).
Metrics: nDCG@10, Recall@100 via ranx.
"""

import logging
from collections import defaultdict

import datasets
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
MAX_SEQ_LENGTH = 512  # keep moderate to fit in VRAM; increase if needed

# Retrieval
TOP_K = 100  # we need top-100 for Recall@100; nDCG@10 uses top-10 subset

# Reproducibility / performance
SEED = 42
FAISS_ON_GPU = True  # use GPU for FAISS index if available

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Load MessIRve qrels (test split only)
# ---------------------------------------------------------------------------
def load_messirve_qrels(country: str, version: str):
    """Return test split of MessIRve and build ranx Qrels."""
    log.info("Loading MessIRve dataset (country=%s, version=%s) …", country, version)
    ds = datasets.load_dataset("spanish-ir/messirve", country, revision=version, trust_remote_code=True)
    test_ds = ds["test"]
    log.info("Test split: %d query-document pairs", len(test_ds))

    # Build qrels dict: {query_id: {doc_id: relevance}}
    # MessIRve has binary relevance (1 = relevant)
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
    """Load the eswiki corpus and return id->text mapping + ordered lists."""
    log.info("Loading eswiki corpus …")
    corpus_ds = datasets.load_dataset(CORPUS_NAME, trust_remote_code=True)
    # The corpus dataset typically has a single split
    if "train" in corpus_ds:
        corpus_split = corpus_ds["train"]
    else:
        # take the first available split
        split_name = list(corpus_ds.keys())[0]
        corpus_split = corpus_ds[split_name]

    log.info("Corpus size: %d documents", len(corpus_split))

    doc_ids: list[str] = []
    doc_texts: list[str] = []

    for row in corpus_split:
        did = str(row["docid"])
        # Combine title + text if title is available
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
    """Load Qwen3-Embedding-8B with SentenceTransformers across 2 GPUs."""
    log.info("Loading model %s …", MODEL_NAME)
    model = SentenceTransformer(
        MODEL_NAME,
        model_kwargs={
            "device_map": "auto",
            "torch_dtype": torch.float16,
        },
        tokenizer_kwargs={"padding_side": "left"},
    )
    log.info("Model loaded successfully.")
    return model


# ---------------------------------------------------------------------------
# 4. Encode corpus in batches and build FAISS index
# ---------------------------------------------------------------------------
def encode_and_index(
    model: SentenceTransformer,
    doc_ids: list[str],
    doc_texts: list[str],
    query_id_to_text: dict[str, str],
    top_k: int,
) -> dict[str, dict[str, float]]:
    """
    Encode all documents, encode queries, retrieve top-k via cosine similarity.
    Uses batched numpy/torch computation to avoid OOM on large corpora.
    Returns run_dict: {qid: {docid: score}}.
    """
    import numpy as np

    num_docs = len(doc_ids)
    log.info("Encoding %d corpus documents …", num_docs)

    # --- Encode documents in batches, store as fp16 numpy on CPU ---
    # We'll compute the embedding dimension from a tiny probe
    probe_emb = model.encode(["probe"], batch_size=1, normalize_embeddings=True)
    emb_dim = probe_emb.shape[1]
    log.info("Embedding dimension: %d", emb_dim)

    # Pre-allocate memmap-like array (float16 to save RAM)
    doc_embeddings = np.zeros((num_docs, emb_dim), dtype=np.float16)

    for start in tqdm(range(0, num_docs, DOC_BATCH_SIZE), desc="Encoding docs", unit="batch"):
        end = min(start + DOC_BATCH_SIZE, num_docs)
        batch_texts = doc_texts[start:end]
        embs = model.encode(
            batch_texts,
            batch_size=DOC_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        doc_embeddings[start:end] = embs.astype(np.float16)

    log.info("Corpus encoding complete.")

    # --- Encode queries ---
    query_ids = list(query_id_to_text.keys())
    query_texts = [query_id_to_text[qid] for qid in query_ids]
    num_queries = len(query_ids)
    log.info("Encoding %d queries …", num_queries)

    query_embeddings = np.zeros((num_queries, emb_dim), dtype=np.float16)
    for start in tqdm(range(0, num_queries, QUERY_BATCH_SIZE), desc="Encoding queries", unit="batch"):
        end = min(start + QUERY_BATCH_SIZE, num_queries)
        batch_texts = query_texts[start:end]
        embs = model.encode(
            batch_texts,
            batch_size=QUERY_BATCH_SIZE,
            prompt_name="query",
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        query_embeddings[start:end] = embs.astype(np.float16)

    log.info("Query encoding complete.")

    # --- Retrieval: batched cosine similarity ---
    # Process queries in chunks to avoid huge score matrices
    QUERY_CHUNK = 256
    run_dict: dict[str, dict[str, float]] = {}

    log.info("Retrieving top-%d documents per query …", top_k)
    # Convert doc embeddings to float32 torch tensor on CPU for matmul
    doc_emb_tensor = torch.from_numpy(doc_embeddings.astype(np.float32))

    for q_start in tqdm(range(0, num_queries, QUERY_CHUNK), desc="Retrieving", unit="chunk"):
        q_end = min(q_start + QUERY_CHUNK, num_queries)
        q_emb = torch.from_numpy(query_embeddings[q_start:q_end].astype(np.float32))
        # Cosine similarity (already normalized)
        # Shape: (chunk_size, num_docs)
        scores = q_emb @ doc_emb_tensor.T  # on CPU

        # Top-k per query
        topk_scores, topk_indices = torch.topk(scores, k=min(top_k, num_docs), dim=1)

        for i in range(q_end - q_start):
            qid = query_ids[q_start + i]
            run_dict[qid] = {}
            for rank_j in range(topk_scores.shape[1]):
                did = doc_ids[topk_indices[i, rank_j].item()]
                run_dict[qid][did] = float(topk_scores[i, rank_j].item())

    log.info("Retrieval complete.")
    return run_dict


# ---------------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------------
def run_evaluation(qrels: Qrels, run_dict: dict[str, dict[str, float]]):
    """Compute nDCG@10 and Recall@100 with ranx."""
    run = Run(run_dict)
    results = evaluate(
        qrels,
        run,
        metrics=["ndcg@10", "recall@100"],
    )
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

    # 1. Qrels
    qrels, query_id_to_text = load_messirve_qrels(COUNTRY, DATASET_VERSION)

    # 2. Corpus
    doc_ids, doc_texts = load_corpus()

    # 3. Model
    model = load_model()

    # 4. Encode + Retrieve
    run_dict = encode_and_index(model, doc_ids, doc_texts, query_id_to_text, TOP_K)

    # 5. Evaluate
    results = run_evaluation(qrels, run_dict)

    return results


if __name__ == "__main__":
    main()
