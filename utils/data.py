"""
MessIRve qrels loading, corpus loading, and word-count filtering.
"""

from __future__ import annotations

import gc
import json
import logging
from collections import defaultdict

import datasets
from ranx import Qrels

from .observability import Timer

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset constants (shared across all models)
# ---------------------------------------------------------------------------
COUNTRY = "full"
DATASET_VERSION = "1.2"
CORPUS_NAME = "spanish-ir/eswiki_20240401_corpus"
MAX_WORD_COUNT = None


# ---------------------------------------------------------------------------
# Qrels loading & pruning
# ---------------------------------------------------------------------------

def load_messirve_qrels(country: str, version: str, num_workers: int):
    """Download the MessIRve test split and build a Qrels object and a query-id → text map.

    Returns:
        qrels: A ranx.Qrels object mapping query IDs to {doc_id: relevance_score}.
        query_id_to_text: A dict mapping each query ID (str) to its query text.
    """
    with Timer(f"Loading raw MessIRve qrels ({country}, v{version})"):
        messirve_dataset = datasets.load_dataset(
            "spanish-ir/messirve", country, revision=version, num_proc=num_workers
        )
        test_split = messirve_dataset["test"]

    raw_query_ids = test_split["id"]
    raw_doc_ids = test_split["docid"]
    raw_query_texts = test_split["query"]

    qrels_dict: dict[str, dict[str, int]] = defaultdict(dict)
    query_id_to_text: dict[str, str] = {}

    for query_id, doc_id, query_text in zip(raw_query_ids, raw_doc_ids, raw_query_texts):
        query_id_str = str(query_id)
        qrels_dict[query_id_str][str(doc_id)] = 1
        query_id_to_text[query_id_str] = query_text

    return Qrels(qrels_dict), query_id_to_text


def get_pruned_qrels_and_queries(
    country: str,
    version: str,
    kept_doc_ids: list[str] | None,
    dataset_cache_dir,
    num_workers: int,
) -> tuple[Qrels, dict[str, str]]:
    """Return qrels and query map pruned to only include documents that survived the word-count filter.

    Loads from cache if available; otherwise builds the pruned qrels from scratch and saves
    them to disk so subsequent runs (and other models) can skip the expensive rebuild.

    Args:
        kept_doc_ids: The list of doc IDs that survived the corpus filter. Only needed on a
                      cache miss — pass None when the cache is known to be valid.
        dataset_cache_dir: Directory where pruned_qrels.json and pruned_q_map.json are stored.
    """
    qrels_cache_path = dataset_cache_dir / "pruned_qrels.json"
    query_map_cache_path = dataset_cache_dir / "pruned_q_map.json"

    # 1. Quick load from cache if available
    if qrels_cache_path.exists() and query_map_cache_path.exists():
        with Timer("Loading cached pruned qrels"):
            with open(qrels_cache_path, "r") as f:
                pruned_qrels_dict = json.load(f)
            with open(query_map_cache_path, "r") as f:
                pruned_query_map = json.load(f)
        return Qrels(pruned_qrels_dict), pruned_query_map

    # 2. Compute from scratch if cache is missing
    if kept_doc_ids is None:
        raise RuntimeError(
            "Cache miss for pruned qrels, but kept_doc_ids were not provided to build it."
        )

    original_qrels, original_query_map = load_messirve_qrels(country, version, num_workers)

    with Timer("Pruning qrels against filtered corpus"):
        valid_doc_ids = set(kept_doc_ids)
        original_qrels_dict = original_qrels.to_dict()
        pruned_qrels_dict: dict[str, dict[str, int]] = defaultdict(dict)

        for query_id, relevance_judgments in original_qrels_dict.items():
            for doc_id, relevance_score in relevance_judgments.items():
                if doc_id in valid_doc_ids:
                    pruned_qrels_dict[query_id][doc_id] = relevance_score

        # Drop queries whose every relevant document was filtered out
        pruned_qrels_dict = {
            query_id: judgments
            for query_id, judgments in pruned_qrels_dict.items()
            if judgments
        }
        pruned_query_map = {
            query_id: original_query_map[query_id]
            for query_id in pruned_qrels_dict.keys()
        }

        log.info(
            "Filtered queries: %d / %d kept after pruning",
            len(pruned_qrels_dict),
            len(original_qrels_dict),
        )

    # 3. Save to cache for the next run and other models that share this dataset config
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)
    with open(qrels_cache_path, "w") as f:
        json.dump(pruned_qrels_dict, f)
    with open(query_map_cache_path, "w") as f:
        json.dump(pruned_query_map, f)

    return Qrels(pruned_qrels_dict), pruned_query_map


# ---------------------------------------------------------------------------
# Corpus loading & filtering
# ---------------------------------------------------------------------------

def _batch_format_and_count(batch: dict) -> dict:
    """Format document text (title + body) and count words for each document in a batch.

    Concatenates title and text into a single 'full_text' field, converts doc IDs to
    strings, and computes word counts. These new fields are used downstream for
    filtering and encoding.

    Assigns three new columns:
      - 'full_text':  The formatted string "Title. Body text" (or just body if no title).
      - 'str_docid':  The document ID cast to a plain Python string.
      - 'word_count': Whitespace-separated word count of 'full_text'.
    """
    titles = batch.get("title", [""] * len(batch["docid"]))
    body_texts = batch.get("text", [""] * len(batch["docid"]))

    full_texts: list[str] = []
    string_doc_ids: list[str] = []
    word_counts: list[int] = []

    for doc_id, title, body_text in zip(batch["docid"], titles, body_texts):
        title = title or ""
        body_text = body_text or ""
        full_text = f"{title}. {body_text}".strip() if title else body_text

        full_texts.append(full_text)
        string_doc_ids.append(str(doc_id))
        word_counts.append(len(full_text.split()))

    batch["full_text"] = full_texts
    batch["str_docid"] = string_doc_ids
    batch["word_count"] = word_counts
    return batch


def load_corpus(num_workers: int):
    """Load the eswiki corpus and format documents.

    When MAX_WORD_COUNT is set, documents exceeding that word count are excluded.
    When MAX_WORD_COUNT is None, all documents are kept.

    Returns:
        doc_ids:   List of document IDs (strings) for the kept documents.
        doc_texts: List of formatted full-text strings for the kept documents.
    """
    with Timer("Loading eswiki corpus"):
        corpus_dataset = datasets.load_dataset(CORPUS_NAME, num_proc=num_workers)
        # Some dataset versions use 'train' as the only split name
        corpus_split = (
            corpus_dataset["train"]
            if "train" in corpus_dataset
            else corpus_dataset[list(corpus_dataset.keys())[0]]
        )

    num_docs_before_filter = len(corpus_split)
    log.info("Original corpus: %d documents", num_docs_before_filter)

    with Timer(f"Formatting and counting words ({num_workers} workers)"):
        corpus_split = corpus_split.map(
            _batch_format_and_count,
            batched=True,
            batch_size=10_000,
            num_proc=num_workers,
            desc="Formatting",
        )

    if MAX_WORD_COUNT is not None:
        with Timer(f"Filtering docs <= {MAX_WORD_COUNT} words"):
            corpus_split = corpus_split.filter(
                lambda batch: [word_count <= MAX_WORD_COUNT for word_count in batch["word_count"]],
                batched=True,
                batch_size=10_000,
                num_proc=num_workers,
                desc="Filtering",
            )

        num_docs_after_filter = len(corpus_split)
        log.info(
            "Filtered corpus: %d / %d documents kept (%.1f%%)",
            num_docs_after_filter,
            num_docs_before_filter,
            (num_docs_after_filter / num_docs_before_filter) * 100 if num_docs_before_filter else 0,
        )
    else:
        log.info("Word-count filter disabled — using all %d documents", num_docs_before_filter)

    doc_ids = corpus_split["str_docid"]
    doc_texts = corpus_split["full_text"]
    del corpus_split, corpus_dataset
    gc.collect()
    return doc_ids, doc_texts
