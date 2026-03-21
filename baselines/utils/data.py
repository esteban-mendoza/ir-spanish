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

from .cache import Timer

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset constants (shared across all models)
# ---------------------------------------------------------------------------
COUNTRY = "full"
DATASET_VERSION = "1.2"
CORPUS_NAME = "spanish-ir/eswiki_20240401_corpus"
MAX_WORD_COUNT = 400


# ---------------------------------------------------------------------------
# Qrels loading & pruning
# ---------------------------------------------------------------------------
def load_messirve_qrels(country: str, version: str, num_workers: int):
    with Timer(f"Loading raw MessIRve qrels ({country}, v{version})"):
        ds = datasets.load_dataset("spanish-ir/messirve", country, revision=version, num_proc=num_workers)
        test = ds["test"]

    raw_ids, raw_docids, raw_queries = test["id"], test["docid"], test["query"]
    qrels_dict, q_map = defaultdict(dict), {}

    for qid, did, qtxt in zip(raw_ids, raw_docids, raw_queries):
        qid_s = str(qid)
        qrels_dict[qid_s][str(did)] = 1
        q_map[qid_s] = qtxt

    return Qrels(qrels_dict), q_map


def get_pruned_qrels_and_queries(
    country: str,
    version: str,
    kept_doc_ids: list[str] | None,
    dataset_cache_dir,
    num_workers: int,
) -> tuple[Qrels, dict[str, str]]:
    qrels_path = dataset_cache_dir / "pruned_qrels.json"
    qmap_path = dataset_cache_dir / "pruned_q_map.json"

    # 1. Quick load from cache if available
    if qrels_path.exists() and qmap_path.exists():
        with Timer("Loading cached pruned qrels"):
            with open(qrels_path, "r") as f:
                pruned_dict = json.load(f)
            with open(qmap_path, "r") as f:
                pruned_q_map = json.load(f)
        return Qrels(pruned_dict), pruned_q_map

    # 2. Compute it if cache is missing
    if kept_doc_ids is None:
        raise RuntimeError("Cache miss for pruned qrels, but kept_doc_ids were not provided to build it.")

    original_qrels, original_q_map = load_messirve_qrels(country, version, num_workers)

    with Timer("Pruning qrels against filtered corpus"):
        kept_set = set(kept_doc_ids)
        qrels_dict = original_qrels.to_dict()
        pruned_dict = defaultdict(dict)

        for qid, judgments in qrels_dict.items():
            for did, score in judgments.items():
                if did in kept_set:
                    pruned_dict[qid][did] = score

        pruned_dict = {qid: judgs for qid, judgs in pruned_dict.items() if judgs}
        pruned_q_map = {qid: original_q_map[qid] for qid in pruned_dict.keys()}

        log.info("Filtered queries: %d / %d", len(pruned_dict), len(qrels_dict))

    # 3. Save to cache for the next run / other models
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)
    with open(qrels_path, "w") as f:
        json.dump(pruned_dict, f)
    with open(qmap_path, "w") as f:
        json.dump(pruned_q_map, f)

    return Qrels(pruned_dict), pruned_q_map


# ---------------------------------------------------------------------------
# Corpus loading & filtering
# ---------------------------------------------------------------------------
def _batch_format_and_count(batch: dict) -> dict:
    titles = batch.get("title", [""] * len(batch["docid"]))
    texts = batch.get("text", [""] * len(batch["docid"]))
    ft, si, wc = [], [], []
    for did, title, text in zip(batch["docid"], titles, texts):
        title, text = title or "", text or ""
        full = f"{title}. {text}".strip() if title else text
        ft.append(full)
        si.append(str(did))
        wc.append(len(full.split()))
    batch["full_text"] = ft
    batch["str_docid"] = si
    batch["word_count"] = wc
    return batch


def load_corpus(num_workers: int):
    with Timer("Loading eswiki corpus"):
        cds = datasets.load_dataset(CORPUS_NAME, num_proc=num_workers)
        split = cds["train"] if "train" in cds else cds[list(cds.keys())[0]]

    n_original = len(split)
    log.info("Original corpus: %d documents", n_original)

    with Timer(f"Formatting and counting words ({num_workers} workers)"):
        split = split.map(
            _batch_format_and_count, batched=True, batch_size=10_000, num_proc=num_workers, desc="Formatting"
        )

    with Timer(f"Filtering docs <= {MAX_WORD_COUNT} words"):
        split = split.filter(
            lambda x: [wc <= MAX_WORD_COUNT for wc in x["word_count"]],
            batched=True,
            batch_size=10_000,
            num_proc=num_workers,
            desc="Filtering",
        )

    n_filtered = len(split)
    log.info(
        "Filtered corpus: %d / %d documents kept (%.1f%%)",
        n_filtered,
        n_original,
        (n_filtered / n_original) * 100 if n_original else 0,
    )

    doc_ids, doc_texts = list(split["str_docid"]), list(split["full_text"])
    del split, cds
    gc.collect()
    return doc_ids, doc_texts
