#!/usr/bin/env python3
"""Measure how many candidate documents each reranker truncates.

The script samples queries from the cached jina-v5-small run, recovers the
text of their top-100 candidates, and tokenizes the pairs with the three
reranker tokenizers.  It reports the length distribution and the fraction of
documents above the caps used in the first evaluation (512 for bge, 300 for
ColBERT) and above the native maxima (8192 for both).

It runs on CPU and does not need the GPUs.  Example:

    python -m scripts.check_reranker_lengths --queries 500
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import statistics
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import datasets
from transformers import AutoTokenizer

from rerankers import context
from utils import cache, data, retrieval

log = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cache" / "messirve_embeddings"
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "reranker_context"
FIRST_STAGE_MODEL = "jinaai/jina-embeddings-v5-text-small-retrieval"
FIRST_STAGE_Q = 32768
FIRST_STAGE_D = 32768

MODELS = {
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    "jina-reranker-v3": "jinaai/jina-reranker-v3",
    "jina-colbert-v2": "jinaai/jina-colbert-v2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=500, help="Number of sampled queries.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_candidates() -> dict[str, list[str]]:
    base = cache.cache_base(
        CACHE_DIR,
        FIRST_STAGE_MODEL,
        data.COUNTRY,
        data.DATASET_VERSION,
        FIRST_STAGE_Q,
        FIRST_STAGE_D,
        data.MAX_WORD_COUNT,
    )
    run_path = cache.run_cache_path(base)
    if not run_path.exists():
        raise SystemExit(f"Missing first-stage run: {run_path}")
    run = retrieval.load_run(run_path)
    return {query_id: list(docs) for query_id, docs in run.to_dict().items()}


def load_documents(doc_ids: set[str]) -> dict[str, str]:
    log.info("Loading corpus and selecting %d candidate documents", len(doc_ids))
    dataset = datasets.load_dataset(data.CORPUS_NAME, split="corpus")
    all_ids = dataset["str_docid"]
    index = {doc_id: position for position, doc_id in enumerate(all_ids) if doc_id in doc_ids}
    missing = doc_ids - set(index)
    if missing:
        log.warning("%d candidate documents are missing from the corpus", len(missing))
    rows = dataset.select(list(index.values()))
    return dict(zip(rows["str_docid"], rows["full_text"]))


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[position]


def summarize(values: list[int], caps: list[int]) -> dict:
    summary = {
        "n": len(values),
        "min": min(values) if values else 0,
        "median": statistics.median(values) if values else 0,
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else 0,
    }
    for cap in caps:
        summary[f"above_{cap}"] = sum(value > cap for value in values)
        summary[f"above_{cap}_pct"] = (
            100.0 * summary[f"above_{cap}"] / len(values) if values else 0.0
        )
    return summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    candidates = load_candidates()
    query_ids = sorted(candidates)
    rng = random.Random(args.seed)
    sampled = rng.sample(query_ids, min(args.queries, len(query_ids)))
    doc_ids = {doc_id for query_id in sampled for doc_id in candidates[query_id]}
    document_texts = load_documents(doc_ids)

    qrels_path = cache.dataset_cache_base(
        CACHE_DIR, data.COUNTRY, data.DATASET_VERSION, data.MAX_WORD_COUNT
    ) / "pruned_q_map.json"
    with qrels_path.open(encoding="utf-8") as handle:
        query_map = json.load(handle)

    tokenizers = {
        name: AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=model == "jinaai/jina-reranker-v3",
            local_files_only=True,
        )
        for name, model in MODELS.items()
    }

    combined_bge: list[int] = []
    doc_colbert: list[int] = []
    doc_v3: list[int] = []
    query_lengths = {name: [] for name in tokenizers}

    for position, query_id in enumerate(sampled, 1):
        query_text = query_map[query_id]
        for name, tokenizer in tokenizers.items():
            query_lengths[name].append(len(tokenizer(query_text)["input_ids"]))

        for doc_id in candidates[query_id]:
            text = document_texts.get(doc_id)
            if text is None:
                continue
            combined_bge.append(
                len(tokenizers["bge-reranker-v2-m3"](query_text, text)["input_ids"])
            )
            doc_colbert.append(len(tokenizers["jina-colbert-v2"](text)["input_ids"]))
            doc_v3.append(len(tokenizers["jina-reranker-v3"](text)["input_ids"]))

        if position % 50 == 0:
            log.info("Tokenized %d/%d sampled queries", position, len(sampled))

    report = {
        "sampled_queries": len(sampled),
        "candidate_documents": len(doc_ids),
        "tokenized_documents": len(doc_v3),
        "query_lengths": {
            name: summarize(lengths, [32, 1024]) for name, lengths in query_lengths.items()
        },
        "bge_combined": summarize(combined_bge, [512, context.BGE_MAX_LENGTH]),
        "colbert_document": summarize(doc_colbert, [300, context.COLBERT_MAX_DOC_LENGTH]),
        "jina_v3_document": summarize(doc_v3, [context.JINA_V3_MAX_DOC_LENGTH]),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "truncation.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    markdown = [
        "# Truncation of reranker inputs",
        "",
        f"- Sampled queries: {report['sampled_queries']}",
        f"- Candidate documents: {report['candidate_documents']}",
        f"- Tokenized documents: {report['tokenized_documents']}",
        "",
        "| Input | n | median | p95 | p99 | max | above 512 | above 300 | above 8192 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, key in (
        ("bge combined (query + document)", "bge_combined"),
        ("ColBERT document", "colbert_document"),
        ("jina-reranker-v3 document", "jina_v3_document"),
    ):
        row = report[key]
        markdown.append(
            f"| {label} | {row['n']} | {row['median']} | {row['p95']} | {row['p99']} | "
            f"{row['max']} | {row.get('above_512_pct', 0):.2f}% | "
            f"{row.get('above_300_pct', 0):.2f}% | {row.get('above_8192_pct', 0):.2f}% |"
        )
    markdown.append("")
    markdown.append("## Query lengths")
    markdown.append("")
    markdown.append("| Tokenizer | median | p95 | max | above 32 | above 1024 |")
    markdown.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, row in report["query_lengths"].items():
        markdown.append(
            f"| {name} | {row['median']} | {row['p95']} | {row['max']} | "
            f"{row['above_32_pct']:.2f}% | {row['above_1024_pct']:.2f}% |"
        )

    markdown_path = OUT_DIR / "truncation.md"
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    log.info("Wrote %s and %s", json_path, markdown_path)
    print("\n".join(markdown))


if __name__ == "__main__":
    main()
