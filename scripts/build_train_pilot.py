#!/usr/bin/env python3
"""Build a small, closed pilot from the MessIRve training partition.

The pilot samples a fixed number of Wikipedia articles that contain at least
one relevant training document, keeps every paragraph of those articles as the
retrieval corpus, and keeps the training queries whose relevant documents all
belong to the sampled articles. This keeps the pilot self-contained: every
relevant document is present, so recall and nDCG are meaningful within the
closed corpus.

The output directory is meant to be used through ``MESSIRVE_PILOT``::

    MESSIRVE_PILOT=$HOME/.cache/messirve_embeddings/pilot \
        python -m baselines.e5_large

The pilot is an exploratory artifact, not a replacement for the full-corpus
selection experiment: absolute metrics are inflated by the much smaller
candidate pool.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Import utils first: it configures NUMA/threading env-vars before datasets/torch.
from utils import data
from utils.observability import Timer

import datasets

log = logging.getLogger(__name__)

DATASET_NAME = "spanish-ir/messirve"
CORPUS_NAME = data.CORPUS_NAME
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = Path.home() / ".cache" / "messirve_embeddings" / "pilot"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _load_train_rows():
    """Load the training split and keep the same constraints as the test split."""
    dataset = datasets.load_dataset(
        DATASET_NAME,
        data.COUNTRY,
        revision=data.DATASET_VERSION,
    )["train"]

    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    query_map: dict[str, str] = {}
    articles_by_query: dict[str, set[str]] = defaultdict(set)

    for raw_id, query, raw_docid, match_score, expanded_search in zip(
        dataset["id"],
        dataset["query"],
        dataset["docid"],
        dataset["match_score"],
        dataset["expanded_search"],
    ):
        if float(match_score) != 1.0 or bool(expanded_search):
            continue
        query_id = str(raw_id)
        doc_id = str(raw_docid)
        qrels[query_id][doc_id] = 1
        query_map[query_id] = query
        articles_by_query[query_id].add(doc_id.split("#")[0])

    return qrels, query_map, articles_by_query


def _select_articles(
    articles_by_query: dict[str, set[str]],
    n_articles: int,
    seed: int,
) -> list[str]:
    articles = sorted({article for articles in articles_by_query.values() for article in articles})
    if n_articles > len(articles):
        raise SystemExit(
            f"Requested {n_articles} articles but only {len(articles)} have relevant training documents"
        )
    rng = random.Random(seed)
    return sorted(rng.sample(articles, n_articles))


def _keep_queries(
    qrels: dict[str, dict[str, int]],
    query_map: dict[str, str],
    articles_by_query: dict[str, set[str]],
    selected_articles: set[str],
) -> tuple[dict[str, dict[str, int]], dict[str, str]]:
    kept_qrels: dict[str, dict[str, int]] = {}
    kept_query_map: dict[str, str] = {}
    for query_id, judgments in qrels.items():
        if articles_by_query[query_id] <= selected_articles:
            kept_qrels[query_id] = judgments
            kept_query_map[query_id] = query_map[query_id]
    return kept_qrels, kept_query_map


def _build_corpus(selected_articles: set[str], num_workers: int):
    corpus_dataset = datasets.load_dataset(CORPUS_NAME, num_proc=num_workers)
    if "train" in corpus_dataset:
        corpus = corpus_dataset["train"]
    elif "corpus" in corpus_dataset:
        corpus = corpus_dataset["corpus"]
    else:
        corpus = corpus_dataset[list(corpus_dataset.keys())[0]]

    with Timer(f"Filtering corpus to {len(selected_articles)} pilot articles"):
        corpus = corpus.filter(
            lambda batch: [
                str(doc_id).split("#")[0] in selected_articles
                for doc_id in batch["docid"]
            ],
            batched=True,
            batch_size=10_000,
            num_proc=num_workers,
            desc="Filtering pilot articles",
        )

    with Timer(f"Formatting pilot corpus ({len(corpus)} documents)"):
        corpus = corpus.map(
            data._batch_format_and_count,
            batched=True,
            batch_size=10_000,
            num_proc=num_workers,
            desc="Formatting pilot corpus",
        )

    return corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Pilot directory (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--n-articles",
        type=int,
        default=10_000,
        help="Number of training articles to sample (default: 10000).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count() or 32,
        help="Processes for dataset filtering and formatting.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing pilot directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    if out_dir.exists():
        if not args.force:
            raise SystemExit(f"{out_dir} already exists; pass --force to rebuild it")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    qrels, query_map, articles_by_query = _load_train_rows()
    log.info("Filtered train rows: %d queries, %d judgments", len(qrels), sum(len(v) for v in qrels.values()))

    selected_articles = _select_articles(articles_by_query, args.n_articles, args.seed)
    selected_set = set(selected_articles)
    kept_qrels, kept_query_map = _keep_queries(qrels, query_map, articles_by_query, selected_set)
    log.info(
        "Pilot: %d articles, %d queries, %d judgments",
        len(selected_articles),
        len(kept_qrels),
        sum(len(v) for v in kept_qrels.values()),
    )

    corpus = _build_corpus(selected_set, args.num_workers)
    corpus_ids = set(corpus["str_docid"])

    missing_docs = {
        doc_id
        for judgments in kept_qrels.values()
        for doc_id in judgments
        if doc_id not in corpus_ids
    }
    if missing_docs:
        raise RuntimeError(
            f"{len(missing_docs)} relevant documents are missing from the pilot corpus; "
            "the article filter and the qrels disagree"
        )

    corpus.save_to_disk(str(out_dir / "corpus"))
    with open(out_dir / "pruned_qrels.json", "w") as handle:
        json.dump(kept_qrels, handle)
    with open(out_dir / "pruned_q_map.json", "w") as handle:
        json.dump(kept_query_map, handle)
    with open(out_dir / "articles.json", "w") as handle:
        json.dump(selected_articles, handle)

    metadata = {
        "kind": "train_pilot",
        "dataset": DATASET_NAME,
        "corpus": CORPUS_NAME,
        "country": data.COUNTRY,
        "revision": data.DATASET_VERSION,
        "filter": {"match_score": 1.0, "expanded_search": False},
        "seed": args.seed,
        "requested_articles": args.n_articles,
        "n_articles": len(selected_articles),
        "n_documents": int(len(corpus)),
        "n_queries": len(kept_qrels),
        "n_judgments": int(sum(len(v) for v in kept_qrels.values())),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
    }
    with open(out_dir / "metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)

    log.info("Pilot written to %s", out_dir)
    log.info("  corpus documents: %d", metadata["n_documents"])
    log.info("  queries:          %d", metadata["n_queries"])
    log.info("  judgments:        %d", metadata["n_judgments"])
    log.info("  articles:         %d", metadata["n_articles"])


if __name__ == "__main__":
    main()
