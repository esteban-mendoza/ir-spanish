#!/usr/bin/env python3
"""
Fuse multiple retrieval runs with RRF (or other ranx-supported strategies)
and evaluate the fused result.

Edit the RUNS and STRATEGY constants below to change which models are fused
and which fusion method is used.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path

# utils.__init__ sets up logging and NUMA/threading env-vars
from utils import cache, data, retrieval

from ranx import fuse

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ALIASES = {
    "bm25_pyserini": "bm25",
    "naver/splade-v3": "splade-v3",
    "Qwen/Qwen3-Embedding-0.6B": "qwen3-0.6b",
    "intfloat/multilingual-e5-large-instruct": "e5-large",
    "BAAI/bge-m3": "bge-m3",
    "jinaai/jina-embeddings-v5-text-small-retrieval": "jina-v5-small",
}


def short_alias(slug: str) -> str:
    return MODEL_ALIASES.get(slug, slug.split("/")[-1].lower())


@dataclass(frozen=True)
class FusionStrategy:
    method: str
    params: dict | None = None


STRATEGIES = {
    "rrf": FusionStrategy(method="rrf", params={"k": 60}),
    "rbc": FusionStrategy(method="rbc", params={"phi": 0.9}),
}

STRATEGY = "rbc"

RUNS = [
    "bm25_pyserini",
    "naver/splade-v3",
    "Qwen/Qwen3-Embedding-0.6B",
    "intfloat/multilingual-e5-large-instruct",
    "BAAI/bge-m3",
    "jinaai/jina-embeddings-v5-text-small-retrieval",
]

MAX_QUERY_LENGTH = 512
MAX_DOC_LENGTH = 512
CACHE_DIR = Path.home() / ".cache" / "messirve_embeddings"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_runs(model_names: list[str]) -> list:
    runs = []
    for model_name in model_names:
        base = cache.cache_base(
            CACHE_DIR,
            model_name,
            data.COUNTRY,
            data.DATASET_VERSION,
            MAX_QUERY_LENGTH,
            MAX_DOC_LENGTH,
            data.MAX_WORD_COUNT,
        )
        run_path = cache.run_cache_path(base)
        log.info("Loading run for %s from %s", model_name, run_path)
        run = retrieval.load_run(run_path)
        run.name = model_name.split("/")[-1]
        runs.append(run)
    return runs


def load_qrels():
    dataset_cache_dir = cache.dataset_cache_base(
        CACHE_DIR, data.COUNTRY, data.DATASET_VERSION, data.MAX_WORD_COUNT
    )
    qrels, _ = data.get_pruned_qrels_and_queries(
        data.COUNTRY,
        data.DATASET_VERSION,
        kept_doc_ids=None,
        dataset_cache_dir=dataset_cache_dir,
        num_workers=1,
    )
    return qrels


def fuse_and_evaluate(qrels, runs: list, strategy: FusionStrategy, *, mode="verbose"):
    log.info(
        "Fusing %d runs with strategy '%s' (params: %s)",
        len(runs),
        strategy.method,
        strategy.params,
    )
    fused_run = fuse(runs=runs, method=strategy.method, params=strategy.params or {})
    fused_run.name = f"fused_{strategy.method}"
    return retrieval.run_evaluation(
        qrels,
        fused_run,
        f"fused({strategy.method})",
        mode=mode,
        strategy=strategy.method,
        params=_format_params(strategy.params),
    )


def _format_params(params: dict | None) -> str:
    if not params:
        return ""
    return ",".join(f"{k}={v}" for k, v in params.items())


def sweep_combinations(
    qrels,
    all_runs: list,
    all_slugs: list[str],
    strategy: FusionStrategy,
) -> str:
    header = "| model | strategy | params | ndcg@10 | recall@100 |"
    separator = "|---|---|---|---|---|"
    rows = [header, separator]

    for size in range(len(all_runs), 1, -1):
        for indices in itertools.combinations(range(len(all_runs)), size):
            combo_runs = [all_runs[i] for i in indices]
            combo_slugs = [all_slugs[i] for i in indices]
            combo_name = "+".join(short_alias(s) for s in combo_slugs)

            fused_run = fuse(
                runs=combo_runs,
                method=strategy.method,
                params=strategy.params or {},
            )
            fused_run.name = combo_name

            results = retrieval.run_evaluation(
                qrels,
                fused_run,
                combo_name,
                mode="short",
                strategy=strategy.method,
                params=_format_params(strategy.params),
            )
            ndcg = results["ndcg@10"]
            recall = results["recall@100"]
            row = f"| {combo_name} | {strategy.method} | {_format_params(strategy.params)} | {ndcg:.4f} | {recall:.4f} |"
            rows.append(row)

    table = "\n".join(rows)
    log.info("Sweep results:\n%s", table)
    return table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    runs = load_runs(RUNS)
    qrels = load_qrels()
    sweep_combinations(qrels, runs, RUNS, STRATEGIES[STRATEGY])


if __name__ == "__main__":
    main()
