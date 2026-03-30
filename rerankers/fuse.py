#!/usr/bin/env python3
"""
Fuse multiple retrieval runs with RRF (or other ranx-supported strategies)
and evaluate the fused result.

Edit the RUNS and STRATEGY constants below to change which models are fused
and which fusion method is used.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from the baselines package (sibling directory)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baselines"))

# utils.__init__ sets up logging and NUMA/threading env-vars
from utils import cache, data, retrieval  # noqa: E402

from ranx import fuse  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RUNS = [
    "intfloat/multilingual-e5-large-instruct",
    "BAAI/bge-m3",
]
STRATEGY = "rrf"
MAX_QUERY_LENGTH = 512
MAX_DOC_LENGTH = 512
CACHE_DIR = Path.home() / ".cache" / "messirve_embeddings"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
import logging  # noqa: E402

log = logging.getLogger(__name__)


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


def evaluate_individual_runs(qrels, model_names: list[str], runs: list):
    for model_name, run in zip(model_names, runs):
        retrieval.run_evaluation(qrels, run, model_name)


def fuse_and_evaluate(qrels, runs: list, strategy: str):
    log.info("Fusing %d runs with strategy '%s'", len(runs), strategy)
    fused_run = fuse(runs=runs, method=strategy)
    fused_run.name = f"fused_{strategy}"
    retrieval.run_evaluation(qrels, fused_run, f"fused({strategy})")


def main():
    runs = load_runs(RUNS)
    qrels = load_qrels()
    evaluate_individual_runs(qrels, RUNS, runs)
    fuse_and_evaluate(qrels, runs, STRATEGY)


if __name__ == "__main__":
    main()
