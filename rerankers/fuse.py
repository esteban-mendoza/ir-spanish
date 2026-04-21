#!/usr/bin/env python3
"""
Fuse multiple retrieval runs with rank fusion strategies and evaluate.

Edit the MODELS, STRATEGY, MIN_COMBO_SIZE, and MAX_COMBO_SIZE constants
below to configure which models are fused and how.
"""

from __future__ import annotations

import itertools
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from ranx import fuse

# utils.__init__ sets up logging and NUMA/threading env-vars
from utils import cache, data, retrieval

log = logging.getLogger(__name__)

NUM_WORKERS = os.cpu_count() or 8


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    name: str  # HuggingFace model name or custom identifier
    alias: str  # short display name for tables
    max_q_len: int  # max_query_length used when the model's run was cached
    max_d_len: int  # max_doc_length used when the model's run was cached


@dataclass(frozen=True)
class FusionStrategy:
    method: str
    params: dict | None = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS = [
    # ModelConfig("bm25_pyserini", "bm25", 512, 512),
    # ModelConfig("naver/splade-v3", "splade-v3", 512, 512),
    ModelConfig("intfloat/multilingual-e5-large-instruct", "e5-large", 512, 512),
    ModelConfig("BAAI/bge-m3", "bge-m3", 8192, 8192),
    # ModelConfig("Qwen/Qwen3-Embedding-0.6B", "qwen3-0.6b", 32768, 32768),
    ModelConfig("jinaai/jina-embeddings-v5-text-small-retrieval", "jina-v5-small", 32768, 32768),
    # ModelConfig("microsoft/harrier-oss-v1-0.6b", "harrier", 32768, 32768),
]

STRATEGIES = {
    "rrf": FusionStrategy(method="rrf", params={"k": 60}),
    "rbc": FusionStrategy(method="rbc", params={"phi": 0.95}),
    "combmnz": FusionStrategy(method="mnz"),
    "isr": FusionStrategy(method="isr"),
    "bordafuse": FusionStrategy(method="bordafuse"),
    "condorcet": FusionStrategy(method="condorcet"),
}

STRATEGY = "combmnz"

MIN_COMBO_SIZE = 2  # minimum number of models in a fusion combination
MAX_COMBO_SIZE = 3  # maximum number of models in a fusion combination

CACHE_DIR = Path.home() / ".cache" / "messirve_embeddings"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_runs(models: list[ModelConfig]) -> list:
    runs = []
    for m in models:
        base = cache.cache_base(
            CACHE_DIR,
            m.name,
            data.COUNTRY,
            data.DATASET_VERSION,
            m.max_q_len,
            m.max_d_len,
            data.MAX_WORD_COUNT,
        )
        run_path = cache.run_cache_path(base)
        log.info("Loading run for %s from %s", m.alias, run_path)
        run = retrieval.load_run(run_path)
        run.name = m.alias
        runs.append(run)
    return runs


def load_qrels():
    dataset_cache_dir = cache.dataset_cache_base(CACHE_DIR, data.COUNTRY, data.DATASET_VERSION, data.MAX_WORD_COUNT)
    qrels, _ = data.get_pruned_qrels_and_queries(
        data.COUNTRY,
        data.DATASET_VERSION,
        kept_doc_ids=None,
        dataset_cache_dir=dataset_cache_dir,
        num_workers=1,
    )
    return qrels


def _format_params(params: dict | None) -> str:
    if not params:
        return ""
    return ",".join(f"{k}={v}" for k, v in params.items())


def fuse_and_evaluate_all(
    qrels,
    all_runs: list,
    models: list[ModelConfig],
    strategy: FusionStrategy,
) -> str:
    """Fuse all combinations of models from MAX_COMBO_SIZE down to
    MIN_COMBO_SIZE and evaluate each. Returns a markdown table of results."""
    data_rows: list[list[str]] = []
    params_str = _format_params(strategy.params)

    for size in range(MAX_COMBO_SIZE, MIN_COMBO_SIZE - 1, -1):
        combos = list(itertools.combinations(range(len(all_runs)), size))
        single_combo = len(combos) == 1 and MIN_COMBO_SIZE == MAX_COMBO_SIZE

        for idx, indices in enumerate(combos, 1):
            combo_runs = [all_runs[i] for i in indices]
            combo_name = "+".join(models[i].alias for i in indices)

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
                mode="verbose" if single_combo else "inline",
                strategy=strategy.method,
                params=params_str,
            )
            data_rows.append(retrieval.results_row(combo_name, strategy.method, params_str, results))

            if not single_combo and idx % 10 == 0:
                log.info("Progress: %d/%d combinations (size=%d)", idx, len(combos), size)

    table = retrieval.md_table(data_rows)
    log.info("Results:\n%s", table)
    return table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    runs = load_runs(MODELS)
    qrels = load_qrels()
    fuse_and_evaluate_all(qrels, runs, MODELS, STRATEGIES[STRATEGY])


if __name__ == "__main__":
    main()
