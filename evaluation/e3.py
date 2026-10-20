"""Experiment 3: best fusion vs. the three neural rerankers.

The family has three comparisons (best fusion against each reranker), so the
Holm-Bonferroni correction is applied over three p-values.  The two reranker
comparisons are not a family of six: the question is whether fusion is
competitive with each reranker, not how the rerankers compare among themselves.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from utils import retrieval

from .compare import load_qrels, per_query_scores, write_csv, write_markdown
from .significance import holm_bonferroni, paired_t_test, permutation_test

log = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cache" / "messirve_embeddings"
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "evaluation"

BEST_FUSION = CACHE_DIR / "fusions_stage_b" / "combmnz__e5-large+bge-m3+jina-v5-small.lz4"
RERANKERS = {
    "bge-reranker-v2-m3": CACHE_DIR
    / "BAAI__bge-reranker-v2-m3"
    / "over__jinaai__jina-embeddings-v5-text-small-retrieval"
    / "full_v1.2_q512_d512_nofilter"
    / "retrieval_run.lz4",
    "jina-reranker-v3": CACHE_DIR
    / "jinaai__jina-reranker-v3"
    / "over__jinaai__jina-embeddings-v5-text-small-retrieval"
    / "full_v1.2_q1024_d8192_nofilter"
    / "retrieval_run.lz4",
    "jina-colbert-v2": CACHE_DIR
    / "jinaai__jina-colbert-v2"
    / "over__jinaai__jina-embeddings-v5-text-small-retrieval"
    / "full_v1.2_q32_d300_nofilter"
    / "retrieval_run.lz4",
}

METRIC = "ndcg@10"
ALPHA = 0.05
N_PERMUTATIONS = 10_000
SEED = 42


def main() -> None:
    if not BEST_FUSION.exists():
        raise SystemExit(f"Missing fused run: {BEST_FUSION}")

    qrels = load_qrels()
    names = ["B1-CombMNZ", *RERANKERS]
    paths = [BEST_FUSION, *RERANKERS.values()]

    runs = []
    for name, path in zip(names, paths):
        log.info("Loading %s from %s", name, path)
        run = retrieval.load_run(path)
        run.name = name
        runs.append(run)

    scores = {name: per_query_scores(qrels, run, METRIC) for name, run in zip(names, runs)}
    base = scores[names[0]]

    rows = []
    for name in names[1:]:
        other = scores[name]
        t_stat, p_student = paired_t_test(base, other)
        p_permutation = permutation_test(
            base,
            other,
            n_permutations=N_PERMUTATIONS,
            seed=SEED,
        )
        rows.append(
            {
                "run_a": names[0],
                "run_b": name,
                "metric": METRIC,
                "n_queries": int(base.size),
                "mean_a": float(np.mean(base)),
                "mean_b": float(np.mean(other)),
                "diff": float(np.mean(base) - np.mean(other)),
                "t_stat": t_stat,
                "p_student": p_student,
                "p_permutation": p_permutation,
                "p_value": float(p_student),
                "p_adjusted": float(p_student),
                "significant": False,
                "test": "both",
                "correction": "holm",
            }
        )

    adjusted = holm_bonferroni(np.array([row["p_value"] for row in rows], dtype=float))
    for row, p_adj in zip(rows, adjusted):
        row["p_adjusted"] = float(p_adj)
        row["significant"] = bool(p_adj <= ALPHA)
        log.info(
            "%s vs %s: Δ=%+.4f p=%.4g p_Holm=%.4g",
            row["run_a"],
            row["run_b"],
            row["diff"],
            row["p_value"],
            row["p_adjusted"],
        )

    csv_path = OUT_DIR / "e3.csv"
    md_path = OUT_DIR / "e3.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    log.info("CSV: %s", csv_path)
    log.info("Markdown: %s", md_path)


if __name__ == "__main__":
    main()
