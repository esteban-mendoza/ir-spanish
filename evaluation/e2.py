"""Experiment 2: paired comparisons among the six fusion algorithms on B1.

All 15 pairs are computed once; the Holm-Bonferroni and Benjamini-Hochberg
adjustments are then applied to the same p-values and written as two tables.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from utils import retrieval

from .compare import compare_runs, load_qrels, write_csv, write_markdown
from .significance import CORRECTIONS

log = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cache" / "messirve_embeddings"
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "evaluation"

COMBO = "e5-large+bge-m3+jina-v5-small"
STAGE_B = CACHE_DIR / "fusions_stage_b"
CONDORCET = CACHE_DIR / "fusions_stage_b_condorcet"

NAMES = ["CombMNZ", "RRF", "ISR", "RBC", "BordaFuse", "Condorcet"]
PATHS = [
    STAGE_B / f"combmnz__{COMBO}.lz4",
    STAGE_B / f"rrf__{COMBO}.lz4",
    STAGE_B / f"isr__{COMBO}.lz4",
    STAGE_B / f"rbc__{COMBO}.lz4",
    STAGE_B / f"bordafuse__{COMBO}.lz4",
    CONDORCET / f"condorcet__{COMBO}.lz4",
]

METRIC = "ndcg@10"
ALPHA = 0.05
N_PERMUTATIONS = 10_000
SEED = 42


def main() -> None:
    missing = [str(path) for path in PATHS if not path.exists()]
    if missing:
        raise SystemExit("Missing runs:\n" + "\n".join(missing))

    qrels = load_qrels()
    runs = []
    for name, path in zip(NAMES, PATHS):
        log.info("Loading %s from %s", name, path)
        run = retrieval.load_run(path)
        run.name = name
        runs.append(run)

    rows = compare_runs(
        qrels=qrels,
        runs=runs,
        names=NAMES,
        metric=METRIC,
        test="both",
        correction="none",
        n_permutations=N_PERMUTATIONS,
        seed=SEED,
        alpha=ALPHA,
    )
    pvalues = np.array([row["p_value"] for row in rows], dtype=float)

    for correction in ("holm", "bh"):
        adjusted = CORRECTIONS[correction](pvalues)
        for row, p_adj in zip(rows, adjusted):
            row["p_adjusted"] = float(p_adj)
            row["significant"] = bool(p_adj <= ALPHA)
            row["correction"] = correction

        csv_path = OUT_DIR / f"e2_{correction}.csv"
        md_path = OUT_DIR / f"e2_{correction}.md"
        write_csv(rows, csv_path)
        write_markdown(rows, md_path)
        log.info("%s: %s", correction, csv_path)
        log.info("%s: %s", correction, md_path)


if __name__ == "__main__":
    main()
