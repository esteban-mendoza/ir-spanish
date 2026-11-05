#!/usr/bin/env python3
"""Write the per-system metrics table for Experiment 3.

The thesis table reports six metrics for the fusion and the three rerankers.
This script evaluates the same runs that ``evaluation.e3`` compares, so the
per-system table and the paired tests share the run paths and the qrels.

Usage:
    python -m evaluation.e3_systems --suffix _maxctx
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from ranx import evaluate

from utils import retrieval

from .compare import load_qrels
from .e3 import BEST_FUSION, RERANKERS

log = logging.getLogger(__name__)

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "evaluation"
METRICS = [
    "ndcg@10",
    "recall@100",
    "mrr@10",
    "map",
    "precision@10",
    "precision@50",
]
LABELS = {
    "B1-CombMNZ": "CombMNZ (fusión)",
    "bge-reranker-v2-m3": "bge-reranker-v2-m3",
    "jina-reranker-v3": "jina-reranker-v3",
    "jina-colbert-v2": "jina-colbert-v2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suffix",
        default="_maxctx",
        help="Suffix for the output files (default: _maxctx).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qrels = load_qrels()

    paths = {"B1-CombMNZ": BEST_FUSION, **RERANKERS}
    rows = []
    for name, path in paths.items():
        if not path.exists():
            raise SystemExit(f"Missing run for {name}: {path}")
        log.info("Evaluating %s from %s", name, path)
        run = retrieval.load_run(path)
        run.name = name
        metrics = evaluate(qrels, run, metrics=METRICS, make_comparable=True)
        rows.append({"sistema": LABELS[name], **{metric: metrics[metric] for metric in METRICS}})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"e3_systems{args.suffix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sistema", *METRICS])
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = OUT_DIR / f"e3_systems{args.suffix}.md"
    lines = [
        "| Sistema | nDCG@10 | Recall@100 | MRR@10 | MAP | P@10 | P@50 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {sistema} | {ndcg:.4f} | {recall:.4f} | {mrr:.4f} | {map:.4f} | "
            "{p10:.4f} | {p50:.4f} |".format(
                sistema=row["sistema"],
                ndcg=row["ndcg@10"],
                recall=row["recall@100"],
                mrr=row["mrr@10"],
                map=row["map"],
                p10=row["precision@10"],
                p50=row["precision@50"],
            )
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Wrote %s and %s", csv_path, markdown_path)


if __name__ == "__main__":
    main()
