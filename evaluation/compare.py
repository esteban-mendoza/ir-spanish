"""Compare retrieval runs with paired tests and multiple-comparison corrections.

Usage (from the repository root)::

    python -m evaluation.compare \
        --runs run_a.lz4 run_b.lz4 \
        --metric ndcg@10 \
        --test both \
        --correction none

For the six-algorithm comparison of Experiment 2 use ``--correction holm``
(primary) and repeat with ``--correction bh`` (sensitivity).
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
from ranx import evaluate

# utils.__init__ sets up logging and NUMA/threading env-vars
from utils import cache, data, retrieval

from .significance import CORRECTIONS, paired_t_test, permutation_test

log = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cache" / "messirve_embeddings"
REPO_ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "run_a",
    "run_b",
    "metric",
    "n_queries",
    "mean_a",
    "mean_b",
    "diff",
    "t_stat",
    "p_student",
    "p_permutation",
    "p_value",
    "p_adjusted",
    "significant",
    "test",
    "correction",
]


def load_qrels():
    dataset_cache_dir = cache.dataset_cache_base(
        CACHE_DIR,
        data.COUNTRY,
        data.DATASET_VERSION,
        data.MAX_WORD_COUNT,
    )
    qrels, _ = data.get_pruned_qrels_and_queries(
        data.COUNTRY,
        data.DATASET_VERSION,
        kept_doc_ids=None,
        dataset_cache_dir=dataset_cache_dir,
        num_workers=1,
    )
    return qrels


def per_query_scores(qrels, run, metric: str) -> np.ndarray:
    """Return one score per query, aligned with the qrels query order.

    ``ranx.evaluate`` 0.3.21 has a bug when a single parametrized metric
    (e.g. ``ndcg@10``) is requested with ``return_mean=False``: it looks up
    the base metric name instead of the full key.  Passing a second metric
    forces the dict branch and avoids the bug.
    """
    metrics = [metric] if metric == "map" else [metric, "map"]
    scores = evaluate(
        qrels=qrels,
        run=run,
        metrics=metrics,
        return_mean=False,
        make_comparable=True,
        save_results_in_run=False,
    )
    if isinstance(scores, dict):
        scores = scores[metric]
    return np.asarray(scores, dtype=np.float64)


def compare_runs(
    qrels,
    runs: list,
    names: list[str],
    metric: str,
    test: str,
    correction: str,
    n_permutations: int,
    seed: int,
    alpha: float,
    limit: int | None = None,
) -> list[dict]:
    scores = {}
    for name, run in zip(names, runs):
        values = per_query_scores(qrels, run, metric)
        if limit is not None:
            values = values[:limit]
        scores[name] = values

    rows: list[dict] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = scores[names[i]]
            b = scores[names[j]]
            t_stat, p_student = paired_t_test(a, b)
            p_permutation = None
            if test in {"permutation", "both"}:
                p_permutation = permutation_test(
                    a,
                    b,
                    n_permutations=n_permutations,
                    seed=seed,
                )

            if test == "permutation":
                p_value = p_permutation
            else:
                p_value = p_student

            rows.append(
                {
                    "run_a": names[i],
                    "run_b": names[j],
                    "metric": metric,
                    "n_queries": int(a.size),
                    "mean_a": float(np.mean(a)),
                    "mean_b": float(np.mean(b)),
                    "diff": float(np.mean(a) - np.mean(b)),
                    "t_stat": t_stat,
                    "p_student": p_student,
                    "p_permutation": p_permutation,
                    "p_value": float(p_value),
                    "p_adjusted": float(p_value),
                    "significant": False,
                    "test": test,
                    "correction": correction,
                }
            )

    pvalues = np.array([row["p_value"] for row in rows], dtype=float)
    adjusted = CORRECTIONS[correction](pvalues)
    for row, p_adj in zip(rows, adjusted):
        row["p_adjusted"] = float(p_adj)
        row["significant"] = bool(p_adj <= alpha)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row[key] is None else row[key]) for key in FIELDS})


def format_p(value: float) -> str:
    """Format a p-value with the thesis convention for the resolution floor."""
    if value < 1e-4:
        return "< 0.0001"
    return f"{value:.4g}"


def write_markdown(rows: list[dict], path: Path) -> None:
    header = (
        "| sistema A | sistema B | media A | media B | Δ | p | p ajustada | significativa |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |\n"
    )
    lines = []
    for row in rows:
        lines.append(
            "| {a} | {b} | {ma:.4f} | {mb:.4f} | {d:+.4f} | {p} | {pa} | {sig} |".format(
                a=row["run_a"],
                b=row["run_b"],
                ma=row["mean_a"],
                mb=row["mean_b"],
                d=row["diff"],
                p=format_p(row["p_value"]),
                pa=format_p(row["p_adjusted"]),
                sig="sí" if row["significant"] else "no",
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="Paths to .lz4 runs.")
    parser.add_argument("--names", nargs="+", default=None, help="Display names (one per run).")
    parser.add_argument("--metric", default="ndcg@10", help="Metric to compare (default: ndcg@10).")
    parser.add_argument(
        "--test",
        choices=["student", "permutation", "both"],
        default="student",
        help="Paired test; 'both' runs the t-test and the permutation test.",
    )
    parser.add_argument(
        "--correction",
        choices=sorted(CORRECTIONS),
        default="none",
        help="Multiple-comparison correction (default: none).",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    parser.add_argument(
        "--permutations",
        type=int,
        default=10_000,
        help="Permutation replicates (default: 10000).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for permutations.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N queries (smoke tests; not for final results).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "evaluation" / "comparisons.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Markdown output path (default: same stem as --output).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = args.names
    if names is not None and len(names) != len(args.runs):
        raise SystemExit("--names must have one entry per --runs path")
    if len(args.runs) < 2:
        raise SystemExit("At least two runs are required")

    log.info("Loading qrels")
    qrels = load_qrels()

    runs = []
    if names is None:
        names = []
        for path in args.runs:
            name = Path(path).stem
            names.append(name)
    for name, path in zip(names, args.runs):
        log.info("Loading %s from %s", name, path)
        run = retrieval.load_run(path)
        run.name = name
        runs.append(run)

    rows = compare_runs(
        qrels=qrels,
        runs=runs,
        names=names,
        metric=args.metric,
        test=args.test,
        correction=args.correction,
        n_permutations=args.permutations,
        seed=args.seed,
        alpha=args.alpha,
        limit=args.limit,
    )

    markdown = args.markdown or args.output.with_suffix(".md")
    write_csv(rows, args.output)
    write_markdown(rows, markdown)

    for row in rows:
        log.info(
            "%s vs %s: Δ=%+.4f p=%.4g p_adj=%.4g %s",
            row["run_a"],
            row["run_b"],
            row["diff"],
            row["p_value"],
            row["p_adjusted"],
            "significant" if row["significant"] else "not significant",
        )
    log.info("CSV: %s", args.output)
    log.info("Markdown: %s", markdown)


if __name__ == "__main__":
    main()
