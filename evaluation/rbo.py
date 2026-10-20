"""Rank-Biased Overlap (RBO) between retrieval runs.

RBO measures the similarity of two ranked lists with a geometric weighting of
depth.  It is a descriptive complementarity measure, not a significance test.
The persistence parameter ``p`` is the probability of continuing to the next
rank; the expected evaluation depth is ``1 / (1 - p)``.

The calculation is delegated to the open-source ``ranked-overlap`` package
(MIT), which implements the extrapolated finite-list equation of Webber,
Moffat and Zobel (2010).  The thesis cites the original RBO paper; the package
is only the implementation used by the repository.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from ranked_overlap import rbo as _rbo

from utils import retrieval

log = logging.getLogger(__name__)

DEFAULT_P = 0.95
DEFAULT_DEPTH = 100


def rbo(a: list[str], b: list[str], p: float = DEFAULT_P, depth: int = DEFAULT_DEPTH) -> float:
    """Return the extrapolated RBO of two ranked lists, truncated to ``depth``."""
    return float(_rbo(a[:depth], b[:depth], p=p))


def rbo_between_runs(
    run_a,
    run_b,
    p: float = DEFAULT_P,
    depth: int = DEFAULT_DEPTH,
    limit: int | None = None,
) -> tuple[float, int]:
    """Average RBO over the queries present in both runs."""
    query_ids = list(run_a.get_query_ids())
    if limit is not None:
        query_ids = query_ids[:limit]

    values = []
    for query_id in query_ids:
        if query_id not in run_b.run:
            continue
        list_a = list(run_a.run[query_id].keys())[:depth]
        list_b = list(run_b.run[query_id].keys())[:depth]
        if not list_a or not list_b:
            continue
        values.append(rbo(list_a, list_b, p=p, depth=depth))

    if not values:
        return 0.0, 0
    return sum(values) / len(values), len(values)


def compare_all_pairs(
    runs: list,
    names: list[str],
    p: float = DEFAULT_P,
    depth: int = DEFAULT_DEPTH,
    limit: int | None = None,
) -> list[dict]:
    rows = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            log.info("RBO %s vs %s", names[i], names[j])
            mean_rbo, n_queries = rbo_between_runs(
                runs[i],
                runs[j],
                p=p,
                depth=depth,
                limit=limit,
            )
            rows.append(
                {
                    "model_a": names[i],
                    "model_b": names[j],
                    "p": p,
                    "depth": depth,
                    "mean_rbo": mean_rbo,
                    "n_queries": n_queries,
                }
            )
            log.info("  RBO = %.4f (%d queries)", mean_rbo, n_queries)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="Paths to .lz4 runs.")
    parser.add_argument("--names", nargs="+", required=True, help="Display names (one per run).")
    parser.add_argument("--p", type=float, default=DEFAULT_P, help="RBO persistence (default: 0.95).")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="Maximum depth (default: 100).")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N queries (smoke tests; not for final results).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "rbo" / "rbo_pairs.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.runs) != len(args.names):
        raise SystemExit("--names must have one entry per --runs path")
    if len(args.runs) < 2:
        raise SystemExit("At least two runs are required")

    runs = []
    for name, path in zip(args.names, args.runs):
        log.info("Loading %s from %s", name, path)
        run = retrieval.load_run(path)
        run.name = name
        runs.append(run)

    rows = compare_all_pairs(
        runs=runs,
        names=args.names,
        p=args.p,
        depth=args.depth,
        limit=args.limit,
    )
    write_csv(rows, args.output)
    log.info("CSV: %s", args.output)


if __name__ == "__main__":
    main()
