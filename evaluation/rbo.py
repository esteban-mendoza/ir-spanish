"""Rank-Biased Overlap (RBO) between retrieval runs.

RBO measures the similarity of two ranked lists with a geometric weighting of
depth.  It is a descriptive complementarity measure, not a significance test.
The persistence parameter ``p`` is the probability of continuing to the next
rank; the expected evaluation depth is ``1 / (1 - p)``.

The calculation uses ``pyterrier_alpha.rbo``, the RBO measure of the official
PyTerrier Alpha extension.  PyTerrier 1.1.2 does not include RBO in its core;
the extension is the PyTerrier implementation of the measure.  It follows the
definition of Webber, Moffat and Zobel (2010), which is the citation required
by the PyTerrier documentation.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import pandas as pd
from pyterrier_alpha import rbo as _rbo

from utils import retrieval

log = logging.getLogger(__name__)

DEFAULT_P = 0.95
DEFAULT_DEPTH = 100
DEFAULT_BATCH_SIZE = 5000


def _frame(run, query_ids: list, depth: int) -> pd.DataFrame:
    """Build the ranking frame expected by ``pyterrier_alpha.rbo``."""
    rows = []
    for query_id in query_ids:
        ranking = run.run[query_id] if query_id in run.run else {}
        for rank, docno in enumerate(list(ranking)[:depth]):
            # The measure sorts by score; -rank preserves the run order even
            # when the original scores contain ties.
            rows.append((query_id, docno, -float(rank)))
    return pd.DataFrame(rows, columns=["qid", "docno", "score"])


def rbo(a: list[str], b: list[str], p: float = DEFAULT_P, depth: int = DEFAULT_DEPTH) -> float:
    """Return the RBO of two ranked lists, truncated to ``depth``."""
    if not a or not b:
        return 0.0
    left = _frame_from_list(a[:depth])
    right = _frame_from_list(b[:depth])
    return float(next(iter(_rbo(left, right, p=p)))[1])


def _frame_from_list(ranking: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "qid": [0] * len(ranking),
            "docno": ranking,
            "score": [-float(rank) for rank in range(len(ranking))],
        }
    )


def rbo_between_runs(
    run_a,
    run_b,
    p: float = DEFAULT_P,
    depth: int = DEFAULT_DEPTH,
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[float, int]:
    """Average RBO over the queries present in both runs."""
    query_ids = [qid for qid in run_a.get_query_ids() if qid in run_b.run]
    if limit is not None:
        query_ids = query_ids[:limit]

    total = 0.0
    count = 0
    for start in range(0, len(query_ids), batch_size):
        batch = query_ids[start : start + batch_size]
        left = _frame(run_a, batch, depth)
        right = _frame(run_b, batch, depth)
        if left.empty or right.empty:
            continue
        values = list(_rbo(left, right, p=p))
        total += sum(value for _, value in values)
        count += len(values)

    if count == 0:
        return 0.0, 0
    return total / count, count


def compare_all_pairs(
    runs: list,
    names: list[str],
    p: float = DEFAULT_P,
    depth: int = DEFAULT_DEPTH,
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
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
                batch_size=batch_size,
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
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Queries per PyTerrier call (default: 5000).",
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
        batch_size=args.batch_size,
    )
    write_csv(rows, args.output)
    log.info("CSV: %s", args.output)


if __name__ == "__main__":
    main()
