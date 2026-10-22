"""Experiment 5: relationship between RBO complementarity and fusion gain.

For every subset of the Stage A sweep, the complementarity of the subset is the
mean RBO of its member pairs.  The gain is the fusion nDCG@10 minus the
nDCG@10 of the best member.  The script reports Pearson and Spearman
correlations with a bootstrap interval over combinations.

The Stage A sweep covers four algorithms (CombMNZ, RBC, RRF and ISR).  The
correlation is computed for each one, so the relationship can be checked for
dependence on the fusion algorithm:

    python -m analysis.rbo_gain --strategy all
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_A = REPO_ROOT / "results" / "fuse_sweep" / "results.csv"
RBO = REPO_ROOT / "results" / "rbo" / "rbo_pairs.csv"
OUT_DIR = REPO_ROOT / "results" / "e5"

INDIVIDUAL = {
    "bm25": (0.1848, 0.5725),
    "splade-v3": (0.1956, 0.5725),
    "qwen3-0.6b": (0.4468, 0.8423),
    "e5-large": (0.4675, 0.8666),
    "bge-m3": (0.4818, 0.8741),
    "jina-v5-small": (0.5115, 0.9039),
}

STRATEGIES = ("combmnz", "rbc", "rrf", "isr")
LABELS = {"combmnz": "CombMNZ", "rbc": "RBC (φ=0.95)", "rrf": "RRF (k=60)", "isr": "ISR"}
N_BOOTSTRAP = 10_000
SEED = 42


def load_rbo() -> dict[tuple[str, str], float]:
    values = {}
    with RBO.open(newline="") as handle:
        for row in csv.DictReader(handle):
            values[tuple(sorted((row["model_a"], row["model_b"])))] = float(row["mean_rbo"])
    return values


def subset_complementarity(members: list[str], rbo: dict[tuple[str, str], float]) -> float:
    pairs = list(itertools.combinations(members, 2))
    return float(np.mean([rbo[tuple(sorted(pair))] for pair in pairs]))


def load_rows(strategy: str = "combmnz") -> list[dict]:
    rows = []
    rbo = load_rbo()
    with STAGE_A.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["strategy"] != strategy:
                continue
            members = row["combo"].split("+")
            ndcg = float(row["ndcg@10"])
            recall = float(row["recall@100"])
            best_ndcg = max(INDIVIDUAL[member][0] for member in members)
            best_recall = max(INDIVIDUAL[member][1] for member in members)
            rows.append(
                {
                    "combo": row["combo"],
                    "size": len(members),
                    "mean_rbo": subset_complementarity(members, rbo),
                    "fusion_ndcg": ndcg,
                    "best_member_ndcg": best_ndcg,
                    "gain_ndcg": ndcg - best_ndcg,
                    "fusion_recall": recall,
                    "best_member_recall": best_recall,
                    "gain_recall": recall - best_recall,
                }
            )
    return rows


def correlations(rows: list[dict], gain_key: str) -> dict:
    x = np.array([row["mean_rbo"] for row in rows], dtype=float)
    y = np.array([row[gain_key] for row in rows], dtype=float)
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return {
        "n": len(rows),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def bootstrap(rows: list[dict], gain_key: str, n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    x = np.array([row["mean_rbo"] for row in rows], dtype=float)
    y = np.array([row[gain_key] for row in rows], dtype=float)
    pearson_values = []
    spearman_values = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(rows), len(rows))
        xs = x[indices]
        ys = y[indices]
        if np.std(xs) == 0 or np.std(ys) == 0:
            continue
        pearson_values.append(pearsonr(xs, ys).statistic)
        spearman_values.append(spearmanr(xs, ys).statistic)
    return {
        "pearson_ci95": [float(np.percentile(pearson_values, 2.5)), float(np.percentile(pearson_values, 97.5))],
        "spearman_ci95": [float(np.percentile(spearman_values, 2.5)), float(np.percentile(spearman_values, 97.5))],
        "n_bootstrap": len(pearson_values),
    }


def run_strategy(strategy: str) -> dict:
    """Compute the RBO-gain analysis for one fusion algorithm."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows(strategy)
    if not rows:
        raise SystemExit(f"No rows for strategy {strategy}")
    rows.sort(key=lambda row: (row["size"], -row["gain_ndcg"]))

    suffix = "" if strategy == "combmnz" else f"_{strategy}"
    csv_path = OUT_DIR / f"rbo_gain{suffix}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    rows_ge_3 = [row for row in rows if row["size"] >= 3]
    summary = {
        "strategy": strategy,
        "label": LABELS[strategy],
        "ndcg": {
            "all": correlations(rows, "gain_ndcg"),
            "size_ge_3": correlations(rows_ge_3, "gain_ndcg"),
            "bootstrap": bootstrap(rows, "gain_ndcg"),
        },
        "recall": {
            "all": correlations(rows, "gain_recall"),
            "size_ge_3": correlations(rows_ge_3, "gain_recall"),
            "bootstrap": bootstrap(rows, "gain_recall"),
        },
    }
    summary_path = OUT_DIR / f"summary{suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"CSV: {csv_path}")
    print(f"JSON: {summary_path}")
    return summary


def write_comparison(summaries: list[dict]) -> None:
    """Write a cross-algorithm comparison table."""
    lines = [
        "# Experimento 5 por algoritmo",
        "",
        "| Algoritmo | n | nDCG: Pearson (IC 95 %) | nDCG: Spearman | Recall: Pearson (IC 95 %) | Recall: Spearman |",
        "| --- | ---: | --- | ---: | --- | ---: |",
    ]
    for s in summaries:
        nd, rc = s["ndcg"], s["recall"]
        lines.append(
            "| {label} | {n} | {r:.3f} ({lo:.3f}, {hi:.3f}) | {rho:.3f} | {rr:.3f} ({rlo:.3f}, {rhi:.3f}) | {rrho:.3f} |".format(
                label=s["label"],
                n=nd["all"]["n"],
                r=nd["all"]["pearson_r"],
                lo=nd["bootstrap"]["pearson_ci95"][0],
                hi=nd["bootstrap"]["pearson_ci95"][1],
                rho=nd["all"]["spearman_rho"],
                rr=rc["all"]["pearson_r"],
                rlo=rc["bootstrap"]["pearson_ci95"][0],
                rhi=rc["bootstrap"]["pearson_ci95"][1],
                rrho=rc["all"]["spearman_rho"],
            )
        )
    path = OUT_DIR / "summary_all.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_DIR / "summary_all.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"TABLA: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        default="combmnz",
        choices=(*STRATEGIES, "all"),
        help="Fusion algorithm to analyse (default: combmnz; 'all' runs the four).",
    )
    args = parser.parse_args()
    if args.strategy == "all":
        summaries = [run_strategy(name) for name in STRATEGIES]
        write_comparison(summaries)
    else:
        run_strategy(args.strategy)


if __name__ == "__main__":
    main()
