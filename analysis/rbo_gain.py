"""Experiment 5: relationship between RBO complementarity and fusion gain.

For every subset of the Stage A sweep (CombMNZ), the complementarity of the
subset is the mean RBO of its member pairs.  The gain is the fusion nDCG@10
minus the nDCG@10 of the best member.  The script reports Pearson and
Spearman correlations with a bootstrap interval over combinations.
"""

from __future__ import annotations

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


def load_rows() -> list[dict]:
    rows = []
    rbo = load_rbo()
    with STAGE_A.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["strategy"] != "combmnz":
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    rows.sort(key=lambda row: (row["size"], -row["gain_ndcg"]))

    csv_path = OUT_DIR / "rbo_gain.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    rows_ge_3 = [row for row in rows if row["size"] >= 3]
    summary = {
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
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"CSV: {csv_path}")
    print(f"JSON: {summary_path}")


if __name__ == "__main__":
    main()
