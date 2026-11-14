#!/usr/bin/env python3
"""Compara SPLADE-v3 en español con los seis modelos de primera etapa.

Escribe en results/splade_v3_spanish/:
  single_models.csv/.md  — las seis métricas de los siete sistemas
  paired.csv/.md         — splade-v3-es frente a cada modelo, con t pareada,
                           IC 95 % y Holm dentro de cada métrica primaria
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np

# utils.__init__ configura logging y las variables de NUMA
from utils import retrieval

from .compare import format_p, per_query_scores
from .significance import holm_bonferroni, paired_mean_ci, paired_t_test
from rerankers.fuse_sweep import MODELS, ModelConfig, load_qrels, load_runs

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "splade_v3_spanish"

NEW_MODEL = ModelConfig("alrolo3/splade-v3-spanish-vllm", "splade-v3-es", 512, 512)
MODELS_ALL = MODELS + [NEW_MODEL]
PRIMARY_METRICS = ["ndcg@10", "recall@100"]
TABLE_METRICS = ["ndcg@10", "recall@100", "mrr@10", "map", "precision@10", "precision@50"]


def family(alias: str) -> str:
    if alias == "bm25":
        return "léxico"
    if alias.startswith("splade"):
        return "disperso"
    return "denso"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    qrels = load_qrels()
    runs = load_runs(MODELS_ALL)
    names = [model.alias for model in MODELS_ALL]

    # --- Métricas de los siete sistemas ---
    rows = []
    for model, run in zip(MODELS_ALL, runs):
        result = retrieval.run_evaluation(qrels, run, model.alias, mode="quiet")
        rows.append(
            {
                "model": model.alias,
                "family": family(model.alias),
                **{metric: float(result[metric]) for metric in TABLE_METRICS},
            }
        )
    rows.sort(key=lambda row: -row["ndcg@10"])

    with (OUT_DIR / "single_models.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "model", "family", *TABLE_METRICS])
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            writer.writerow({"rank": rank, **row})
            log.info("%2d  %-20s %8.4f %8.4f", rank, row["model"], row["ndcg@10"], row["recall@100"])

    header = "| # | modelo | familia | nDCG@10 | Recall@100 | MRR@10 | MAP | P@10 | P@50 |\n"
    header += "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    lines = [
        "| {rank} | `{model}` | {family} | {ndcg@10:.4f} | {recall@100:.4f} | {mrr@10:.4f} | "
        "{map:.4f} | {precision@10:.4f} | {precision@50:.4f} |".format(rank=rank, **row)
        for rank, row in enumerate(rows, 1)
    ]
    (OUT_DIR / "single_models.md").write_text(header + "\n".join(lines) + "\n")

    # --- Comparaciones pareadas del modelo nuevo contra cada base ---
    scores = {
        metric: {name: per_query_scores(qrels, run, metric) for name, run in zip(names, runs)}
        for metric in PRIMARY_METRICS
    }

    paired = []
    for metric in PRIMARY_METRICS:
        a = scores[metric][NEW_MODEL.alias]
        comparisons = []
        for name in names:
            if name == NEW_MODEL.alias:
                continue
            b = scores[metric][name]
            t_stat, p_value = paired_t_test(a, b)
            ci = paired_mean_ci(a, b)
            comparisons.append(
                {
                    "metric": metric,
                    "system_a": NEW_MODEL.alias,
                    "system_b": name,
                    "mean_a": float(np.mean(a)),
                    "mean_b": float(np.mean(b)),
                    "diff": float(np.mean(a) - np.mean(b)),
                    "ci_low": float(ci["low"]),
                    "ci_high": float(ci["high"]),
                    "t_stat": t_stat,
                    "p_student": p_value,
                }
            )
        adjusted = holm_bonferroni(np.array([row["p_student"] for row in comparisons]))
        for row, p_adjusted in zip(comparisons, adjusted):
            row["p_holm"] = float(p_adjusted)
            row["significant"] = bool(p_adjusted <= 0.05)
            paired.append(row)
            log.info(
                "%-9s %-20s Δ=%+.4f IC=[%+.4f, %+.4f] p_holm=%.4g sig=%s",
                metric, row["system_b"], row["diff"], row["ci_low"], row["ci_high"],
                row["p_holm"], row["significant"],
            )

    fields = [
        "metric", "system_a", "system_b", "mean_a", "mean_b", "diff",
        "ci_low", "ci_high", "t_stat", "p_student", "p_holm", "significant",
    ]
    with (OUT_DIR / "paired.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in paired:
            writer.writerow(row)

    md = (
        "| métrica | A | B | media A | media B | Δ | IC 95 % | p Holm | significativa |\n"
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | :---: |\n"
    )
    md += "\n".join(
        "| {metric} | {system_a} | {system_b} | {mean_a:.4f} | {mean_b:.4f} | {diff:+.4f} | "
        "[{ci_low:+.4f}, {ci_high:+.4f}] | {p} | {sig} |".format(
            p=format_p(row["p_holm"]), sig="sí" if row["significant"] else "no", **row
        )
        for row in paired
    )
    (OUT_DIR / "paired.md").write_text(md + "\n")
    log.info("Escrito en %s", OUT_DIR)


if __name__ == "__main__":
    main()
