"""Experiment 1: paired comparison of Recall@100 against the best first-stage model.

The six Stage B fusion algorithms on the dense trio are compared with
``jina-v5-small``, the best individual first-stage model.  For each fusion the
script reports the paired difference in Recall@100 and its 95 % confidence
interval; for the principal comparison (CombMNZ) it also reports the paired
t-test and permutation p-values.  Recall@100 is one of the two primary
effectiveness metrics of the thesis.

The interval is the paired Student t interval over the 170,055 test queries.
It is descriptive for the five non-principal fusions and inferential for the
principal comparison, which is corrected with the nDCG@10 p-value under
Holm-Bonferroni (the family of the two primary metrics).
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np

from utils import retrieval

from .compare import format_p, load_qrels, per_query_scores
from .significance import holm_bonferroni, paired_mean_ci, paired_t_test, permutation_test

log = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cache" / "messirve_embeddings"
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "evaluation"

COMBO = "e5-large+bge-m3+jina-v5-small"
STAGE_B = CACHE_DIR / "fusions_stage_b"
CONDORCET = CACHE_DIR / "fusions_stage_b_condorcet"

METRIC = "recall@100"
ALPHA = 0.05
N_PERMUTATIONS = 10_000
SEED = 42
N_QUERIES = 170_055

REFERENCE = (
    "jina-v5-small",
    CACHE_DIR
    / "jinaai__jina-embeddings-v5-text-small-retrieval"
    / "full_v1.2_q32768_d32768_nofilter"
    / "retrieval_run.lz4",
)

FUSIONS = [
    ("CombMNZ", STAGE_B / f"combmnz__{COMBO}.lz4"),
    ("RBC", STAGE_B / f"rbc__{COMBO}.lz4"),
    ("RRF", STAGE_B / f"rrf__{COMBO}.lz4"),
    ("Condorcet", CONDORCET / f"condorcet__{COMBO}.lz4"),
    ("BordaFuse", STAGE_B / f"bordafuse__{COMBO}.lz4"),
    ("ISR", STAGE_B / f"isr__{COMBO}.lz4"),
]

EXPECTED_RECALL = {
    "CombMNZ": 0.9231,
    "RBC": 0.9206,
    "RRF": 0.9234,
    "Condorcet": 0.8918,
    "BordaFuse": 0.9234,
    "ISR": 0.9230,
}
EXPECTED_REFERENCE = 0.9039
EXPECTED_NDCG_CI = (0.0391, 0.0413)

FIELDS = [
    "fusion",
    "metric",
    "n_queries",
    "mean_fusion",
    "mean_reference",
    "diff",
    "ci_low",
    "ci_high",
    "includes_zero",
    "p_student",
    "p_permutation",
    "p_holm",
    "significant",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N queries (smoke test; not for final results).",
    )
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Significance level.")
    parser.add_argument(
        "--permutations",
        type=int,
        default=N_PERMUTATIONS,
        help="Permutation replicates (default: 10000).",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    return parser.parse_args()


def _check_mean(value: float, expected: float, label: str, tolerance: float = 5e-4) -> None:
    if abs(value - expected) > tolerance:
        raise SystemExit(f"{label}: mean {value:.6f} does not match the expected {expected:.4f}")


def main() -> None:
    args = parse_args()
    smoke = args.limit is not None

    missing = [str(path) for path in [REFERENCE[1], *[path for _, path in FUSIONS]] if not path.exists()]
    if missing:
        raise SystemExit("Missing runs:\n" + "\n".join(missing))

    qrels = load_qrels()

    log.info("Loading reference %s from %s", REFERENCE[0], REFERENCE[1])
    reference_run = retrieval.load_run(REFERENCE[1])
    reference_run.name = REFERENCE[0]
    reference_scores = per_query_scores(qrels, reference_run, METRIC)
    if smoke:
        reference_scores = reference_scores[: args.limit]
    if not smoke:
        if reference_scores.size != N_QUERIES:
            raise SystemExit(f"Unexpected number of queries: {reference_scores.size}")
        _check_mean(float(np.mean(reference_scores)), EXPECTED_REFERENCE, REFERENCE[0])

    rows: list[dict] = []
    ndcg_sanity: tuple[float, float, float, float] | None = None
    for name, path in FUSIONS:
        log.info("Loading %s from %s", name, path)
        run = retrieval.load_run(path)
        run.name = name
        scores = per_query_scores(qrels, run, METRIC)
        if smoke:
            scores = scores[: args.limit]
        if scores.size != reference_scores.size:
            raise SystemExit(f"{name}: {scores.size} queries, expected {reference_scores.size}")
        if not smoke:
            _check_mean(float(np.mean(scores)), EXPECTED_RECALL[name], name)

        ci = paired_mean_ci(scores, reference_scores, alpha=args.alpha)
        p_student = None
        p_permutation = None
        p_holm = None
        significant = None
        if name == "CombMNZ":
            _, p_student = paired_t_test(scores, reference_scores)
            p_permutation = permutation_test(
                scores,
                reference_scores,
                n_permutations=args.permutations,
                seed=args.seed,
            )
            if not smoke:
                ndcg_fusion = per_query_scores(qrels, run, "ndcg@10")
                ndcg_reference = per_query_scores(qrels, reference_run, "ndcg@10")
                ndcg_ci = paired_mean_ci(ndcg_fusion, ndcg_reference, alpha=args.alpha)
                _, ndcg_p = paired_t_test(ndcg_fusion, ndcg_reference)
                if abs(ndcg_ci["low"] - EXPECTED_NDCG_CI[0]) > 5e-4 or abs(
                    ndcg_ci["high"] - EXPECTED_NDCG_CI[1]
                ) > 5e-4:
                    raise SystemExit(
                        "nDCG@10 sanity check failed: "
                        f"[{ndcg_ci['low']:.4f}, {ndcg_ci['high']:.4f}] "
                        f"vs {EXPECTED_NDCG_CI}"
                    )
                adjusted = holm_bonferroni(np.array([ndcg_p, p_student], dtype=float))
                ndcg_sanity = (
                    float(ndcg_ci["low"]),
                    float(ndcg_ci["high"]),
                    float(ndcg_p),
                    float(adjusted[0]),
                )
                p_holm = float(adjusted[1])
                significant = bool(p_holm <= args.alpha)
                log.info(
                    "nDCG@10 sanity: CI [%.4f, %.4f], p=%.4g, p_Holm=%.4g",
                    ndcg_ci["low"],
                    ndcg_ci["high"],
                    ndcg_p,
                    adjusted[0],
                )
                log.info(
                    "Recall@100 principal: Δ=%+.4f CI [%.4f, %.4f] p=%.4g p_Holm=%.4g",
                    ci["mean"],
                    ci["low"],
                    ci["high"],
                    p_student,
                    p_holm,
                )

        rows.append(
            {
                "fusion": name,
                "metric": METRIC,
                "n_queries": int(reference_scores.size),
                "mean_fusion": float(np.mean(scores)),
                "mean_reference": float(np.mean(reference_scores)),
                "diff": float(ci["mean"]),
                "ci_low": float(ci["low"]),
                "ci_high": float(ci["high"]),
                "includes_zero": bool(ci["low"] <= 0.0 <= ci["high"]),
                "p_student": p_student,
                "p_permutation": p_permutation,
                "p_holm": p_holm,
                "significant": significant,
            }
        )
        log.info(
            "%s: Recall@100=%.4f Δ=%+.4f CI [%.4f, %.4f]",
            name,
            float(np.mean(scores)),
            ci["mean"],
            ci["low"],
            ci["high"],
        )

    csv_path = OUT_DIR / "e1_recall.csv"
    md_path = OUT_DIR / "e1_recall.md"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row[key] is None else row[key]) for key in FIELDS})

    lines = [
        "| fusión | Recall@100 | Δ vs. jina | IC 95 % | ¿incluye 0? |",
        "| --- | ---: | ---: | --- | :---: |",
    ]
    for row in rows:
        lines.append(
            "| {name} | {mean:.4f} | {diff:+.4f} | [{low:.4f}, {high:.4f}] | {zero} |".format(
                name=row["fusion"],
                mean=row["mean_fusion"],
                diff=row["diff"],
                low=row["ci_low"],
                high=row["ci_high"],
                zero="sí" if row["includes_zero"] else "no",
            )
        )
    principal = next(row for row in rows if row["fusion"] == "CombMNZ")
    if not smoke:
        lines += [
            "",
            (
                "Referencia: jina-v5-small ({ref:.4f}); Δ = fusión − jina. "
                "El IC es el de la t pareada por consulta ({n} consultas). "
                "La comparación principal es CombMNZ: p de la t = {p}; "
                "p de la permutación = {perm}; p de Holm sobre nDCG@10 y Recall@100 = {ph}."
            ).format(
                ref=principal["mean_reference"],
                n=principal["n_queries"],
                p=format_p(principal["p_student"]),
                perm=format_p(principal["p_permutation"]),
                ph=format_p(principal["p_holm"]),
            ),
            (
                "Comprobación de sanidad: el IC de nDCG@10 de CombMNZ frente a jina es "
                f"[{ndcg_sanity[0]:.4f}, {ndcg_sanity[1]:.4f}]."
            ),
        ]
    else:
        lines.append("")
        lines.append(f"Prueba de humo con {principal['n_queries']} consultas; no usar como resultado final.")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    log.info("CSV: %s", csv_path)
    log.info("Markdown: %s", md_path)


if __name__ == "__main__":
    main()
