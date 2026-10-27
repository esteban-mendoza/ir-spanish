#!/usr/bin/env python3
"""Measure whether the closed training pilot reproduces the test ordering.

The pilot is the selection environment; the test partition is used here only
descriptively, to check that the ordering of models, fusion combinations and
algorithms transfers to the full-corpus evaluation. No selection happens in
this script.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scipy.stats import kendalltau, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
TRIO = "e5-large+bge-m3+jina-v5-small"

# nDCG@10 on the test partition, from the tables already reported in the thesis
# (experiments 1 and 2). Used only to compare the ordering, not to select.
TEST_ALGORITHMS = {
    "combmnz": 0.5517,
    "rbc": 0.5485,
    "rrf": 0.5450,
    "condorcet": 0.5422,
    "bordafuse": 0.5414,
    "isr": 0.5382,
}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def ranking(rows: list[dict], key: str) -> dict[str, float]:
    return {row[key]: float(row["ndcg@10"]) for row in rows}


def top_set(values: dict[str, float], k: int) -> set[str]:
    return set(sorted(values, key=lambda name: -values[name])[:k])


def correlations(common: list[str], pilot: dict[str, float], test: dict[str, float]) -> dict:
    a = [pilot[name] for name in common]
    b = [test[name] for name in common]
    spearman = spearmanr(a, b).statistic
    kendall = kendalltau(a, b).statistic
    return {"spearman": float(spearman), "kendall": float(kendall)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-single", type=Path, default=REPO_ROOT / "results/train_pilot/single_models.csv")
    parser.add_argument("--test-single", type=Path, default=REPO_ROOT / "results/test_selection/single_models.csv")
    parser.add_argument("--pilot-sweep", type=Path, default=REPO_ROOT / "results/train_pilot/sweep_all/results.csv")
    parser.add_argument("--test-sweep", type=Path, default=REPO_ROOT / "results/fuse_sweep/results.csv")
    parser.add_argument("--pilot-algorithms", type=Path, default=REPO_ROOT / "results/train_pilot/algorithm_check_six/results.csv")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results/train_pilot")
    args = parser.parse_args()

    pilot_single = ranking(read_csv(args.pilot_single), "model")
    test_single = ranking(read_csv(args.test_single), "model")
    single_common = sorted(set(pilot_single) & set(test_single), key=lambda name: -pilot_single[name])
    single_corr = correlations(single_common, pilot_single, test_single)

    pilot_fusion = ranking(
        [row for row in read_csv(args.pilot_sweep) if row["strategy"] == "combmnz"],
        "combo",
    )
    test_fusion = ranking(
        [row for row in read_csv(args.test_sweep) if row["strategy"] == "combmnz"],
        "combo",
    )
    fusion_common = sorted(set(pilot_fusion) & set(test_fusion), key=lambda name: -pilot_fusion[name])
    fusion_corr = correlations(fusion_common, pilot_fusion, test_fusion)
    fusion_top = {
        k: {
            "pilot": sorted(top_set(pilot_fusion, k)),
            "test": sorted(top_set(test_fusion, k)),
            "overlap": len(top_set(pilot_fusion, k) & top_set(test_fusion, k)),
        }
        for k in (1, 3, 5)
    }

    pilot_alg = ranking(read_csv(args.pilot_algorithms), "strategy")
    alg_common = sorted(set(pilot_alg) & set(TEST_ALGORITHMS), key=lambda name: -pilot_alg[name])
    alg_corr = correlations(alg_common, pilot_alg, TEST_ALGORITHMS)

    pilot_best = max(pilot_fusion, key=lambda name: pilot_fusion[name])
    test_best = max(test_fusion, key=lambda name: test_fusion[name])
    result = {
        "n_queries_pilot": None,
        "individual": {"pilot": pilot_single, "test": test_single, **single_corr},
        "fusion": {
            "pilot": pilot_fusion,
            "test": test_fusion,
            **fusion_corr,
            "top": fusion_top,
            "pilot_best": pilot_best,
            "test_best": test_best,
            "trio_pilot_rank": 1 + sorted(pilot_fusion.values(), reverse=True).index(pilot_fusion[TRIO]),
            "trio_test_rank": 1 + sorted(test_fusion.values(), reverse=True).index(test_fusion[TRIO]),
        },
        "algorithms": {"pilot": pilot_alg, "test": TEST_ALGORITHMS, **alg_corr},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "stability.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    lines = ["# Estabilidad piloto–prueba", ""]
    lines.append(
        "El piloto es el entorno de selección; la prueba se usa aquí solo de forma descriptiva "
        "para comprobar si el orden se transfiere al corpus completo. Ninguna elección de este "
        "informe usa la partición de prueba."
    )
    lines.append("")
    lines.append("## Modelos individuales")
    lines.append("")
    lines.append("| modelo | piloto | prueba |")
    lines.append("| --- | ---: | ---: |")
    for name in single_common:
        lines.append(f"| `{name}` | {pilot_single[name]:.4f} | {test_single[name]:.4f} |")
    lines.append("")
    lines.append(
        f"Spearman = {single_corr['spearman']:.4f}; Kendall = {single_corr['kendall']:.4f}. "
        f"Top-3 piloto: {', '.join(sorted(top_set(pilot_single, 3)))}; "
        f"top-3 prueba: {', '.join(sorted(top_set(test_single, 3)))}."
    )
    lines.append("")
    lines.append("## Fusiones (57 subconjuntos, CombMNZ)")
    lines.append("")
    lines.append("| # | combinación | piloto | prueba |")
    lines.append("| ---: | --- | ---: | ---: |")
    for rank, name in enumerate(sorted(fusion_common, key=lambda n: -pilot_fusion[n])[:10], 1):
        lines.append(f"| {rank} | {name} | {pilot_fusion[name]:.4f} | {test_fusion[name]:.4f} |")
    lines.append("")
    lines.append(f"Spearman = {fusion_corr['spearman']:.4f}; Kendall = {fusion_corr['kendall']:.4f}.")
    lines.append(
        f"Mejor piloto: {pilot_best}; mejor prueba: {test_best}. "
        f"El trío ocupa la posición {result['fusion']['trio_pilot_rank']} en el piloto y "
        f"{result['fusion']['trio_test_rank']} en prueba."
    )
    for k in (1, 3, 5):
        lines.append(
            f"Coincidencia del top-{k}: {fusion_top[k]['overlap']}/{k}."
        )
    lines.append("")
    lines.append("## Algoritmos sobre el trío")
    lines.append("")
    lines.append("| algoritmo | piloto | prueba |")
    lines.append("| --- | ---: | ---: |")
    for name in alg_common:
        lines.append(f"| {name} | {pilot_alg[name]:.4f} | {TEST_ALGORITHMS[name]:.4f} |")
    lines.append("")
    lines.append(f"Spearman = {alg_corr['spearman']:.4f}; Kendall = {alg_corr['kendall']:.4f}.")
    lines.append("")
    lines.append("## Lectura")
    lines.append("")
    lines.append(
        "Si las correlaciones son altas y el top-1 coincide, el piloto reproduce el orden de la "
        "evaluación completa y sirve para seleccionar sin usar la partición de prueba. Aun así, "
        "el piloto es un corpus cerrado (2.3 % del corpus) y la transferencia es descriptiva, no "
        "una garantía: esa es la limitación residual que hay que declarar."
    )
    (args.out_dir / "stability.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: result[k] for k in ("individual", "fusion", "algorithms")}, indent=2, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
