#!/usr/bin/env python3
"""Summarize a selection run on the MessIRve training partition.

The script reads the six first-stage runs, the 57-subset CombMNZ sweep and the
four-algorithm check, and writes a compact report plus machine-readable CSVs.
It works both for the closed-corpus pilot (``MESSIRVE_PILOT``) and, later, for
the full-corpus training selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path

import numpy as np

from utils import data, retrieval
from utils.observability import Timer

from evaluation.significance import paired_mean_ci
from rerankers.fuse_sweep import MODELS, load_qrels, load_runs

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "train_pilot"
TRIO_ALIASES = ("e5-large", "bge-m3", "jina-v5-small")
TRIO_COMBO = "+".join(TRIO_ALIASES)
EXPECTED_FIRST_STAGE = "jina-v5-small"
REFERENCE_STRATEGY = "combmnz"


def family(alias: str) -> str:
    if alias == "bm25":
        return "léxico"
    if alias == "splade-v3":
        return "disperso"
    return "denso"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing {path}; run the pilot sweep first")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_single_models(qrels, models) -> list[dict]:
    runs = load_runs(models)
    rows = []
    for model, run in zip(models, runs):
        result = retrieval.run_evaluation(qrels, run, model.alias, mode="quiet")
        rows.append(
            {
                "model": model.alias,
                "family": family(model.alias),
                "ndcg@10": float(result["ndcg@10"]),
                "recall@100": float(result["recall@100"]),
            }
        )
    rows.sort(key=lambda row: -row["ndcg@10"])
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def load_sweep(path: Path) -> list[dict]:
    rows = []
    for raw in read_csv(path):
        rows.append(
            {
                "combo": raw["combo"],
                "strategy": raw["strategy"],
                "ndcg@10": float(raw["ndcg@10"]),
                "recall@100": float(raw["recall@100"]),
                "per_query_path": raw.get("per_query_path", ""),
            }
        )
    rows.sort(key=lambda row: -row["ndcg@10"])
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def load_algorithm_check(path: Path) -> list[dict]:
    rows = []
    for raw in read_csv(path):
        rows.append(
            {
                "strategy": raw["strategy"],
                "ndcg@10": float(raw["ndcg@10"]),
                "recall@100": float(raw["recall@100"]),
                "per_query_path": raw.get("per_query_path", ""),
            }
        )
    rows.sort(key=lambda row: -row["ndcg@10"])
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def add_cis(rows: list[dict], per_query_dir: Path) -> None:
    """Attach paired 95 % CIs (vs. the best configuration) to the top rows."""
    if not rows:
        return
    best_path = rows[0].get("per_query_path") or str(per_query_dir / f"{rows[0]['strategy']}__{rows[0]['combo']}.npy")
    if not Path(best_path).exists():
        return
    best_scores = np.load(best_path)
    for row in rows[:5]:
        path = row.get("per_query_path") or str(per_query_dir / f"{row['strategy']}__{row['combo']}.npy")
        if not Path(path).exists():
            continue
        scores = np.load(path)
        ci = paired_mean_ci(scores, best_scores)
        row["diff_vs_best"] = float(np.mean(scores) - np.mean(best_scores))
        row["ci_low"] = float(ci["low"])
        row["ci_high"] = float(ci["high"])


def write_single_models(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rank", "model", "family", "ndcg@10", "recall@100"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "rank": row["rank"],
                    "model": row["model"],
                    "family": row["family"],
                    "ndcg@10": f"{row['ndcg@10']:.6f}",
                    "recall@100": f"{row['recall@100']:.6f}",
                }
            )



def write_markdown(
    path: Path,
    single: list[dict],
    sweep: list[dict],
    algorithms: list[dict],
    metadata: dict,
    agreement: dict,
) -> None:
    is_pilot = metadata.get("kind") == "train_pilot"
    lines = []
    if is_pilot:
        lines.append("# Piloto de selección sobre la partición de entrenamiento")
        lines.append("")
        lines.append(
            "> **Piloto exploratorio.** Corpus cerrado de "
            f"{metadata['n_articles']} artículos y {metadata['n_documents']} documentos. "
            "Las métricas absolutas no son comparables con las de prueba y este informe "
            "no es evidencia para la tesis; sirve para validar la tubería y ver una primera señal."
        )
    else:
        lines.append("# Selección sobre la partición de entrenamiento")
    lines.append("")

    if metadata:
        lines.append("## Partición")
        lines.append("")
        lines.append(f"- Artículos: {metadata.get('n_articles', '—')}")
        lines.append(f"- Documentos: {metadata.get('n_documents', '—')}")
        lines.append(f"- Consultas: {metadata.get('n_queries', '—')}")
        lines.append(f"- Juicios: {metadata.get('n_judgments', '—')}")
        lines.append(f"- Filtro: {metadata.get('filter', '—')}")
        lines.append(f"- Semilla: {metadata.get('seed', '—')}")
        lines.append(f"- Modelos: {', '.join('`' + row['model'] + '`' for row in single)}")
        lines.append(f"- Commit: `{metadata.get('git_commit', '—')}`")
        lines.append("")

    lines.append("## Modelos individuales")
    lines.append("")
    lines.append("| # | modelo | familia | nDCG@10 | Recall@100 |")
    lines.append("| ---: | --- | --- | ---: | ---: |")
    for row in single:
        lines.append(
            f"| {row['rank']} | `{row['model']}` | {row['family']} | "
            f"{row['ndcg@10']:.4f} | {row['recall@100']:.4f} |"
        )
    lines.append("")

    lines.append("## Mejores fusiones con el algoritmo de referencia")
    lines.append("")
    has_cis = any("ci_low" in row for row in sweep[:10])
    if has_cis:
        lines.append("| # | combinación | algoritmo | nDCG@10 | Recall@100 | Δ vs. la mejor | IC 95 % |")
        lines.append("| ---: | --- | --- | ---: | ---: | ---: | --- |")
    else:
        lines.append("| # | combinación | algoritmo | nDCG@10 | Recall@100 | Δ vs. la mejor |")
        lines.append("| ---: | --- | --- | ---: | ---: | ---: |")
    best = sweep[0]["ndcg@10"] if sweep else None
    for row in sweep[:10]:
        gap = "—" if best is None else f"{row['ndcg@10'] - best:+.4f}"
        if has_cis:
            ci = (
                f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
                if "ci_low" in row
                else "—"
            )
            lines.append(
                f"| {row['rank']} | {row['combo']} | {row['strategy']} | "
                f"{row['ndcg@10']:.4f} | {row['recall@100']:.4f} | {gap} | {ci} |"
            )
        else:
            lines.append(
                f"| {row['rank']} | {row['combo']} | {row['strategy']} | "
                f"{row['ndcg@10']:.4f} | {row['recall@100']:.4f} | {gap} |"
            )
    lines.append("")

    lines.append("## Comprobación de algoritmos sobre el trío")
    lines.append("")
    lines.append("| # | algoritmo | nDCG@10 | Recall@100 |")
    lines.append("| ---: | --- | ---: | ---: |")
    for row in algorithms:
        lines.append(
            f"| {row['rank']} | {row['strategy']} | {row['ndcg@10']:.4f} | "
            f"{row['recall@100']:.4f} |"
        )
    lines.append("")

    trio = next((row for row in sweep if row["combo"] == TRIO_COMBO), None)
    combmnz = next((row for row in algorithms if row["strategy"] == REFERENCE_STRATEGY), None)
    lines.append("## Lectura")
    lines.append("")
    lines.append(f"- Mejor modelo individual: `{single[0]['model']}` ({single[0]['ndcg@10']:.4f}).")
    top3 = ", ".join(f"`{row['model']}`" for row in single[:3])
    lines.append(f"- Tres mejores modelos individuales: {top3}.")
    if trio is None:
        lines.append(f"- El trío `{TRIO_COMBO}` no aparece en el barrido.")
    else:
        gap = trio["ndcg@10"] - sweep[0]["ndcg@10"]
        lines.append(
            f"- El trío `{TRIO_COMBO}` ocupa la posición {trio['rank']} de {len(sweep)} "
            f"con {trio['ndcg@10']:.4f} de nDCG@10 (Δ = {gap:+.4f} frente a la mejor)."
        )
    if combmnz is None:
        lines.append("- CombMNZ no aparece en la comprobación de algoritmos.")
    else:
        gap = combmnz["ndcg@10"] - algorithms[0]["ndcg@10"]
        lines.append(
            f"- CombMNZ ocupa la posición {combmnz['rank']} de {len(algorithms)} en la "
            f"comprobación de algoritmos (Δ = {gap:+.4f} frente al mejor)."
        )
    lines.append("")
    lines.append("## Coincidencia con los sistemas reportados")
    lines.append("")
    lines.append(f"- ¿El top-3 individual es el trío denso? **{'sí' if agreement['top3_matches'] else 'no'}**.")
    lines.append(
        f"- ¿El mejor modelo individual es `{EXPECTED_FIRST_STAGE}`? "
        f"**{'sí' if agreement['first_stage_matches'] else 'no'}**."
    )
    lines.append(
        f"- ¿El trío es la mejor fusión con el algoritmo de referencia? "
        f"**{'sí' if agreement['trio_is_best_fusion'] else 'no'}**."
    )
    lines.append(
        f"- ¿El algoritmo de referencia es el mejor de los cuatro? "
        f"**{'sí' if agreement['reference_algorithm_is_best'] else 'no'}**."
    )
    if is_pilot:
        lines.append("")
        lines.append(
            "En el piloto estas coincidencias son solo una señal: el corpus cerrado "
            "puede cambiar el orden de los sistemas. La decisión definitiva requiere "
            "el corpus completo."
        )

    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for the report (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model aliases to include (default: all six).",
    )
    parser.add_argument(
        "--single-models-only",
        action="store_true",
        help="Only evaluate the individual models (useful for the test partition).",
    )
    parser.add_argument(
        "--per-query-dir",
        type=Path,
        default=None,
        help="Directory with the per-query .npy files (default: <out-dir>/per_query).",
    )
    parser.add_argument(
        "--sweep",
        type=Path,
        default=None,
        help="CSV from the 57-subset sweep (default: <out-dir>/sweep/results.csv).",
    )
    parser.add_argument(
        "--algorithm-check",
        type=Path,
        default=None,
        help="CSV from the four-algorithm check (default: <out-dir>/algorithm_check/results.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    sweep_path = args.sweep or out_dir / "sweep" / "results.csv"
    algorithm_path = args.algorithm_check or out_dir / "algorithm_check" / "results.csv"

    models = MODELS
    if args.models.strip():
        wanted = [value.strip() for value in args.models.split(",") if value.strip()]
        known = {model.alias for model in MODELS}
        unknown = [value for value in wanted if value not in known]
        if unknown:
            raise SystemExit(f"Unknown models: {unknown}. Available: {sorted(known)}")
        models = [model for model in MODELS if model.alias in set(wanted)]
        if len(models) < 2:
            raise SystemExit("Need at least two models")

    with Timer("Loading training qrels"):
        qrels = load_qrels()

    with Timer("Evaluating the individual models"):
        single = load_single_models(qrels, models)

    if args.single_models_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_single_models(single, out_dir / "single_models.csv")
        log.info("Single models: %s", out_dir / "single_models.csv")
        return

    per_query_dir = args.per_query_dir or (out_dir / "per_query")
    sweep = load_sweep(sweep_path)
    add_cis(sweep, per_query_dir)
    algorithms = load_algorithm_check(algorithm_path)

    trio = next((row for row in sweep if row["combo"] == TRIO_COMBO), None)
    combmnz = next((row for row in algorithms if row["strategy"] == REFERENCE_STRATEGY), None)
    agreement = {
        "top3_matches": {row["model"] for row in single[:3]} == set(TRIO_ALIASES),
        "first_stage_matches": single[0]["model"] == EXPECTED_FIRST_STAGE,
        "trio_is_best_fusion": bool(trio and trio["rank"] == 1),
        "reference_algorithm_is_best": bool(combmnz and combmnz["rank"] == 1),
    }

    pilot_dir = os.environ.get(data.PILOT_ENV)
    metadata = {}
    if pilot_dir:
        metadata_path = Path(pilot_dir) / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())

    out_dir.mkdir(parents=True, exist_ok=True)
    write_single_models(single, out_dir / "single_models.csv")
    write_markdown(out_dir / "selection.md", single, sweep, algorithms, metadata, agreement)
    (out_dir / "selection.json").write_text(
        json.dumps(
            {
                "pilot": bool(pilot_dir),
                "metadata": metadata,
                "single_models": single,
                "top_fusions": sweep[:10],
                "algorithms": algorithms,
                "trio": trio,
                "reference_algorithm": combmnz,
                "agreement": agreement,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    log.info("Report: %s", out_dir / "selection.md")
    log.info("Single models: %s", out_dir / "single_models.csv")
    log.info("Agreement: %s", agreement)


if __name__ == "__main__":
    main()
