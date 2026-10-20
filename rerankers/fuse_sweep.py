#!/usr/bin/env python3
"""Exhaustive rank-fusion sweep over every non-trivial subset of the base models.

The reference run is ``logs/rerankers-fuse-rbc0.9.log``: all combinations of
sizes 6 down to 2 for one strategy.  This script generalises it to several
strategies, writes one CSV row per (combination, strategy), and optionally
saves the fused runs so that paired significance tests can be computed later
without fusing again.

Usage (from the repository root)::

    python -m rerankers.fuse_sweep --strategies combmnz,rbc

The default configuration is the one reported in the thesis: the six base
models at their **native** maximum sequence lengths (the same runs used in
the ``max_seq_length`` table of ``notes/results.md``).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from ranx import fuse

# utils.__init__ sets up logging and NUMA/threading env-vars
from utils import cache, data, retrieval

log = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cache" / "messirve_embeddings"
REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    name: str  # HuggingFace model name or custom identifier
    alias: str  # short display name for tables
    max_q_len: int  # max_query_length used when the model's run was cached
    max_d_len: int  # max_doc_length used when the model's run was cached


# Order matters: it fixes the order of the model names in combination names.
MODELS = [
    ModelConfig("bm25_pyserini", "bm25", 512, 512),
    ModelConfig("naver/splade-v3", "splade-v3", 512, 512),
    ModelConfig("Qwen/Qwen3-Embedding-0.6B", "qwen3-0.6b", 32768, 32768),
    ModelConfig("intfloat/multilingual-e5-large-instruct", "e5-large", 512, 512),
    ModelConfig("BAAI/bge-m3", "bge-m3", 8192, 8192),
    ModelConfig("jinaai/jina-embeddings-v5-text-small-retrieval", "jina-v5-small", 32768, 32768),
]

STRATEGIES = {
    "combmnz": ("mnz", None),
    "rbc": ("rbc", {"phi": 0.95}),
    "rrf": ("rrf", {"k": 60}),
    "isr": ("isr", None),
    "bordafuse": ("bordafuse", None),
    "condorcet": ("condorcet", None),
}

CSV_FIELDS = [
    "combo",
    "strategy",
    "params",
    "ndcg@10",
    "recall@100",
    "mrr@10",
    "map",
    "precision@10",
    "precision@50",
    "elapsed_s",
    "run_path",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_runs(models: list[ModelConfig]) -> list:
    runs = []
    for m in models:
        base = cache.cache_base(
            CACHE_DIR,
            m.name,
            data.COUNTRY,
            data.DATASET_VERSION,
            m.max_q_len,
            m.max_d_len,
            data.MAX_WORD_COUNT,
        )
        run_path = cache.run_cache_path(base)
        log.info("Loading run for %s from %s", m.alias, run_path)
        run = retrieval.load_run(run_path)
        run.name = m.alias
        runs.append(run)
    return runs


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


def format_params(params: dict | None) -> str:
    if not params:
        return ""
    return ",".join(f"{k}={v}" for k, v in params.items())


def run_key(strategy: str, combo: str) -> str:
    return f"{strategy}|{combo}"


def read_done(csv_path: Path) -> set[str]:
    """Return the set of (strategy, combo) pairs already present in the CSV."""
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="") as f:
        return {run_key(row["strategy"], row["combo"]) for row in csv.DictReader(f)}


def write_markdown(csv_path: Path, md_path: Path) -> None:
    if not csv_path.exists():
        return
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (r["strategy"], -float(r["ndcg@10"])))

    header = (
        "| combinación | estrategia | parámetros | nDCG@10 | Recall@100 | MRR@10 | MAP | P@10 | P@50 |\n"
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    body = "".join(
        "| {combo} | {strategy} | {params} | {ndcg@10} | {recall@100} | {mrr@10} | "
        "{map} | {precision@10} | {precision@50} |\n".format(**row)
        for row in rows
    )
    md_path.write_text(header + body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategies",
        default="combmnz,rbc",
        help="Comma-separated strategy keys (default: combmnz,rbc).",
    )
    parser.add_argument(
        "--combos",
        default="",
        help="Comma-separated combination names to run (default: all).",
    )
    parser.add_argument(
        "--save-combos",
        default="",
        help="Comma-separated combinations whose fused runs are saved (default: all).",
    )
    parser.add_argument("--min-size", type=int, default=2, help="Minimum number of fused runs.")
    parser.add_argument("--max-size", type=int, default=6, help="Maximum number of fused runs.")
    parser.add_argument(
        "--norm",
        default="min-max",
        help="Score normalization passed to ranx.fuse (default: min-max). Use 'none' to disable.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "fuse_sweep",
        help="Directory for the CSV and markdown report.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=CACHE_DIR / "fusions",
        help="Directory for the fused run files (kept out of git).",
    )
    parser.add_argument(
        "--no-save-runs",
        action="store_true",
        help="Do not write the fused runs to disk.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore the existing CSV and recompute every combination.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    strategy_keys = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = [s for s in strategy_keys if s not in STRATEGIES]
    if unknown:
        raise SystemExit(f"Unknown strategies: {unknown}. Available: {sorted(STRATEGIES)}")
    if not (2 <= args.min_size <= args.max_size <= len(MODELS)):
        raise SystemExit("Need 2 <= min-size <= max-size <= number of models")

    alias_to_index = {m.alias: i for i, m in enumerate(MODELS)}
    selected = set()
    if args.combos.strip():
        for raw in args.combos.split(","):
            parts = [part.strip() for part in raw.split("+") if part.strip()]
            if len(parts) < 2:
                raise SystemExit(f"Invalid combination: {raw!r}")
            unknown_models = [part for part in parts if part not in alias_to_index]
            if unknown_models:
                raise SystemExit(f"Unknown models in {raw!r}: {unknown_models}")
            indices = tuple(sorted(alias_to_index[part] for part in parts))
            if not (args.min_size <= len(indices) <= args.max_size):
                raise SystemExit(f"Combination {raw!r} is outside the size range")
            selected.add("+".join(MODELS[i].alias for i in indices))

    save_selected = set()
    if args.save_combos.strip():
        for raw in args.save_combos.split(","):
            parts = [part.strip() for part in raw.split("+") if part.strip()]
            if len(parts) < 2:
                raise SystemExit(f"Invalid save combination: {raw!r}")
            unknown_models = [part for part in parts if part not in alias_to_index]
            if unknown_models:
                raise SystemExit(f"Unknown models in {raw!r}: {unknown_models}")
            indices = tuple(sorted(alias_to_index[part] for part in parts))
            save_selected.add("+".join(MODELS[i].alias for i in indices))

    norm = None if args.norm.lower() == "none" else args.norm
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_save_runs:
        args.runs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "results.csv"
    md_path = args.out_dir / "results.md"

    done = set() if args.no_resume else read_done(csv_path)

    log.info("Models (native lengths): %s", ", ".join(f"{m.alias}[{m.max_q_len}/{m.max_d_len}]" for m in MODELS))
    log.info("Strategies: %s", ", ".join(strategy_keys))
    log.info("Normalization: %s", norm or "none")
    log.info("Combination sizes: %d..%d", args.min_size, args.max_size)

    runs = load_runs(MODELS)
    qrels = load_qrels()

    if selected:
        total = len(selected) * len(strategy_keys)
    else:
        total = sum(
            len(list(itertools.combinations(range(len(MODELS)), size)))
            for size in range(args.max_size, args.min_size - 1, -1)
        ) * len(strategy_keys)
    log.info("Total (combination, strategy) pairs: %d", total)

    new_csv = not csv_path.exists()
    csv_file = csv_path.open("a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if new_csv:
        writer.writeheader()
        csv_file.flush()

    index = 0
    try:
        for strategy_key in strategy_keys:
            method, params = STRATEGIES[strategy_key]
            params_str = format_params(params)

            for size in range(args.max_size, args.min_size - 1, -1):
                combos = list(itertools.combinations(range(len(MODELS)), size))
                for indices in combos:
                    combo = "+".join(MODELS[i].alias for i in indices)
                    if selected and combo not in selected:
                        continue
                    index += 1
                    key = run_key(strategy_key, combo)

                    save_run = not args.no_save_runs and (not save_selected or combo in save_selected)
                    run_path = args.runs_dir / f"{strategy_key}__{combo}.lz4" if save_run else None
                    if key in done and (not save_run or run_path.exists()):
                        log.info("[%d/%d] skip %s (%s)", index, total, combo, strategy_key)
                        continue

                    started = time.perf_counter()
                    log.info("[%d/%d] fusing %s (%s, %s)", index, total, combo, method, params_str or "-")
                    fused_run = fuse(
                        runs=[runs[i] for i in indices],
                        norm=norm,
                        method=method,
                        params=params or {},
                    )
                    fused_run.name = f"{combo}__{strategy_key}"

                    results = retrieval.run_evaluation(
                        qrels,
                        fused_run,
                        combo,
                        mode="inline",
                        strategy=method,
                        params=params_str,
                    )
                    elapsed = time.perf_counter() - started

                    written_path = ""
                    if save_run:
                        fused_run.save(str(run_path))
                        written_path = str(run_path)

                    writer.writerow(
                        {
                            "combo": combo,
                            "strategy": strategy_key,
                            "params": params_str,
                            "ndcg@10": f"{results['ndcg@10']:.6f}",
                            "recall@100": f"{results['recall@100']:.6f}",
                            "mrr@10": f"{results['mrr@10']:.6f}",
                            "map": f"{results['map']:.6f}",
                            "precision@10": f"{results['precision@10']:.6f}",
                            "precision@50": f"{results['precision@50']:.6f}",
                            "elapsed_s": f"{elapsed:.1f}",
                            "run_path": written_path,
                        }
                    )
                    csv_file.flush()
                    log.info(
                        "[%d/%d] done %s (%s): nDCG@10=%.4f Recall@100=%.4f (%.1fs)",
                        index,
                        total,
                        combo,
                        strategy_key,
                        results["ndcg@10"],
                        results["recall@100"],
                        elapsed,
                    )
    finally:
        csv_file.close()
        write_markdown(csv_path, md_path)
        log.info("CSV: %s", csv_path)
        log.info("Markdown: %s", md_path)


if __name__ == "__main__":
    main()
