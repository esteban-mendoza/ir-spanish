"""Figures of fusion gain against RBO, split by pair category (Experiment 5).

The unit is the fusion combination, as in ``analysis/rbo_gain.py``.  For each
combination and each category of member pair (lexical--sparse, lexical--dense,
sparse--dense, dense--dense) the horizontal axis is the mean RBO of *only* the
pairs of that category inside the combination; the vertical axis is the gain of
the fusion over its best member, which does not change.  A combination
contributes one point to every category it contains.

Each figure is a 4 x 4 grid: one fusion algorithm per row (CombMNZ, RBC, RRF and
ISR) and one category per column.  The color of the points encodes the number of
models in the combination, with the same white-to-blue scale as the other
figures of the experiment.  Every panel carries the Pearson coefficient of its
scatter.

The lexical models are BM25, the sparse model is SPLADE-v3 and the dense models
are qwen3-0.6b, e5-large, bge-m3 and jina-v5-small.  The lexical--sparse
category has a single pair (BM25--SPLADE), so its RBO is constant across the
sixteen combinations that contain it and no correlation can be computed.

Typography follows ``analysis/rbo_gain_figure.py``: Latin Modern Roman (the body
font of ``tesis/tesis.tex``) and the text width of the thesis.

Usage (from ``ir-spanish/``)::

    python analysis/rbo_gain_category_figure.py
"""

from __future__ import annotations

import csv
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter

from rbo_gain_figure import ols, pearson, rankdata, setup_style

ANALYSIS = Path(__file__).resolve().parent
REPO = ANALYSIS.parent
RESULTS = REPO / "results" / "e5"
FIGURES = ANALYSIS / "figures"

FAMILY = {
    "bm25": "lex",
    "splade-v3": "sparse",
    "qwen3-0.6b": "dense",
    "e5-large": "dense",
    "bge-m3": "dense",
    "jina-v5-small": "dense",
}

CATEGORIES = (
    ("lex", "sparse", "léxico–disperso"),
    ("lex", "dense", "léxico–denso"),
    ("sparse", "dense", "disperso–denso"),
    ("dense", "dense", "denso–denso"),
)

STRATEGIES = ("combmnz", "rbc", "rrf", "isr")
LABELS = {"combmnz": "CombMNZ", "rbc": "RBC", "rrf": "RRF", "isr": "ISR"}
SIZES = (2, 3, 4, 5, 6)
# Sequential blue scale, the same as in analysis/rbo_gain_figure.py.
SIZE_COLORS = {
    2: "#B4C9E2",
    3: "#87A9CC",
    4: "#4C72B0",
    5: "#31527F",
    6: "#1F3557",
}
DATA_FILES = {
    "combmnz": "rbo_gain.csv",
    "rbc": "rbo_gain_rbc.csv",
    "rrf": "rbo_gain_rrf.csv",
    "isr": "rbo_gain_isr.csv",
}

METRICS = (
    ("gain_ndcg", "nDCG@10", "rbo-ganancia-categorias-ndcg.pdf"),
    ("gain_recall", "Recall@100", "rbo-ganancia-categorias-recall.pdf"),
)


def load_pair_rbo() -> dict[tuple[str, str], float]:
    """RBO of every unordered pair of first-stage models."""
    values = {}
    with (REPO / "results" / "rbo" / "rbo_pairs.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values[tuple(sorted((row["model_a"], row["model_b"])))] = float(row["mean_rbo"])
    return values


def category_rbo(members: list[str], first: str, second: str, pairs: dict) -> float | None:
    """Mean RBO of the member pairs that join the two given families."""
    values = []
    for model_a, model_b in itertools.combinations(members, 2):
        family_a, family_b = FAMILY[model_a], FAMILY[model_b]
        if (first == second and family_a == first and family_b == second) or (
            first != second and {family_a, family_b} == {first, second}
        ):
            values.append(pairs[tuple(sorted((model_a, model_b)))])
    return sum(values) / len(values) if values else None


def load_points(strategy: str, pairs: dict) -> list[dict]:
    """One record per combination, with its per-category RBO and its gains."""
    path = RESULTS / DATA_FILES[strategy]
    points = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            members = row["combo"].split("+")
            point = {
                "combo": row["combo"],
                "size": int(row["size"]),
                "gain_ndcg": float(row["gain_ndcg"]),
                "gain_recall": float(row["gain_recall"]),
            }
            for first, second, label in CATEGORIES:
                point[label] = category_rbo(members, first, second, pairs)
            points.append(point)
    return points


def statistics(points: list[dict], category: str, metric: str) -> dict | None:
    """Pearson and Spearman coefficients for one category and one strategy."""
    x = [point[category] for point in points if point[category] is not None]
    y = [point[metric] for point in points if point[category] is not None]
    if not x or max(x) - min(x) == 0:
        return None
    return {
        "n": len(x),
        "pearson": pearson(x, y),
        "spearman": pearson(rankdata(x), rankdata(y)),
        "x": x,
        "y": y,
    }


def plot_metric(metric_key: str, metric_label: str, filename: str) -> dict:
    """Render the 4 x 4 grid for one metric and return the coefficients."""
    pairs = load_pair_rbo()
    points = {strategy: load_points(strategy, pairs) for strategy in STRATEGIES}

    fig, axes = plt.subplots(4, 4, figsize=(6.14, 6.5), sharex="col", sharey=True)
    fig.subplots_adjust(left=0.105, right=0.99, top=0.895, bottom=0.145, wspace=0.10, hspace=0.14)

    summary: dict[str, dict] = {}
    for row, strategy in enumerate(STRATEGIES):
        for col, (first, second, category) in enumerate(CATEGORIES):
            ax = axes[row, col]
            ax.axhline(0.0, color="0.62", linewidth=0.7, zorder=1)
            ax.grid(color="0.93", linewidth=0.5, zorder=0)
            ax.set_axisbelow(True)

            values = statistics(points[strategy], category, metric_key)
            summary.setdefault(category, {})[strategy] = values

            subset = [point for point in points[strategy] if point[category] is not None]
            x = [point[category] for point in subset]
            y = [point[metric_key] for point in subset]
            ax.scatter(
                x, y, s=13, c=[SIZE_COLORS[point["size"]] for point in subset],
                edgecolor="white", linewidth=0.25, zorder=3,
            )

            if values is None:
                label = None
            else:
                slope, intercept = ols(x, y)
                ax.plot(
                    [min(x), max(x)],
                    [intercept + slope * min(x), intercept + slope * max(x)],
                    color="0.25", linewidth=0.9, linestyle="--", zorder=2,
                )
                label = f"$r$ = {values['pearson']:+.3f}"

            if label is not None:
                ax.text(
                    0.96, 0.94, label, transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="0.85",
                              linewidth=0.5, alpha=0.9),
                )
            ax.tick_params(labelsize=6.8, length=2, pad=1.5)
            if row == 0:
                if values is None:
                    title = f"{category}\nRBO constante ($n$ = {len(x)})"
                else:
                    title = f"{category}\n($n$ = {len(x)})"
                ax.set_title(title, fontsize=7.5, pad=3)
            if col == 0:
                ax.set_ylabel(LABELS[strategy], fontsize=8, labelpad=2)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    fig.text(
        0.012, 0.5, f"Ganancia en {metric_label}\n(fusión $-$ mejor miembro)",
        rotation=90, va="center", ha="center", fontsize=8.5,
    )
    fig.text(
        0.55, 0.088, "RBO promedio de los pares de la categoría",
        ha="center", va="bottom", fontsize=8.5,
    )
    handles = [
        plt.Line2D(
            [], [], linestyle="", marker="o", markersize=5, color=SIZE_COLORS[size],
            markeredgecolor="white", markeredgewidth=0.4, label=f"{size}",
        )
        for size in SIZES
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8,
        bbox_to_anchor=(0.5, 0.004), title="Modelos en la combinación",
        title_fontsize=8, handletextpad=0.4, columnspacing=1.4,
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / filename, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return summary


def main() -> None:
    setup_style()
    for metric_key, metric_label, filename in METRICS:
        summary = plot_metric(metric_key, metric_label, filename)
        print(f"\n{metric_label} -> figures/{filename}")
        for _, _, category in CATEGORIES:
            parts = []
            n = 16 if all(summary[category][s] is None for s in STRATEGIES) else 0
            for strategy in STRATEGIES:
                values = summary[category][strategy]
                if values is None:
                    parts.append(f"{LABELS[strategy]}: sin varianza")
                else:
                    n = values["n"]
                    parts.append(
                        f"{LABELS[strategy]}: r={values['pearson']:+.3f} rho={values['spearman']:+.3f}"
                    )
            print(f"  {category:17s} n={n:2d} | " + " | ".join(parts))


if __name__ == "__main__":
    main()
