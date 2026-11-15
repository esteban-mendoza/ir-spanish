"""Figures for the RBO--gain analysis of Experiment 5.

Each panel is a scatter plot of the 57 fusion combinations: the mean RBO of the
combination's member pairs on the x axis and the gain of its fusion over its
best member on the y axis, with the fitted line and the Pearson and Spearman
coefficients annotated.  One figure per primary metric, one panel per fusion
algorithm.

The points are read from the per-strategy CSVs that ``analysis/rbo_gain.py``
writes to ``results/e5/``, so the figures and the numbers in the thesis table
come from the same artifact.  The typography follows
``analysis/distributions.ipynb``: Latin Modern Roman (the body font of
``tesis/tesis.tex``), 10 pt, and the text width of the thesis (15.59 cm).

Usage (from ``ir-spanish/``)::

    python analysis/rbo_gain_figure.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter

ANALYSIS = Path(__file__).resolve().parent
REPO = ANALYSIS.parent
RESULTS = REPO / "results" / "e5"
FIGURES = ANALYSIS / "figures"
FONT = ANALYSIS / "fonts" / "lmroman10-regular.ttf"

STRATEGIES = ("combmnz", "rbc", "rrf", "isr")
LABELS = {"combmnz": "CombMNZ", "rbc": "RBC", "rrf": "RRF", "isr": "ISR"}
DATA_FILES = {
    "combmnz": "rbo_gain.csv",
    "rbc": "rbo_gain_rbc.csv",
    "rrf": "rbo_gain_rrf.csv",
    "isr": "rbo_gain_isr.csv",
}

METRICS = (
    ("gain_ndcg", "nDCG@10", "rbo-ganancia-ndcg.pdf"),
    ("gain_recall", "Recall@100", "rbo-ganancia-recall.pdf"),
)

SIZES = (2, 3, 4, 5, 6)
# Sequential blue scale: the size of the combination is a magnitude.
SIZE_COLORS = {
    2: "#B4C9E2",
    3: "#87A9CC",
    4: "#4C72B0",
    5: "#31527F",
    6: "#1F3557",
}


def setup_style() -> None:
    """Register the thesis font and the plotting conventions of the thesis."""
    if not FONT.exists():
        raise FileNotFoundError(f"Falta la fuente de las figuras: {FONT.resolve()}")
    font_manager.fontManager.addfont(FONT)
    plt.rcParams.update(
        {
            "font.family": "Latin Modern Roman",
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def load_points(strategy: str) -> list[dict]:
    """Read the per-combination rows written by ``analysis/rbo_gain.py``."""
    path = RESULTS / DATA_FILES[strategy]
    if not path.exists():
        raise FileNotFoundError(f"Falta el artefacto del análisis: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    points = []
    for row in rows:
        points.append(
            {
                "combo": row["combo"],
                "size": int(row["size"]),
                "mean_rbo": float(row["mean_rbo"]),
                "gain_ndcg": float(row["gain_ndcg"]),
                "gain_recall": float(row["gain_recall"]),
            }
        )
    return points


def rankdata(values: list[float]) -> list[float]:
    """Average ranks, so that ties do not inflate Spearman's coefficient."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    """Pearson correlation, used here over ranks for Spearman as well."""
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    sxy = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    sxx = sum((a - mean_x) ** 2 for a in x)
    syy = sum((b - mean_y) ** 2 for b in y)
    return sxy / (sxx * syy) ** 0.5


def ols(x: list[float], y: list[float]) -> tuple[float, float]:
    """Least-squares slope and intercept."""
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    slope = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / sum(
        (a - mean_x) ** 2 for a in x
    )
    return slope, mean_y - slope * mean_x


def plot_metric(metric_key: str, metric_label: str, filename: str) -> dict[str, dict]:
    """Render one figure (four panels) and return the coefficients."""
    fig, axes = plt.subplots(2, 2, figsize=(6.14, 4.85), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.135, right=0.985, top=0.945, bottom=0.205, wspace=0.10, hspace=0.22)

    coefficients: dict[str, dict] = {}
    for ax, strategy in zip(axes.flat, STRATEGIES):
        points = load_points(strategy)
        x = [point["mean_rbo"] for point in points]
        y = [point[metric_key] for point in points]

        ax.axhline(0.0, color="0.62", linewidth=0.8, zorder=1)
        for size in SIZES:
            xs = [point["mean_rbo"] for point in points if point["size"] == size]
            ys = [point[metric_key] for point in points if point["size"] == size]
            ax.scatter(
                xs, ys, s=24, color=SIZE_COLORS[size], edgecolor="white",
                linewidth=0.5, zorder=3,
            )

        slope, intercept = ols(x, y)
        ax.plot(
            [min(x), max(x)], [intercept + slope * min(x), intercept + slope * max(x)],
            color="0.25", linewidth=1.0, linestyle="--", zorder=2,
        )

        r = pearson(x, y)
        rho = pearson(rankdata(x), rankdata(y))
        coefficients[strategy] = {"r": r, "rho": rho, "slope": slope, "n": len(points)}
        ax.text(
            0.04, 0.96,
            f"$r$ = {r:.3f}\n$\\rho$ = {rho:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="0.8", linewidth=0.6),
        )

        ax.set_title(LABELS[strategy], pad=4)
        ax.grid(color="0.92", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(length=2.5, pad=2)

    for ax in axes[1, :]:
        ax.set_xlabel("RBO promedio de los pares")
    fig.text(
        0.05, 0.55, f"Ganancia en {metric_label}\n(fusión $-$ mejor miembro)",
        rotation=90, va="center", ha="center", fontsize=10,
    )

    handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="", markersize=5.5, color=SIZE_COLORS[size],
            markeredgecolor="white", markeredgewidth=0.5, label=f"{size}",
        )
        for size in SIZES
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=5, frameon=False,
        bbox_to_anchor=(0.5, 0.005), title="Modelos en la combinación",
        title_fontsize=9, handletextpad=0.3, columnspacing=1.2,
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / filename, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return coefficients


def main() -> None:
    setup_style()
    for metric_key, metric_label, filename in METRICS:
        coefficients = plot_metric(metric_key, metric_label, filename)
        print(f"\n{metric_label} -> figures/{filename}")
        for strategy in STRATEGIES:
            values = coefficients[strategy]
            print(
                f"  {LABELS[strategy]:8s} r={values['r']:.3f} "
                f"rho={values['rho']:.3f} pendiente={values['slope']:.3f} n={values['n']}"
            )


if __name__ == "__main__":
    main()
