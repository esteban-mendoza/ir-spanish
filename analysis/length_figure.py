"""Figura resumen de la longitud en palabras de consultas y documentos.

Una sola gráfica con las cinco filas de la tabla 4.2 de la tesis: el bigote
gris va del mínimo al máximo, la banda oscura de P50 a P95, la banda clara de
P95 a P99 y los marcadores señalan P50, P95 y P99. El eje es logarítmico
porque las consultas (2--22 palabras) y los documentos (5--9 757) no caben en
una escala lineal común.

Las cifras son las de la tabla 4.2 del capítulo 4, calculadas en las celdas
§5--§7 de ``analysis/distributions.ipynb`` (consultas y documentos relevantes
de los splits de entrenamiento y prueba de MessIRve ``full`` 1.2, y corpus
``eswiki_20240401``). Si el conjunto de datos cambia, hay que actualizar el
cuaderno, la tabla y ``ROWS`` al mismo tiempo.

Sustituye a las cinco figuras individuales de longitud que producían las celdas
§5--§7 y §10--§11 de ``analysis/distributions.ipynb``, que retiramos de
``analysis/figures/`` y de la tesis.

La tipografía sigue ``analysis/rbo_gain_figure.py``: Latin Modern Roman (la
fuente del cuerpo de ``tesis/tesis.tex``), 10 pt y el ancho de la caja de texto
de la tesis (15.59 cm).

Uso (desde ``ir-spanish/``)::

    python analysis/length_figure.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ANALYSIS = Path(__file__).resolve().parent
FIGURES = ANALYSIS / "figures"
FONT = ANALYSIS / "fonts" / "lmroman10-regular.ttf"

FIG_ANCHO, FIG_ALTO = 6.14, 3.15
AZUL = "#4C72B0"
NARANJA = "#DD8452"
GRIS = "#8A8A8A"
GRIS_LEYENDA = "#6E6E6E"


@dataclass(frozen=True)
class Fila:
    """Resumen de una distribución: los cuantiles de una fila de la tabla 4.2."""

    etiqueta: str
    minimo: int
    p50: int
    p95: int
    p99: int
    maximo: int
    color: str


CONSULTAS = [Fila("Consultas (prueba)", 3, 5, 9, 12, 22, AZUL),
             Fila("Consultas (entrenamiento)", 2, 6, 9, 12, 22, AZUL)]
DOCUMENTOS = [Fila("Documentos (prueba)", 5, 68, 168, 245, 2226, NARANJA),
              Fila("Documentos (entrenamiento)", 5, 64, 170, 252, 2101, NARANJA),
              Fila("Documentos (corpus)", 5, 47, 153, 233, 9757, NARANJA)]
ROWS = CONSULTAS + DOCUMENTOS

# Posición vertical de cada fila. El hueco entre consultas y documentos (1.4)
# es el doble del que separa dos filas del mismo grupo (0.7).
YS_CONSULTAS = (4.0, 3.3)
YS_DOCUMENTOS = (1.9, 1.2, 0.5)


def setup_style() -> None:
    """Registra la fuente de la tesis y las convenciones de las figuras."""
    if not FONT.exists():
        raise FileNotFoundError(f"Falta la fuente de las figuras: {FONT.resolve()}")
    font_manager.fontManager.addfont(FONT)
    plt.rcParams.update(
        {
            "font.family": "Latin Modern Roman",
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,  # fuente TrueType incrustada, no Type 3
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def draw_row(ax: plt.Axes, y: float, row: Fila) -> None:
    """Dibuja una fila: bigote mín--máx, bandas de cuantiles y marcadores."""
    ax.hlines(y, row.minimo, row.maximo, color=GRIS, lw=1.0, zorder=2)
    ax.vlines([row.minimo, row.maximo], y - 0.11, y + 0.11, color=GRIS, lw=1.0, zorder=2)
    ax.hlines(y, row.p50, row.p99, color=row.color, lw=7, alpha=0.25, zorder=3)
    ax.hlines(y, row.p50, row.p95, color=row.color, lw=7, alpha=0.60, zorder=4)
    ax.plot(row.p50, y, "o", ms=5.5, color=row.color, mec="black", mew=0.55, zorder=6)
    ax.plot(row.p95, y, "o", ms=4.8, mfc="white", mec=row.color, mew=1.2, zorder=6)
    ax.plot(row.p99, y, "|", ms=8, color=row.color, mew=1.8, zorder=6)
    # P50 y P99 arriba; P95 abajo en gris, para no encimar las tres etiquetas.
    ax.annotate(f"{row.p50:,}", (row.p50, y), xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5)
    ax.annotate(f"{row.p99:,}", (row.p99, y), xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5)
    ax.annotate(f"{row.p95:,}", (row.p95, y), xytext=(0, -7), textcoords="offset points",
                ha="center", va="top", fontsize=7.5, color="#444444")


def draw_legend(ax: plt.Axes) -> None:
    """Leyenda de los elementos de la fila, en el hueco superior derecho."""
    handles = [
        Line2D([], [], color=GRIS, lw=1.0, label="mín–máx"),
        Patch(color=GRIS_LEYENDA, alpha=0.60, label="P50–P95"),
        Patch(color=GRIS_LEYENDA, alpha=0.25, label="P95–P99"),
        Line2D([], [], marker="o", ls="", ms=5, mfc=GRIS_LEYENDA, mec="black", label="P50"),
        Line2D([], [], marker="o", ls="", ms=5, mfc="white", mec=GRIS_LEYENDA, label="P95"),
        Line2D([], [], marker="|", ls="", ms=7, color=GRIS_LEYENDA, label="P99"),
    ]
    ax.legend(handles=handles, ncol=2, frameon=False, handlelength=1.4,
              columnspacing=1.0, handletextpad=0.5, loc="upper right",
              bbox_to_anchor=(1.0, 1.02))


def main() -> None:
    setup_style()
    FIGURES.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(FIG_ANCHO, FIG_ALTO))
    for y, row in zip(YS_CONSULTAS + YS_DOCUMENTOS, ROWS):
        draw_row(ax, y, row)

    ax.set_xscale("log")
    ax.set_xlim(1.7, 12000)
    ax.set_ylim(0.1, 4.6)
    ax.set_yticks(YS_CONSULTAS + YS_DOCUMENTOS)
    ax.set_yticklabels([row.etiqueta for row in ROWS])
    ax.set_xlabel("Número de palabras (escala logarítmica)")
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.set_xticklabels(["1", "10", "100", "1000", "10 000"])
    ax.xaxis.grid(True, which="major", color="#DEDEDE", lw=0.6, zorder=0)
    ax.tick_params(length=3)
    draw_legend(ax)

    fig.tight_layout()
    out = FIGURES / "messirve-longitudes-resumen.pdf"
    fig.savefig(out, bbox_inches="tight")
    png = out.with_suffix(".png")
    fig.savefig(png, dpi=240, bbox_inches="tight")
    print(f"Figura: {out}")
    print(f"Vista previa: {png}")


if __name__ == "__main__":
    main()
