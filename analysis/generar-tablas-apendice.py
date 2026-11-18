"""Genera las tablas de resultados del apéndice A de la tesis.

Lee los artefactos de ``results/`` y reescribe solo los bloques
marcados de ``../tesis/apendices/resultados.tex``: A.3 (fusiones), A.5
(Recall@100 del experimento 1), A.6 (15 pares del experimento 2), A.7
(contrastes del experimento 3) y A.8/A.9 (RBO y ganancia por combinación).
Las tablas A.1, A.2 y A.4 se escriben a mano en el .tex.

Uso (desde ``ir-spanish/``)::

    python3 analysis/generar-tablas-apendice.py

Los marcadores son ``% >>> TABLA-A3`` y ``% <<< TABLA-A3`` (y los análogos).
El script no toca la prosa ni las tablas estáticas.
"""

from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = REPO / "results"
TESIS = REPO.parent / "tesis"
TARGET = TESIS / "apendices" / "resultados.tex"

CODES = {
    "bm25": "BM",
    "splade-v3": "SP",
    "qwen3-0.6b": "QW",
    "e5-large": "E5",
    "bge-m3": "BG",
    "jina-v5-small": "JI",
}

ALGORITHMS = (
    ("combmnz", "CombMNZ"),
    ("rbc", "RBC"),
    ("rrf", "RRF"),
    ("isr", "ISR"),
)

BORDA = {
    "e5-large+bge-m3+jina-v5-small": "B1",
    "bm25+splade-v3+qwen3-0.6b+e5-large+bge-m3+jina-v5-small": "B2",
    "splade-v3+qwen3-0.6b+e5-large+bge-m3+jina-v5-small": "B3",
    "splade-v3+e5-large+jina-v5-small": "B4",
    "splade-v3+jina-v5-small": "B5",
    "qwen3-0.6b+e5-large+bge-m3+jina-v5-small": "B6",
}

E2_ORDER = ("CombMNZ", "RBC", "RRF", "Condorcet", "BordaFuse", "ISR")

METRICS = (
    ("ndcg@10", "nDCG@10"),
    ("recall@100", "Recall@100"),
    ("mrr@10", "MRR@10"),
    ("map", "MAP"),
    ("precision@10", "P@10"),
    ("precision@50", "P@50"),
)


def read_csv(name: str) -> list[dict[str, str]]:
    with open(RESULTS / name, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fp(value: float | str, digits: int = 4, sign: bool = False) -> str:
    number = float(value)
    if round(number, digits) == 0:
        number = 0.0
    text = f"{number:.{digits}f}"
    if sign and number > 0:
        text = "+" + text
    return text


def fci(low: str, high: str, digits: int = 4) -> str:
    return f"$[{fp(low, digits)}, {fp(high, digits)}]$"


def fdelta(value: float | str, digits: int = 4) -> str:
    return f"${fp(value, digits, sign=True)}$"


def ftime(value: float | str) -> str:
    number = float(value)
    if number != number:  # NaN
        return "---"
    return f"{number:,.1f}".replace(",", r"\,")


def fpvalue(value: float | str) -> str:
    number = float(value)
    if number < 0.0001:
        return r"$<0.0001$"
    return f"{number:.4f}"


def codes(combo: str) -> str:
    return "+".join(CODES[model] for model in combo.split("+"))


def block(tag: str, body: str) -> str:
    return f"% >>> {tag}\n{body.rstrip()}\n% <<< {tag}"


def replace_block(text: str, tag: str, body: str) -> str:
    start = f"% >>> {tag}"
    end = f"% <<< {tag}"
    before = text.index(start) + len(start)
    after = text.index(end)
    return text[:before] + "\n" + body.rstrip() + "\n" + text[after:]


def table_a3() -> str:
    sweep = read_csv("fuse_sweep/results.csv")
    lines = [
        r"{\footnotesize\setlength{\tabcolsep}{2.5pt}",
        r"\begin{longtable}{@{}l r r r r r r r@{}}",
        r"\caption{Resultados de las 235 configuraciones de fusión sobre las 170\,055 consultas de la partición de prueba. Los cuatro primeros bloques son la evaluación exhaustiva ---57 subconjuntos de dos a seis modelos cada uno--- y están ordenados por nDCG@10; el bloque de BordaFuse también está ordenado por nDCG@10, con B2 al final porque no produjo resultado, y el de Condorcet contiene una sola combinación. La combinación se escribe con los códigos de la \cref{tab:ap-codigos}; en BordaFuse y Condorcet, el prefijo B1--B6 identifica la combinación. Reportamos cuatro decimales y el tiempo con un decimal. El símbolo \dag{} marca el tiempo transcurrido hasta el agotamiento de la memoria, sin resultado. Las celdas sin medición quedan con --- y no las imputamos.}"
        r"\label{tab:ap-fusiones}\\",
        r"\toprule",
        r"Combinación & nDCG@10 & Recall@100 & MRR@10 & MAP & P@10 & P@50 & Tiempo (s) \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Combinación & nDCG@10 & Recall@100 & MRR@10 & MAP & P@10 & P@50 & Tiempo (s) \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    for strategy, label in ALGORITHMS:
        rows = [row for row in sweep if row["strategy"] == strategy]
        if len(rows) != 57:
            raise SystemExit(f"{strategy}: se esperaban 57 filas y hay {len(rows)}")
        rows.sort(key=lambda row: (-float(row["ndcg@10"]), row["combo"]))
        lines.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\textbf{{{label}}}}}\\*")
        lines.append(r"\midrule")
        for row in rows:
            metrics = " & ".join(fp(row[key]) for key, _ in METRICS)
            lines.append(f"{codes(row['combo'])} & {metrics} & {ftime(row['elapsed_s'])} \\\\")
        lines.append(r"\midrule")

    stage_b = read_csv("fuse_stage_b/results.csv")
    borda = {row["combo"]: row for row in stage_b if row["strategy"] == "bordafuse"}
    if len(borda) != 5:
        raise SystemExit(f"bordafuse: se esperaban 5 filas y hay {len(borda)}")
    lines.append(r"\multicolumn{8}{@{}l}{\textbf{BordaFuse}}\\*")
    lines.append(r"\midrule")
    borda_rows = []
    for combo, label in BORDA.items():
        if label == "B2":
            continue
        row = borda[combo]
        borda_rows.append((float(row["ndcg@10"]), label, combo, row))
    borda_rows.sort(key=lambda item: (-item[0], item[1]))
    for _, label, combo, row in borda_rows:
        metrics = " & ".join(fp(row[key]) for key, _ in METRICS)
        lines.append(f"B{label[1:]}~{codes(combo)} & {metrics} & {ftime(row['elapsed_s'])} \\\\")
    lines.append(
        r"B2~BM+SP+QW+E5+BG+JI & --- & --- & --- & --- & --- & --- & 1\,519\dag"
        r"\rlap{\hspace{0.15cm}\parbox[t]{2.3cm}{\raggedright\footnotesize\itshape B2 no produjo resultado: el proceso agotó la memoria (OOM).}} \\"
    )
    lines.append(r"\midrule")

    condorcet = read_csv("fuse_stage_b_condorcet/results.csv")
    if len(condorcet) != 1:
        raise SystemExit(f"condorcet: se esperaba 1 fila y hay {len(condorcet)}")
    row = condorcet[0]
    metrics = " & ".join(fp(row[key]) for key, _ in METRICS)
    lines.append(r"\multicolumn{8}{@{}l}{\textbf{Condorcet}}\\*")
    lines.append(r"\midrule")
    lines.append(f"B1~{codes(row['combo'])} & {metrics} & {ftime(row['elapsed_s'])} \\\\")
    lines.append(r"\end{longtable}}")
    return "\n".join(lines)


def table_a5() -> str:
    rows = read_csv("evaluation/e1_recall.csv")
    if len(rows) != 6:
        raise SystemExit(f"E1 Recall: se esperaban 6 filas y hay {len(rows)}")
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}l r c c@{}}",
        r"\toprule",
        r"Fusión & $\Delta$ & IC 95\,\% & $p$ (Holm--Bonferroni) \\",
        r"\midrule",
    ]
    for row in rows:
        p = fpvalue(row["p_holm"]) if row["p_holm"].strip() else "---"
        lines.append(
            f"{row['fusion']} & {fdelta(row['diff'])} & {fci(row['ci_low'], row['ci_high'])} & {p} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Diferencias pareadas de Recall@100 de las seis fusiones frente a \mbox{\cod{jina-v5-small}} sobre las 170\,055 consultas de prueba. $\Delta$ es la diferencia entre la fusión y la referencia, y el intervalo es el del 95\,\% sin ajustar. Solo la comparación principal (CombMNZ) lleva prueba de significancia; las otras cinco filas son descriptivas. El valor $p$ corresponde a la corrección de Holm--Bonferroni.}",
        r"\label{tab:ap-e1-recall}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def pair_row(rows: list[dict[str, str]], first: str, second: str) -> tuple[dict[str, str], int]:
    for row in rows:
        if row["run_a"] == first and row["run_b"] == second:
            return row, 1
        if row["run_a"] == second and row["run_b"] == first:
            return row, -1
    raise SystemExit(f"no encontré el par {first}--{second}")


def table_a6() -> str:
    holm = {
        "nDCG@10": read_csv("evaluation/e2_holm.csv"),
        "Recall@100": read_csv("evaluation/e2_holm_recall.csv"),
    }
    bh = {
        "nDCG@10": read_csv("evaluation/e2_bh.csv"),
        "Recall@100": read_csv("evaluation/e2_bh_recall.csv"),
    }
    for metric in holm:
        if len(holm[metric]) != 15 or len(bh[metric]) != 15:
            raise SystemExit(f"E2 {metric}: se esperaban 15 pares")

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}l r c c c l@{}}",
        r"\toprule",
        r"Par & $\Delta$ & IC 95\,\% & $p$ (Holm) & $p$ (BH) & ¿Significativa? \\",
        r"\midrule",
    ]
    for metric, label in (("nDCG@10", "nDCG@10"), ("Recall@100", "Recall@100")):
        lines.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\textbf{{{label}}}}}\\")
        lines.append(r"\midrule")
        for i, first in enumerate(E2_ORDER):
            for second in E2_ORDER[i + 1 :]:
                hrow, sign = pair_row(holm[metric], first, second)
                brow, _ = pair_row(bh[metric], first, second)
                diff = sign * float(hrow["diff"])
                low = -float(hrow["ci_high"]) if sign < 0 else float(hrow["ci_low"])
                high = -float(hrow["ci_low"]) if sign < 0 else float(hrow["ci_high"])
                decision = "sí" if hrow["significant"] == "True" else "no"
                lines.append(
                    f"{first}--{second} & ${fp(diff, 5, sign=True)}$ & "
                    f"$[{fp(low, 5)}, {fp(high, 5)}]$ & "
                    f"{fpvalue(hrow['p_adjusted'])} & {fpvalue(brow['p_adjusted'])} & {decision} \\\\"
                )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [
        r"\end{tabular}",
        r"\caption{Contrastes pareados de los seis algoritmos de fusión sobre el trío \mbox{\cod{e5-large}}, \mbox{\cod{bge-m3}} y \mbox{\cod{jina-v5-small}}, en las dos métricas primarias. $\Delta$ es la diferencia entre el primer algoritmo y el segundo; ordenamos los pares por el nDCG@10 de la \cref{tab:resultados-e1} y normalizamos el signo y los extremos del intervalo cuando el par aparece en el orden inverso. Los intervalos son los del 95\,\% sin ajustar; los valores $p$ corresponden a las correcciones de Holm--Bonferroni y de Benjamini--Hochberg sobre los 15 pares de cada métrica. Reportamos cinco decimales porque las diferencias son del orden de $10^{-4}$. La columna de decisión usa la corrección de Holm--Bonferroni; la de Benjamini--Hochberg conduce a las mismas decisiones.}",
        r"\label{tab:ap-e2}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def table_a7() -> str:
    files = (
        ("nDCG@10", "evaluation/e3_maxctx.csv"),
        ("Recall@100", "evaluation/e3_maxctx_recall.csv"),
    )
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l r c c l@{}}",
        r"\toprule",
        r"Comparación & $\Delta$ & IC 95\,\% & $p$ (Holm) & ¿Significativa? \\",
        r"\midrule",
    ]
    for label, name in files:
        rows = read_csv(name)
        if len(rows) != 3:
            raise SystemExit(f"E3 {label}: se esperaban 3 comparaciones")
        lines.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\textbf{{{label}}}}}\\")
        lines.append(r"\midrule")
        for row in rows:
            decision = "sí" if row["significant"] == "True" else "no"
            lines.append(
                f"Fusión vs. \\mbox{{\\cod{{{row['run_b']}}}}} & "
                f"{fdelta(row['diff'])} & {fci(row['ci_low'], row['ci_high'])} & "
                f"{fpvalue(row['p_adjusted'])} & {decision} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [
        r"\end{tabular}",
        r"\caption{Contrastes pareados de la fusión de referencia frente a los tres reordenadores, en las dos métricas primarias. $\Delta$ es la diferencia entre la fusión y el reordenador; los intervalos son los del 95\,\% sin ajustar y los valores $p$ corresponden a la corrección de Holm--Bonferroni sobre las tres comparaciones de cada métrica. En Recall@100 las tres filas coinciden porque los reordenadores comparten la lista de entrada. Los valores medios están en la \cref{tab:ap-reranking}.}",
        r"\label{tab:ap-e3}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def rbo_block(metric: str, caption_title: str, table_label: str, caption_metric: str) -> str:
    sources = {
        "CombMNZ": "e5/rbo_gain.csv",
        "RBC": "e5/rbo_gain_rbc.csv",
        "RRF": "e5/rbo_gain_rrf.csv",
        "ISR": "e5/rbo_gain_isr.csv",
    }
    data: dict[str, dict[str, dict[str, str]]] = {}
    for algorithm, name in sources.items():
        rows = read_csv(name)
        if len(rows) != 57:
            raise SystemExit(f"{name}: se esperaban 57 filas")
        data[algorithm] = {row["combo"]: row for row in rows}

    combos = list(data["CombMNZ"])
    for algorithm in sources:
        if set(combos) != set(data[algorithm]):
            raise SystemExit(f"{algorithm}: las combinaciones no coinciden")
        for combo in combos:
            if abs(
                float(data["CombMNZ"][combo]["mean_rbo"])
                - float(data[algorithm][combo]["mean_rbo"])
            ) > 1e-12:
                raise SystemExit(f"{algorithm}: el RBO de {combo} no coincide")

    key = "gain_ndcg" if metric == "ndcg" else "gain_recall"
    combos.sort(
        key=lambda combo: (
            -float(data["CombMNZ"][combo]["mean_rbo"]),
            int(data["CombMNZ"][combo]["size"]),
            combo,
        )
    )
    lines = [
        r"{\footnotesize\setlength{\tabcolsep}{2.5pt}",
        r"\begin{longtable}{@{}l c r r r r r@{}}",
        rf"\caption{{{caption_title} para los cuatro algoritmos de la evaluación exhaustiva. Usamos $p = 0.95$ y los top~100 de los modelos; las 57 combinaciones están ordenadas por RBO descendente. La ganancia es la diferencia entre la fusión y su mejor miembro en {caption_metric}. La matriz de RBO por pares está en la \cref{{tab:resultados-rbo}}.}}",
        rf"\label{{{table_label}}}\\",
        r"\toprule",
        r"Combinación & Modelos & RBO & CombMNZ & RBC & RRF & ISR \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Combinación & Modelos & RBO & CombMNZ & RBC & RRF & ISR \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for combo in combos:
        row = data["CombMNZ"][combo]
        gains = " & ".join(
            f"${fp(data[algorithm][combo][key], 4, sign=True)}$" for algorithm in sources
        )
        lines.append(
            f"{codes(combo)} & {row['size']} & {fp(row['mean_rbo'])} & {gains} \\\\"
        )
    lines.append(r"\end{longtable}}")
    return "\n".join(lines)


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    text = replace_block(text, "TABLA-A3", table_a3())
    text = replace_block(text, "TABLA-A5", table_a5())
    text = replace_block(text, "TABLA-A6", table_a6())
    text = replace_block(text, "TABLA-A7", table_a7())
    text = replace_block(
        text,
        "TABLA-A8",
        rbo_block("ndcg", "RBO y ganancia en nDCG@10", "tab:ap-rbo-ndcg", "nDCG@10"),
    )
    text = replace_block(
        text,
        "TABLA-A9",
        rbo_block("recall", "RBO y ganancia en Recall@100", "tab:ap-rbo-recall", "Recall@100"),
    )
    TARGET.write_text(text, encoding="utf-8")
    print("Tablas regeneradas:")
    print("  A.3 228 configuraciones exhaustivas + 6 BordaFuse + 1 Condorcet = 235 filas")
    print("  A.5 6 fusiones; A.6 15 pares x 2 métricas; A.7 3 comparaciones x 2 métricas")
    print("  A.8 y A.9 57 combinaciones cada una")


if __name__ == "__main__":
    main()
