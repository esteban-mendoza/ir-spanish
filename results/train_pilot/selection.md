# Piloto de selección sobre la partición de entrenamiento

> **Piloto de selección.** Corpus cerrado de 10000 artículos y 317750 documentos. Las métricas absolutas no son comparables con las de prueba; el piloto es el entorno de selección. La transferencia al corpus completo se mide en `stability.md`.

## Partición

- Artículos: 10000
- Documentos: 317750
- Consultas: 156949
- Juicios: 161644
- Filtro: {'match_score': 1.0, 'expanded_search': False}
- Semilla: 42
- Modelos: `jina-v5-small`, `bge-m3`, `e5-large`, `qwen3-0.6b`, `splade-v3`, `bm25`
- Commit: `db24667`

## Modelos individuales

| # | modelo | familia | nDCG@10 | Recall@100 |
| ---: | --- | --- | ---: | ---: |
| 1 | `jina-v5-small` | denso | 0.6075 | 0.9758 |
| 2 | `bge-m3` | denso | 0.5629 | 0.9477 |
| 3 | `e5-large` | denso | 0.5452 | 0.9513 |
| 4 | `qwen3-0.6b` | denso | 0.5317 | 0.9307 |
| 5 | `splade-v3` | disperso | 0.2380 | 0.6928 |
| 6 | `bm25` | léxico | 0.2251 | 0.6779 |

## Mejores fusiones con el algoritmo de referencia

| # | combinación | algoritmo | nDCG@10 | Recall@100 | Δ vs. la mejor | IC 95 % |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 1 | e5-large+bge-m3+jina-v5-small | combmnz | 0.6253 | 0.9809 | +0.0000 | [0.0000, 0.0000] |
| 2 | bge-m3+jina-v5-small | combmnz | 0.6210 | 0.9789 | -0.0043 | [-0.0050, -0.0036] |
| 3 | qwen3-0.6b+e5-large+bge-m3+jina-v5-small | combmnz | 0.6207 | 0.9798 | -0.0047 | [-0.0052, -0.0041] |
| 4 | e5-large+jina-v5-small | combmnz | 0.6172 | 0.9792 | -0.0081 | [-0.0088, -0.0074] |
| 5 | qwen3-0.6b+bge-m3+jina-v5-small | combmnz | 0.6131 | 0.9774 | -0.0122 | [-0.0130, -0.0113] |
| 6 | qwen3-0.6b+e5-large+jina-v5-small | combmnz | 0.6124 | 0.9781 | -0.0130 | — |
| 7 | qwen3-0.6b+e5-large+bge-m3 | combmnz | 0.6032 | 0.9688 | -0.0221 | — |
| 8 | splade-v3+qwen3-0.6b+e5-large+bge-m3+jina-v5-small | combmnz | 0.6030 | 0.9789 | -0.0223 | — |
| 9 | splade-v3+e5-large+bge-m3+jina-v5-small | combmnz | 0.5979 | 0.9789 | -0.0275 | — |
| 10 | bm25+qwen3-0.6b+e5-large+bge-m3+jina-v5-small | combmnz | 0.5970 | 0.9796 | -0.0283 | — |

## Comprobación de algoritmos sobre el trío

| # | algoritmo | nDCG@10 | Recall@100 |
| ---: | --- | ---: | ---: |
| 1 | combmnz | 0.6253 | 0.9809 |
| 2 | rbc | 0.6220 | 0.9799 |
| 3 | rrf | 0.6186 | 0.9806 |
| 4 | condorcet | 0.6183 | 0.9678 |
| 5 | isr | 0.6163 | 0.9806 |
| 6 | bordafuse | 0.6154 | 0.9806 |

## Lectura

- Mejor modelo individual: `jina-v5-small` (0.6075).
- Tres mejores modelos individuales: `jina-v5-small`, `bge-m3`, `e5-large`.
- El trío `e5-large+bge-m3+jina-v5-small` ocupa la posición 1 de 57 con 0.6253 de nDCG@10 (Δ = +0.0000 frente a la mejor).
- CombMNZ ocupa la posición 1 de 6 en la comprobación de algoritmos (Δ = +0.0000 frente al mejor).

## Coincidencia con los sistemas reportados

- ¿El top-3 individual es el trío denso? **sí**.
- ¿El mejor modelo individual es `jina-v5-small`? **sí**.
- ¿El trío es la mejor fusión con el algoritmo de referencia? **sí**.
- ¿El algoritmo de referencia es el mejor de los seis? **sí**.

Estas coincidencias son la evidencia de selección. El análisis de estabilidad (`stability.md`) comprueba que el orden se reproduce en la partición de prueba; la diferencia del top-5 es la limitación residual.
