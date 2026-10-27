# Piloto de selección sobre la partición de entrenamiento

> **Piloto exploratorio.** Corpus cerrado de 10000 artículos y 317750 documentos. Las métricas absolutas no son comparables con las de prueba y este informe no es evidencia para la tesis; sirve para validar la tubería y ver una primera señal.

## Partición

- Artículos: 10000
- Documentos: 317750
- Consultas: 156949
- Juicios: 161644
- Filtro: {'match_score': 1.0, 'expanded_search': False}
- Semilla: 42
- Modelos: `jina-v5-small`, `bge-m3`, `e5-large`, `qwen3-0.6b`
- Commit: `db24667`

## Modelos individuales

| # | modelo | familia | nDCG@10 | Recall@100 |
| ---: | --- | --- | ---: | ---: |
| 1 | `jina-v5-small` | denso | 0.6075 | 0.9758 |
| 2 | `bge-m3` | denso | 0.5629 | 0.9477 |
| 3 | `e5-large` | denso | 0.5452 | 0.9513 |
| 4 | `qwen3-0.6b` | denso | 0.5317 | 0.9307 |

## Mejores fusiones con el algoritmo de referencia

| # | combinación | algoritmo | nDCG@10 | Recall@100 | Δ vs. la mejor |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | e5-large+bge-m3+jina-v5-small | combmnz | 0.6253 | 0.9809 | +0.0000 |
| 2 | bge-m3+jina-v5-small | combmnz | 0.6210 | 0.9789 | -0.0043 |
| 3 | qwen3-0.6b+e5-large+bge-m3+jina-v5-small | combmnz | 0.6207 | 0.9798 | -0.0047 |
| 4 | e5-large+jina-v5-small | combmnz | 0.6172 | 0.9792 | -0.0081 |
| 5 | qwen3-0.6b+bge-m3+jina-v5-small | combmnz | 0.6131 | 0.9774 | -0.0122 |
| 6 | qwen3-0.6b+e5-large+jina-v5-small | combmnz | 0.6124 | 0.9781 | -0.0130 |
| 7 | qwen3-0.6b+e5-large+bge-m3 | combmnz | 0.6032 | 0.9688 | -0.0221 |
| 8 | qwen3-0.6b+jina-v5-small | combmnz | 0.5942 | 0.9735 | -0.0311 |
| 9 | e5-large+bge-m3 | combmnz | 0.5928 | 0.9661 | -0.0325 |
| 10 | qwen3-0.6b+bge-m3 | combmnz | 0.5881 | 0.9555 | -0.0372 |

## Comprobación de algoritmos sobre el trío

| # | algoritmo | nDCG@10 | Recall@100 |
| ---: | --- | ---: | ---: |
| 1 | combmnz | 0.6253 | 0.9809 |
| 2 | rbc | 0.6220 | 0.9799 |
| 3 | rrf | 0.6186 | 0.9806 |
| 4 | isr | 0.6163 | 0.9806 |

## Lectura

- Mejor modelo individual: `jina-v5-small` (0.6075).
- Tres mejores modelos individuales: `jina-v5-small`, `bge-m3`, `e5-large`.
- El trío `e5-large+bge-m3+jina-v5-small` ocupa la posición 1 de 11 con 0.6253 de nDCG@10 (Δ = +0.0000 frente a la mejor).
- CombMNZ ocupa la posición 1 de 4 en la comprobación de algoritmos (Δ = +0.0000 frente al mejor).

## Coincidencia con los sistemas reportados

- ¿El top-3 individual es el trío denso? **sí**.
- ¿El mejor modelo individual es `jina-v5-small`? **sí**.
- ¿El trío es la mejor fusión con el algoritmo de referencia? **sí**.
- ¿El algoritmo de referencia es el mejor de los cuatro? **sí**.

En el piloto estas coincidencias son solo una señal: el corpus cerrado puede cambiar el orden de los sistemas. La decisión definitiva requiere el corpus completo.
