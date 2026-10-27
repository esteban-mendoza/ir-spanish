# Estabilidad piloto–prueba

El piloto es el entorno de selección; la prueba se usa aquí solo de forma descriptiva para comprobar si el orden se transfiere al corpus completo. Ninguna elección de este informe usa la partición de prueba.

## Modelos individuales

| modelo | piloto | prueba |
| --- | ---: | ---: |
| `jina-v5-small` | 0.6075 | 0.5115 |
| `bge-m3` | 0.5629 | 0.4818 |
| `e5-large` | 0.5452 | 0.4675 |
| `qwen3-0.6b` | 0.5317 | 0.4468 |
| `splade-v3` | 0.2380 | 0.1956 |
| `bm25` | 0.2251 | 0.1848 |

Spearman = 1.0000; Kendall = 1.0000. Top-3 piloto: jina-v5-small, bge-m3, e5-large; top-3 prueba: jina-v5-small, bge-m3, e5-large.

## Fusiones (57 subconjuntos, CombMNZ)

| # | combinación | piloto | prueba |
| ---: | --- | ---: | ---: |
| 1 | e5-large+bge-m3+jina-v5-small | 0.6253 | 0.5517 |
| 2 | bge-m3+jina-v5-small | 0.6210 | 0.5445 |
| 3 | qwen3-0.6b+e5-large+bge-m3+jina-v5-small | 0.6207 | 0.5482 |
| 4 | e5-large+jina-v5-small | 0.6172 | 0.5370 |
| 5 | qwen3-0.6b+bge-m3+jina-v5-small | 0.6131 | 0.5379 |
| 6 | qwen3-0.6b+e5-large+jina-v5-small | 0.6124 | 0.5340 |
| 7 | qwen3-0.6b+e5-large+bge-m3 | 0.6032 | 0.5328 |
| 8 | splade-v3+qwen3-0.6b+e5-large+bge-m3+jina-v5-small | 0.6030 | 0.5400 |
| 9 | splade-v3+e5-large+bge-m3+jina-v5-small | 0.5979 | 0.5382 |
| 10 | bm25+qwen3-0.6b+e5-large+bge-m3+jina-v5-small | 0.5970 | 0.5372 |

Spearman = 0.9726; Kendall = 0.8784.
Mejor piloto: e5-large+bge-m3+jina-v5-small; mejor prueba: e5-large+bge-m3+jina-v5-small. El trío ocupa la posición 1 en el piloto y 1 en prueba.
Coincidencia del top-1: 1/1.
Coincidencia del top-3: 3/3.
Coincidencia del top-5: 3/5.

## Algoritmos sobre el trío

| algoritmo | piloto | prueba |
| --- | ---: | ---: |
| combmnz | 0.6253 | 0.5517 |
| rbc | 0.6220 | 0.5485 |
| rrf | 0.6186 | 0.5450 |
| condorcet | 0.6183 | 0.5422 |
| isr | 0.6163 | 0.5382 |
| bordafuse | 0.6154 | 0.5414 |

Spearman = 0.9429; Kendall = 0.8667.

## Lectura

Si las correlaciones son altas y el top-1 coincide, el piloto reproduce el orden de la evaluación completa y sirve para seleccionar sin usar la partición de prueba. Aun así, el piloto es un corpus cerrado (2.3 % del corpus) y la transferencia es descriptiva, no una garantía: esa es la limitación residual que hay que declarar.
