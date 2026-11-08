# Costo de codificar documentos e índices

Muestra: 20000 documentos, 10 deciles de longitud, semilla 42.

| Modelo | Tope nativo | Seq. usada | Lote | docs/s | IC 95 % | tok/s | ms/doc | Horas a 14 M | Trunc. % | VRAM (MiB) | Bytes/vector | Un índice (GiB) |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e5-large-instruct | 512 | 512 | 128 | 615.0 | [554.8, 684.4] | 55950 | 1.63 | 6.35 | 0.14% | 2519 | 4096 | 53.59 |
| bge-m3 | 8192 | 2048 | 32 | 560.4 | [514.3, 601.9] | 51129 | 1.78 | 6.96 | 0.01% | 2629 | 4096 | 53.59 |
| jina-v5-small | 32768 | 2048 | 32 | 213.8 | [197.4, 229.4] | 22444 | 4.68 | 18.25 | 0.00% | 9957 | 4096 | 53.59 |

**Ruta de 1 modelo (`jina-v5-small`):** 18.25 h y 53.59 GiB.
**Ruta de 3 modelos:** 31.56 h y 160.76 GiB (tres índices).
**Costo adicional:** 13.31 h y 107.18 GiB.

## Codificación de consultas

| Modelo | Mediana b1 (ms) | P95 b1 (ms) | QPS b32 |
| --- | ---: | ---: | ---: |
| e5-large-instruct | 14.80 | 15.10 | 1598.8 |
| bge-m3 | 13.19 | 13.48 | 2098.9 |
| jina-v5-small | 30.31 | 30.61 | 915.8 |

**1 modelo:** 30.31 ms/consulta (b1), 1.09 ms (b32).
**3 modelos:** 58.30 ms/consulta (b1), 2.19 ms (b32).

El costo de documentos e índices es fuera de línea; el de consultas es en línea. No se suman.
