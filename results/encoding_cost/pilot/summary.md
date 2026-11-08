# Costo de codificar documentos e índices

Muestra: 1000 documentos, 10 deciles de longitud, semilla 42.

| Modelo | Tope nativo | Seq. usada | Lote | docs/s | IC 95 % | tok/s | ms/doc | Horas a 14 M | Trunc. % | VRAM (MiB) | Bytes/vector | Un índice (GiB) |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e5-large-instruct | 512 | 512 | 128 | 559.4 | [349.3, 963.6] | 50436 | 1.79 | 6.98 | 0.00% | 2139 | 4096 | 53.59 |
| bge-m3 | 8192 | 512 | 128 | 558.4 | [351.7, 951.0] | 50342 | 1.79 | 6.99 | 0.00% | 2153 | 4096 | 53.59 |
| jina-v5-small | 32768 | 512 | 128 | 201.8 | [128.1, 343.2] | 20923 | 4.96 | 19.34 | 0.10% | 8348 | 4096 | 53.59 |

**Ruta de 1 modelo (`jina-v5-small`):** 19.34 h y 53.59 GiB.
**Ruta de 3 modelos:** 33.30 h y 160.76 GiB (tres índices).
**Costo adicional:** 13.96 h y 107.18 GiB.

## Codificación de consultas

| Modelo | Mediana b1 (ms) | P95 b1 (ms) | QPS b32 |
| --- | ---: | ---: | ---: |
| e5-large-instruct | 14.59 | 14.86 | 1616.3 |
| bge-m3 | 13.38 | 13.80 | 2037.8 |
| jina-v5-small | 30.43 | 30.79 | 707.2 |

**1 modelo:** 30.43 ms/consulta (b1), 1.41 ms (b32).
**3 modelos:** 58.39 ms/consulta (b1), 2.52 ms (b32).

El costo de documentos e índices es fuera de línea; el de consultas es en línea. No se suman.
