| Modelo | Mediana (ms) | P95 (ms) | QPS b1 | QPS b32 | RSS pico (MiB) | VRAM pico (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| e5 | 14.649 | 15.965 | 67.27 | 1436.70 | 2358 | 1113 |
| bge | 13.306 | 14.813 | 73.93 | 1700.67 | 4077 | 1111 |
| jina | 29.574 | 31.774 | 33.40 | 890.70 | 2209 | 1275 |

Costo adicional de codificación (e5 + bge, lote 32): **1.28 ms/consulta**.
Codificación de jina-v5 (lote 32): **1.12 ms/consulta**.
Codificación total de la primera etapa (e5 + bge + jina, lote 32): **2.41 ms/consulta**.
