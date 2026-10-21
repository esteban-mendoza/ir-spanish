# Results

## Resultados previos

| Model                         | nDCG@10 | Recall@100 |
| ----------------------------- | ------: | ---------: |
| BM25                          |   0.179 |      0.558 |
| MIRACL-mdpr-es                |   0.284 |      0.658 |
| E5-large                      |   0.463 |      0.865 |
| E5-large-ft-messirve          |   0.491 |      0.887 |
| OpenAI-text-embedding-3-large |   0.476 |      0.916 |

## Mis resultados (max_seq_length)

| Model                                   | length | nDCG@10 | Recall@100 | MRR@10 |    MAP |   P@10 |   P@50 |
| --------------------------------------- | -----: | ------: | ---------: | -----: | -----: | -----: | -----: |
| BM25                                    |    N/A |  0.1848 |     0.5725 | 0.1483 | 0.1575 | 0.0311 | 0.0102 |
| splade-v3                               |    512 |  0.1956 |     0.5725 | 0.1580 | 0.1666 | 0.0326 | 0.0103 |
| qwen3-embedding-0.6b                    | 32,768 |  0.4468 |     0.8423 | 0.3886 | 0.3947 | 0.0651 | 0.0163 |
| multilingual-e5-large-instruct          |    512 |  0.4675 |     0.8666 | 0.4079 | 0.4140 | 0.0676 | 0.0168 |
| bge-m3                                  |  8,192 |  0.4818 |     0.8741 | 0.4206 | 0.4264 | 0.0696 | 0.0171 |
| jina-embeddings-v5-text-small-retrieval | 32,768 |  0.5115 |     0.9039 | 0.4481 | 0.4538 | 0.0733 | 0.0178 |

## Rerankings

| Model              |                                    from | nDCG@10 | Recall@100 | MRR@10 |    MAP |   P@10 |   P@50 |
| ------------------ | --------------------------------------: | ------: | ---------: | -----: | -----: | -----: | -----: |
| jina-reranker-v3   | jina-embeddings-v5-text-small-retrieval |  0.5792 |     0.9039 | 0.5167 | 0.5201 | 0.0798 | 0.0182 |
| bge-reranker-v2-m3 | jina-embeddings-v5-text-small-retrieval |  0.5770 |     0.9039 | 0.5124 | 0.5159 | 0.0802 | 0.0183 |
| jina-colbert-v2    | jina-embeddings-v5-text-small-retrieval |  0.5019 |     0.9039 | 0.4358 | 0.4423 | 0.0731 | 0.0180 |

## Hybrid
| Model              |                          from | nDCG@10 | Recall@100 | MRR@10 |    MAP |   P@10 |   P@50 |
| ------------------ | ----------------------------: | ------: | ---------: | -----: | -----: | -----: | -----: |
| jina-reranker-v3   | e5-large+bge-m3+jina-v5-small |  0.5185 |     0.9275 | 0.4532 | 0.4595 | 0.0747 | 0.0183 |

## Fusiones (max_seq_length)

| model                         | strategy  | params   | ndcg@10 | recall@100 | mrr@10 | map    | precision@10 | precision@50 |
| ----------------------------- | --------- | -------- | ------- | ---------- | ------ | ------ | ------------ | ------------ |
| e5-large+bge-m3+jina-v5-small | mnz       |          | 0.5517  | 0.9231     | 0.4878 | 0.4929 | 0.0775       | 0.0183       |
| e5-large+bge-m3+jina-v5-small | rbc       | phi=0.95 | 0.5485  | 0.9206     | 0.4835 | 0.4884 | 0.0775       | 0.0182       |
| e5-large+bge-m3+jina-v5-small | rrf       | k=60     | 0.5450  | 0.9234     | 0.4806 | 0.4859 | 0.0770       | 0.0183       |
| e5-large+bge-m3+jina-v5-small | isr       |          | 0.5382  | 0.9230     | 0.4743 | 0.4798 | 0.0763       | 0.0183       |
| e5-large+bge-m3+jina-v5-small | bordafuse |          | 0.5414  | 0.9234     | 0.4776 | 0.4831 | 0.0764       | 0.0183       |
| e5-large+bge-m3+jina-v5-small | condorcet |          | 0.5422  | 0.8918     | 0.4791 | 0.4837 | 0.0763       | 0.0177       |

## Etapa B: BordaFuse por combinación

| # | Combinación | Modelos | Tiempo (s) | nDCG@10 | Recall@100 | Estado |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| B1 | e5-large+bge-m3+jina-v5-small | 3 | 595.0 | 0.5414 | 0.9234 | completa |
| B2 | bm25+splade-v3+qwen3-0.6b+e5-large+bge-m3+jina-v5-small | 6 | 1 519 (hasta el OOM) | — | — | OOM, sin resultado |
| B3 | splade-v3+qwen3-0.6b+e5-large+bge-m3+jina-v5-small | 5 | 1 259.8 | 0.4962 | 0.9237 | completa |
| B4 | splade-v3+e5-large+jina-v5-small | 3 | 651.2 | 0.4658 | 0.9119 | completa |
| B5 | splade-v3+jina-v5-small | 2 | 394.1 | 0.3858 | 0.8904 | completa |
| B6 | qwen3-0.6b+e5-large+bge-m3+jina-v5-small | 4 | 846.5 | 0.5363 | 0.9228 | completa |

B2 con los cuatro algoritmos rápidos sí terminó (CombMNZ 0.5277, RBC 0.5309, ISR 0.5234 y RRF 0.4985 de nDCG@10), así que la inviabilidad es específica de BordaFuse con seis listas, no de la combinación de seis modelos. La combinación completa tampoco mejora el nDCG@10 de B1 con esos algoritmos.

## Fusiones exhaustivas (57 subconjuntos, longitudes nativas)

Barrido completo con CombMNZ y RBC $\varphi = 0.95$. El CSV completo está en `results/fuse_sweep/results.csv`.

### Mejores por nDCG@10

| Combinación | Estrategia | nDCG@10 | Recall@100 |
| --- | --- | ---: | ---: |
| e5-large+bge-m3+jina-v5-small | combmnz | 0.5517 | 0.9231 |
| e5-large+bge-m3+jina-v5-small | rbc | 0.5485 | 0.9206 |
| qwen3-0.6b+e5-large+bge-m3+jina-v5-small | combmnz | 0.5482 | 0.9228 |
| qwen3-0.6b+e5-large+bge-m3+jina-v5-small | rbc | 0.5462 | 0.9193 |
| bge-m3+jina-v5-small | combmnz | 0.5445 | 0.9184 |
| splade-v3+qwen3-0.6b+e5-large+bge-m3+jina-v5-small | combmnz | 0.5400 | 0.9225 |
| splade-v3+qwen3-0.6b+e5-large+bge-m3+jina-v5-small | rbc | 0.5396 | 0.9157 |
| bge-m3+jina-v5-small | rbc | 0.5388 | 0.9177 |
| splade-v3+e5-large+bge-m3+jina-v5-small | combmnz | 0.5382 | 0.9217 |
| qwen3-0.6b+bge-m3+jina-v5-small | combmnz | 0.5379 | 0.9164 |

### Mejores por Recall@100

| Combinación | Estrategia | nDCG@10 | Recall@100 |
| --- | --- | ---: | ---: |
| bm25+qwen3-0.6b+e5-large+bge-m3+jina-v5-small | combmnz | 0.5372 | 0.9269 |
| bm25+e5-large+bge-m3+jina-v5-small | combmnz | 0.5339 | 0.9259 |
| bm25+splade-v3+qwen3-0.6b+e5-large+bge-m3+jina-v5-small | combmnz | 0.5277 | 0.9256 |
| bm25+splade-v3+e5-large+bge-m3+jina-v5-small | combmnz | 0.5208 | 0.9241 |
| e5-large+bge-m3+jina-v5-small | combmnz | 0.5517 | 0.9231 |

## RBO entre modelos de primera etapa (p = 0.95, top-100)

Promedio sobre las consultas de prueba. Los pares con BM25 usan 169 963 consultas (BM25 no recupera 92). Calculado con `ranked-overlap` 0.1.0 (MIT), que implementa la ecuación de Webber et al. (2010).

| Modelo A | Modelo B | RBO |
| --- | --- | ---: |
| bm25 | splade-v3 | 0.1704 |
| bm25 | qwen3-0.6b | 0.1740 |
| bm25 | e5-large | 0.1617 |
| bm25 | bge-m3 | 0.1598 |
| bm25 | jina-v5-small | 0.1721 |
| splade-v3 | qwen3-0.6b | 0.1718 |
| splade-v3 | e5-large | 0.1515 |
| splade-v3 | bge-m3 | 0.1668 |
| splade-v3 | jina-v5-small | 0.1777 |
| qwen3-0.6b | e5-large | 0.3804 |
| qwen3-0.6b | bge-m3 | 0.3905 |
| qwen3-0.6b | jina-v5-small | 0.4849 |
| e5-large | bge-m3 | 0.4118 |
| e5-large | jina-v5-small | 0.4081 |
| bge-m3 | jina-v5-small | 0.4176 |

## E5: RBO y ganancia

La ganancia se mide como nDCG@10 de la fusión menos nDCG@10 del mejor miembro; la complementariedad de una combinación es el promedio de los RBO de sus pares. Sobre las 57 combinaciones de la etapa A (CombMNZ):

- nDCG@10: Pearson 0.678 (IC 95 % 0.521–0.808), Spearman 0.716 (0.508–0.852).
- Recall@100: Pearson 0.253 (IC 95 % −0.025–0.553), Spearman 0.360 (0.076–0.588).
- En el subconjunto de tres o más modelos, la correlación con nDCG@10 se mantiene (0.682) y la de recall deja de ser significativa.

La relación es **positiva**: las combinaciones con RBO alto —los tríos y cuartetos densos— son las que más mejoran sobre su mejor miembro. La complementariedad léxico-semántica por sí sola no explica la ganancia; los pares mixtos tienen RBO bajo pero no siempre superan a su mejor miembro denso. Archivos: `results/e5/rbo_gain.csv` y `results/e5/summary.json`.

## E4: líneas base

La tabla descriptiva está en `results/e4/baselines.md`: la mejor fusión (0.5517 / 0.9231) supera a `text-embedding-3-large` (0.476 / 0.916); el mejor reranker (0.5792 / 0.9039) la supera en nDCG@10 pero no en Recall@100.

## E3: eficiencia de la segunda etapa

Banco controlado de 500 consultas de prueba, tres repeticiones y orden intercalado. La latencia se midió con lote 1 en una sola GPU y el throughput con lote 1 y 32. La fusión corre en CPU con un hilo; los rerankers, en `cuda:0`. Los tiempos excluyen la carga de modelos, la carga del corpus y la primera etapa; la evaluación de métricas queda fuera del cronómetro. Los valores son la mediana de las tres repeticiones; los intervalos son un *bootstrap* de 10 000 remuestras sobre las consultas.

| Sistema | Mediana (ms) | P95 (ms) | Razón mediana | IC 95 % | Razón P95 | Throughput b1 rel. | Throughput b32 rel. | VRAM pico (MiB) | RSS pico (MiB) |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Fusión B1 (CombMNZ) | 1.903 | 2.348 | 1.000 | [1.000, 1.000] | 1.000 | 1.000 | 1.000 | — | 8 351 |
| `bge-reranker-v2-m3` | 2 247.752 | 2 938.009 | 1 182.020 | [1 145.317, 1 205.856] | 1 251.526 | 0.0013 | 0.0219 | 4 401 | 9 432 |
| `jina-reranker-v3` | 793.453 | 1 329.855 | 416.966 | [401.792, 428.151] | 565.580 | 0.0035 | 0.0256 | 3 153 | 8 403 |
| `jina-colbert-v2` | 410.756 | 477.672 | 215.855 | [213.504, 218.308] | 203.477 | 0.0072 | 0.0524 | 1 385 | 8 343 |

En la **segunda etapa**, la fusión es **dos o tres órdenes de magnitud más barata** que los rerankers. La mediana de latencia pasa de 1.9 ms a 0.41–2.25 s; los cross-encoders son 1 182 y 417 veces más lentos, y el más barato de los tres, `jina-colbert-v2`, 216 veces. Con lote 32 la brecha se reduce a 19–46 veces, pero no desaparece. La ganancia de nDCG@10 de los cross-encoders (+0.025 a +0.028) se paga con ese costo; `jina-colbert-v2` es más barato pero también menos efectivo (0.5019, por debajo de la fusión).

**Alcance del costo.** Estas razones son de la **segunda etapa**: suponen las listas de candidatos ya calculadas. La fusión necesita tres recuperadores de primera etapa (`e5-large`, `bge-m3` y `jina-v5-small`) y los rerankers solo uno (`jina-v5-small`). La codificación de los tres se midió aparte (sección siguiente): **57.53 ms/consulta con lote 1** y **2.41 ms/consulta con lote 32**; de ellos, **27.96 ms** y **1.28 ms** corresponden a los dos recuperadores extra de la fusión. Es una cota inferior: no incluye la búsqueda en los índices ni el costo de mantener tres índices. Sumando la codificación total, la fusión queda en **~59.4 ms** (lote 1) o **~23.8 ms** (lote 32), frente a 410–2 248 ms de los rerankers. La ventaja relativa baja de 1 182× a **~38×** frente a `bge-reranker-v2-m3`, **~14×** frente a `jina-reranker-v3` y **~7×** frente a `jina-colbert-v2` (lote 1); con lote 32, a ~41×, ~35× y ~17×. La ventaja se reduce, pero no desaparece.

La razón de throughput del lote 1 no coincide exactamente con la inversa de la razón de latencia mediana porque el throughput usa la media de las latencias y la fusión tiene valores atípicos (una consulta de 4.9 s en la primera repetición; mediana 1.9 ms, p95 2.5 ms). Por eso la comparación principal es la mediana y el p95; el throughput se reporta como complemento. La memoria también importa: `bge-reranker-v2-m3` reserva 5 832 MiB de VRAM y la fusión de seis listas con BordaFuse no cabe en 125 GiB (B2), mientras que B3, con cinco, queda al límite.

Archivos: `results/efficiency/summary.csv`, `summary.json`, `summary.md`, `raw/` y `metadata/`; log en `logs/efficiency_run.log`.

## E3: primera etapa (codificación de consultas)

La diferencia de primera etapa entre las dos rutas son los dos recuperadores extra de la fusión: `e5-large` y `bge-m3`. `jina-v5-small` es común y se cancela. Medimos la codificación de consultas sobre las mismas 500 consultas, con una GPU, lote 1 y lote 32.

| Modelo | Mediana b1 (ms) | P95 b1 (ms) | QPS b1 | QPS b32 | RSS pico (MiB) | VRAM pico (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `e5-large` | 14.649 | 15.965 | 67.27 | 1 436.70 | 2 358 | 1 113 |
| `bge-m3` | 13.306 | 14.813 | 73.93 | 1 700.67 | 4 077 | 1 111 |
| `jina-v5-small` | 29.574 | 31.774 | 33.40 | 890.70 | 2 209 | 1 275 |

Costo adicional de los dos recuperadores extra (e5 + bge): **27.96 ms/consulta en lote 1** y **1.28 ms/consulta en lote 32**. Codificación total de la primera etapa (e5 + bge + jina): **57.53 ms/consulta en lote 1** y **2.41 ms/consulta en lote 32**.

Es una **cota inferior** de la primera etapa: no incluye la búsqueda en los índices ni el costo de almacenar y mantener tres índices. Con esa cota, el costo de la fusión queda en ~59.4 ms/consulta (lote 1) o ~23.8 ms (lote 32) frente a 410–2 248 ms de los rerankers. La ventaja relativa baja de 1 182× a ~38× en el caso más desfavorable, pero sigue siendo de varias veces. Para el costo total habría que medir la búsqueda en los índices; no está incluida.

Archivos: `results/first_stage/summary.csv`, `summary.json`, `summary.md`, `e5.json`, `bge.json`, `jina.json`, `e5_batch1.csv`, `bge_batch1.csv`, `jina_batch1.csv`; log en `logs/first_stage.log`.

## Índices y almacenamiento

Los 8 351 MiB de RSS de la fusión son los tres *runs* cargados, no los índices. El corpus tiene 14 047 759 documentos y los tres modelos densos producen vectores de 1 024 dimensiones; con `faiss.IndexFlatIP`:

| Representación | Un índice | Tres índices |
| --- | ---: | ---: |
| `float32` (CPU/disco) | 53.59 GiB | 160.76 GiB |
| `float16` (GPU, sharded) | 26.79 GiB | 80.38 GiB |
| `int8` (cuantización, no usada) | 13.40 GiB | 40.19 GiB |

La fusión necesita los tres índices: 160.8 GiB no caben en 125 GiB de RAM, y 80.4 GiB no caben en 48 GiB de las dos A5000. Un solo índice (53.6 GiB en RAM o 26.8 GiB en GPU, 13.4 por GPU) sí cabe: la ruta del reranker es viable en este servidor; la fusión con índices planos, no. El disco sí alcanza (399 GB libres; 172.6 GB de índices), pero el límite es la memoria de trabajo.

Consecuencia: la fusión puede ser más barata **por consulta** y, a la vez, no ser desplegable en este servidor. El análisis completo está en `tesis/knowledge/indices-almacenamiento-fusion.md`; una variante comprimida (SQ8, IVF-PQ) requeriría reconstruir los índices y repetir la evaluación, no se puede extrapolar.

## Límites de recursos de la etapa B

La etapa B se ejecutó en el servidor del IIMAS (125 GiB de RAM, sin swap). B2 —BordaFuse sobre los seis modelos— no terminó: el RSS muestreado llegó a 123 196 256 kB (117.5 GiB) y el 2026-10-20, a las 21:54:29, el cgroup de la sesión registró `oom_kill=1`. El proceso desapareció a las 21:59:40 con código de salida 137 y sin traza de Python. La memoria disponible llegó a 0 kB y el servidor dejó de responder por varios minutos. El pico es una cota inferior: el muestreo era cada 15 s.

Las configuraciones que sí terminaron fueron B1, B3, B4, B5 y B6. B6 (cuatro modelos) tardó 846.5 s; B3 (cinco modelos) tardó 1 259.8 s y alcanzó un RSS de 105 018 624 kB (100.2 GiB), con 19.4 GB de memoria disponible en el pico; B1, B4 y B5 (tres, tres y dos modelos) tardaron 595.0 s, 651.2 s y 394.1 s. El contraste es nítido: con BordaFuse, cinco modelos caben con poco margen y seis no caben. El costo de BordaFuse no es lineal con el número de listas y la combinación completa no es viable en este servidor; no se imputa el resultado de B2.

Condorcet B1 (tres modelos) terminó en 6 567 s (1.82 h) con 0.5422 de nDCG@10 y 0.8918 de Recall@100. No se ejecutó en combinaciones mayores. No hay una medición de OOM para Condorcet: se descarta como alternativa práctica por el tiempo observado y por su menor recall, no por una falla de memoria comprobada.

Evidencia: `logs/fuse_stage_b_b2_oom_watch.log`, `logs/fuse_stage_b_b2_rss.log`, `logs/fuse_stage_b_b2_memory.log`, `logs/fuse_stage_b_b3_oom_watch.log`, `logs/fuse_stage_b_b3_rss.log`, `logs/fuse_stage_b_b3_memory.log` y `logs/run_b2_b3.log`. En el monitor de B3, `oom_kill=1` es el contador heredado de B2 y no aumentó durante B3.
