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
| harrier-oss-v1-0.6b                     | 32,768 |  0.4519 |     0.8768 | 0.3902 | 0.3972 | 0.0668 | 0.0170 |
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
| e5-large+bge-m3+jina-v5-small | condorcet |          | 0.5415  | 0.8913     | 0.4785 | 0.4831 | 0.0762       | 0.0177       |

## Fusiones (q512, d512)

| Model           |     strategy | nDCG@10 | Recall@100 |
| --------------- | -----------: | ------: | ---------: |
| e5 + bge        |   rrf (k=60) |  0.5128 |     0.8997 |
| e5 + bge + jina |   rrf (k=60) |  0.5450 |     0.9232 |
| e5 + bge + jina |  rbc (φ=0.8) |  0.5434 |     0.9191 |
| e5 + bge + jina |  rbc (φ=0.9) |  0.5478 |     0.9194 |
| e5 + bge + jina | rbc (φ=0.95) |  0.5485 |     0.9204 |
| e5 + bge + jina | rbc (φ=0.98) |  0.5460 |     0.9225 |
| e5 + bge + jina |      CombMNZ |  0.5518 |     0.9230 |
| e5 + bge + jina |          ISR |  0.5382 |     0.9228 |
| e5 + bge + jina |    BordaFuse |  0.5413 |     0.9233 |
| e5 + bge + jina |    Condorcet |  0.5418 |     0.8905 |

## Mis resultados (q512, d512)

| Model                                   | nDCG@10 | Recall@100 |
| --------------------------------------- | ------: | ---------: |
| BM25                                    |  0.1848 |     0.5725 |
| splade-v3                               |  0.1956 |     0.5759 |
| qwen3-embedding-0.6b                    |  0.4468 |     0.8422 |
| harrier-oss-v1-0.6b                     |  0.4520 |     0.8769 |
| multilingual-e5-large-instruct          |  0.4675 |     0.8666 |
| bge-m3                                  |  0.4818 |     0.8741 |
| jina-embeddings-v5-text-small-retrieval |  0.5111 |     0.9037 |

## Mis resultados (q64, d256)

| Model                                   | nDCG@10 | Recall@100 |
| --------------------------------------- | ------: | ---------: |
| BM25                                    |  0.1848 |     0.5725 |
| splade-v3                               |  0.1964 |     0.5759 |
| qwen3-embedding-0.6b                    |  0.4463 |     0.8408 |
| harrier-oss-v1-0.6b                     |  0.xxxx |     0.xxxx |
| multilingual-e5-large-instruct          |  0.4625 |     0.8591 |
| bge-m3                                  |  0.4816 |     0.8735 |
| jina-embeddings-v5-text-small-retrieval |  0.xxxx |     0.xxxx |

## Mis resultados (400 words)

| Model                          | nDCG@10 | Recall@100 |
| ------------------------------ | ------: | ---------: |
| BM25                           |  0.1848 |     0.5726 |
| multilingual-e5-large-instruct |  0.4679 |     0.8669 |
| BGE-M3                         |  0.4821 |     0.8743 |

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
