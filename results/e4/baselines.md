# Experimento 4: sistemas abiertos frente a líneas base

Comparación descriptiva; las líneas base de MessIRve solo publican promedios y no puntuaciones por consulta, así que no se aplica una prueba de significancia.

| Sistema | Tipo | nDCG@10 | Recall@100 |
| --- | --- | ---: | ---: |
| OpenAI text-embedding-3-large | propietario | 0.476 | 0.916 |
| E5-large-ft-messirve | abierto, ajustado en MessIRve | 0.491 | 0.887 |
| E5-large | abierto | 0.463 | 0.865 |
| MIRACL-mdpr-es | abierto | 0.284 | 0.658 |
| BM25 (MessIRve) | abierto | 0.179 | 0.558 |
| B1-CombMNZ (fusión) | abierto | **0.5517** | **0.9231** |
| jina-reranker-v3 | abierto | **0.5792** | 0.9039 |
| jina-v5-small | abierto | 0.5115 | 0.9039 |

La mejor fusión supera a la línea base propietaria en nDCG@10 y Recall@100. El mejor reranker abierto supera a la línea propietaria en nDCG@10, pero no en Recall@100.
