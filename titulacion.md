# Titulación

## Tareas

- Experimentos
  - Implementar rerankers:
    - Interacción tardía:
      - Jina-ColBERT-v2
    - Cross-encoders:
      - bge-reranker-v2-m3
      - jina-reranker-v3
  - Comparar rankings entre sí
  - Comparar con text-embedding-3-large

- Protocolo
  - Obtener fundamentos teóricos de:
    - Embeddings
    - Probabilistic relevance framework
  - Rerankers:
    - RRF
    - Rank-Biased Centroids (rbc)
    - CombMNZ
    - Condorcet-fuse
    - Borda-fuse

- Tesis
  - Estudiar distribución del número de documentos relevantes por query para explicar métricas
  - Revisar resultados y métodos:
    - Artículos utilizando MessIRve
    - MTEB y La Leaderboard
    - Comparar los resultados obtenidos con resultados en otras lenguas para todos los modelos involucrados

## Registro

1. Formato de registro firmado por estudiante y asesor(a).
2. Historial académico SIAE (por lo menos con el 75% de los créditos)
3. Constancia de conclusión de servicio social
4. Propuesta de tema de Tesis
    - Título tentativo
    - Justificación
    - Objetivos
    - Índice tentativo
    - Bibliografía básica en orden alfabético.
    - Firmada de forma autógrafa por el estudiante y por el tutor.

_INCLUIR LOS 4 DOCUMENTOS EN UN SOLO PDF._

1. CV resumido y firmado.
2. Copia de su último grado obtenido (título, anverso y reverso, no cédula).
3. Copia de CURP.

_INCLUIR LOS 3 DOCUMENTOS EN UN SOLO PDF._

## Experimentos

- Léxicos
  - BM25
- Dual-encoders:
  - multilingual-e5-large-instruct
  - BGE-M3
  - Qwen3-Embedding-8B
  - jina-embeddings-v5-text-small
- Dispersos:
  - SPLADE-v3
- Interacción tardía:
  - Jina-ColBERT-v2
- Cross-encoders:
  - bge-reranker-v2-m3
  - jina-reranker-v3

- Rerankers:
  - RRF
  - Rank-Biased Centroids (rbc)
  - CombMNZ
  - Condorcet-fuse
  - Borda-fuse

- `ranx`: biblioteca para rerankings y pruebas de significancia estadística
- `sentence-transformers`: biblioteca de Hugging Face para modelos pre-entrenados
- `pyserini`: biblioteca para bm25

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
| BM25                                    |    N/A |  0.1848 |     0.5725 | 0.xxxx | 0.xxxx | 0.xxxx | 0.xxxx |
| splade-v3                               |    512 |  0.1964 |     0.5759 | 0.xxxx | 0.xxxx | 0.xxxx | 0.xxxx |
| qwen3-embedding-0.6b                    | 32,768 |  0.4468 |     0.8423 | 0.3886 | 0.3947 | 0.0651 | 0.0163 |
| harrier-oss-v1-0.6b                     | 32,768 |  0.4519 |     0.8768 | 0.3902 | 0.3972 | 0.0668 | 0.0170 |
| multilingual-e5-large-instruct          |    512 |  0.4675 |     0.8666 | 0.4079 | 0.4140 | 0.0676 | 0.0168 |
| bge-m3                                  |  8,192 |  0.4818 |     0.8741 | 0.4206 | 0.4264 | 0.0696 | 0.0171 |
| jina-embeddings-v5-text-small-retrieval | 32,768 |  0.5115 |     0.9039 | 0.4481 | 0.4538 | 0.0733 | 0.0178 |

## Mis resultados (q512, d512)

| Model                                   | nDCG@10 | Recall@100 |
| --------------------------------------- | ------: | ---------: |
| BM25                                    |  0.1848 |     0.5725 |
| splade-v3                               |  0.1956 |     0.5725 |
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

## Mis resultados (400 words)

| Model                          | nDCG@10 | Recall@100 |
| ------------------------------ | ------: | ---------: |
| BM25                           |  0.1848 |     0.5726 |
| multilingual-e5-large-instruct |  0.4679 |     0.8669 |
| BGE-M3                         |  0.4821 |     0.8743 |

## Docs

Protocolo: <https://www.overleaf.com/project/69af3f611c674ab6c9bfd1f3>

## Notas

Truncating at 400 words:

- Filtered corpus: 14,037,518 / 14,047,759 documents kept (0.99927099)
- Filtered queries: 169,894 / 170,055 queries kept (0.99905325)
- Filtered qrels: 173914 / 174078 qrels kept (0.99905789)

qrel: query relevance judgment

- Encoder (BERT):
  - Takes a full sequence and builds a rich representation of each token using full context.
  - Every token can attend to each other.
  - It cannot generate text natively.
  - It produces embeddings/representations.
  - Its training objective is masked language modeling (MLM). Predicts masked words.
  - Used for classification, named entity recognition, sentence similarity, extractive QA (find an answer in text).

- Decoder (GPT-2, GPT-3, LLaMA)
  - Attention is unidirectional: each token can attend itself and previous tokens.
  - Predicts the next token.
  - Its training objective is next token prediction (causal LM).
  - Used for text generation, conversational AI, code generation, general-purpose tasks.
  - With enough scale they have proven they can handle tasks that were traditionally encoder-only or encoder-decoder territory by framing everything as "complete this text."

- Encoder-decoder (T5, BART)
  - Attention in the encoder is bidirectional; the decoder is causal but also cross-attends to the encoder's output.
  - There's a clear separation between input and output.
  - Used for translation, summarization, abstractive QA (generate an answer), sequence-to-sequence tasks.

- Arquitecturas
  - Dual-encoders:
    - Query and document are separately embedded with the same model, then we compare each other with dot product
  - Cross-encoders + reranking:
    - Query and document are jointly embedded with the same model at inference time.
    - We use an efficient dual-encoder to process candidates, then use a cross-encoder to rerank top-k candidates.
  - Dispersos:
    - Modelos como SPLADE proyectan representaciones densas en el mismo espacio del vocabulario obteniendo representaciones dispersas.
    - Las representaciones dispersas se indexan con los mismos índices invertidos de BM25, lo que permite escalabilidad.
  - Interacción tardía:
    - Modelos como COLBERT generan un embedding por cada token (en vez de un embedding por consulta/documento).
    - Calcula similitud a nivel de token entre consultas y documentos.

## TeX

```bash
brew install --cask basictex

sudo tlmgr update --self && sudo tlmgr update --all
tlmgr search --global --file mypackage.sty
sudo tlmgr install <package1> <package2>
tlmgr list --only-installed
```

## Cmds

```bash
ps -o user PID

cd /media/discoexterno/

lscpu
free -h
nvidia-smi
df -h

du -sh .[^.]* * 2>/dev/null | sort -hr

ps aux | grep baselines
kill <PIDs>
kill -9 <PIDs>

tail -f /home/jmendoza/ir-spanish/logs/qwen3-embedding-0.6b

cd /home/jmendoza/ir-spanish && nohup /home/jmendoza/miniconda3/envs/proyecto/bin/python -m baselines.qwen > logs/qwen3-embedding-0.6b 2>&1 &

cd /home/jmendoza/ir-spanish && nohup /home/jmendoza/miniconda3/envs/proyecto/bin/python -m rerankers.fuse > logs/rerankers-fuse-borda.log 2>&1 &
```
