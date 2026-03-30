# Titulación

## Tareas

- Experimentos
  - Baselines
  - Comparar rankings entre sí
    - 
  - Comparar con text-embedding-3-large

- Revisar resultados y métodos:
  - Paper de MessIRve
  - Artículos utilizando MessIRve
  - MTEB y La Leaderboard

- Redacción
  - Hipótesis
  - Preguntas de investigación
  - Contribución
  - Marco teórico

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
  - BayesFuse (bayesfuse)
  - MAPFuse (mapfuse)
  - Condorcet-fuse
  - Borda-fuse

- `ranx`: biblioteca para rerankings y pruebas de significancia estadística
- `sentence-transformers`: biblioteca de Hugging Face para modelos pre-entrenados
- `pyserini`: biblioteca para bm25

## Resultados previos

| Model                         | nDCG@10 | Recall@100 |
|-------------------------------|--------:|-----------:|
| BM25                          | 0.179   | 0.558      |
| MIRACL-mdpr-es                | 0.284   | 0.658      |
| E5-large                      | 0.463   | 0.865      |
| E5-large-ft-messirve          | 0.491   | 0.887      |
| OpenAI-text-embedding-3-large | 0.476   | 0.916      |

## Mis resultados (q512, d512)

| Model                                   | nDCG@10 | Recall@100 |
|-----------------------------------------|--------:|-----------:|
| BM25 (no filters)                       | 0.1848  | 0.5725     |
| splade-v3                               | 0.1956  | 0.5725     |
| qwen3-embedding-0.6b                    | 0.4468  | 0.8422     |
| multilingual-e5-large-instruct          | 0.4675  | 0.8666     |
| bge-m3                                  | 0.4818  | 0.8741     |
| jina-embeddings-v5-text-small-retrieval | 0.5111  | 0.9037     |

## Fusiones (q512, d512)

| Model                            | strategy   | nDCG@10 | Recall@100 |
|----------------------------------|-----------:|--------:|-----------:|
| e5 + bge                         | rrf (k=60) | 0.5128  | 0.8997     |
| e5 + bge + jina                  | rrf (k=60) | 0.5450  | 0.9232     |
| e5 + bge + jina                  | rbc (φ=0.9)| 0.5478  | 0.9194     |
| e5 + bge + jina                  | rbc (φ=0.8)| 0.5434  | 0.9191     |

## Mis resultados (q64, d256)

| Model                            | nDCG@10 | Recall@100 |
|----------------------------------|--------:|-----------:|
| BM25 (no filters)                | 0.1848  | 0.5725     |
| splade-v3                        | 0.1964  | 0.5759     |
| qwen3-embedding-0.6b             | 0.4463  | 0.8408     |
| multilingual-e5-large-instruct   | 0.4625  | 0.8591     |
| BGE-M3                           | 0.4816  | 0.8735     |

## Mis resultados (400 words)

| Model                            | nDCG@10 | Recall@100 |
|----------------------------------|--------:|-----------:|
| BM25                             | 0.1848  | 0.5726     |
| multilingual-e5-large-instruct   | 0.4679  | 0.8669     |
| BGE-M3                           | 0.4821  | 0.8743     |

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

## Cmds

```bash
sudo tlmgr update --self && sudo tlmgr update --all
tlmgr search --global --file mypackage.sty
sudo tlmgr install <package1> <package2>
tlmgr list --only-installed

ps -o user PID

cd /media/discoexterno/

lscpu
free -h
nvidia-smi
df -h

du -sh .[^.]* * 2>/dev/null | sort -hr

ps aux | grep proyecto
kill <PIDs>
kill -9 <PIDs>

tail -f /home/jmendoza/proyecto/logs/rerankers-fuse.log

cd /home/jmendoza/proyecto && nohup /home/jmendoza/miniconda3/envs/proyecto/bin/python baselines/jina_v5_small.py >logs/jina-embeddings-v5-text-small-retrieval.log 2>&1 &

cd /home/jmendoza/proyecto && conda activate proyecto && nohup /home/jmendoza/miniconda3/envs/proyecto/bin/python -m rerankers.fuse > logs/rerankers-fuse-rrf60.log 2>&1 &

cd /home/jmendoza/proyecto && conda activate proyecto && nohup /home/jmendoza/miniconda3/envs/proyecto/bin/python -m rerankers.bge_reranker_v2_m3 > logs/bge-reranker-v2-m3.log 2>&1 &
```
