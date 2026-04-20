# Titulación

## Tareas

- Experimentos
  - Implementar rerankers:
    - Interacción tardía:
      - Jina-ColBERT-v2
    - Cross-encoders:
      - bge-reranker-v2-m3
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
  - BM25 ✅
- Dual-encoders:
  - multilingual-e5-large-instruct ✅
  - BGE-M3 ✅
  - Qwen3-Embedding-8B ✅
  - jina-embeddings-v5-text-small ✅
- Dispersos:
  - SPLADE-v3 ✅
- Interacción tardía:
  - Jina-ColBERT-v2
- Cross-encoders:
  - bge-reranker-v2-m3
  - jina-reranker-v3 ✅

- Rerankers:
  - Rank-Biased Centroids (rbc)
  - RRF
  - CombMNZ
  - Condorcet-fuse
  - Borda-fuse

- `ranx`: biblioteca para rerankings y pruebas de significancia estadística
- `sentence-transformers`: biblioteca de Hugging Face para modelos pre-entrenados
- `pyserini`: biblioteca para bm25

## Docs

Protocolo: <https://www.overleaf.com/project/69af3f611c674ab6c9bfd1f3>

## Notas

- RanX:
  - Qrel: query relevance judgment
    - query_id → document_id → relevance_score
  - Run: output of a retrieval system
    - query_id → document_id → score
  - Both have the following structure:
    ```python
    {
        "q_1": {"doc_a": 2, "doc_b": 1, "doc_c": 0},
        "q_2": {"doc_x": 1, "doc_y": 0},
    }
    ```

- Arquitecturas
  - Dual-encoders:
    - Query and document are separately embedded with the same model, then we compare each other with dot product
  - Dispersos:
    - Modelos como SPLADE proyectan representaciones densas en el mismo espacio del vocabulario obteniendo representaciones dispersas.
    - Las representaciones dispersas se indexan con los mismos índices invertidos de BM25, lo que permite escalabilidad.
  - Cross-encoders + reranking:
    - Query and document are jointly embedded with the same model at inference time.
    - We use an efficient dual-encoder to process candidates, then use a cross-encoder to rerank top-k candidates.
  - Interacción tardía:
    - Modelos como COLBERT generan un embedding por cada token (en vez de un embedding por consulta/documento).
    - Calcula similitud a nivel de token entre consultas y documentos.

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

