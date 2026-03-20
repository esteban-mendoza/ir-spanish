# Rank Fusion for Spanish Information Retrieval

**Evaluación de estrategias de fusión de rangos (*rank fusion*) en la recuperación de información en español: un análisis sobre el conjunto de datos MessIRve**

Thesis project by **Jorge Esteban Mendoza Ortiz**, advised by **Dra. Helena Gómez Adorno** (IIMAS-UNAM).

---

## Overview

This project investigates whether combining the ranked results of multiple open-source retrieval models through rank fusion algorithms can improve information retrieval effectiveness in Spanish, and whether such hybrid systems can match or surpass proprietary baselines reported in the literature.

We evaluate on [**MessIRve**](https://huggingface.co/datasets/spanish-ir/messirve) — a large-scale Spanish IR dataset with ~700,000 queries derived from real search intents via Google's autocomplete API, paired with relevant passages from Spanish Wikipedia.

**Metrics:** nDCG@10 and Recall@100.

## Research Questions

1. Does rank fusion of lexical and semantic retrieval models improve relevance over individual models for Spanish IR?
2. How do the Borda and Condorcet fusion methods compare against the standard Reciprocal Rank Fusion (RRF)?
3. Can open-source hybrid systems match or surpass proprietary baselines (e.g., OpenAI's `text-embedding-3-large`) on MessIRve?

## Retrieval Models

### Lexical

| Model | Type | Reference |
|-------|------|-----------|
| [BM25](https://github.com/castorini/pyserini) | Sparse lexical | Robertson & Zaragoza, 2009 |

### Dense (Dual-Encoders)

| Model | Parameters | Reference |
|-------|-----------|-----------|
| [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) | 8B | Qwen, 2025 |
| [BGE-M3](https://huggingface.co/BAAI/bge-m3) | 568M | BAAI, 2024 |
| [multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct) | 560M | Microsoft, 2024 |

### Sparse (Learned Sparse)

| Model | Type | Reference |
|-------|------|-----------|
| [SPLADE-v3](https://huggingface.co/naver/splade-v3) | Learned sparse representations | Naver Labs, 2024 |

### Late Interaction

| Model | Type | Reference |
|-------|------|-----------|
| [Jina-ColBERT-v2](https://huggingface.co/jinaai/jina-colbert-v2) | ColBERT-style late interaction | Jina AI, 2024 |

### Cross-Encoders (Rerankers)

| Model | Reference |
|-------|-----------|
| [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | BAAI, 2024 |
| [jina-reranker-v3](https://huggingface.co/jinaai/jina-reranker-v3) | Jina AI, 2024 |

## Rank Fusion Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| **Reciprocal Rank Fusion (RRF)** | Score-based | Cormack et al., 2009 |
| **Borda-fuse** | Score-based (positional) | Aslam & Montague, 2001 |
| **Condorcet-fuse** | Graph-based (pairwise majority) | Montague & Aslam, 2002 |

## Experimental Design

| Experiment | Description |
|------------|-------------|
| **Experiment 1** | Fusion vs. individual models — does combining ranked lists improve over the best single model? |
| **Experiment 2** | Comparison of fusion algorithms — RRF vs. Borda vs. Condorcet |
| **Experiment 3** | Open-source hybrid systems vs. proprietary baselines reported on MessIRve |

## Dataset

- **MessIRve v1.2** ([HuggingFace](https://huggingface.co/datasets/spanish-ir/messirve))
- **Corpus:** [eswiki_20240401_corpus](https://huggingface.co/datasets/spanish-ir/eswiki_20240401_corpus) — Spanish Wikipedia paragraphs
- **Split:** ~80% train / ~20% test (partitioned by Wikipedia article)
- **Test set constraints:** `match_score = 1`, `expanded_search = False`

## Tech Stack

| Tool | Purpose |
|------|---------|
| [`ranx`](https://github.com/AmenRa/ranx) | Rank fusion, evaluation metrics, and statistical significance tests |
| [`sentence-transformers`](https://www.sbert.net/) | Dense and sparse embedding models |
| [`pyserini`](https://github.com/castorini/pyserini) | BM25 indexing and retrieval |
| [`datasets`](https://huggingface.co/docs/datasets) | Loading MessIRve and the Wikipedia corpus from HuggingFace |

## Hardware

| Component | Specification |
|-----------|--------------|
| GPU 0 | NVIDIA RTX A5000 (24 GB VRAM) |
| GPU 1 | NVIDIA RTX A5000 (24 GB VRAM) |
| CUDA | 12.2 |
| Driver | 535.154.05 |

## Setup

Requires Python ≥ 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd proyecto
uv sync