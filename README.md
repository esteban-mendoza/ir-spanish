# Rank Fusion for Spanish Information Retrieval

**Evaluación de estrategias de fusión de rangos (*rank fusion*) en la recuperación de información en español: un análisis sobre el conjunto de datos MessIRve**

Thesis project by **Jorge Esteban Mendoza Ortiz**, advised by **Dra. Helena Gómez Adorno** (IIMAS-UNAM).

---

## Overview

This project investigates whether combining the ranked results of multiple open-source retrieval models through rank fusion algorithms can improve information retrieval effectiveness in Spanish, and whether such hybrid systems can match or surpass proprietary baselines reported in the literature.

We evaluate on [**MessIRve**](https://huggingface.co/datasets/spanish-ir/messirve) — a large-scale Spanish IR dataset with ~700,000 queries derived from real search intents via Google's autocomplete API, paired with relevant passages from Spanish Wikipedia.

**Metrics:** nDCG@10, Recall@100, MRR@10, MAP, Precision@10, and Precision@50.

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
| [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 600M | Qwen, 2025 |
| [BGE-M3](https://huggingface.co/BAAI/bge-m3) | 568M | BAAI, 2024 |
| [multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct) | 560M | Microsoft, 2024 |
| [jina-embeddings-v5-text-small](https://huggingface.co/jinaai/jina-embeddings-v5-text-small-retrieval) | 677M | Jina AI, 2025 |
| [Harrier-oss-v1-0.6B](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) | 600M | Microsoft, 2025 |

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
| **Experiment 3** | Fusion vs. late interaction and cross-encoders — how do rank fusion methods compare against Jina-ColBERT-v2 (late interaction), bge-reranker-v2-m3, and jina-reranker-v3 (cross-encoders)? |
| **Experiment 4** | Open-source hybrid systems vs. proprietary baselines reported on MessIRve |

## Metrics

| Metric | Description |
|--------|-------------|
| **nDCG@10** | Normalized Discounted Cumulative Gain at rank 10 — measures ranking quality with graded relevance, giving higher weight to top positions |
| **Recall@100** | Fraction of relevant documents retrieved in the top 100 results |
| **MRR@10** | Mean Reciprocal Rank at 10 — average of the reciprocal rank of the first relevant result, considering only the top 10 |
| **MAP** | Mean Average Precision — mean of average precision scores across all queries |
| **Precision@10** | Fraction of relevant documents in the top 10 results |
| **Precision@50** | Fraction of relevant documents in the top 50 results |

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
| [`faiss-gpu`](https://github.com/facebookresearch/faiss) | GPU-accelerated exact nearest-neighbour search |

## Hardware

| Component | Specification |
|-----------|--------------|
| GPU 0 | NVIDIA RTX A5000 (24 GB VRAM) |
| GPU 1 | NVIDIA RTX A5000 (24 GB VRAM) |
| CUDA | 12.2 |
| Driver | 535.154.05 |

## Setup

Requires [Miniconda](https://docs.anaconda.com/miniconda/) (or any conda distribution).

**Create and populate the environment:**

```bash
git clone <repo-url>
cd proyecto
bash setup_env.sh
```

This creates a conda environment named `ir-messirve`, installs `faiss-gpu` via conda,
and installs all remaining dependencies via pip from `requirements.txt`.

**Activate the environment:**

```bash
conda activate ir-messirve
```

**Run a baseline:**

```bash
python baselines/e5_large.py
```

**Add a new dependency:**

```bash
# If available on pip:
pip install <package>
# Then pin it:
echo "<package>>=<version>" >> requirements.txt

# If it requires conda (e.g. another GPU library):
conda install -c <channel> <package>
```

**Remove the environment entirely:**

```bash
conda remove -n ir-messirve --all -y
```