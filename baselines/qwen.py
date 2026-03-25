#!/usr/bin/env python3
"""
Baseline evaluation of Qwen3-Embedding-8B on MessIRve (Spanish IR).
Metrics: nDCG@10, Recall@100 via ranx.

Features:
- Filters out documents > 400 words to prevent truncation & OOM.
- Prunes qrels/queries missing from the filtered corpus to maintain metric validity (Cached).
- Chunked document encoding with crash-safe resumption.
- float32 `.npy` caching to avoid Parquet bottleneck (Model & Sequence length aware).
- SDPA attention & NUMA-aware multi-processing.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

# utils.__init__ sets NUMA/threading env-vars before torch is imported
from utils.model import BaseEmbeddingModel
from utils.workflow import BaseWorkflow

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
QUERY_BATCH_SIZE = 32
DOC_BATCH_SIZE = 8
# 64 effective tokens + ~20 tokens consumed by the instruct prefix
MAX_QUERY_LENGTH = 84
MAX_DOC_LENGTH = 256

GPU_DEVICES = ["cuda:0", "cuda:1"]
SEED = 42
NUM_WORKERS = os.cpu_count() or 32
CACHE_DIR = Path.home() / ".cache" / "messirve_embeddings"
DOC_CHUNK_SIZE = 50_000


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class EmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str, devices: list[str]):
        super().__init__(model_name, devices, DOC_BATCH_SIZE, QUERY_BATCH_SIZE, MAX_DOC_LENGTH, MAX_QUERY_LENGTH)

    def get_model_kwargs(self) -> dict:
        return {"torch_dtype": torch.float16, "attn_implementation": "sdpa"}

    def get_tokenizer_kwargs(self) -> dict:
        return {"padding_side": "left"}

    def setup_prompts(self):
        prompt = self.model.prompts.get("query", "") if hasattr(self.model, "prompts") else ""
        if prompt.startswith("Instruct: ") and "\nQuery: " in prompt:
            self.task_description = prompt.removeprefix("Instruct: ").split("\nQuery: ")[0]
        else:
            self.task_description = "Given a web search query, retrieve relevant passages that answer the query"
        super().setup_prompts()


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------
class Workflow(BaseWorkflow):
    def __init__(self):
        super().__init__(MODEL_NAME, MAX_QUERY_LENGTH, MAX_DOC_LENGTH, CACHE_DIR, GPU_DEVICES, SEED, NUM_WORKERS, DOC_CHUNK_SIZE)

    def create_model(self) -> EmbeddingModel:
        return EmbeddingModel(self.model_name, self.gpu_devices)


if __name__ == "__main__":
    Workflow().run()
