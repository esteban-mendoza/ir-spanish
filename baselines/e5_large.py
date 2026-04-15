#!/usr/bin/env python3
"""
Baseline evaluation of multilingual-e5-large-instruct on MessIRve (Spanish IR).
Metrics: nDCG@10, Recall@100, MRR@10, MAP, P@10, P@50 via ranx.

Features:
- Chunked document encoding with crash-safe resumption.
- float32 `.npy` caching to avoid Parquet bottleneck.
- NUMA-aware multi-processing.
"""

from __future__ import annotations

import os
from pathlib import Path

# utils.__init__ sets NUMA/threading env-vars before torch is imported
from utils.model import BaseEmbeddingModel
from utils.workflow import BaseWorkflow

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
MODEL_MAX_LENGTH = 512
QUERY_BATCH_SIZE = 512
DOC_BATCH_SIZE = 128

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
        super().__init__(model_name, devices, DOC_BATCH_SIZE, QUERY_BATCH_SIZE, MODEL_MAX_LENGTH, MODEL_MAX_LENGTH)

    def setup_prompts(self):
        self.task_description = "Given a web search query, retrieve relevant passages that answer the query"
        super().setup_prompts()


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------
class Workflow(BaseWorkflow):
    def __init__(self):
        super().__init__(MODEL_NAME, MODEL_MAX_LENGTH, MODEL_MAX_LENGTH, CACHE_DIR, GPU_DEVICES, SEED, NUM_WORKERS, DOC_CHUNK_SIZE)

    def create_model(self) -> EmbeddingModel:
        return EmbeddingModel(self.model_name, self.gpu_devices)


if __name__ == "__main__":
    Workflow().run()
