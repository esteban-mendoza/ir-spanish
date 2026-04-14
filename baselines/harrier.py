#!/usr/bin/env python3
"""
Baseline evaluation of harrier-oss-v1-0.6b on MessIRve (Spanish IR).
Metrics: nDCG@10, Recall@100 via ranx.

Features:
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
MODEL_NAME = "microsoft/harrier-oss-v1-0.6b"
MODEL_MAX_LENGTH = 32768
QUERY_BATCH_SIZE = 512
DOC_BATCH_SIZE = 16

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

    def get_model_kwargs(self) -> dict:
        return {"torch_dtype": torch.float16, "attn_implementation": "sdpa"}

    def get_tokenizer_kwargs(self) -> dict:
        return {"padding_side": "left"}

    def setup_prompts(self):
        self.query_prompt_name = "web_search_query"


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
