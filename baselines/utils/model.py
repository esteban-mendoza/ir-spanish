"""
Base class for SentenceTransformer-based embedding models.

Subclass BaseEmbeddingModel and override the hook methods to configure
model-specific behaviour:

  get_model_kwargs()     — extra kwargs for SentenceTransformer(model_kwargs=...)
  get_tokenizer_kwargs() — kwargs for SentenceTransformer(tokenizer_kwargs=...)
  setup_prompts()        — called after the model loads; set self.query_prompt here
"""

from __future__ import annotations

import gc
import logging

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .cache import Timer

log = logging.getLogger(__name__)


class BaseEmbeddingModel:
    def __init__(
        self,
        model_name: str,
        devices: list[str],
        doc_batch_size: int,
        query_batch_size: int,
        max_seq_length: int = 512,
    ):
        self.model_name = model_name
        self.devices = devices
        self.doc_batch_size = doc_batch_size
        self.query_batch_size = query_batch_size
        self.max_seq_length = max_seq_length
        self.model = None
        self.pool = None
        self.query_prompt = ""

    # ------------------------------------------------------------------
    # Hooks — override in subclasses
    # ------------------------------------------------------------------

    def get_model_kwargs(self) -> dict:
        return {"torch_dtype": torch.float16}

    def get_tokenizer_kwargs(self) -> dict:
        return {}

    def setup_prompts(self):
        """Set self.query_prompt after the model has been loaded."""
        pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        tokenizer_kwargs = self.get_tokenizer_kwargs()
        extra = {"tokenizer_kwargs": tokenizer_kwargs} if tokenizer_kwargs else {}

        with Timer("Loading SentenceTransformer"):
            self.model = SentenceTransformer(
                self.model_name,
                device="cpu",
                model_kwargs=self.get_model_kwargs(),
                **extra,
            )
            self.model.max_seq_length = self.max_seq_length

        self.setup_prompts()

        with Timer(f"Starting pool on {self.devices}"):
            self.pool = self.model.start_multi_process_pool(target_devices=self.devices)

    def stop(self):
        if self.pool is not None:
            self.model.stop_multi_process_pool(self.pool)
            self.pool = None
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, texts: list[str], batch_size: int, is_query: bool = False) -> np.ndarray:
        prefix = self.query_prompt if is_query else ""
        if prefix:
            texts = [prefix + t for t in texts]

        emb = self.model.encode(
            sentences=texts,
            pool=self.pool,
            batch_size=batch_size,
            normalize_embeddings=True,
            prompt_name=None,
        )
        return emb.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
