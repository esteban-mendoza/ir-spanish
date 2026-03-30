"""
Base class for SentenceTransformer-based embedding models.

Subclass BaseEmbeddingModel and override the hook methods to configure
model-specific behaviour:

  get_model_kwargs()     — extra kwargs for SentenceTransformer(model_kwargs=...)
  get_tokenizer_kwargs() — kwargs for SentenceTransformer(tokenizer_kwargs=...)
  setup_prompts()        — called after the model loads; set self.task_description
                           and it will be registered into model.prompts["query"]
                           automatically, or set self.query_prompt_name directly
                           for models with built-in prompts
"""

from __future__ import annotations

import gc
import logging

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .observability import Timer

log = logging.getLogger(__name__)


def format_query_prompt(task_description: str) -> str:
    """Format a task description into the e5-instruct query prompt template."""
    return f"Instruct: {task_description}\nQuery: "


class BaseEmbeddingModel:
    def __init__(
        self,
        model_name: str,
        devices: list[str],
        doc_batch_size: int,
        query_batch_size: int,
        max_doc_length: int = 256,
        max_query_length: int = 64,
    ):
        self.model_name = model_name
        self.devices = devices
        self.doc_batch_size = doc_batch_size
        self.query_batch_size = query_batch_size
        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length
        self.model = None
        self.pool = None
        self.task_description = ""
        self.query_prompt_name: str | None = None
        self.doc_prompt_name: str | None = None

    # ------------------------------------------------------------------
    # Hooks — override in subclasses
    # ------------------------------------------------------------------

    def get_model_kwargs(self) -> dict:
        """Return extra kwargs to pass as model_kwargs= when loading SentenceTransformer."""
        return {"torch_dtype": torch.float16}

    def get_tokenizer_kwargs(self) -> dict:
        """Return extra kwargs to pass as tokenizer_kwargs= when loading SentenceTransformer."""
        return {}

    def setup_prompts(self):
        """Register the query prompt after the model has been loaded.

        Called automatically by start(). Override this in subclasses to set
        self.task_description, then call super().setup_prompts() to have the
        prompt registered into model.prompts["query"] automatically.
        """
        if self.task_description:
            self.model.prompts["query"] = format_query_prompt(self.task_description)
            self.query_prompt_name = "query"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Load the model onto CPU and start the multi-GPU process pool."""
        tokenizer_kwargs = self.get_tokenizer_kwargs()
        optional_tokenizer_kwargs = {"tokenizer_kwargs": tokenizer_kwargs} if tokenizer_kwargs else {}

        with Timer("Loading SentenceTransformer"):
            self.model = SentenceTransformer(
                self.model_name,
                device="cpu",
                model_kwargs=self.get_model_kwargs(),
                **optional_tokenizer_kwargs,
            )
            self.model.max_seq_length = self.max_doc_length

        self.setup_prompts()

        with Timer(f"Starting pool on {self.devices}"):
            self.pool = self.model.start_multi_process_pool(target_devices=self.devices)

    def _ensure_seq_length(self, length: int):
        """Restart the multi-GPU pool if the current max_seq_length differs from *length*.

        Pool workers inherit max_seq_length at fork time, so the pool must be
        recycled whenever the truncation limit changes (e.g. switching from
        document encoding to query encoding).
        """
        if self.model.max_seq_length == length:
            return
        log.info("Switching max_seq_length %d → %d — restarting pool", self.model.max_seq_length, length)
        self.model.stop_multi_process_pool(self.pool)
        self.model.max_seq_length = length
        self.pool = self.model.start_multi_process_pool(target_devices=self.devices)

    def stop(self):
        """Shut down the GPU pool and free all model memory."""
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
        """Encode a list of texts into normalized float32 embeddings.

        Args:
            texts: Raw strings to encode (documents or queries).
            batch_size: Number of texts per encoding batch.
            is_query: If True, use self.query_prompt_name so the model prepends
                      the registered query prompt automatically.
        """
        target_length = self.max_query_length if is_query else self.max_doc_length
        self._ensure_seq_length(target_length)

        embeddings = self.model.encode(
            sentences=texts,
            pool=self.pool,
            batch_size=batch_size,
            normalize_embeddings=True,
            prompt_name=self.query_prompt_name if is_query else self.doc_prompt_name,
        )
        return embeddings.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
