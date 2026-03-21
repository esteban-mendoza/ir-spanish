"""
BaseWorkflow: encapsulates the end-to-end evaluation pipeline.

Subclass BaseWorkflow and implement create_model() to return an instance of
the model-specific EmbeddingModel. Then call run() to execute the pipeline.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import datasets
import torch

from . import cache, data, encoding, retrieval

log = logging.getLogger(__name__)


class BaseWorkflow:
    def __init__(
        self,
        model_name: str,
        max_seq_length: int,
        cache_dir: Path,
        gpu_devices: list[str],
        seed: int,
        num_workers: int,
        doc_chunk_size: int,
    ):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.cache_dir = cache_dir
        self.gpu_devices = gpu_devices
        self.seed = seed
        self.num_workers = num_workers
        self.doc_chunk_size = doc_chunk_size

        self.base = cache.cache_base(
            cache_dir, model_name, data.COUNTRY, data.DATASET_VERSION, max_seq_length, data.MAX_WORD_COUNT
        )
        self.doc_dir = cache.emb_dir(self.base, "doc")
        self.query_dir = cache.emb_dir(self.base, "query")
        self.dataset_cache_dir = cache.dataset_cache_base(
            cache_dir, data.COUNTRY, data.DATASET_VERSION, data.MAX_WORD_COUNT
        )

    # ------------------------------------------------------------------
    # Hook — implement in subclasses
    # ------------------------------------------------------------------

    def create_model(self):
        """Return an initialised (but not yet started) EmbeddingModel."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Workflow steps
    # ------------------------------------------------------------------

    def setup(self):
        """Configure global settings and log current cache state."""
        datasets.config.NUM_PROC = self.num_workers
        torch.manual_seed(self.seed)
        cache.log_cache_status(self.base, self.dataset_cache_dir)

    def _check_cache(self) -> tuple[bool, bool, bool]:
        """Return (need_docs, need_queries, need_pruned_qrels) flags."""
        need_docs = not cache.is_complete(self.doc_dir)
        need_queries = not cache.is_complete(self.query_dir)
        need_pruned_qrels = not (
            (self.dataset_cache_dir / "pruned_qrels.json").exists()
            and (self.dataset_cache_dir / "pruned_q_map.json").exists()
        )
        return need_docs, need_queries, need_pruned_qrels

    def load_corpus(self, need_docs: bool, need_pruned_qrels: bool):
        """Load (or read cached IDs of) the filtered corpus."""
        if need_docs:
            return data.load_corpus(self.num_workers)
        return cache.load_ids(self.doc_dir / "ids.json"), None

    def load_qrels(self, doc_ids: list[str], need_pruned_qrels: bool):
        """Load pruned qrels and query map, building the cache if required."""
        return data.get_pruned_qrels_and_queries(
            data.COUNTRY,
            data.DATASET_VERSION,
            doc_ids if need_pruned_qrels else None,
            self.dataset_cache_dir,
            self.num_workers,
        )

    def encode(self, need_docs: bool, need_queries: bool, doc_ids, doc_texts, q_map):
        """Encode documents and queries, loading from cache when available."""
        doc_emb = query_emb = query_ids = None

        if need_docs or need_queries:
            with self.create_model() as model:
                if need_docs:
                    doc_ids, doc_emb = encoding.encode_documents_chunked(
                        doc_ids, doc_texts, self.doc_dir, model, self.doc_chunk_size
                    )
                    del doc_texts
                    gc.collect()

                if need_queries:
                    query_ids, query_emb = encoding.encode_queries(q_map, self.query_dir, model)

        if doc_emb is None:
            doc_ids, doc_emb = cache.load_cached(self.doc_dir)
        if query_emb is None:
            query_ids, query_emb = cache.load_cached(self.query_dir)

        return doc_ids, doc_emb, query_ids, query_emb

    def retrieve(self, query_ids, query_emb, doc_ids, doc_emb):
        """Run FAISS nearest-neighbour search and return the run dict."""
        return retrieval.retrieve(query_ids, query_emb, doc_ids, doc_emb, retrieval.TOP_K, self.num_workers)

    def evaluate(self, qrels, run_dict):
        """Compute and log nDCG@10 and Recall@100."""
        return retrieval.run_evaluation(qrels, run_dict, self.model_name)

    # ------------------------------------------------------------------
    # Top-level orchestration
    # ------------------------------------------------------------------

    def run(self):
        self.setup()

        need_docs, need_queries, need_pruned_qrels = self._check_cache()
        doc_ids, doc_texts = self.load_corpus(need_docs, need_pruned_qrels)
        qrels, q_map = self.load_qrels(doc_ids, need_pruned_qrels)
        doc_ids, doc_emb, query_ids, query_emb = self.encode(
            need_docs, need_queries, doc_ids, doc_texts, q_map
        )
        run_dict = self.retrieve(query_ids, query_emb, doc_ids, doc_emb)
        self.evaluate(qrels, run_dict)
