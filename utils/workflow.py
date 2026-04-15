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
        max_query_length: int,
        max_doc_length: int,
        cache_dir: Path,
        gpu_devices: list[str],
        seed: int,
        num_workers: int,
        doc_chunk_size: int,
    ):
        self.model_name = model_name
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.cache_dir = cache_dir
        self.gpu_devices = gpu_devices
        self.seed = seed
        self.num_workers = num_workers
        self.doc_chunk_size = doc_chunk_size

        # Pre-compute all cache directory paths so each step can check / write them
        self.model_cache_base = cache.cache_base(
            cache_dir, model_name, data.COUNTRY, data.DATASET_VERSION,
            max_query_length, max_doc_length, data.MAX_WORD_COUNT
        )
        self.doc_embedding_dir = cache.emb_dir(self.model_cache_base, "doc")
        self.query_embedding_dir = cache.emb_dir(self.model_cache_base, "query")
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
        cache.log_cache_status(self.model_cache_base, self.dataset_cache_dir)

    def _check_cache(self) -> tuple[bool, bool, bool, bool]:
        """Check which pipeline outputs still need to be computed.

        Returns:
            need_doc_embeddings:   True if document embeddings are not yet cached.
            need_query_embeddings: True if query embeddings are not yet cached.
            need_pruned_qrels:     True if the pruned qrels JSON files are not yet cached.
            need_retrieval_run:    True if the retrieval run file is not yet cached.
        """
        need_doc_embeddings = not cache.is_complete(self.doc_embedding_dir)
        need_query_embeddings = not cache.is_complete(self.query_embedding_dir)
        need_pruned_qrels = not cache.are_qrels_cached(self.dataset_cache_dir)
        need_retrieval_run = not cache.run_cache_path(self.model_cache_base).exists()
        return need_doc_embeddings, need_query_embeddings, need_pruned_qrels, need_retrieval_run

    def load_corpus(self, need_doc_embeddings: bool, need_pruned_qrels: bool):
        """Load the filtered corpus texts, or just the cached doc IDs if embeddings already exist.

        When document embeddings are already cached, only the list of IDs is needed (to pass
        to qrels pruning), so we skip the expensive full corpus load.

        Returns:
            doc_ids:     List of document ID strings.
            doc_texts:   List of formatted document strings, or None if loaded from cache.
            word_counts: List of word counts per document, or None if loaded from cache.
        """
        if need_doc_embeddings:
            return data.load_corpus(self.num_workers)
        # Embeddings exist — we only need the IDs for potential qrels pruning
        return cache.load_ids(self.doc_embedding_dir / "ids.json"), None, None

    def load_qrels(self, doc_ids: list[str] | None, need_pruned_qrels: bool):
        """Load pruned qrels and the query-id-to-text map, building the cache if required.

        Returns:
            qrels:         A ranx.Qrels object with relevance judgments.
            query_id_to_text: A dict mapping query IDs to their query text strings.
        """
        return data.get_pruned_qrels_and_queries(
            data.COUNTRY,
            data.DATASET_VERSION,
            # Only pass doc_ids when the pruned qrels need to be (re)built from scratch
            doc_ids if need_pruned_qrels else None,
            self.dataset_cache_dir,
            self.num_workers,
        )

    def encode(
        self,
        need_doc_embeddings: bool,
        need_query_embeddings: bool,
        doc_ids,
        doc_texts,
        word_counts,
        query_id_to_text,
    ):
        """Encode documents and queries, loading each from cache when already available.

        The model is only loaded onto GPU when at least one encoding step is needed.

        Returns:
            doc_ids:           List of document ID strings.
            doc_embeddings:    Float32 array of shape (num_docs, embedding_dim).
            query_ids:         List of query ID strings.
            query_embeddings:  Float32 array of shape (num_queries, embedding_dim).
        """
        doc_embeddings = query_embeddings = query_ids = None

        if need_doc_embeddings or need_query_embeddings:
            with self.create_model() as model:
                if need_doc_embeddings:
                    doc_ids, doc_embeddings = encoding.encode_documents_chunked(
                        doc_ids, doc_texts, word_counts,
                        self.doc_embedding_dir, model, self.doc_chunk_size,
                        self.max_doc_length,
                    )
                    del doc_texts, word_counts
                    gc.collect()

                if need_query_embeddings:
                    query_ids, query_embeddings = encoding.encode_queries(
                        query_id_to_text, self.query_embedding_dir, model
                    )

        if doc_embeddings is None:
            doc_ids, doc_embeddings = cache.load_cached(self.doc_embedding_dir)
        if query_embeddings is None:
            query_ids, query_embeddings = cache.load_cached(self.query_embedding_dir)

        return doc_ids, doc_embeddings, query_ids, query_embeddings

    def retrieve(self, query_ids, query_embeddings, doc_ids, doc_embeddings):
        """Run FAISS nearest-neighbour search and return the retrieval run dict."""
        return retrieval.retrieve(
            query_ids, query_embeddings, doc_ids, doc_embeddings, retrieval.TOP_K, self.num_workers
        )

    def evaluate(self, qrels, retrieval_run):
        """Compute and log nDCG@10 and Recall@100."""
        return retrieval.run_evaluation(qrels, retrieval_run, self.model_name)

    # ------------------------------------------------------------------
    # Top-level orchestration
    # ------------------------------------------------------------------

    def run(self):
        """Execute the full evaluation pipeline end-to-end."""
        self.setup()

        need_doc_embeddings, need_query_embeddings, need_pruned_qrels, need_retrieval_run = (
            self._check_cache()
        )

        run_path = cache.run_cache_path(self.model_cache_base)

        if need_retrieval_run:
            doc_ids, doc_texts, word_counts = self.load_corpus(need_doc_embeddings, need_pruned_qrels)
            qrels, query_id_to_text = self.load_qrels(doc_ids, need_pruned_qrels)
            doc_ids, doc_embeddings, query_ids, query_embeddings = self.encode(
                need_doc_embeddings, need_query_embeddings,
                doc_ids, doc_texts, word_counts, query_id_to_text
            )
            retrieval_run = self.retrieve(query_ids, query_embeddings, doc_ids, doc_embeddings)
            run_path.parent.mkdir(parents=True, exist_ok=True)
            ranx_run = retrieval.save_run(retrieval_run, self.model_name, run_path)
            log.info("Saved retrieval run to %s", run_path)
        else:
            log.info("Loading cached retrieval run from %s", run_path)
            ranx_run = retrieval.load_run(run_path)
            qrels, _ = self.load_qrels(None, need_pruned_qrels)

        self.evaluate(qrels, ranx_run)
