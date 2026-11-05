"""Context lengths and length-aware batching for the neural rerankers.

The values below are the native maxima that each model declares, not the
shorter caps used in the first evaluation:

* ``bge-reranker-v2-m3``: 8,192 tokens for the concatenated query--document
  input (``max_position_embeddings = 8194`` in its ``config.json``).
* ``jina-reranker-v3``: 131,072 shared tokens, with a maximum document length
  of 8,192 tokens (paper, Table 5).  The 1,024-token query cap already exceeds
  the longest MessIRve query (22 words), so it does not truncate.
* ``jina-colbert-v2``: 8,192 tokens per sequence.  The 32-token query budget is
  ColBERT's query-expansion length, not a context limit.

A full 8,192-token batch at the old batch sizes would exceed the 24 GiB of each
A5000: attention memory grows with ``batch * length**2``.  ``length_aware_batches``
therefore groups pairs by estimated length and caps the quadratic token budget,
so short documents keep large batches while long ones use small batches.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Sequence

# Native maxima, overridable for pilots or ablations.
BGE_MAX_LENGTH = int(os.environ.get("BGE_MAX_LENGTH", "8192"))
JINA_V3_MAX_QUERY_LENGTH = int(os.environ.get("JINA_V3_MAX_QUERY_LENGTH", "1024"))
JINA_V3_MAX_DOC_LENGTH = int(os.environ.get("JINA_V3_MAX_DOC_LENGTH", "8192"))
COLBERT_MAX_QUERY_LENGTH = int(os.environ.get("COLBERT_MAX_QUERY_LENGTH", "32"))
COLBERT_MAX_DOC_LENGTH = int(os.environ.get("COLBERT_MAX_DOC_LENGTH", "8192"))

# ``batch * length**2`` budget.  2**26 keeps the longest 8192-token batch at
# one sequence (about 2.1 GiB of attention scores for 16 heads in float32).
# We used 2**27 first; with four grouped queries and another user on GPU 1 it
# still caused OOM on an A5000.
QUADRATIC_TOKEN_BUDGET = int(
    os.environ.get("RERANKER_QUADRATIC_TOKEN_BUDGET", str(2**26))
)
MAX_BATCH = int(os.environ.get("RERANKER_MAX_BATCH", "256"))

# Optional tag appended to the cache directory, for smoke tests that must not
# look like a complete production run.
RUN_TAG = os.environ.get("RERANKER_RUN_TAG", "")


def estimate_tokens(text: str) -> int:
    """Conservative token estimate for a Spanish text.

    The fast tokenizers of the three models use roughly 1.2--1.3 tokens per
    word.  We use 1.4 plus a fixed overhead so that the batch planner never
    underestimates the padded length.  The model tokenizer still applies the
    exact truncation; this estimate only decides the batch size.
    """
    return int(len(text.split()) * 1.4) + 16


def length_aware_batches(
    lengths: Sequence[int],
    *,
    quadratic_budget: int = QUADRATIC_TOKEN_BUDGET,
    max_batch: int = MAX_BATCH,
) -> Iterator[list[int]]:
    """Yield batches of indices whose padded quadratic cost fits the budget.

    Indices are sorted by length so that each batch contains documents of
    similar size.  A batch is closed when adding the next document would make
    ``len(batch) * max_length**2`` exceed ``quadratic_budget`` or when it
    reaches ``max_batch``.
    """
    order = sorted(range(len(lengths)), key=lambda index: lengths[index])
    batch: list[int] = []
    batch_max = 0

    for index in order:
        length = max(1, int(lengths[index]))
        candidate_max = max(batch_max, length)
        if batch and (
            len(batch) >= max_batch
            or (len(batch) + 1) * candidate_max * candidate_max > quadratic_budget
        ):
            yield batch
            batch = []
            candidate_max = length
        batch.append(index)
        batch_max = candidate_max

    if batch:
        yield batch


def pair_lengths(
    queries: Iterable[str],
    documents: Iterable[str],
    max_length: int,
) -> list[int]:
    """Estimated combined length of each query--document pair, capped."""
    return [
        min(max_length, estimate_tokens(query) + estimate_tokens(document))
        for query, document in zip(queries, documents)
    ]
