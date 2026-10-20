"""Statistical comparison utilities for retrieval runs.

Pairwise tests and multiple-comparison corrections used in the thesis:

- paired two-sided Student t-test for mean effectiveness;
- sign-flip permutation test as a robust confirmation;
- Holm-Bonferroni (FWER) and Benjamini-Hochberg (FDR) corrections.

The module is intentionally independent of ``ranx.compare`` so that the
corrections are explicit and reproducible.  See
``tesis/knowledge/urbano-et-al-2019-pruebas-estadisticas.md`` for the rationale.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def paired_t_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sided paired Student t-test on per-query scores.

    Returns the t statistic and the p-value.
    """
    result = stats.ttest_rel(a, b, nan_policy="raise")
    return float(result.statistic), float(result.pvalue)


def permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    n_permutations: int = 10_000,
    seed: int = 42,
    chunk_size: int = 1_000,
) -> float:
    """Two-sided sign-flip permutation test for the difference of means.

    Under the null hypothesis the two systems are exchangeable, so the sign of
    each per-query difference is equally likely.  The p-value is the fraction
    of sign-flipped replicates whose absolute mean difference is at least the
    observed one.  Replicates are generated in chunks to bound memory use.
    """
    differences = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    observed = float(np.mean(differences, dtype=np.float64))
    rng = np.random.default_rng(seed)
    extreme = 0
    remaining = n_permutations
    while remaining > 0:
        size = min(chunk_size, remaining)
        signs = rng.integers(0, 2, size=(size, differences.size), dtype=np.int8)
        signs = signs.astype(np.float32) * 2.0 - 1.0
        replicated = signs @ differences / differences.size
        extreme += int(np.sum(np.abs(replicated) >= abs(observed)))
        remaining -= size
    return (extreme + 1) / (n_permutations + 1)


def holm_bonferroni(pvalues: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (FWER control)."""
    p = np.asarray(pvalues, dtype=float)
    if p.size == 0:
        return p
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (p.size - rank) * p[index]
        running = max(running, candidate)
        adjusted[index] = running
    return np.minimum(adjusted, 1.0)


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up adjusted p-values (FDR control)."""
    p = np.asarray(pvalues, dtype=float)
    if p.size == 0:
        return p
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 1.0
    for rank in range(p.size - 1, -1, -1):
        index = order[rank]
        candidate = p.size * p[index] / (rank + 1)
        running = min(running, candidate)
        adjusted[index] = running
    return np.minimum(adjusted, 1.0)


CORRECTIONS = {
    "none": lambda p: np.asarray(p, dtype=float),
    "holm": holm_bonferroni,
    "bh": benjamini_hochberg,
}
