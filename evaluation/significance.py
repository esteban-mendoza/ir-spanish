"""Statistical comparison utilities for retrieval runs.

Thin wrappers around maintained implementations:

- paired two-sided Student t-test: ``scipy.stats.ttest_rel``;
- paired mean difference and its confidence interval:
  ``statsmodels.stats.weightstats.DescrStatsW``;
- paired permutation test:
  ``scipy.stats.permutation_test`` (sign-flip, ``permutation_type="samples"``);
- Holm-Bonferroni and Benjamini-Hochberg adjusted p-values:
  ``statsmodels.stats.multitest.multipletests``.

The public API is kept stable for the experiment scripts.  See
``tesis/knowledge/metricas-y-correcciones-multiples.md`` for the rationale.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.weightstats import DescrStatsW


def paired_t_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sided paired Student t-test on per-query scores.

    Returns the t statistic and the p-value.
    """
    result = stats.ttest_rel(a, b, nan_policy="raise")
    return float(result.statistic), float(result.pvalue)


def paired_mean_ci(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Mean paired difference and its two-sided Student t confidence interval.

    ``a`` and ``b`` are aligned per-query scores.  The interval is the
    ``tconfint_mean`` interval of ``statsmodels`` for the paired differences.
    """
    differences = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if differences.size < 2:
        raise ValueError("At least two paired observations are required")
    if not np.all(np.isfinite(differences)):
        raise ValueError("Paired differences contain non-finite values")

    descr = DescrStatsW(differences)
    low, high = descr.tconfint_mean(alpha=alpha)
    return {
        "mean": float(descr.mean),
        "se": float(descr.std_mean),
        "low": float(low),
        "high": float(high),
        "n": int(differences.size),
    }


def permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    n_permutations: int = 10_000,
    seed: int = 42,
    chunk_size: int = 1_000,
) -> float:
    """Two-sided paired permutation test for the difference of means.

    The paired differences are sign-flipped at random
    (``permutation_type="samples"``) and the p-value is the fraction of
    replicates whose absolute mean is at least the observed one.  The
    ``chunk_size`` argument is passed to SciPy as ``batch`` and bounds the
    memory used per batch.
    """
    differences = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if differences.size < 2:
        raise ValueError("At least two paired observations are required")
    if not np.all(np.isfinite(differences)):
        raise ValueError("Paired differences contain non-finite values")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    result = stats.permutation_test(
        (differences,),
        np.mean,
        permutation_type="samples",
        n_resamples=n_permutations,
        alternative="two-sided",
        random_state=seed,
        batch=chunk_size,
    )
    return float(result.pvalue)


def _adjusted(pvalues: np.ndarray, method: str) -> np.ndarray:
    """Adjusted p-values for a multiple-testing method, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    if p.size == 0:
        return p
    if not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p-values must be finite and in [0, 1]")
    _, corrected, _, _ = multipletests(
        p,
        alpha=0.05,
        method=method,
        returnsorted=False,
    )
    return np.asarray(corrected, dtype=float)


def holm_bonferroni(pvalues: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (FWER control)."""
    return _adjusted(pvalues, "holm")


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up adjusted p-values (FDR control)."""
    return _adjusted(pvalues, "fdr_bh")


CORRECTIONS = {
    "none": lambda p: np.asarray(p, dtype=float),
    "holm": holm_bonferroni,
    "bh": benjamini_hochberg,
}
