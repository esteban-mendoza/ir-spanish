"""Evaluation utilities for the thesis experiments.

This package implements the statistical protocol decided for Experiments 1–3:

- paired two-sided Student t-test for mean effectiveness;
- sign-flip permutation test as confirmation;
- Holm-Bonferroni (FWER) and Benjamini-Hochberg (FDR) corrections for the
  six-algorithm comparison of Experiment 2.

See ``tesis/knowledge/urbano-et-al-2019-pruebas-estadisticas.md``.
"""

from .significance import (
    benjamini_hochberg,
    holm_bonferroni,
    paired_t_test,
    permutation_test,
)

__all__ = [
    "benjamini_hochberg",
    "compare_runs",
    "holm_bonferroni",
    "load_qrels",
    "paired_t_test",
    "per_query_scores",
    "permutation_test",
]


def __getattr__(name: str):
    """Import the run-level helpers lazily.

    Eagerly importing ``compare`` from this module triggers a runpy warning
    when the CLI is executed with ``python -m evaluation.compare``.
    """
    if name in {"compare_runs", "load_qrels", "per_query_scores"}:
        from . import compare

        return getattr(compare, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
