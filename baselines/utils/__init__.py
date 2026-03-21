"""
Shared utilities for baseline evaluations.

Importing this package sets NUMA/threading env-vars before any heavy
libraries (torch, datasets, etc.) are loaded.
"""

import logging
import os

# ---------------------------------------------------------------------------
# NUMA / threading env-vars — MUST be set BEFORE other imports
# ---------------------------------------------------------------------------
_NCPU = str(os.cpu_count() or 32)
os.environ.setdefault("OMP_NUM_THREADS", _NCPU)
os.environ.setdefault("MKL_NUM_THREADS", _NCPU)
os.environ.setdefault("OMP_PROC_BIND", "spread")
os.environ.setdefault("OMP_PLACES", "threads")
os.environ.setdefault("GOMP_CPU_AFFINITY", f"0-{int(_NCPU) - 1}")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
