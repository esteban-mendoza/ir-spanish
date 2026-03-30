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
_num_cpus: int = os.cpu_count() or 32
_num_cpus_str: str = str(_num_cpus)

os.environ.setdefault("OMP_NUM_THREADS", _num_cpus_str)
os.environ.setdefault("MKL_NUM_THREADS", _num_cpus_str)
os.environ.setdefault("OMP_PROC_BIND", "spread")
os.environ.setdefault("OMP_PLACES", "threads")
os.environ.setdefault("GOMP_CPU_AFFINITY", f"0-{_num_cpus - 1}")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
