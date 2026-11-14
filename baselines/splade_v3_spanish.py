#!/usr/bin/env python3
"""
Prototipo: SPLADE-v3 adaptado al español sobre MessIRve.

Reutiliza el pipeline de ``baselines/splade_v3.py`` tal cual —misma
codificación por lotes, misma caché reanudable, misma matmul dispersa en dos
GPUs y misma evaluación con ranx— y solo cambia el checkpoint por
``alrolo3/splade-v3-spanish-vllm``.

Uso (desde la raíz del repositorio)::

    python -m baselines.splade_v3_spanish
"""

from __future__ import annotations

from baselines.splade_v3 import run

MODEL_NAME = "alrolo3/splade-v3-spanish-vllm"
MODEL_ALIAS = "splade-v3-es"


if __name__ == "__main__":
    run(MODEL_NAME, MODEL_ALIAS)
