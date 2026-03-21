"""
Timing and logging utilities
"""

import logging
import logging
import time

import torch

log = logging.getLogger(__name__)


class Timer:
    def __init__(self, description: str):
        self.description = description
        self.elapsed: float | None = None

    def __enter__(self):
        log.info("⏱  START: %s", self.description)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._t0
        log.info("⏱  DONE:  %s (%.1fs)", self.description, self.elapsed)


def log_gpu_memory():
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        a = torch.cuda.memory_allocated(i) / (1024**3)
        r = torch.cuda.memory_reserved(i) / (1024**3)
        t = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        log.info(
            "GPU %d (%s): %.1f/%.1f/%.1f GB (alloc/reserved/total)",
            i,
            torch.cuda.get_device_name(i),
            a,
            r,
            t,
        )


def log_ram_usage():
    try:
        import resource

        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        log.info("RAM (peak RSS): %.1f GB", kb / (1024**2))
    except ImportError:
        pass
