"""
Timing and logging utilities.
"""

import logging
import time

import torch

log = logging.getLogger(__name__)


class Timer:
    """Context manager that logs the start and elapsed time of a named operation."""

    def __init__(self, description: str):
        self.description = description
        self.elapsed: float | None = None

    def __enter__(self):
        log.info("⏱  START: %s", self.description)
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start_time
        log.info("⏱  DONE:  %s (%.1fs)", self.description, self.elapsed)


def log_gpu_memory():
    """Log allocated, reserved, and total GPU memory for each visible CUDA device."""
    if not torch.cuda.is_available():
        return
    for device_index in range(torch.cuda.device_count()):
        allocated_gb = torch.cuda.memory_allocated(device_index) / (1024**3)
        reserved_gb = torch.cuda.memory_reserved(device_index) / (1024**3)
        total_gb = torch.cuda.get_device_properties(device_index).total_memory / (1024**3)
        log.info(
            "GPU %d (%s): %.1f/%.1f/%.1f GB (alloc/reserved/total)",
            device_index,
            torch.cuda.get_device_name(device_index),
            allocated_gb,
            reserved_gb,
            total_gb,
        )


def log_ram_usage():
    """Log peak resident set size (RSS) memory usage of the current process."""
    try:
        import resource

        peak_rss_kilobytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        log.info("RAM (peak RSS): %.1f GB", peak_rss_kilobytes / (1024**2))
    except ImportError:
        pass
