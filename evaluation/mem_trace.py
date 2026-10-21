#!/usr/bin/env python3
"""Historical memory tracing for the efficiency benchmarks.

A before/after snapshot cannot attribute memory to a phase, misses the transient
peak of ``index_cpu_to_all_gpus`` (which holds the CPU index and the GPU copies
at the same time) and is polluted by other users of a shared GPU.  This module
samples memory on a fixed interval in a background thread and writes one CSV row
per sample, tagged with the current phase.

Two layers are recorded and must not be confused:

- process layer (``torch.cuda.memory_allocated`` / ``reserved``): only this
  process, immune to other tenants;
- device layer (``nvidia-smi``): whole GPU, includes other users;
- own-process layer (``nvidia-smi --query-compute-apps``): only the CUDA contexts
  that belong to this PID, which is what isolates FAISS allocations.

FAISS uses its own caching allocator, so ``torch.cuda.memory_allocated`` does not
see the indexes; the own-process column is the one that attributes them.

CPU side records ``VmRSS`` (current), ``VmHWM`` (peak) and ``VmSize`` (virtual).
``VmHWM`` is a global high-water mark: in the per-phase summary it is the peak
*up to the end of that phase*, not the peak *of* that phase.  Use ``rss_mib``
for per-phase attribution.
"""

from __future__ import annotations

import csv
import logging
import os
import subprocess
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

FIELDS = [
    "elapsed_s",
    "phase",
    "rss_mib",
    "vmsize_mib",
    "vmhwm_mib",
    "cuda_allocated_mib",
    "cuda_reserved_mib",
    "gpu_used_mib",
    "own_gpu_used_mib",
]


def _proc_status() -> dict[str, float]:
    values: dict[str, float] = {}
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(("VmRSS:", "VmHWM:", "VmSize:")):
                    key, _, rest = line.partition(":")
                    values[key] = int(rest.split()[0]) / 1024.0
    except OSError:
        pass
    return values


def _gpu_used() -> list[int]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return [int(line.strip()) for line in output.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


_GPU_INDEX_BY_UUID: dict[str, int] = {}


def _gpu_index_by_uuid() -> dict[str, int]:
    if _GPU_INDEX_BY_UUID:
        return _GPU_INDEX_BY_UUID
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in output.splitlines():
            index_text, _, uuid = line.partition(",")
            _GPU_INDEX_BY_UUID[uuid.strip()] = int(index_text.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return _GPU_INDEX_BY_UUID


def _own_gpu_used() -> list[int]:
    """GPU memory held by this process, per device index."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    mapping = _gpu_index_by_uuid()
    sizes: dict[int, int] = {}
    my_pid = os.getpid()
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        pid_text, uuid, used_text = parts
        if not pid_text.isdigit() or int(pid_text) != my_pid:
            continue
        index = mapping.get(uuid)
        if index is None:
            continue
        sizes[index] = sizes.get(index, 0) + int(used_text)
    if not sizes:
        return []
    return [sizes.get(index, 0) for index in range(max(sizes) + 1)]


def _cuda_memory() -> tuple[float, float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0, 0.0
        return (
            torch.cuda.memory_allocated() / (1024**2),
            torch.cuda.memory_reserved() / (1024**2),
        )
    except Exception:  # pragma: no cover - torch always present in practice
        return 0.0, 0.0


class MemoryTrace:
    """Sample memory every ``interval_s`` seconds into ``path``."""

    def __init__(self, path: Path, interval_s: float = 1.0):
        self.path = Path(path)
        self.interval_s = interval_s
        self.phase = "start"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        self._file = None
        self._writer = None

    def __enter__(self) -> "MemoryTrace":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        self._writer.writeheader()
        self._started_at = time.perf_counter()
        self._stop.clear()
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set_phase(self, phase: str) -> None:
        if phase != self.phase:
            self.phase = phase
            self._sample()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def _sample(self) -> None:
        if self._writer is None:
            return
        status = _proc_status()
        allocated, reserved = _cuda_memory()
        gpu = _gpu_used()
        own = _own_gpu_used()
        self._writer.writerow(
            {
                "elapsed_s": f"{time.perf_counter() - self._started_at:.3f}",
                "phase": self.phase,
                "rss_mib": f"{status.get('VmRSS', 0.0):.1f}",
                "vmsize_mib": f"{status.get('VmSize', 0.0):.1f}",
                "vmhwm_mib": f"{status.get('VmHWM', 0.0):.1f}",
                "cuda_allocated_mib": f"{allocated:.1f}",
                "cuda_reserved_mib": f"{reserved:.1f}",
                "gpu_used_mib": ";".join(str(value) for value in gpu),
                "own_gpu_used_mib": ";".join(str(value) for value in own),
            }
        )
        self._file.flush()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._sample()
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


def summarize(path: Path) -> list[dict]:
    """Return per-phase peaks from a trace CSV.

    ``rss_mib_max`` and ``cuda_*_max`` are peaks *within* the phase.
    ``vmhwm_mib_max`` is the cumulative high-water mark at the end of the phase.
    """
    phases: dict[str, dict] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            phase = row["phase"]
            entry = phases.setdefault(
                phase,
                {
                    "phase": phase,
                    "samples": 0,
                    "rss_mib_max": 0.0,
                    "vmhwm_mib_max": 0.0,
                    "cuda_allocated_mib_max": 0.0,
                    "cuda_reserved_mib_max": 0.0,
                    "gpu_used_mib_max": [],
                    "own_gpu_used_mib_max": [],
                },
            )
            entry["samples"] += 1
            entry["rss_mib_max"] = max(entry["rss_mib_max"], float(row["rss_mib"]))
            entry["vmhwm_mib_max"] = max(entry["vmhwm_mib_max"], float(row["vmhwm_mib"]))
            entry["cuda_allocated_mib_max"] = max(
                entry["cuda_allocated_mib_max"], float(row["cuda_allocated_mib"])
            )
            entry["cuda_reserved_mib_max"] = max(
                entry["cuda_reserved_mib_max"], float(row["cuda_reserved_mib"])
            )
            for column in ("gpu_used_mib", "own_gpu_used_mib"):
                values = [int(value) for value in row[column].split(";") if value]
                key = f"{column}_max"
                if len(values) > len(entry[key]):
                    entry[key] = values[:]
                else:
                    entry[key] = [max(a, b) for a, b in zip(entry[key], values)]
    return list(phases.values())
