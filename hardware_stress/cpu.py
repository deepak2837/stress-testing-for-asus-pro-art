"""CPU stress worker management."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from dataclasses import dataclass, field

from .utils import set_single_thread_blas

set_single_thread_blas()

import numpy as np  # noqa: E402
import psutil  # noqa: E402


LOG = logging.getLogger(__name__)


def _cpu_worker(core_index: int, matrix_size: int, stop_event: mp.Event) -> None:
    set_single_thread_blas()
    rng = np.random.default_rng(seed=os.getpid() ^ (core_index << 16))
    left = rng.random((matrix_size, matrix_size), dtype=np.float32)
    right = rng.random((matrix_size, matrix_size), dtype=np.float32)
    output = np.empty_like(left)

    try:
        process = psutil.Process()
        affinity = getattr(process, "cpu_affinity", None)
        if callable(affinity):
            available = psutil.cpu_count(logical=True) or 1
            process.cpu_affinity([core_index % available])
    except Exception:
        pass

    while not stop_event.is_set():
        np.matmul(left, right, out=output)
        left, right, output = right, output, left


@dataclass(slots=True)
class CPUStressManager:
    worker_count: int
    matrix_size: int
    _context: mp.context.BaseContext = field(init=False, repr=False)
    _stop_event: mp.Event = field(init=False, repr=False)
    _processes: list[mp.Process] = field(init=False, repr=False, default_factory=list)

    def __post_init__(self) -> None:
        self._context = mp.get_context("spawn")
        self._stop_event = self._context.Event()

    def start(self) -> None:
        if self._processes:
            return
        for core_index in range(self.worker_count):
            process = self._context.Process(
                target=_cpu_worker,
                name=f"cpu-worker-{core_index}",
                args=(core_index, self.matrix_size, self._stop_event),
                daemon=True,
            )
            process.start()
            self._processes.append(process)
        LOG.info("Started %d CPU workers", len(self._processes))

    def stop(self) -> None:
        self._stop_event.set()
        for process in self._processes:
            if process.is_alive():
                process.join(timeout=3.0)
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        self._processes.clear()

    def is_running(self) -> bool:
        return any(process.is_alive() for process in self._processes)