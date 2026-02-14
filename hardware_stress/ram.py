"""RAM stress worker management."""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass, field

import psutil

from .utils import bytes_for_mb


LOG = logging.getLogger(__name__)


def _ram_worker(target_bytes: int, chunk_bytes: int, stop_event: mp.Event) -> None:
    available = psutil.virtual_memory().available
    safety_margin = 2 * 1024 * 1024 * 1024
    target_bytes = min(target_bytes, max(0, available - safety_margin))
    if target_bytes <= 0:
        return

    chunks: list[bytearray] = []
    allocated = 0
    while allocated < target_bytes and not stop_event.is_set():
        size = min(chunk_bytes, target_bytes - allocated)
        try:
            chunk = bytearray(size)
        except MemoryError:
            break
        chunks.append(chunk)
        allocated += size

    page_step = 4096
    while not stop_event.is_set() and chunks:
        for chunk in chunks:
            limit = len(chunk)
            for offset in range(0, limit, page_step):
                value = chunk[offset]
                chunk[offset] = (value + 1) & 0xFF
            if stop_event.is_set():
                break


@dataclass(slots=True)
class RAMStressManager:
    fraction: float
    chunk_mb: int
    safety_margin_gb: float
    _context: mp.context.BaseContext = field(init=False, repr=False)
    _stop_event: mp.Event = field(init=False, repr=False)
    _process: mp.Process | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self._context = mp.get_context("spawn")
        self._stop_event = self._context.Event()

    def start(self) -> None:
        if self._process and self._process.is_alive():
            return
        target_bytes = int(psutil.virtual_memory().available * self.fraction)
        process = self._context.Process(
            target=_ram_worker,
            name="ram-worker",
            args=(target_bytes, bytes_for_mb(self.chunk_mb), self._stop_event),
            daemon=True,
        )
        process.start()
        self._process = process
        LOG.info("Started RAM worker targeting %.0f%% of available memory", self.fraction * 100.0)

    def stop(self) -> None:
        self._stop_event.set()
        if self._process and self._process.is_alive():
            self._process.join(timeout=3.0)
            if self._process.is_alive():
                self._process.terminate()
        self._process = None

    def is_running(self) -> bool:
        return bool(self._process and self._process.is_alive())