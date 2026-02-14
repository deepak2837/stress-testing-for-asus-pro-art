"""Disk stress worker management."""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .utils import bytes_for_mb


LOG = logging.getLogger(__name__)


def _disk_worker(file_path: Path, file_size: int, stop_flag: threading.Event) -> None:
    block = os.urandom(min(1024 * 1024, max(64 * 1024, file_size // 8)))
    with open(file_path, "w+b", buffering=0) as handle:
        handle.seek(file_size - 1)
        handle.write(b"\0")
        handle.flush()

        while not stop_flag.is_set():
            handle.seek(0)
            remaining = file_size
            while remaining > 0 and not stop_flag.is_set():
                size = min(len(block), remaining)
                handle.write(block[:size])
                remaining -= size
            handle.flush()
            os.fsync(handle.fileno())

            handle.seek(0)
            while not stop_flag.is_set():
                data = handle.read(len(block))
                if not data:
                    break


@dataclass(slots=True)
class DiskStressManager:
    worker_count: int
    file_count: int
    file_size_mb: int
    _temp_dir: tempfile.TemporaryDirectory[str] | None = field(default=None, init=False)
    _executor: ThreadPoolExecutor | None = field(default=None, init=False)
    _stop_event: threading.Event | None = field(default=None, init=False)
    _futures: list = field(default_factory=list, init=False)

    def start(self) -> None:
        if self._executor is not None:
            return
        self._temp_dir = tempfile.TemporaryDirectory(prefix="hardware-stress-")
        self._stop_event = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=self.worker_count, thread_name_prefix="disk-worker")
        file_size = bytes_for_mb(self.file_size_mb)
        base_dir = Path(self._temp_dir.name)
        paths = [base_dir / f"stress-{index:02d}.tmp" for index in range(self.file_count)]
        for path in paths:
            future = self._executor.submit(_disk_worker, path, file_size, self._stop_event)
            self._futures.append(future)
        LOG.info("Started %d disk workers using %d temp files", self.worker_count, self.file_count)

    def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        self._executor = None
        self._futures.clear()
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def is_running(self) -> bool:
        return self._executor is not None