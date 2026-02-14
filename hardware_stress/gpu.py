"""GPU stress worker management with PyTorch or CuPy backends."""

from __future__ import annotations

import importlib.util
import logging
import multiprocessing as mp
from dataclasses import dataclass, field

LOG = logging.getLogger(__name__)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _torch_worker(target_fraction: float, matrix_size: int, stop_event: mp.Event) -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    total_memory = int(properties.total_memory)
    safety_margin = max(512 * 1024 * 1024, int(total_memory * 0.12))
    target_bytes = min(int(total_memory * target_fraction), total_memory - safety_margin)
    element_size = 2
    compute_bytes = matrix_size * matrix_size * element_size * 3
    ballast_target = max(0, target_bytes - compute_bytes)

    dtype = torch.float16
    left = torch.randn((matrix_size, matrix_size), device=device, dtype=dtype)
    right = torch.randn((matrix_size, matrix_size), device=device, dtype=dtype)
    output = torch.empty_like(left)

    ballast: list[torch.Tensor] = []
    chunk_bytes = min(256 * 1024 * 1024, max(32 * 1024 * 1024, ballast_target // 12 or 32 * 1024 * 1024))
    chunk_elements = max(1, chunk_bytes // element_size)

    try:
        allocated = int(left.numel() * element_size * 3)
        while allocated < ballast_target and not stop_event.is_set():
            try:
                chunk = torch.empty((chunk_elements,), device=device, dtype=dtype)
                chunk.uniform_(-1.0, 1.0)
                ballast.append(chunk)
                allocated += int(chunk.numel() * element_size)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    break
                raise

        while not stop_event.is_set():
            torch.matmul(left, right, out=output)
            left, right, output = right, output, left
            torch.cuda.synchronize()
    finally:
        ballast.clear()
        del left, right, output
        torch.cuda.empty_cache()


def _cupy_worker(target_fraction: float, matrix_size: int, stop_event: mp.Event) -> None:
    import cupy as cp

    device = cp.cuda.Device(0)
    with device:
        props = cp.cuda.runtime.getDeviceProperties(0)
        total_memory = int(props["totalGlobalMem"])
        safety_margin = max(512 * 1024 * 1024, int(total_memory * 0.12))
        target_bytes = min(int(total_memory * target_fraction), total_memory - safety_margin)
        element_size = 2
        compute_bytes = matrix_size * matrix_size * element_size * 3
        ballast_target = max(0, target_bytes - compute_bytes)

        left = cp.random.random((matrix_size, matrix_size), dtype=cp.float16)
        right = cp.random.random((matrix_size, matrix_size), dtype=cp.float16)
        output = cp.empty_like(left)

        ballast: list[cp.ndarray] = []
        chunk_bytes = min(256 * 1024 * 1024, max(32 * 1024 * 1024, ballast_target // 12 or 32 * 1024 * 1024))
        chunk_elements = max(1, chunk_bytes // element_size)

        try:
            allocated = int(left.size * element_size * 3)
            while allocated < ballast_target and not stop_event.is_set():
                try:
                    chunk = cp.empty((chunk_elements,), dtype=cp.float16)
                    chunk.fill(0.5)
                    ballast.append(chunk)
                    allocated += int(chunk.size * element_size)
                except cp.cuda.memory.OutOfMemoryError:
                    break

            while not stop_event.is_set():
                cp.matmul(left, right, out=output)
                left, right, output = right, output, left
                cp.cuda.Stream.null.synchronize()
        finally:
            ballast.clear()
            cp.get_default_memory_pool().free_all_blocks()


@dataclass(slots=True)
class GPUStressManager:
    target_fraction: float
    matrix_size: int
    backend_order: tuple[str, ...] = ("torch", "cupy")
    _context: mp.context.BaseContext = field(init=False, repr=False)
    _stop_event: mp.Event = field(init=False, repr=False)
    _process: mp.Process | None = field(init=False, repr=False, default=None)
    backend: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._context = mp.get_context("spawn")
        self._stop_event = self._context.Event()

    def _select_backend(self) -> str | None:
        for candidate in self.backend_order:
            if candidate == "torch" and _module_available("torch"):
                return "torch"
            if candidate == "cupy" and _module_available("cupy"):
                return "cupy"
        return None

    def start(self) -> None:
        if self._process and self._process.is_alive():
            return
        backend = self._select_backend()
        if backend is None:
            LOG.warning("No GPU backend found. Install a CUDA-enabled torch or CuPy wheel to enable GPU stress.")
            return
        target = _torch_worker if backend == "torch" else _cupy_worker
        process = self._context.Process(
            target=target,
            name=f"gpu-{backend}-worker",
            args=(self.target_fraction, self.matrix_size, self._stop_event),
            daemon=True,
        )
        process.start()
        self.backend = backend
        self._process = process
        LOG.info("Started GPU worker using %s backend", backend)

    def stop(self) -> None:
        self._stop_event.set()
        if self._process and self._process.is_alive():
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()
        self._process = None
        self.backend = None

    def is_running(self) -> bool:
        return bool(self._process and self._process.is_alive())