"""Application orchestration for the stress test suite."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from rich.console import Console
from rich.live import Live

from .cpu import CPUStressManager
from .disk import DiskStressManager
from .gpu import GPUStressManager
from .monitoring import HardwareMonitor, render_dashboard
from .ram import RAMStressManager
from .utils import play_beep


LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class StressController:
    cpu_workers: int
    cpu_matrix_size: int
    gpu_matrix_size: int
    gpu_target_fraction: float
    ram_fraction: float
    ram_chunk_mb: int
    disk_workers: int
    disk_file_size_mb: int
    disk_file_count: int
    gpu_backend_order: tuple[str, ...]
    safety_ram_gb: float
    alert_after_seconds: float | None = None
    beep_frequency_hz: int = 880
    beep_duration_ms: int = 250
    beep_count: int = 1
    console: Console = field(init=False, repr=False)
    monitor: HardwareMonitor = field(init=False, repr=False)
    cpu: CPUStressManager = field(init=False, repr=False)
    gpu: GPUStressManager = field(init=False, repr=False)
    ram: RAMStressManager = field(init=False, repr=False)
    disk: DiskStressManager = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.console = Console()
        self.monitor = HardwareMonitor()
        self.cpu = CPUStressManager(worker_count=self.cpu_workers, matrix_size=self.cpu_matrix_size)
        self.gpu = GPUStressManager(
            target_fraction=self.gpu_target_fraction,
            matrix_size=self.gpu_matrix_size,
            backend_order=self.gpu_backend_order,
        )
        self.ram = RAMStressManager(
            fraction=self.ram_fraction,
            chunk_mb=self.ram_chunk_mb,
            safety_margin_gb=self.safety_ram_gb,
        )
        self.disk = DiskStressManager(
            worker_count=self.disk_workers,
            file_count=self.disk_file_count,
            file_size_mb=self.disk_file_size_mb,
        )

    def start(self) -> None:
        self.cpu.start()
        self.gpu.start()
        self.ram.start()
        self.disk.start()

    def stop(self) -> None:
        self.disk.stop()
        self.ram.stop()
        self.gpu.stop()
        self.cpu.stop()
        self.monitor.close()

    def run(self, refresh_seconds: float = 1.0, duration_seconds: float | None = None) -> int:
        self.start()
        start = time.monotonic()
        alert_triggered = False
        try:
            with Live(console=self.console, refresh_per_second=4, screen=True) as live:
                while True:
                    elapsed = time.monotonic() - start
                    snapshot = self.monitor.collect(
                        cpu_workers_running=self.cpu.is_running(),
                        gpu_worker_running=self.gpu.is_running(),
                        ram_worker_running=self.ram.is_running(),
                        disk_worker_running=self.disk.is_running(),
                    )
                    live.update(render_dashboard(snapshot))

                    if (
                        self.alert_after_seconds is not None
                        and not alert_triggered
                        and elapsed >= self.alert_after_seconds
                    ):
                        play_beep(
                            frequency_hz=self.beep_frequency_hz,
                            duration_ms=self.beep_duration_ms,
                            count=self.beep_count,
                        )
                        alert_triggered = True

                    if duration_seconds is not None and elapsed >= duration_seconds:
                        break
                    if not self.cpu.is_running() or not self.ram.is_running() or not self.disk.is_running():
                        LOG.error("One or more workers exited unexpectedly; stopping the run.")
                        break
                    if self.gpu.backend is not None and not self.gpu.is_running():
                        LOG.error("GPU worker exited unexpectedly; stopping the run.")
                        break
                    time.sleep(refresh_seconds)
        except KeyboardInterrupt:
            LOG.info("Ctrl+C received; shutting down gracefully.")
        finally:
            self.stop()
        return 0