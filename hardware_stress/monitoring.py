"""Monitoring and dashboard rendering."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterable

import psutil
from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table

try:
    import GPUtil
except Exception:
    GPUtil = None

from .utils import format_bytes, format_duration, format_percent, format_rate, safe_int


LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class MonitoringSnapshot:
    runtime_seconds: float
    cpu_total_percent: float
    per_core_percent: list[float]
    cpu_frequency_mhz: float | None
    cpu_temperature_c: float | None
    ram_used_bytes: int
    ram_total_bytes: int
    ram_percent: float
    gpu_name: str | None
    gpu_utilization_percent: float | None
    gpu_temperature_c: float | None
    gpu_power_w: float | None
    gpu_memory_used_bytes: int | None
    gpu_memory_total_bytes: int | None
    gpu_clock_mhz: int | None
    gpu_memory_clock_mhz: int | None
    disk_read_bps: float
    disk_write_bps: float
    disk_read_total_bytes: int
    disk_write_total_bytes: int
    cpu_workers_running: bool
    gpu_worker_running: bool
    ram_worker_running: bool
    disk_worker_running: bool


class HardwareMonitor:
    def __init__(self) -> None:
        self._start_time = time.monotonic()
        self._previous_io = psutil.disk_io_counters()
        self._previous_time = time.monotonic()
        self._nvml = None
        self._gpu_handle = None
        self._gpu_name: str | None = None

        try:
            import pynvml

            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            if count > 0:
                self._nvml = pynvml
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._gpu_name = pynvml.nvmlDeviceGetName(self._gpu_handle).decode("utf-8", errors="ignore")
        except Exception:
            if GPUtil is not None:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        self._gpu_name = gpus[0].name
                except Exception:
                    pass

        psutil.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None, percpu=True)

    def close(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

    def __enter__(self) -> "HardwareMonitor":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def collect(
        self,
        cpu_workers_running: bool,
        gpu_worker_running: bool,
        ram_worker_running: bool,
        disk_worker_running: bool,
    ) -> MonitoringSnapshot:
        now = time.monotonic()
        cpu_total = psutil.cpu_percent(interval=None)
        per_core = psutil.cpu_percent(interval=None, percpu=True)
        cpu_freq = psutil.cpu_freq()
        cpu_temp = self._read_cpu_temperature()
        memory = psutil.virtual_memory()
        disk_read_bps, disk_write_bps, total_read, total_write = self._read_disk_io(now)
        gpu_metrics = self._read_gpu_metrics()

        return MonitoringSnapshot(
            runtime_seconds=now - self._start_time,
            cpu_total_percent=float(cpu_total),
            per_core_percent=[float(value) for value in per_core],
            cpu_frequency_mhz=float(cpu_freq.current) if cpu_freq else None,
            cpu_temperature_c=cpu_temp,
            ram_used_bytes=int(memory.used),
            ram_total_bytes=int(memory.total),
            ram_percent=float(memory.percent),
            gpu_name=self._gpu_name,
            gpu_utilization_percent=gpu_metrics[0],
            gpu_temperature_c=gpu_metrics[1],
            gpu_power_w=gpu_metrics[2],
            gpu_memory_used_bytes=gpu_metrics[3],
            gpu_memory_total_bytes=gpu_metrics[4],
            gpu_clock_mhz=gpu_metrics[5],
            gpu_memory_clock_mhz=gpu_metrics[6],
            disk_read_bps=disk_read_bps,
            disk_write_bps=disk_write_bps,
            disk_read_total_bytes=total_read,
            disk_write_total_bytes=total_write,
            cpu_workers_running=cpu_workers_running,
            gpu_worker_running=gpu_worker_running,
            ram_worker_running=ram_worker_running,
            disk_worker_running=disk_worker_running,
        )

    def _read_cpu_temperature(self) -> float | None:
        try:
            temps = psutil.sensors_temperatures(fahrenheit=False)
        except Exception:
            return None
        if not temps:
            return None
        for key in ("cpu_thermal", "coretemp", "k10temp", "acpitz", "cpu-thermal", "soc_thermal"):
            entries = temps.get(key)
            if entries:
                values = [entry.current for entry in entries if entry.current is not None]
                if values:
                    return float(max(values))
        for entries in temps.values():
            values = [entry.current for entry in entries if entry.current is not None]
            if values:
                return float(max(values))
        return None

    def _read_disk_io(self, now: float) -> tuple[float, float, int, int]:
        current = psutil.disk_io_counters()
        previous = self._previous_io
        previous_time = self._previous_time
        self._previous_io = current
        self._previous_time = now
        if previous is None or current is None:
            return 0.0, 0.0, 0, 0
        elapsed = max(0.001, now - previous_time)
        read_delta = max(0, current.read_bytes - previous.read_bytes)
        write_delta = max(0, current.write_bytes - previous.write_bytes)
        return read_delta / elapsed, write_delta / elapsed, int(current.read_bytes), int(current.write_bytes)

    def _read_gpu_metrics(self) -> tuple[float | None, float | None, float | None, int | None, int | None, int | None, int | None]:
        if self._nvml is None or self._gpu_handle is None:
            return None, None, None, None, None, None, None
        try:
            utilization = self._nvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
            temperature = self._nvml.nvmlDeviceGetTemperature(self._gpu_handle, self._nvml.NVML_TEMPERATURE_GPU)
            memory = self._nvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
            power = self._nvml.nvmlDeviceGetPowerUsage(self._gpu_handle) / 1000.0
            graphics_clock = self._nvml.nvmlDeviceGetClockInfo(self._gpu_handle, self._nvml.NVML_CLOCK_GRAPHICS)
            memory_clock = self._nvml.nvmlDeviceGetClockInfo(self._gpu_handle, self._nvml.NVML_CLOCK_MEM)
            return (
                float(utilization.gpu),
                float(temperature),
                float(power),
                int(memory.used),
                int(memory.total),
                int(graphics_clock),
                int(memory_clock),
            )
        except Exception:
            return None, None, None, None, None, None, None


def _meter(value: float | None, width: int = 18) -> str:
    if value is None:
        return "n/a"
    clamped = max(0.0, min(100.0, value))
    filled = int(round((clamped / 100.0) * width))
    return f"[{('=' * filled).ljust(width, '-')}] {clamped:5.1f}%"


def _core_rows(per_core: Iterable[float]) -> list[tuple[str, str]]:
    rows = []
    for index, value in enumerate(per_core):
        rows.append((f"Core {index:02d}", _meter(value, width=16)))
    return rows


def render_dashboard(snapshot: MonitoringSnapshot) -> Panel:
    cpu_line = (
        f"[bold]Runtime[/bold] {format_duration(snapshot.runtime_seconds)}\n"
        f"[bold]CPU[/bold] {format_percent(snapshot.cpu_total_percent)}"
    )
    if snapshot.cpu_frequency_mhz is not None:
        cpu_line += f" at {snapshot.cpu_frequency_mhz:.0f} MHz"

    header = Table.grid(expand=True)
    header.add_column(ratio=1)
    header.add_column(ratio=1)
    header.add_row(
        cpu_line,
        f"[bold]RAM[/bold] {format_percent(snapshot.ram_percent)} ({format_bytes(snapshot.ram_used_bytes)} / {format_bytes(snapshot.ram_total_bytes)})\n"
        f"[bold]Disk[/bold] R {format_rate(snapshot.disk_read_bps)} | W {format_rate(snapshot.disk_write_bps)}",
    )

    gpu_table = Table.grid(expand=True)
    gpu_table.add_column(ratio=1)
    gpu_table.add_column(ratio=1)
    gpu_table.add_row(
        f"[bold]GPU[/bold] {snapshot.gpu_name or 'unavailable'}\n"
        f"Util {format_percent(snapshot.gpu_utilization_percent)} | Temp {format_temperature(snapshot.gpu_temperature_c)}",
        f"Power {format_power(snapshot.gpu_power_w)}\n"
        f"VRAM {format_bytes(snapshot.gpu_memory_used_bytes)} / {format_bytes(snapshot.gpu_memory_total_bytes)}\n"
        f"Clocks {safe_int(snapshot.gpu_clock_mhz)} / {safe_int(snapshot.gpu_memory_clock_mhz)} MHz",
    )

    status = Table.grid(expand=True)
    status.add_column(ratio=1)
    status.add_column(ratio=1)
    status.add_row(
        f"[bold]Workers[/bold] CPU={'up' if snapshot.cpu_workers_running else 'down'}  GPU={'up' if snapshot.gpu_worker_running else 'down'}",
        f"[bold]Workers[/bold] RAM={'up' if snapshot.ram_worker_running else 'down'}  Disk={'up' if snapshot.disk_worker_running else 'down'}",
    )

    core_table = Table(
        title="Per-core CPU",
        box=box.SIMPLE,
        expand=True,
        show_header=True,
        header_style="bold cyan",
    )
    core_table.add_column("Core", style="bold")
    core_table.add_column("Usage", justify="left")
    for label, meter in _core_rows(snapshot.per_core_percent):
        core_table.add_row(label, meter)

    content = Group(
        Panel(header, title="System Summary", box=box.ROUNDED),
        Panel(gpu_table, title="GPU Summary", box=box.ROUNDED),
        Panel(status, title="Worker Health", box=box.ROUNDED),
        core_table,
    )
    return Panel(content, title="Hardware Stress Dashboard", box=box.DOUBLE)


def format_temperature(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f} C"


def format_power(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f} W"