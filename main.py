"""Command-line entry point for the hardware stress testing application."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from typing import Sequence

import psutil

from hardware_stress.config import StressConfig, StressLevel
from hardware_stress.controller import StressController
from hardware_stress.utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Professional-grade Windows hardware stress tester")
    parser.add_argument("--level", choices=[level.value for level in StressLevel], default=StressLevel.MEDIUM.value)
    parser.add_argument("--duration", type=float, default=None, help="Optional duration in seconds before auto-stop")
    parser.add_argument(
        "--alert-after",
        type=float,
        default=None,
        help="Play a completion beep once the run reaches this many seconds",
    )
    parser.add_argument("--beep-frequency", type=int, default=880, help="Completion beep frequency in Hz")
    parser.add_argument("--beep-duration-ms", type=int, default=250, help="Completion beep duration in ms")
    parser.add_argument("--beep-count", type=int, default=1, help="Number of beep pulses to play")
    parser.add_argument("--cpu-workers", type=int, default=None, help="Override logical CPU worker count")
    parser.add_argument("--refresh", type=float, default=1.0, help="Dashboard refresh interval in seconds")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser


def _pick_cpu_workers(requested: int | None) -> int:
    logical = psutil.cpu_count(logical=True) or 1
    if requested is None:
        return logical
    return max(1, min(logical, requested))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    level = StressLevel(args.level)
    cpu_workers = _pick_cpu_workers(args.cpu_workers)
    config = StressConfig.for_level(level=level, cpu_workers=cpu_workers, duration_seconds=args.duration)

    controller = StressController(
        cpu_workers=config.cpu_workers,
        cpu_matrix_size=config.cpu_matrix_size,
        gpu_matrix_size=config.gpu_matrix_size,
        gpu_target_fraction=config.gpu_target_fraction,
        ram_fraction=config.ram_fraction,
        ram_chunk_mb=config.ram_chunk_mb,
        disk_workers=config.disk_workers,
        disk_file_size_mb=config.disk_file_size_mb,
        disk_file_count=config.disk_file_count,
        gpu_backend_order=config.gpu_backend_order,
        safety_ram_gb=config.safety_ram_gb,
        alert_after_seconds=args.alert_after,
        beep_frequency_hz=args.beep_frequency,
        beep_duration_ms=args.beep_duration_ms,
        beep_count=args.beep_count,
    )
    return controller.run(refresh_seconds=args.refresh, duration_seconds=config.duration_seconds)


if __name__ == "__main__":
    mp.freeze_support()
    os.environ.setdefault("PYTHONUTF8", "1")
    raise SystemExit(main())