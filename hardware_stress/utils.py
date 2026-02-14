"""Shared utilities for logging, formatting, and process helpers."""

from __future__ import annotations

import logging
import os
from datetime import timedelta

from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> None:
    """Configure a single rich console logger."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )


def format_bytes(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    amount = float(value)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} TB"


def format_rate(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    amount = float(value)
    units = ["B/s", "KB/s", "MB/s", "GB/s"]
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} GB/s"


def format_duration(seconds: float) -> str:
    delta = timedelta(seconds=max(0.0, seconds))
    total_seconds = int(delta.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_percent(value: float | int | None) -> str:
    return "n/a" if value is None else f"{float(value):5.1f}%"


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def set_single_thread_blas() -> None:
    """Keep NumPy/BLAS from oversubscribing CPU workers."""

    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")


def bytes_for_mb(value_mb: int) -> int:
    return int(value_mb * 1024 * 1024)


def safe_int(value: float | int | None, fallback: int = 0) -> int:
    if value is None:
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def play_beep(frequency_hz: int = 880, duration_ms: int = 250, count: int = 1) -> None:
    """Play a short completion beep on Windows, with a console fallback."""

    try:
        import winsound

        for _ in range(max(1, count)):
            winsound.Beep(max(37, frequency_hz), max(1, duration_ms))
        return
    except (ImportError, RuntimeError, OSError):
        pass

    for _ in range(max(1, count)):
        print("\a", end="", flush=True)