"""Configuration models and stress-level presets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class StressLevel(str, Enum):
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"
    EXTREME = "extreme"


@dataclass(slots=True, frozen=True)
class StressConfig:
    level: StressLevel
    cpu_workers: int
    cpu_matrix_size: int
    gpu_matrix_size: int
    gpu_target_fraction: float
    ram_fraction: float
    ram_chunk_mb: int
    disk_workers: int
    disk_file_size_mb: int
    disk_file_count: int
    dashboard_refresh_seconds: float = 1.0
    safety_ram_gb: float = 2.0
    duration_seconds: float | None = None
    gpu_backend_order: tuple[str, ...] = ("torch", "cupy")

    @classmethod
    def for_level(
        cls,
        level: StressLevel,
        cpu_workers: int,
        duration_seconds: float | None = None,
    ) -> "StressConfig":
        presets = {
            StressLevel.LIGHT: {
                "cpu_matrix_size": 256,
                "gpu_matrix_size": 2048,
                "gpu_target_fraction": 0.28,
                "ram_fraction": 0.20,
                "ram_chunk_mb": 8,
                "disk_workers": 1,
                "disk_file_size_mb": 32,
                "disk_file_count": 1,
            },
            StressLevel.MEDIUM: {
                "cpu_matrix_size": 384,
                "gpu_matrix_size": 3072,
                "gpu_target_fraction": 0.45,
                "ram_fraction": 0.40,
                "ram_chunk_mb": 8,
                "disk_workers": 2,
                "disk_file_size_mb": 64,
                "disk_file_count": 2,
            },
            StressLevel.HEAVY: {
                "cpu_matrix_size": 512,
                "gpu_matrix_size": 4096,
                "gpu_target_fraction": 0.65,
                "ram_fraction": 0.60,
                "ram_chunk_mb": 16,
                "disk_workers": 4,
                "disk_file_size_mb": 96,
                "disk_file_count": 4,
            },
            StressLevel.EXTREME: {
                "cpu_matrix_size": 640,
                "gpu_matrix_size": 6144,
                "gpu_target_fraction": 0.78,
                "ram_fraction": 0.80,
                "ram_chunk_mb": 16,
                "disk_workers": 4,
                "disk_file_size_mb": 128,
                "disk_file_count": 4,
            },
        }
        preset = presets[level]
        return cls(
            level=level,
            cpu_workers=cpu_workers,
            cpu_matrix_size=preset["cpu_matrix_size"],
            gpu_matrix_size=preset["gpu_matrix_size"],
            gpu_target_fraction=preset["gpu_target_fraction"],
            ram_fraction=preset["ram_fraction"],
            ram_chunk_mb=preset["ram_chunk_mb"],
            disk_workers=preset["disk_workers"],
            disk_file_size_mb=preset["disk_file_size_mb"],
            disk_file_count=preset["disk_file_count"],
            duration_seconds=duration_seconds,
        )