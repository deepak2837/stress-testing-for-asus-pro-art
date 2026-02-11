# Hardware Stress Tester

Windows 11 Python application for simultaneous CPU, GPU, RAM, and disk stress testing with a live monitoring dashboard.

## Features

- CPU saturation through one NumPy worker process per logical core.
- GPU stress through PyTorch CUDA or CuPy, with large matrix multiplications and VRAM ballast.
- RAM stress through a dedicated process that allocates and continuously touches memory pages.
- Disk stress through parallel temporary-file read/write workers.
- Live Rich dashboard with CPU, RAM, GPU, disk, and runtime metrics.
- Graceful shutdown with Ctrl+C and automatic cleanup of processes and temporary files.

## Project Structure

```text
main.py
requirements.txt
README.md
hardware_stress/
  __init__.py
  config.py
  controller.py
  cpu.py
  disk.py
  gpu.py
  monitoring.py
  ram.py
  utils.py
```

## Installation

```powershell
pip install -r requirements.txt
```

For GPU stress, install one CUDA backend if it is not already available:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

or:

```powershell
pip install cupy-cuda12x
```

## Run

```powershell
python main.py --level heavy
```

Optional flags:

- `--level`: `light`, `medium`, `heavy`, `extreme`
- `--duration`: auto-stop after the given number of seconds
- `--alert-after`: play a beep once the run reaches the given number of seconds
- `--beep-frequency`, `--beep-duration-ms`, `--beep-count`: tune the completion beep
- `--cpu-workers`: override the number of CPU worker processes
- `--refresh`: dashboard refresh interval in seconds

To run a 30-minute stress session and beep when the timer is reached:

```powershell
python main.py --level heavy --duration 1800 --alert-after 1800
```

## Safety Notes

- The app is designed to maximize utilization without intentionally corrupting data or disabling system controls.
- Use `light` or `medium` first to validate thermals and stability.
- The GPU worker keeps a safety margin to reduce the chance of CUDA out-of-memory errors.
- Temporary disk files are removed automatically on exit.

## Monitoring

The dashboard updates every second and reports:

- CPU usage, per-core usage, and CPU frequency
- CPU temperature when available from system sensors
- RAM usage
- GPU utilization, temperature, power draw, VRAM usage, and clocks
- Disk read/write throughput
- Runtime

## Notes

- If no CUDA backend is installed, the application still runs and reports that GPU stress is unavailable.
- Some CPU temperature sensors are not exposed by all laptop BIOS or driver combinations on Windows.