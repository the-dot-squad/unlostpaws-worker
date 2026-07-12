"""Hardware detection, profile recommendation, and preflight validation."""

from __future__ import annotations

import os
import platform
import shutil
from importlib import reload

from tools._console import err, out
from tools._paths import ensure_import_path

# Estimated Hugging Face cache size on first model download (match + safety).
_PROFILE_DISK_HINT_MB: dict[str, int] = {
    "dedup-only": 0,
    "standard": 1024,
    "quality": 1536,
}

_doctor_mode = False


def _print_profile_resources(profile_name: str) -> None:
    from app.config.profiles import get_preset

    preset = get_preset(profile_name)
    disk_mb = _PROFILE_DISK_HINT_MB.get(profile_name, 0)
    out(f"Profile Resource Requirements ({profile_name}):")
    out(f"  - Min RAM Required:     {preset.min_ram_mb} MB")
    if preset.min_vram_mb:
        out(f"  - Min VRAM Required:    {preset.min_vram_mb} MB (Optional)")
    if disk_mb:
        out(f"  - HF Cache (first run): ~{disk_mb / 1024:.1f} GB")


def print_requirements_checklist() -> None:
    # OS & Architecture
    os_name = platform.system()
    arch_name = platform.machine()

    # Python Version Compatibility
    py_ver = platform.python_version()
    py_major, py_minor, _ = platform.python_version_tuple()
    py_ok = py_major == "3" and py_minor == "12"

    # System RAM Detection
    ram_gb = 0.0
    try:
        pagesize = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        ram_gb = (pagesize * pages) / (1024**3)
    except Exception:
        try:
            import subprocess

            out_bytes = subprocess.check_output(["sysctl", "-n", "hw.memsize"])
            ram_gb = int(out_bytes.strip()) / (1024**3)
        except Exception:
            pass

    ram_ok = ram_gb >= 1.0

    # Docker Engine Installation
    has_docker = shutil.which("docker") is not None

    # NVIDIA CUDA availability
    has_cuda = False
    gpu_desc = ""
    try:
        import torch

        has_cuda = torch.cuda.is_available()
        if has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            gpu_desc = f"Available ({gpu_name}, {vram_mb / 1024:.1f} GB VRAM)"
    except Exception:
        pass

    out("System & Requirements Check:")
    out(f"  [✓] OS / Arch:          {os_name} ({arch_name})")

    if py_ok:
        out(f"  [✓] Python Version:     {py_ver} (Passed)")
    else:
        out(f"  [✗] Python Version:     {py_ver} (Failed - Python 3.12 is required)")

    if ram_ok:
        out(f"  [✓] System RAM:         {ram_gb:.2f} GB (Passed)")
    else:
        if ram_gb > 0:
            out(
                f"  [✗] System RAM:         {ram_gb:.2f} GB (Failed - Min 1 GB required)"
            )
        else:
            out("  [✗] System RAM:         Unknown (Failed)")

    if has_docker:
        out("  [✓] Docker Engine:      Installed (Passed)")
    else:
        out(
            "  [ ] Docker Engine:      Not Found (Optional - only required for Docker container)"
        )

    if has_cuda:
        out(f"  [✓] NVIDIA GPU (CUDA):  {gpu_desc} (Passed)")
    else:
        out("  [ ] NVIDIA GPU (CUDA):  Not Available (Optional)")
    out()


def print_report(hardware, config) -> None:
    global _doctor_mode
    _doctor_mode = True

    out("UnLostPaws Vision Worker — Hardware Doctor")
    out("=" * 50)

    # Print the requirements checklist
    print_requirements_checklist()

    # Print the recommended config
    out("Recommended Configuration:")
    out(f"  VISION_PROFILE={config.vision_profile}")
    out(f"  INFERENCE_RUNTIME={config.inference_runtime}")
    if config.inference_runtime == "onnx":
        out(f"  ORT_EXECUTION_PROVIDER={config.execution_provider}")
    else:
        out(f"  DEVICE={config.device}")
    out(f"  Run Hint:               {config.run_hint}")

    if config.warnings:
        out("\nWarnings:")
        for warning in config.warnings:
            out(f"  [!] {warning}")
    out()


def validate_profile(profile: str) -> None:
    """Run preflight hardware checks for the given profile; exit 1 on mismatch."""
    ensure_import_path()
    os.environ["VISION_PROFILE"] = profile

    import app.config.settings as settings_mod

    reload(settings_mod)

    from app.config.runtime_validation import (
        RuntimeValidationError,
        detect_hardware,
        validate_runtime,
    )
    from app.config.settings import load_settings

    cfg = load_settings()
    hardware = detect_hardware()
    try:
        validate_runtime(cfg, hardware, phase="preflight")
        out("Preflight Validation:")
        out(f"  [✓] Preflight Check:    Passed for VISION_PROFILE={profile}")
        out()
        _print_profile_resources(profile)

        # If run from main doctor, print checklist at the very end
        if _doctor_mode:
            out()
            out("First-run Checklist:")
            out(
                "  1. Run './tools/run.sh setup' to automatically generate .env and docker-compose.yml."
            )
            out(
                "  2. Ensure your Redis server is active and reachable via the REDIS_URL in .env."
            )
            out(
                "  3. Start the worker container: 'docker compose up -d' (or python app/main.py)."
            )
            out()
    except RuntimeValidationError as exc:
        err("Preflight Validation:")
        err(f"  [✗] Preflight Check:    FAILED for VISION_PROFILE={profile}")
        err(f"  Reason: {exc}")
        raise SystemExit(1) from exc


def detect_and_recommend():
    ensure_import_path()
    from app.config.runtime_validation import detect_hardware, recommend_config

    hardware = detect_hardware()
    config = recommend_config(hardware)
    return hardware, config
