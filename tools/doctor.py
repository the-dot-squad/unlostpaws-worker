"""Hardware detection, profile recommendation, and preflight validation."""

from __future__ import annotations

import os
from importlib import reload

from tools._console import err, out
from tools._paths import ensure_import_path

# Estimated Hugging Face cache size on first model download (match + safety).
_PROFILE_DISK_HINT_MB: dict[str, int] = {
    "dedup-only": 0,
    "standard": 1024,
    "quality": 1536,
}


def _print_profile_resources(profile_name: str) -> None:
    from app.config.profiles import get_preset

    preset = get_preset(profile_name)
    disk_mb = _PROFILE_DISK_HINT_MB.get(profile_name, 0)
    out(f"Profile resources ({profile_name}):")
    out(f"  Min RAM: {preset.min_ram_mb} MB")
    if preset.min_vram_mb:
        out(f"  Min GPU VRAM (optional): {preset.min_vram_mb} MB")
    if disk_mb:
        out(f"  HF model cache (first run): ~{disk_mb / 1024:.1f} GB")


def print_report(hardware, config) -> None:
    from app.config.runtime_validation import format_hardware_summary

    out("UnLostPaws Vision Worker — hardware doctor")
    out("=" * 50)
    out(f"Hardware: {format_hardware_summary(hardware)}")
    out("Recommended configuration:")
    out(f"  VISION_PROFILE={config.vision_profile}")
    out(f"  INFERENCE_RUNTIME={config.inference_runtime}")
    if config.inference_runtime == "onnx":
        out(f"  ORT_EXECUTION_PROVIDER={config.execution_provider}")
    else:
        out(f"  DEVICE={config.device}")
    out(f"Run: {config.run_hint}")
    for warning in config.warnings:
        out(f"Warning: {warning}")
    out()
    out("First-run checklist:")
    out("  1. cp .env.example .env")
    out("  2. Edit REDIS_URL (use rediss:// for Upstash TLS)")
    out(f"  3. Set VISION_PROFILE={config.vision_profile}")
    out(f"  4. {config.run_hint}")


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
        out(f"Preflight OK for VISION_PROFILE={profile}")
        _print_profile_resources(profile)
    except RuntimeValidationError as exc:
        err(f"Preflight FAILED for VISION_PROFILE={profile}:\n{exc}")
        raise SystemExit(1) from exc


def detect_and_recommend():
    ensure_import_path()
    from app.config.runtime_validation import detect_hardware, recommend_config

    hardware = detect_hardware()
    config = recommend_config(hardware)
    return hardware, config
