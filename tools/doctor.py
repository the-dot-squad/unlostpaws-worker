"""Hardware detection, profile recommendation, and preflight validation."""

from __future__ import annotations

import os
from importlib import reload

from tools._console import err, out
from tools._paths import ensure_import_path


def print_report(hardware, profile: str, run_hint: str, warnings: list[str]) -> None:
    from app.config.runtime_validation import format_hardware_summary

    out("UnLostPaws Vision Worker — hardware doctor")
    out("=" * 50)
    out(f"Hardware: {format_hardware_summary(hardware)}")
    out(f"Recommended profile: VISION_PROFILE={profile}")
    out(f"Run: {run_hint}")
    for warning in warnings:
        out(f"Warning: {warning}")
    out()
    out("First-run checklist:")
    out("  1. cp .env.example .env")
    out("  2. Edit REDIS_URL (use rediss:// for Upstash TLS)")
    out(f"  3. Set VISION_PROFILE={profile}  (GPU compose bakes this in)")
    out(f"  4. {run_hint}")


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
    except RuntimeValidationError as exc:
        err(f"Preflight FAILED for VISION_PROFILE={profile}:\n{exc}")
        raise SystemExit(1) from exc


def detect_and_recommend() -> tuple[object, str, str, list[str]]:
    ensure_import_path()
    from app.config.runtime_validation import detect_hardware, recommend_profile

    hardware = detect_hardware()
    profile, run_hint, warnings = recommend_profile(hardware)
    return hardware, profile, run_hint, warnings
