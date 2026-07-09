"""
Runtime hardware validation — fail-fast checks before the worker consumes jobs.

Preflight validation runs before model warmup; post-warmup validation confirms
models loaded on the expected device/execution provider. Misconfigured GPU
deployments must exit with an actionable error instead of silently falling
back to CPU.
"""

from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from app.config.settings import Settings
from app.models.execution_providers import (
    COREML_EP,
    CPU_EP,
    CUDA_EP,
    OPENVINO_EP,
    QNN_EP,
    TENSORRT_EP,
    first_resolvable_provider,
    get_available_ort_providers,
)

logger = logging.getLogger(__name__)

ValidationPhase = Literal["preflight", "post_warmup"]

# Map profile execution_provider aliases to the ORT class name that must be primary.
_EP_PRIMARY: dict[str, str] = {
    "cpu": CPU_EP,
    "cuda": CUDA_EP,
    "tensorrt": TENSORRT_EP,
    "openvino": OPENVINO_EP,
    "coreml": COREML_EP,
    "qnn": QNN_EP,
}

_ARM_ARCHES = frozenset({"aarch64", "arm64", "armv8", "armv7l"})


class RuntimeValidationError(RuntimeError):
    """Raised when VISION_PROFILE does not match available hardware."""


@dataclass(frozen=True)
class HardwareInfo:
    """Snapshot of host hardware relevant to profile selection."""

    arch: str
    platform_system: str
    is_docker: bool
    image_variant: str
    cuda_available: bool
    ort_providers: tuple[str, ...]


def detect_hardware() -> HardwareInfo:
    """Probe the current host for CUDA, ORT providers, and container context."""
    variant = os.getenv("WORKER_IMAGE_VARIANT", "").strip().lower() or "unknown"
    is_docker = Path("/.dockerenv").exists() or os.getenv(
        "RUNNING_IN_DOCKER", ""
    ).lower() in ("1", "true", "yes")
    return HardwareInfo(
        arch=platform.machine().lower(),
        platform_system=platform.system(),
        is_docker=is_docker,
        image_variant=variant,
        cuda_available=torch.cuda.is_available(),
        ort_providers=tuple(get_available_ort_providers()),
    )


def recommend_profile(
    hardware: HardwareInfo | None = None,
) -> tuple[str, str, list[str]]:
    """
    Suggest a VISION_PROFILE and how to run it.

    Returns (profile_name, run_hint, warnings).
    """
    hw = hardware or detect_hardware()
    warnings: list[str] = []

    if hw.image_variant == "gpu" and hw.cuda_available:
        return (
            "gpu-standard",
            "docker compose -f docker-compose.gpu.yml up -d",
            warnings,
        )

    if hw.platform_system == "Darwin" and hw.arch in _ARM_ARCHES:
        if COREML_EP in hw.ort_providers:
            if hw.is_docker:
                warnings.append(
                    "CoreML is not available inside Linux containers — run Python natively on macOS."
                )
            return (
                "onnx-apple",
                "pip install -r requirements-torch.txt && python app/main.py",
                warnings,
            )
        warnings.append(
            "Apple Silicon detected but CoreML EP is not installed — using CPU ONNX profile."
        )
        return (
            "onnx-cpu-quality",
            "pip install -r requirements-torch.txt && python app/main.py",
            warnings,
        )

    if hw.arch in _ARM_ARCHES:
        return "onnx-cpu-quality", "docker compose up -d", warnings

    if OPENVINO_EP in hw.ort_providers:
        return (
            "onnx-intel",
            "docker compose up -d  # pip install onnxruntime-openvino",
            warnings,
        )

    if QNN_EP in hw.ort_providers:
        return (
            "onnx-qualcomm",
            "pip install onnxruntime-qnn && python app/main.py",
            warnings,
        )

    if hw.cuda_available:
        if hw.is_docker and hw.image_variant == "cpu":
            warnings.append(
                "CUDA is visible but WORKER_IMAGE_VARIANT=cpu — use docker-compose.gpu.yml."
            )
        return (
            "gpu-standard",
            "docker compose -f docker-compose.gpu.yml up -d",
            warnings,
        )

    return "cpu-quality", "docker compose up -d", warnings


def _requires_ep(profile_name: str, execution_provider: str) -> str | None:
    """Return the ORT provider class that must be primary, or None if CPU-only."""
    if execution_provider in _EP_PRIMARY:
        return _EP_PRIMARY[execution_provider]
    return None


def _validate_image_variant(settings: Settings, hardware: HardwareInfo) -> None:
    profile = settings.profile
    if not profile.requires_cuda:
        return
    if hardware.image_variant == "cpu":
        raise RuntimeValidationError(
            f"VISION_PROFILE={profile.name!r} requires CUDA but WORKER_IMAGE_VARIANT=cpu.\n"
            "You are running the CPU Docker image with a GPU profile.\n"
            "Fix: use docker-compose.gpu.yml and image tag :latest-gpu, "
            "or set VISION_PROFILE to a CPU profile (e.g. cpu-quality)."
        )


def _validate_torch_cuda(settings: Settings, hardware: HardwareInfo) -> None:
    profile = settings.profile
    if settings.runtime != "torch":
        return
    if profile.device != "cuda" and settings.device != "cuda":
        return
    if not hardware.cuda_available:
        raise RuntimeValidationError(
            f"VISION_PROFILE={profile.name!r} requires CUDA but torch.cuda.is_available() is False.\n"
            "Fix: deploy with docker-compose.gpu.yml on an NVIDIA GPU host, "
            "install NVIDIA drivers and the Container Toolkit, "
            "or switch to a CPU profile (e.g. cpu-quality)."
        )


def _validate_ort_ep_preflight(settings: Settings, hardware: HardwareInfo) -> None:
    if settings.runtime != "onnx":
        return

    profile = settings.profile
    ep_alias = settings.execution_provider

    if profile.name == "onnx-apple" and (
        hardware.is_docker or hardware.platform_system != "Darwin"
    ):
        raise RuntimeValidationError(
            f"VISION_PROFILE={profile.name!r} requires Apple CoreML on native macOS.\n"
            "CoreML is not available inside Linux Docker containers.\n"
            "Fix: run Python directly on an Apple Silicon Mac with onnx-apple."
        )

    required_primary = _requires_ep(profile.name, ep_alias)
    if required_primary is None:
        return

    available = set(hardware.ort_providers)
    if required_primary not in available:
        hints = {
            CUDA_EP: "Use docker-compose.gpu.yml / :latest-gpu and NVIDIA drivers.",
            TENSORRT_EP: "TensorRT EP requires the GPU image and TensorRT libraries.",
            OPENVINO_EP: "pip install onnxruntime-openvino",
            COREML_EP: "Run natively on macOS with CoreML support.",
            QNN_EP: "pip install onnxruntime-qnn on Qualcomm Windows ARM64.",
        }
        raise RuntimeValidationError(
            f"VISION_PROFILE={profile.name!r} requires {required_primary} "
            f"but available ORT providers are: {list(hardware.ort_providers)}.\n"
            f"Fix: {hints.get(required_primary, 'Install the matching ONNX Runtime EP package.')}"
        )

    # Ensure the EP chain would not silently degrade to CPU for GPU-bound aliases.
    if ep_alias in ("cuda", "tensorrt", "openvino", "coreml", "qnn"):
        primary = first_resolvable_provider(ep_alias, available)
        if primary == CPU_EP and ep_alias != "cpu":
            raise RuntimeValidationError(
                f"VISION_PROFILE={profile.name!r} requested execution_provider={ep_alias!r} "
                f"but only {CPU_EP} is resolvable.\n"
                "Fix: install the required execution provider package or choose a CPU profile."
            )


def _validate_post_warmup(settings: Settings, hardware: HardwareInfo) -> None:
    from app.models.registry import health_models

    health = health_models(settings)
    profile = settings.profile

    if settings.embed_enabled and not health.get("matchLoaded"):
        raise RuntimeValidationError(
            f"Embed model failed to load for VISION_PROFILE={profile.name!r}."
        )
    if settings.safety_enabled and not health.get("safetyLoaded"):
        raise RuntimeValidationError(
            f"Safety model failed to load for VISION_PROFILE={profile.name!r}."
        )

    if settings.runtime == "torch" and profile.requires_cuda:
        resolved = health.get("device", "")
        if resolved != "cuda":
            raise RuntimeValidationError(
                f"VISION_PROFILE={profile.name!r} requires CUDA but resolved device={resolved!r}."
            )

    if settings.runtime == "onnx":
        ep_alias = settings.execution_provider
        required_primary = _requires_ep(profile.name, ep_alias)
        if required_primary is None:
            return

        active = health.get("executionProvider") or ""
        active_chain = health.get("activeProviders") or []
        primary = active or (active_chain[0] if active_chain else "")

        if primary == CPU_EP and required_primary != CPU_EP:
            raise RuntimeValidationError(
                f"VISION_PROFILE={profile.name!r} expected primary ORT provider "
                f"{required_primary} but got {primary!r}.\n"
                "Fix: verify GPU drivers, EP packages, and WORKER_IMAGE_VARIANT."
            )


def validate_runtime(
    settings: Settings,
    hardware: HardwareInfo | None = None,
    *,
    phase: ValidationPhase = "preflight",
) -> None:
    """
    Validate that settings match hardware. Raises RuntimeValidationError on mismatch.

    phase="preflight"  — before model warmup (image variant, CUDA, EP availability).
    phase="post_warmup" — after warmup (models loaded, correct active provider).
    """
    hw = hardware or detect_hardware()
    profile = settings.profile

    # dedup-only and other non-ML profiles skip hardware validation.
    if not settings.embed_enabled and not settings.safety_enabled:
        return

    if phase == "preflight":
        _validate_image_variant(settings, hw)
        _validate_torch_cuda(settings, hw)
        _validate_ort_ep_preflight(settings, hw)

        if not profile.requires_cuda and hw.cuda_available:
            logger.info(
                "CUDA is available but VISION_PROFILE=%s is CPU-oriented — using CPU as configured.",
                profile.name,
            )
        return

    if phase == "post_warmup":
        _validate_post_warmup(settings, hw)


def format_hardware_summary(hardware: HardwareInfo) -> str:
    """One-line hardware summary for doctor CLI output."""
    return (
        f"arch={hardware.arch}, os={hardware.platform_system}, "
        f"docker={hardware.is_docker}, image={hardware.image_variant}, "
        f"cuda={hardware.cuda_available}, ort=[{', '.join(hardware.ort_providers)}]"
    )
