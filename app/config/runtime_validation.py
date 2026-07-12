"""
Runtime hardware validation — fail-fast checks before the worker consumes jobs.
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
    """Raised when runtime env does not match available hardware."""


@dataclass(frozen=True)
class HardwareInfo:
    arch: str
    platform_system: str
    is_docker: bool
    image_variant: str
    cuda_available: bool
    ort_providers: tuple[str, ...]


@dataclass(frozen=True)
class RecommendedConfig:
    vision_profile: str
    inference_runtime: str
    device: str
    execution_provider: str
    run_hint: str
    warnings: tuple[str, ...]


def detect_hardware() -> HardwareInfo:
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


def recommend_config(hardware: HardwareInfo | None = None) -> RecommendedConfig:
    hw = hardware or detect_hardware()
    warnings: list[str] = []

    if hw.image_variant == "gpu" and hw.cuda_available:
        return RecommendedConfig(
            vision_profile="quality",
            inference_runtime="torch",
            device="cuda",
            execution_provider="cuda",
            run_hint="docker compose -f docker-compose.gpu.yml up -d",
            warnings=tuple(warnings),
        )

    if hw.platform_system == "Darwin" and hw.arch in _ARM_ARCHES:
        if COREML_EP in hw.ort_providers:
            if hw.is_docker:
                warnings.append(
                    "CoreML is not available inside Linux containers — run Python natively on macOS."
                )
                return RecommendedConfig(
                    vision_profile="quality",
                    inference_runtime="onnx",
                    device="cpu",
                    execution_provider="cpu",
                    run_hint="docker compose up -d",
                    warnings=tuple(warnings),
                )
            return RecommendedConfig(
                vision_profile="quality",
                inference_runtime="onnx",
                device="cpu",
                execution_provider="coreml",
                run_hint="pip install -e '.[dev]' && python app/main.py",
                warnings=tuple(warnings),
            )
        warnings.append(
            "Apple Silicon detected but CoreML EP is not installed — using CPU ONNX."
        )
        return RecommendedConfig(
            vision_profile="quality",
            inference_runtime="onnx",
            device="cpu",
            execution_provider="cpu",
            run_hint="pip install -e '.[dev]' && python app/main.py",
            warnings=tuple(warnings),
        )

    if hw.arch in _ARM_ARCHES:
        return RecommendedConfig(
            vision_profile="quality",
            inference_runtime="onnx",
            device="cpu",
            execution_provider="cpu",
            run_hint="docker compose up -d",
            warnings=tuple(warnings),
        )

    if OPENVINO_EP in hw.ort_providers and not hw.is_docker:
        return RecommendedConfig(
            vision_profile="standard",
            inference_runtime="onnx",
            device="cpu",
            execution_provider="openvino",
            run_hint="pip install onnxruntime-openvino && python app/main.py",
            warnings=tuple(warnings),
        )

    if QNN_EP in hw.ort_providers:
        return RecommendedConfig(
            vision_profile="quality",
            inference_runtime="onnx",
            device="cpu",
            execution_provider="qnn",
            run_hint="pip install onnxruntime-qnn && python app/main.py",
            warnings=tuple(warnings),
        )

    if hw.cuda_available:
        if hw.is_docker and hw.image_variant == "cpu":
            warnings.append(
                "CUDA is visible but WORKER_IMAGE_VARIANT=cpu — use docker-compose.gpu.yml."
            )
        return RecommendedConfig(
            vision_profile="quality",
            inference_runtime="torch",
            device="cuda",
            execution_provider="cuda",
            run_hint="docker compose -f docker-compose.gpu.yml up -d",
            warnings=tuple(warnings),
        )

    return RecommendedConfig(
        vision_profile="quality",
        inference_runtime="torch",
        device="cpu",
        execution_provider="cpu",
        run_hint="docker compose up -d",
        warnings=tuple(warnings),
    )


def recommend_profile(
    hardware: HardwareInfo | None = None,
) -> tuple[str, str, list[str]]:
    """Backward-compatible wrapper returning (profile_name, run_hint, warnings)."""
    cfg = recommend_config(hardware)
    return cfg.vision_profile, cfg.run_hint, list(cfg.warnings)


def _requires_ep(execution_provider: str) -> str | None:
    return _EP_PRIMARY.get(execution_provider)


def _validate_image_variant(settings: Settings, hardware: HardwareInfo) -> None:
    if not settings.requires_cuda:
        return
    if hardware.image_variant == "cpu":
        raise RuntimeValidationError(
            "CUDA or TensorRT requested but WORKER_IMAGE_VARIANT=cpu.\n"
            "You are running the CPU Docker image with GPU settings.\n"
            "Fix: use docker-compose.gpu.yml and image tag :latest-gpu, "
            "or set DEVICE=cpu and ORT_EXECUTION_PROVIDER=cpu."
        )


def _validate_torch_cuda(settings: Settings, hardware: HardwareInfo) -> None:
    if settings.runtime != "torch":
        return
    if settings.device != "cuda":
        return
    if not hardware.cuda_available:
        raise RuntimeValidationError(
            "DEVICE=cuda requested but torch.cuda.is_available() is False.\n"
            "Fix: deploy with docker-compose.gpu.yml on an NVIDIA GPU host, "
            "or set DEVICE=cpu."
        )


def _validate_ort_ep_preflight(settings: Settings, hardware: HardwareInfo) -> None:
    if settings.runtime != "onnx":
        return

    ep_alias = settings.execution_provider

    if ep_alias == "coreml" and (
        hardware.is_docker or hardware.platform_system != "Darwin"
    ):
        raise RuntimeValidationError(
            "ORT_EXECUTION_PROVIDER=coreml requires Apple CoreML on native macOS.\n"
            "CoreML is not available inside Linux Docker containers.\n"
            "Fix: run Python directly on an Apple Silicon Mac, or use ORT_EXECUTION_PROVIDER=cpu."
        )

    if ep_alias in ("cuda", "tensorrt") and not hardware.cuda_available:
        raise RuntimeValidationError(
            f"ORT_EXECUTION_PROVIDER={ep_alias!r} requires CUDA but "
            "torch.cuda.is_available() is False.\n"
            "Fix: use docker-compose.gpu.yml on an NVIDIA GPU host, "
            "or set ORT_EXECUTION_PROVIDER=cpu."
        )

    required_primary = _requires_ep(ep_alias)
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
            f"ORT_EXECUTION_PROVIDER={ep_alias!r} requires {required_primary} "
            f"but available ORT providers are: {list(hardware.ort_providers)}.\n"
            f"Fix: {hints.get(required_primary, 'Install the matching ONNX Runtime EP package.')}"
        )

    if ep_alias in ("cuda", "tensorrt", "openvino", "coreml", "qnn"):
        primary = first_resolvable_provider(ep_alias, available)
        if primary == CPU_EP and ep_alias != "cpu":
            raise RuntimeValidationError(
                f"ORT_EXECUTION_PROVIDER={ep_alias!r} requested but only "
                f"{CPU_EP} is resolvable.\n"
                "Fix: install the required execution provider package or use cpu."
            )


def _validate_post_warmup(settings: Settings, hardware: HardwareInfo) -> None:
    from app.models.registry import health_models

    health = health_models(settings)

    if settings.embed_enabled and not health.get("matchLoaded"):
        raise RuntimeValidationError(
            f"Embed model failed to load for VISION_PROFILE={settings.profile.name!r}."
        )
    if settings.safety_enabled and not health.get("safetyLoaded"):
        raise RuntimeValidationError(
            f"Safety model failed to load for VISION_PROFILE={settings.profile.name!r}."
        )

    if settings.runtime == "torch" and settings.requires_cuda:
        resolved = health.get("device", "")
        if resolved != "cuda":
            raise RuntimeValidationError(
                f"CUDA required but resolved torch device={resolved!r}."
            )

    if settings.runtime == "onnx":
        ep_alias = settings.execution_provider
        required_primary = _requires_ep(ep_alias)
        if required_primary is None:
            return

        active = health.get("executionProvider") or ""
        active_chain = health.get("activeProviders") or []
        primary = active or (active_chain[0] if active_chain else "")

        if primary == CPU_EP and required_primary != CPU_EP:
            raise RuntimeValidationError(
                f"Expected primary ORT provider {required_primary} but got {primary!r}.\n"
                "Fix: verify GPU drivers, EP packages, and WORKER_IMAGE_VARIANT."
            )


def validate_runtime(
    settings: Settings,
    hardware: HardwareInfo | None = None,
    *,
    phase: ValidationPhase = "preflight",
) -> None:
    hw = hardware or detect_hardware()

    if not settings.embed_enabled and not settings.safety_enabled:
        return

    if phase == "preflight":
        _validate_image_variant(settings, hw)
        _validate_torch_cuda(settings, hw)
        _validate_ort_ep_preflight(settings, hw)

        if not settings.requires_cuda and hw.cuda_available:
            logger.info(
                "CUDA is available but configured for CPU — using CPU as configured "
                "(profile=%s runtime=%s ep=%s).",
                settings.profile.name,
                settings.runtime,
                settings.execution_provider,
            )
        return

    if phase == "post_warmup":
        _validate_post_warmup(settings, hw)


def format_hardware_summary(hardware: HardwareInfo) -> str:
    return (
        f"arch={hardware.arch}, os={hardware.platform_system}, "
        f"docker={hardware.is_docker}, image={hardware.image_variant}, "
        f"cuda={hardware.cuda_available}, ort=[{', '.join(hardware.ort_providers)}]"
    )
