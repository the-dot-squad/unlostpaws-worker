"""Tests for runtime hardware validation and fail-fast behavior."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from app.config.profiles import get_preset
from app.config.runtime_validation import (
    HardwareInfo,
    RuntimeValidationError,
    detect_hardware,
    recommend_profile,
    validate_runtime,
)
from app.config.settings import Settings, load_settings
from app.models.execution_providers import CPU_EP, CUDA_EP, OPENVINO_EP
from app.models.factory import resolve_torch_device


def _settings(profile: str) -> Settings:
    os.environ["VISION_PROFILE"] = profile
    return load_settings()


def _hardware(
    *,
    arch: str = "x86_64",
    platform_system: str = "Linux",
    is_docker: bool = False,
    image_variant: str = "unknown",
    cuda_available: bool = False,
    ort_providers: tuple[str, ...] = (CPU_EP,),
) -> HardwareInfo:
    return HardwareInfo(
        arch=arch,
        platform_system=platform_system,
        is_docker=is_docker,
        image_variant=image_variant,
        cuda_available=cuda_available,
        ort_providers=ort_providers,
    )


def test_gpu_profile_fails_without_cuda(monkeypatch):
    monkeypatch.setenv("VISION_PROFILE", "gpu-standard")
    cfg = load_settings()
    hw = _hardware(cuda_available=False, image_variant="gpu")
    with pytest.raises(RuntimeValidationError, match="requires CUDA"):
        validate_runtime(cfg, hw, phase="preflight")


def test_cpu_profile_passes_without_cuda(monkeypatch):
    monkeypatch.setenv("VISION_PROFILE", "cpu-quality")
    cfg = load_settings()
    hw = _hardware(cuda_available=False, image_variant="cpu")
    validate_runtime(cfg, hw, phase="preflight")


def test_image_variant_mismatch(monkeypatch):
    monkeypatch.setenv("VISION_PROFILE", "gpu-standard")
    cfg = load_settings()
    hw = _hardware(cuda_available=True, image_variant="cpu")
    with pytest.raises(RuntimeValidationError, match="WORKER_IMAGE_VARIANT=cpu"):
        validate_runtime(cfg, hw, phase="preflight")


def test_onnx_gpu_fails_without_cuda_ep(monkeypatch):
    monkeypatch.setenv("VISION_PROFILE", "onnx-gpu-standard")
    cfg = load_settings()
    hw = _hardware(image_variant="gpu", ort_providers=(CPU_EP,))
    with pytest.raises(RuntimeValidationError, match="CUDAExecutionProvider"):
        validate_runtime(cfg, hw, phase="preflight")


def test_onnx_trt_fails_without_trt_ep(monkeypatch):
    monkeypatch.setenv("VISION_PROFILE", "onnx-trt-standard")
    cfg = load_settings()
    hw = _hardware(
        image_variant="gpu",
        cuda_available=True,
        ort_providers=(CUDA_EP, CPU_EP),
    )
    with pytest.raises(RuntimeValidationError, match="TensorrtExecutionProvider"):
        validate_runtime(cfg, hw, phase="preflight")


def test_onnx_apple_fails_in_linux_docker(monkeypatch):
    monkeypatch.setenv("VISION_PROFILE", "onnx-apple")
    cfg = load_settings()
    hw = _hardware(is_docker=True, platform_system="Linux", ort_providers=(CPU_EP,))
    with pytest.raises(RuntimeValidationError, match="CoreML"):
        validate_runtime(cfg, hw, phase="preflight")


def test_recommend_profile_arm64():
    hw = _hardware(arch="aarch64", ort_providers=(CPU_EP,))
    profile, _, _ = recommend_profile(hw)
    assert profile == "onnx-cpu-quality"


def test_recommend_profile_default_cpu():
    hw = _hardware(arch="x86_64", ort_providers=(CPU_EP,))
    profile, run_hint, _ = recommend_profile(hw)
    assert profile == "cpu-quality"
    assert "docker compose up" in run_hint


def test_recommend_profile_gpu_image_with_cuda():
    hw = _hardware(
        image_variant="gpu", cuda_available=True, ort_providers=(CUDA_EP, CPU_EP)
    )
    profile, run_hint, _ = recommend_profile(hw)
    assert profile == "gpu-standard"
    assert "docker-compose.gpu.yml" in run_hint


def test_recommend_profile_openvino():
    hw = _hardware(ort_providers=(OPENVINO_EP, CPU_EP))
    profile, _, _ = recommend_profile(hw)
    assert profile == "onnx-intel"


def test_resolve_torch_device_strict_raises():
    import torch

    torch.cuda.is_available.return_value = False
    with pytest.raises(RuntimeValidationError, match="torch.cuda.is_available"):
        resolve_torch_device("cuda")


def test_resolve_torch_device_cpu():
    import torch

    torch.cuda.is_available.return_value = False
    assert resolve_torch_device("cpu") == "cpu"


def test_resolve_torch_device_cuda_when_available():
    import torch

    torch.cuda.is_available.return_value = True
    assert resolve_torch_device("cuda") == "cuda"


def test_requires_cuda_property():
    assert get_preset("gpu-standard").requires_cuda is True
    assert get_preset("onnx-trt-standard").requires_cuda is True
    assert get_preset("cpu-quality").requires_cuda is False
    assert get_preset("onnx-cpu-quality").requires_cuda is False


def test_dedup_only_skips_validation(monkeypatch):
    monkeypatch.setenv("VISION_PROFILE", "dedup-only")
    cfg = load_settings()
    hw = _hardware(image_variant="cpu")
    validate_runtime(cfg, hw, phase="preflight")


def test_post_warmup_fails_when_models_not_loaded(monkeypatch):
    monkeypatch.setenv("VISION_PROFILE", "cpu-quality")
    cfg = load_settings()
    hw = _hardware(image_variant="cpu")

    fake_health = {
        "device": "cpu",
        "matchLoaded": False,
        "safetyLoaded": True,
        "executionProvider": CPU_EP,
        "activeProviders": [CPU_EP],
    }
    with patch("app.models.registry.health_models", return_value=fake_health):
        with pytest.raises(RuntimeValidationError, match="Embed model failed"):
            validate_runtime(cfg, hw, phase="post_warmup")


def test_first_resolvable_provider():
    from app.models.execution_providers import first_resolvable_provider

    assert first_resolvable_provider("cuda", {CUDA_EP, CPU_EP}) == CUDA_EP
    assert first_resolvable_provider("cuda", {CPU_EP}) == CPU_EP


def test_detect_hardware_reads_image_variant(monkeypatch):
    monkeypatch.setenv("WORKER_IMAGE_VARIANT", "cpu")
    monkeypatch.delenv("RUNNING_IN_DOCKER", raising=False)
    with patch("pathlib.Path.exists", return_value=False):
        hw = detect_hardware()
    assert hw.image_variant == "cpu"
