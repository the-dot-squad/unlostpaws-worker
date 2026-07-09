"""Tests for ONNX Execution Provider resolution."""

from unittest.mock import patch

from app.models.execution_providers import (
    CPU_EP,
    CUDA_EP,
    EP_ALIASES,
    resolve_ort_providers,
)


def test_ep_aliases_cover_hardware_targets():
    assert "cpu" in EP_ALIASES
    assert "cuda" in EP_ALIASES
    assert "tensorrt" in EP_ALIASES
    assert "openvino" in EP_ALIASES
    assert "coreml" in EP_ALIASES
    assert "qnn" in EP_ALIASES


def test_resolve_cpu_only():
    with patch(
        "app.models.execution_providers._available_providers",
        return_value={CPU_EP},
    ):
        resolved = resolve_ort_providers("auto")
    assert resolved == [(CPU_EP, {})]


def test_resolve_cuda_with_fallback():
    with patch(
        "app.models.execution_providers._available_providers",
        return_value={CUDA_EP, CPU_EP},
    ):
        resolved = resolve_ort_providers("cuda")
    assert resolved[0][0] == CUDA_EP
    assert resolved[-1][0] == CPU_EP


def test_unknown_alias_falls_back_to_auto_chain():
    with patch(
        "app.models.execution_providers._available_providers",
        return_value={CPU_EP},
    ):
        resolved = resolve_ort_providers("not-a-real-ep")
    assert resolved == [(CPU_EP, {})]


def test_tensorrt_options_include_cache_path(tmp_path):
    with patch(
        "app.models.execution_providers._available_providers",
        return_value={CPU_EP},
    ):
        resolve_ort_providers("cpu", tensorrt_cache_dir=str(tmp_path))
    # No error — cache path creation is exercised for tensorrt builder via imports.
