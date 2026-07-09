"""
ONNX Runtime Execution Provider (EP) resolution.

An **Execution Provider** is ONNX Runtime's plugin for a hardware backend
(CPU, NVIDIA CUDA, TensorRT, Intel OpenVINO, Apple CoreML, Qualcomm QNN).

``VisionProfile.execution_provider`` stores a short alias (e.g. ``cuda``,
``openvino``). This module maps aliases to ORT provider names, attaches
per-vendor options, and builds a fallback chain ending at CPU if a plugin
is not installed on the host.

Only used when ``Settings.runtime == "onnx"``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Canonical ORT provider class names.
CUDA_EP = "CUDAExecutionProvider"
CPU_EP = "CPUExecutionProvider"
TENSORRT_EP = "TensorrtExecutionProvider"
OPENVINO_EP = "OpenVINOExecutionProvider"
COREML_EP = "CoreMLExecutionProvider"
QNN_EP = "QNNExecutionProvider"

EP_ALIASES: dict[str, list[str]] = {
    "auto": [
        TENSORRT_EP,
        CUDA_EP,
        OPENVINO_EP,
        COREML_EP,
        QNN_EP,
        CPU_EP,
    ],
    "cpu": [CPU_EP],
    "cuda": [CUDA_EP, CPU_EP],
    "tensorrt": [TENSORRT_EP, CUDA_EP, CPU_EP],
    "openvino": [OPENVINO_EP, CPU_EP],
    "coreml": [COREML_EP, CPU_EP],
    "qnn": [QNN_EP, CPU_EP],
}

_last_resolved_providers: list[str] = []
_last_active_provider: str = CPU_EP


def get_last_resolved_providers() -> list[str]:
    return list(_last_resolved_providers)


def get_active_provider() -> str:
    return _last_active_provider


def _available_providers() -> set[str]:
    try:
        import onnxruntime as ort

        return set(ort.get_available_providers())
    except ImportError:
        logger.warning("onnxruntime not installed — ONNX backend unavailable")
        return {CPU_EP}


def get_available_ort_providers() -> list[str]:
    """Return installed ONNX Runtime execution provider names."""
    return sorted(_available_providers())


def first_resolvable_provider(alias: str, available: set[str] | None = None) -> str:
    """
    Return the first provider name from an alias chain that is installed.

    Used by startup validation to detect silent CPU degradation before inference.
    """
    providers = available if available is not None else _available_providers()
    chain = EP_ALIASES.get(alias.strip().lower(), EP_ALIASES["auto"])
    for provider in chain:
        if provider in providers:
            return provider
    return CPU_EP


def build_openvino_options(device_type: str = "CPU") -> dict[str, Any]:
    """OpenVINO EP options — use NPU on Intel Meteor Lake+ when available."""
    return {"device_type": device_type}


def build_coreml_options() -> dict[str, Any]:
    """CoreML EP options — prefer Apple Neural Engine when present."""
    return {"MLComputeUnits": "CPUAndNeuralEngine"}


def build_qnn_options() -> dict[str, Any]:
    """QNN EP options — Hexagon NPU via HTP backend."""
    return {"backend_type": "htp"}


def build_tensorrt_options(cache_dir: str) -> dict[str, Any]:
    """TensorRT EP options with persistent engine cache."""
    os.makedirs(cache_dir, exist_ok=True)
    return {
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": cache_dir,
    }


def build_cuda_options() -> dict[str, Any]:
    return {}


def _provider_options(
    provider: str,
    *,
    tensorrt_cache_dir: str,
    openvino_device: str = "CPU",
) -> dict[str, Any]:
    if provider == TENSORRT_EP:
        return build_tensorrt_options(tensorrt_cache_dir)
    if provider == OPENVINO_EP:
        return build_openvino_options(openvino_device)
    if provider == COREML_EP:
        return build_coreml_options()
    if provider == QNN_EP:
        return build_qnn_options()
    if provider == CUDA_EP:
        return build_cuda_options()
    return {}


def resolve_ort_providers(
    requested: str,
    *,
    tensorrt_cache_dir: str = "/app/.cache/tensorrt",
    openvino_device: str = "CPU",
) -> list[tuple[str, dict[str, Any]]]:
    """
    Build an ordered list of (provider_name, options) for ORT session creation.

    Providers not installed on the host are skipped with a debug log.
    """
    global _last_resolved_providers, _last_active_provider

    alias = requested.strip().lower()
    chain = EP_ALIASES.get(alias, EP_ALIASES["auto"])
    available = _available_providers()

    resolved: list[tuple[str, dict[str, Any]]] = []
    for provider in chain:
        if provider not in available:
            logger.debug("ORT provider unavailable, skipping: %s", provider)
            continue
        opts = _provider_options(
            provider,
            tensorrt_cache_dir=tensorrt_cache_dir,
            openvino_device=openvino_device,
        )
        resolved.append((provider, opts))

    if not resolved:
        resolved = [(CPU_EP, {})]

    _last_resolved_providers = [name for name, _ in resolved]
    _last_active_provider = _last_resolved_providers[0]
    logger.info("Resolved ORT providers: %s", " -> ".join(_last_resolved_providers))
    return resolved
