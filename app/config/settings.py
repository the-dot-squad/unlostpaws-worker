"""
Application settings loaded from environment variables.

``VISION_PROFILE`` selects capability (dedup / standard / quality).
Hardware/runtime is configured separately:

  INFERENCE_RUNTIME=torch|onnx
  DEVICE=cpu|cuda           (torch)
  ORT_EXECUTION_PROVIDER=cpu|cuda|tensorrt|coreml|openvino|qnn  (onnx)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.config.profiles import VisionProfile, get_preset

RuntimeKind = Literal["torch", "onnx"]
PrecisionKind = Literal["fp32", "fp16", "int8"]

_GPU_EPS = frozenset({"cuda", "tensorrt"})


def _default_hf_home() -> str:
    explicit = os.getenv("HF_HOME", "").strip()
    if explicit:
        return explicit
    if os.getenv("RUNNING_IN_DOCKER", "").lower() in ("1", "true", "yes"):
        return "/app/.cache/huggingface"
    if Path("/.dockerenv").exists():
        return "/app/.cache/huggingface"
    return str(Path.home() / ".cache" / "huggingface")


def default_precision_for_ep(execution_provider: str) -> PrecisionKind:
    if execution_provider in ("cuda", "tensorrt", "coreml"):
        return "fp16"
    if execution_provider in ("cpu", "openvino", "qnn"):
        return "int8"
    return "fp32"


def default_batch_size(
    profile: VisionProfile, device: str, runtime: RuntimeKind
) -> int:
    if not profile.embed_enabled:
        return 0
    if runtime == "onnx":
        return 4 if device == "cuda" or profile.name == "quality" else 2
    return 4 if device == "cuda" else 1


@dataclass(frozen=True)
class Settings:
    """Immutable application configuration container."""

    worker_version: str
    redis_url: str
    stream_key: str
    dlq_stream_key: str
    consumer_group: str
    consumer_name: str
    max_attempts: int
    profile: VisionProfile
    match_model: str | None
    safety_model: str | None
    embed_enabled: bool
    safety_enabled: bool
    relevance_enabled: bool
    device: str
    batch_size: int
    download_timeout: float
    callback_timeout: float
    hf_home: str
    max_concurrent_downloads: int
    runtime: RuntimeKind
    execution_provider: str
    precision: PrecisionKind
    torch_compile: bool
    model_cache_dir: str
    tensorrt_cache_dir: str
    openvino_device: str
    relevance_formulation: Literal["baseline", "unified_softmax"]
    relevance_temp_scale: float
    relevance_threshold: float
    relevance_margin_threshold: float
    max_image_pixels: int

    @property
    def requires_cuda(self) -> bool:
        if self.runtime == "torch":
            return self.device == "cuda"
        return self.execution_provider in _GPU_EPS


def _parse_bool(value: str, default: bool) -> bool:
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    return default


def _resolve_torch_device(device_override: str, execution_provider: str) -> str:
    if device_override in ("cpu", "cuda"):
        return device_override
    if execution_provider in _GPU_EPS:
        return "cuda"
    return "cpu"


def load_settings() -> Settings:
    from app import __version__

    profile_name = os.getenv("VISION_PROFILE", "quality")
    profile = get_preset(profile_name)

    runtime_raw = os.getenv("INFERENCE_RUNTIME", "torch").strip().lower()
    runtime: RuntimeKind = runtime_raw if runtime_raw in ("torch", "onnx") else "torch"

    execution_provider = os.getenv("ORT_EXECUTION_PROVIDER", "cpu").strip().lower()
    if execution_provider == "auto":
        execution_provider = "cpu"

    device_override = os.getenv("DEVICE", "auto").strip().lower()
    if runtime == "torch":
        device = _resolve_torch_device(device_override, execution_provider)
    else:
        device = "cuda" if execution_provider in _GPU_EPS else "cpu"

    precision_override = os.getenv("MODEL_PRECISION", "").strip().lower()
    precision: PrecisionKind = (
        precision_override
        if precision_override in ("fp32", "fp16", "int8")
        else default_precision_for_ep(execution_provider)
    )

    batch_override = os.getenv("BATCH_SIZE", "").strip()
    batch_size = (
        int(batch_override)
        if batch_override
        else default_batch_size(profile, device, runtime)
    )

    match_override = os.getenv("MATCH_MODEL", "").strip()
    safety_override = os.getenv("SAFETY_MODEL", "").strip()
    match_model = match_override or profile.match_model
    safety_model = safety_override or profile.safety_model

    embed_enabled = profile.embed_enabled
    safety_enabled = profile.safety_enabled
    relevance_enabled = profile.relevance_enabled

    torch_compile = _parse_bool(
        os.getenv("TORCH_COMPILE", str(profile.default_torch_compile)),
        profile.default_torch_compile,
    )
    if runtime == "onnx":
        torch_compile = False

    hf_home = _default_hf_home()
    openvino_device = os.getenv(
        "OPENVINO_DEVICE", "NPU" if execution_provider == "openvino" else "CPU"
    )

    relevance_formulation = (
        os.getenv("RELEVANCE_FORMULATION", "unified_softmax").strip().lower()
    )
    if relevance_formulation not in ("baseline", "unified_softmax"):
        relevance_formulation = "unified_softmax"

    relevance_temp_scale_raw = os.getenv("RELEVANCE_TEMP_SCALE", "").strip()
    relevance_temp_scale = (
        float(relevance_temp_scale_raw) if relevance_temp_scale_raw else 1.5
    )

    relevance_threshold_raw = os.getenv("RELEVANCE_THRESHOLD", "").strip()
    if relevance_threshold_raw:
        relevance_threshold = float(relevance_threshold_raw)
    else:
        relevance_threshold = (
            0.32 if relevance_formulation == "unified_softmax" else 0.30
        )

    relevance_margin_threshold_raw = os.getenv("RELEVANCE_MARGIN_THRESHOLD", "").strip()
    if relevance_margin_threshold_raw:
        relevance_margin_threshold = float(relevance_margin_threshold_raw)
    else:
        relevance_margin_threshold = (
            0.40 if relevance_formulation == "unified_softmax" else 0.75
        )

    return Settings(
        worker_version=__version__,
        redis_url=os.getenv("REDIS_URL") or os.getenv("UPSTASH_REDIS_URL") or "",
        stream_key=os.getenv("STREAM_KEY", "unlostpaws:stream:vision-processing"),
        dlq_stream_key=os.getenv(
            "DLQ_STREAM_KEY", "unlostpaws:stream:vision-processing:dlq"
        ),
        consumer_group=os.getenv("CONSUMER_GROUP", "vision-worker"),
        consumer_name=os.getenv("CONSUMER_NAME", "worker-1"),
        max_attempts=int(os.getenv("MAX_JOB_ATTEMPTS", "3")),
        profile=profile,
        match_model=match_model if embed_enabled else None,
        safety_model=safety_model if safety_enabled else None,
        embed_enabled=embed_enabled,
        safety_enabled=safety_enabled,
        relevance_enabled=relevance_enabled,
        device=device,
        batch_size=batch_size,
        download_timeout=float(os.getenv("DOWNLOAD_TIMEOUT", "30")),
        callback_timeout=float(os.getenv("CALLBACK_TIMEOUT", "60")),
        hf_home=hf_home,
        max_concurrent_downloads=int(os.getenv("MAX_CONCURRENT_DOWNLOADS", "4")),
        runtime=runtime,
        execution_provider=execution_provider,
        precision=precision,
        torch_compile=torch_compile,
        model_cache_dir=os.getenv("MODEL_CACHE_DIR", f"{hf_home}/onnx"),
        tensorrt_cache_dir=os.getenv("TENSORRT_CACHE_DIR", f"{hf_home}/tensorrt"),
        openvino_device=openvino_device,
        relevance_formulation=relevance_formulation,
        relevance_temp_scale=relevance_temp_scale,
        relevance_threshold=relevance_threshold,
        relevance_margin_threshold=relevance_margin_threshold,
        max_image_pixels=int(os.getenv("MAX_IMAGE_PIXELS", "89478485")),
    )


settings = load_settings()
