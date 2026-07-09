"""
Application settings loaded from environment variables.

``VISION_PROFILE`` is the primary knob — it loads a :class:`VisionProfile` preset
that already defines runtime, execution provider, precision, stages, and models.

Optional overrides (maintainers only — profiles already set these):
  INFERENCE_RUNTIME, ORT_EXECUTION_PROVIDER, MODEL_PRECISION, TORCH_COMPILE,
  DEVICE, BATCH_SIZE

See docs/GUIDE.md for when overrides are useful.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.config.profiles import PrecisionKind, RuntimeKind, VisionProfile, get_preset


def _default_hf_home() -> str:
    """Pick a writable Hugging Face cache directory for Docker vs local dev."""
    explicit = os.getenv("HF_HOME", "").strip()
    if explicit:
        return explicit
    if os.getenv("RUNNING_IN_DOCKER", "").lower() in ("1", "true", "yes"):
        return "/app/.cache/huggingface"
    if Path("/.dockerenv").exists():
        return "/app/.cache/huggingface"
    return str(Path.home() / ".cache" / "huggingface")


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
    # Inference backend settings.
    runtime: RuntimeKind
    execution_provider: str
    precision: PrecisionKind
    torch_compile: bool
    model_cache_dir: str
    tensorrt_cache_dir: str


def _resolve_device(profile: VisionProfile, override: str) -> str:
    if override == "auto":
        return profile.device
    return override


def _parse_bool(value: str, default: bool) -> bool:
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    return default


def load_settings() -> Settings:
    from app import __version__

    profile_name = os.getenv("VISION_PROFILE", "cpu-quality")
    profile = get_preset(profile_name)

    match_override = os.getenv("MATCH_MODEL", "").strip()
    safety_override = os.getenv("SAFETY_MODEL", "").strip()
    match_model = match_override or profile.match_model
    safety_model = safety_override or profile.safety_model

    # Stage toggles come from the profile preset (single knob: VISION_PROFILE).
    embed_enabled = profile.embed_enabled
    safety_enabled = profile.safety_enabled
    relevance_enabled = profile.relevance_enabled

    device = _resolve_device(profile, os.getenv("DEVICE", "auto"))
    batch_size = int(os.getenv("BATCH_SIZE", str(profile.batch_size or 1)))

    hf_home = _default_hf_home()

    runtime_override = os.getenv("INFERENCE_RUNTIME", "").strip().lower()
    runtime: RuntimeKind = (
        runtime_override if runtime_override in ("torch", "onnx") else profile.runtime
    )

    execution_provider = (
        os.getenv("ORT_EXECUTION_PROVIDER", profile.execution_provider).strip().lower()
    )

    precision_override = os.getenv("MODEL_PRECISION", "").strip().lower()
    precision: PrecisionKind = (
        precision_override
        if precision_override in ("fp32", "fp16", "int8")
        else profile.precision
    )

    torch_compile = _parse_bool(
        os.getenv("TORCH_COMPILE", str(profile.torch_compile)),
        profile.torch_compile,
    )

    model_cache_dir = os.getenv("MODEL_CACHE_DIR", f"{hf_home}/onnx")
    tensorrt_cache_dir = os.getenv("TENSORRT_CACHE_DIR", f"{hf_home}/tensorrt")

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
        model_cache_dir=model_cache_dir,
        tensorrt_cache_dir=tensorrt_cache_dir,
    )


settings = load_settings()
