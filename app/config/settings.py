"""
Environment configuration — loads VISION_PROFILE and shared worker secrets.

This module acts as the configuration hub for the worker application. It parses
environment variables, maps them to type-safe settings, merges profile-specific
presets, resolves execution devices (CPU vs. CUDA/GPU), and exposes a global
singleton `settings` object.
"""

import os
from dataclasses import dataclass

from app.config.profiles import VisionProfile, get_preset


@dataclass(frozen=True)
class Settings:
    """
    Immutable application configuration container.
    """

    # Version identifier derived from package metadata
    worker_version: str

    # Connection URL for the shared Redis instance
    redis_url: str

    # Stream queue where the worker consumes pet processing jobs
    stream_key: str

    # Dead Letter Queue stream where failed jobs are sent
    dlq_stream_key: str

    # Consumer group name registered for message tracking
    consumer_group: str

    # Unique name identifying this worker instance
    consumer_name: str

    # Maximum execution attempts before writing to the DLQ stream
    max_attempts: int

    # Active execution preset profile loaded from profiles registry
    profile: VisionProfile

    # Hugging Face model ID used for matching/embeddings (resolved or overridden)
    match_model: str | None

    # Hugging Face model ID used for NSFW classification (resolved or overridden)
    safety_model: str | None

    # Feature flag to enable/disable visual embedding vector stage
    embed_enabled: bool

    # Feature flag to enable/disable NSFW safety detection stage
    safety_enabled: bool

    # Feature flag to enable/disable zero-shot pet verification stage
    relevance_enabled: bool

    # Target execution device ('cpu' or 'cuda')
    device: str

    # Number of images processed concurrently during ML model inference
    batch_size: int

    # HTTP timeout limit in seconds for downloading image assets
    download_timeout: float

    # HTTP timeout limit in seconds when posting callbacks back to Next.js
    callback_timeout: float

    # Filesystem path where Hugging Face downloads and caches model parameters
    hf_home: str

    # Maximum number of concurrent HTTP image downloads
    max_concurrent_downloads: int


def _resolve_device(profile: VisionProfile, override: str) -> str:
    """
    Determines whether to execute models on CPU or GPU.
    If 'auto' is specified, it falls back to the preset profile's default device.
    """
    if override == "auto":
        return profile.device
    return override


def load_settings() -> Settings:
    """
    Loads environment configurations, resolves overrides, and constructs
    the global Settings instance.
    """
    from app import __version__

    # 1. Resolve Profile Name and fetch preset configurations
    profile_name = os.getenv("VISION_PROFILE", "cpu-quality")
    profile = get_preset(profile_name)

    # 2. Extract model override parameters if specified in .env
    match_override = os.getenv("MATCH_MODEL", "").strip()
    safety_override = os.getenv("SAFETY_MODEL", "").strip()
    match_model = match_override or profile.match_model
    safety_model = safety_override or profile.safety_model

    # 3. Parse features toggle flags
    embed_enabled = os.getenv("EMBED_ENABLED", "true").lower() == "true"
    safety_enabled = os.getenv("SAFETY_ENABLED", "true").lower() == "true"
    relevance_enabled = os.getenv("RELEVANCE_ENABLED", "true").lower() == "true"

    # 4. Apply profile specific constraints (e.g. disable stages on basic profiles)
    if profile_name == "dedup-only":
        embed_enabled = False
        safety_enabled = False
        relevance_enabled = False
    elif profile_name == "cpu-light":
        embed_enabled = False
        relevance_enabled = False

    # 5. Enforce profile hardware capabilities constraints
    if not profile.embed_enabled:
        embed_enabled = False
    if not profile.safety_enabled:
        safety_enabled = False
    if not profile.relevance_enabled:
        relevance_enabled = False

    # 6. Resolve execution device and batch sizes
    device = _resolve_device(profile, os.getenv("DEVICE", "auto"))
    batch_size = int(os.getenv("BATCH_SIZE", str(profile.batch_size or 1)))

    # 7. Construct and return configuration settings dataclass
    return Settings(
        worker_version=__version__,
        # Resolve Redis url check against upstash fallbacks
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
        hf_home=os.getenv("HF_HOME", "/app/.cache/huggingface"),
        max_concurrent_downloads=int(os.getenv("MAX_CONCURRENT_DOWNLOADS", "4")),
    )


# Instantiate the global configuration singleton
settings = load_settings()
