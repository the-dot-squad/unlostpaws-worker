"""
Lazy Model Registry — manages model instantiation and device allocation.

Selects torch or ONNX backends via the factory based on VISION_PROFILE settings.
Heavy inference runs on a dedicated ThreadPoolExecutor to avoid blocking asyncio.
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from app.config.settings import Settings, settings
from app.models.execution_providers import (
    get_active_provider,
    get_last_resolved_providers,
)
from app.models.factory import create_classifier, create_embedder, resolve_torch_device

__all__ = ["resolve_torch_device", "create_embedder", "create_classifier"]
from app.models.protocols import ClassifierBackend, EmbedderBackend

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=max(2, os.cpu_count() or 2))

_match_embedder: EmbedderBackend | None = None
_nsfw_classifier: ClassifierBackend | None = None
_active_device: str | None = None


def _get_device(cfg: Settings = settings) -> str:
    global _active_device
    if _active_device is None:
        if cfg.runtime == "onnx":
            _active_device = cfg.execution_provider
        else:
            _active_device = resolve_torch_device(cfg.device)
    return _active_device


def get_match_embedder(cfg: Settings = settings) -> EmbedderBackend | None:
    global _match_embedder
    if not cfg.embed_enabled or not cfg.match_model:
        return None
    if _match_embedder is None:
        _match_embedder = create_embedder(cfg)
    return _match_embedder


def get_nsfw_classifier(cfg: Settings = settings) -> ClassifierBackend | None:
    global _nsfw_classifier
    if not cfg.safety_enabled or not cfg.safety_model:
        return None
    if _nsfw_classifier is None:
        _nsfw_classifier = create_classifier(cfg)
    return _nsfw_classifier


async def run_in_executor(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, fn, *args)


async def warmup(cfg: Settings = settings) -> None:
    """Download/load models and log resolved backend configuration."""

    def _warm():
        embedder = get_match_embedder(cfg)
        if embedder:
            embedder.load()

        classifier = get_nsfw_classifier(cfg)
        if classifier:
            classifier.load()

    if cfg.embed_enabled or cfg.safety_enabled:
        await run_in_executor(_warm)
        logger.info(
            "Warmup complete profile=%s runtime=%s device=%s precision=%s "
            "match=%s safety=%s active_ep=%s",
            cfg.profile.name,
            cfg.runtime,
            _get_device(cfg),
            cfg.precision,
            cfg.match_model,
            cfg.safety_model,
            get_active_provider() if cfg.runtime == "onnx" else "n/a",
        )


def health_models(cfg: Settings = settings) -> dict:
    embedder = get_match_embedder(cfg)
    classifier = get_nsfw_classifier(cfg)

    active_ep = ""
    if cfg.runtime == "onnx":
        if embedder and hasattr(embedder, "active_provider"):
            active_ep = embedder.active_provider
        elif classifier and hasattr(classifier, "active_provider"):
            active_ep = classifier.active_provider
        else:
            active_ep = get_active_provider()

    return {
        "device": _get_device(cfg),
        "runtime": cfg.runtime,
        "executionProvider": active_ep,
        "precision": cfg.precision,
        "activeProviders": get_last_resolved_providers()
        if cfg.runtime == "onnx"
        else [],
        "torchCompile": cfg.torch_compile,
        "matchModel": cfg.match_model,
        "matchLoaded": embedder.is_loaded if embedder else False,
        "safetyModel": cfg.safety_model,
        "safetyLoaded": classifier.is_loaded if classifier else False,
        "relevanceEnabled": cfg.relevance_enabled,
    }


def get_executor() -> ThreadPoolExecutor:
    return _executor
