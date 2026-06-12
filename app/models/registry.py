"""
Lazy Model Registry — manages model instantiation and device allocation.

This module acts as a registry to manage singleton instances of ML models
(NSFW Classifier, SigLIP2 Embedder). It isolates heavy Torch model loading
operations using a dedicated background ThreadPoolExecutor to prevent blocking
the main asyncio event loop during start-up or lazy execution.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import torch

from app.config.settings import Settings, settings
from app.models.nsfw import NsfwClassifier
from app.models.siglip import SiglipEmbedder

logger = logging.getLogger(__name__)

# Dedicated thread pool executor for running CPU-bound model parameters loading
# and heavy PyTorch inference calculations without blocking the async event loop.
_executor = ThreadPoolExecutor(max_workers=2)

# Singleton registry storage slots
_match_embedder: SiglipEmbedder | None = None
_nsfw_classifier: NsfwClassifier | None = None
_active_device: str | None = None


def resolve_torch_device(preferred: str) -> str:
    """
    Evaluates system hardware to determine if CUDA GPU acceleration is available.
    Falls back to 'cpu' if CUDA is unavailable or not requested.
    """
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_device(cfg: Settings = settings) -> str:
    """
    Internal helper to resolve and cache the active PyTorch hardware device.
    """
    global _active_device
    if _active_device is None:
        _active_device = resolve_torch_device(cfg.device)
    return _active_device


def get_match_embedder(cfg: Settings = settings) -> SiglipEmbedder | None:
    """
    Singleton retriever for the SigLIP2 image embedding/matching model.
    Instantiates the model wrapper class only if embedding feature is enabled.
    """
    global _match_embedder
    if not cfg.embed_enabled or not cfg.match_model:
        return None
    if _match_embedder is None:
        _match_embedder = SiglipEmbedder(
            cfg.match_model, _get_device(cfg), cfg.batch_size
        )
    return _match_embedder


def get_nsfw_classifier(cfg: Settings = settings) -> NsfwClassifier | None:
    """
    Singleton retriever for the NSFW safety classification model.
    Instantiates the classifier wrapper class only if safety checking is enabled.
    """
    global _nsfw_classifier
    if not cfg.safety_enabled or not cfg.safety_model:
        return None
    if _nsfw_classifier is None:
        _nsfw_classifier = NsfwClassifier(cfg.safety_model, _get_device(cfg))
    return _nsfw_classifier


async def run_in_executor(fn, *args):
    """
    Utility wrapper to run a blocking CPU/PyTorch function on the shared ThreadPoolExecutor.
    Awaits the completion of the execution asynchronously.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, fn, *args)


async def warmup(cfg: Settings = settings) -> None:
    """
    Triggers model parameter downloads and loads weights into memory.
    Runs on worker start-up to prevent latency spikes during the first job execution.
    """

    def _warm():
        # Retrieve and trigger load on the embedder
        embedder = get_match_embedder(cfg)
        if embedder:
            embedder.load()

        # Retrieve and trigger load on the classifier
        classifier = get_nsfw_classifier(cfg)
        if classifier:
            classifier.load()

    # Offload loading task to the ThreadPoolExecutor
    if cfg.embed_enabled or cfg.safety_enabled:
        await run_in_executor(_warm)
        logger.info(
            "Warmup complete profile=%s device=%s match=%s safety=%s",
            cfg.profile.name,
            _get_device(cfg),
            cfg.match_model,
            cfg.safety_model,
        )


def health_models(cfg: Settings = settings) -> dict:
    """
    Compiles diagnostic statuses of the active models for worker system health checks.
    """
    embedder = get_match_embedder(cfg)
    classifier = get_nsfw_classifier(cfg)
    return {
        "device": _get_device(cfg),
        "matchModel": cfg.match_model,
        "matchLoaded": embedder.is_loaded if embedder else False,
        "safetyModel": cfg.safety_model,
        "safetyLoaded": classifier.is_loaded if classifier else False,
        "relevanceEnabled": cfg.relevance_enabled,
    }


def get_executor() -> ThreadPoolExecutor:
    """
    Exposes the thread pool executor reference.
    """
    return _executor
