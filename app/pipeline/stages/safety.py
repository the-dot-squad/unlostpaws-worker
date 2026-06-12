"""
Safety Classification Stage.

Dispatches a batch of PIL images to the active NSFW model classifier.
Offloads prediction calculations to a background thread pool executor
to avoid blocking the main asyncio event loop.
"""

from PIL import Image

from app.config.settings import Settings, settings
from app.models.registry import get_nsfw_classifier, run_in_executor
from app.schemas.result import SafetyResult


def run_safety(
    images: list[Image.Image], cfg: Settings = settings
) -> list[SafetyResult]:
    """
    Retrieves the singleton classifier, runs inference, and wraps results
    in SafetyResult schemas.
    """
    classifier = get_nsfw_classifier(cfg)
    if not classifier:
        return []
    # Execute model prediction on image batch
    preds = classifier.predict(images)
    return [
        SafetyResult(
            nsfwScore=p.nsfw_score,
            label=p.label,
            model=cfg.safety_model or "",
        )
        for p in preds
    ]


async def safety_stage(
    images: list[Image.Image], cfg: Settings = settings
) -> list[SafetyResult]:
    """
    Asynchronous entrypoint for the safety pipeline stage.
    """
    if not cfg.safety_enabled:
        return []
    # Offload the blocking CPU/GPU inference code to the thread executor
    return await run_in_executor(run_safety, images, cfg)
