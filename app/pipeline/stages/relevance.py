"""
Zero-Shot Pet Relevance Verification Stage.

Analyzes if the uploaded image matches standard pet characteristics
and validates target classifications (e.g., dog, cat) using SigLIP2.
Gated by RELEVANCE_ENABLED setting.
"""

from PIL import Image

from app.config.settings import Settings, settings
from app.models.registry import get_match_embedder, run_in_executor
from app.schemas.result import RelevanceResult


async def relevance_stage(
    images: list[Image.Image], pet_type: str = "", cfg: Settings = settings
) -> list[RelevanceResult]:
    """
    Asynchronous entrypoint for pet relevance check.

    If disabled or if the active embedder model does not support relevance checks
    (e.g., DINOv2-small), returns empty array.
    """
    if not cfg.relevance_enabled or not cfg.embed_enabled:
        return []

    # Retrieve match model embedder singleton (e.g., SigLIP2)
    embedder = get_match_embedder(cfg)
    if not embedder or not getattr(embedder, "supports_relevance", False):
        return []

    def _run():
        preds = embedder.relevance_batch(images, pet_type)
        return [
            RelevanceResult(petLikelihood=p.pet_likelihood, topLabel=p.top_label)
            for p in preds
        ]

    # Offload zero-shot text-image calculations to ThreadPoolExecutor
    return await run_in_executor(_run)
