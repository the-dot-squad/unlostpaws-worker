"""
Fused SigLIP match stage — embedding and relevance in one forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from app.config.settings import Settings, settings
from app.models.registry import get_match_embedder, run_in_executor
from app.schemas.result import RelevanceResult


@dataclass(frozen=True)
class MatchStageResult:
    embedding: list[float]
    relevance: RelevanceResult | None = None


async def match_stage(
    images: list[Image.Image],
    pet_type: str = "",
    cfg: Settings = settings,
) -> list[MatchStageResult]:
    """Run fused SigLIP inference for embeddings and optional relevance.

    ``pet_type`` is an optional species hint from the job payload (``petType``).
    Empty string runs zero-shot; known hints stabilize ``topLabel`` when uncertain.
    """
    if not cfg.embed_enabled:
        return [MatchStageResult(embedding=[]) for _ in images]

    embedder = get_match_embedder(cfg)
    if not embedder:
        return [MatchStageResult(embedding=[]) for _ in images]

    include_relevance = cfg.relevance_enabled and getattr(
        embedder, "supports_relevance", False
    )

    def _run() -> list[MatchStageResult]:
        if hasattr(embedder, "match_batch"):
            preds = embedder.match_batch(
                images, pet_type, include_relevance=include_relevance
            )
        else:
            embed_preds = embedder.embed_batch(images)
            rel_preds = (
                embedder.relevance_batch(images, pet_type)
                if include_relevance
                else [None] * len(images)
            )
            from app.models.types import MatchPrediction

            preds = [
                MatchPrediction(
                    embedding=embed_pred.embedding,
                    relevance=rel_pred,
                )
                for embed_pred, rel_pred in zip(embed_preds, rel_preds, strict=True)
            ]

        results: list[MatchStageResult] = []
        for pred in preds:
            rel_result = None
            if pred.relevance is not None:
                rel_result = RelevanceResult(
                    petLikelihood=pred.relevance.pet_likelihood,
                    topLabel=pred.relevance.top_label,
                )
            results.append(
                MatchStageResult(embedding=pred.embedding, relevance=rel_result)
            )
        return results

    return await run_in_executor(_run)
