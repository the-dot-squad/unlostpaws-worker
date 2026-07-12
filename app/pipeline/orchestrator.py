"""
Pipeline Orchestrator — downloads images once, then coordinates execution stages.

Moderation-first sequence:
  Phase 1. Moderation (quality, safety on originals)
  Phase 2. Fingerprint + fused match (embed + relevance on full-frame image)
"""

import asyncio
import logging
import time
from typing import Any

from app.config.settings import Settings, settings
from app.models.registry import health_models, run_in_executor
from app.pipeline.download import DecodedImage, download_all
from app.pipeline.quality import assess_quality
from app.pipeline.stages.fingerprint import fingerprint_image
from app.pipeline.stages.match import MatchStageResult, match_stage
from app.pipeline.stages.safety import safety_stage
from app.schemas.result import (
    CallbackPayload,
    ImageError,
    JobResult,
    ProcessedImageResult,
    QualityResult,
)

logger = logging.getLogger(__name__)


def resolve_stages(job: dict, cfg: Settings) -> tuple[str, ...]:
    """Use job pipeline override or default profile stages."""
    override = job.get("pipeline")
    if override:
        return tuple(override)
    return cfg.profile.stages


def _run_fingerprint(decoded: list[DecodedImage]) -> list[tuple[str, str]]:
    return [fingerprint_image(d.raw_bytes, d.image) for d in decoded]


def _run_quality(decoded: list[DecodedImage]) -> list[QualityResult]:
    return [QualityResult(**assess_quality(d.image)) for d in decoded]


async def _run_stage_batch(
    stage_tasks: dict[str, Any],
) -> dict[str, Any]:
    if not stage_tasks:
        return {}
    keys = list(stage_tasks.keys())
    values = await asyncio.gather(*stage_tasks.values())
    return dict(zip(keys, values))


async def run_pipeline(job: dict, cfg: Settings = settings) -> JobResult:
    """Main pipeline executor."""
    job_type = job.get("jobType", "listing")
    image_urls = job.get("imageUrls", [])
    pet_type = job.get("petType", "")  # optional species hint; empty = zero-shot
    stages = resolve_stages(job, cfg)

    t0 = time.perf_counter()
    decoded, errors = await download_all(image_urls)
    logger.info(
        "Downloaded %d/%d images in %.2fs. Active stages: %s",
        len(decoded),
        len(image_urls),
        time.perf_counter() - t0,
        stages,
    )

    result = JobResult(
        job_type=job_type,
        listing_id=job.get("listingId"),
        owned_pet_id=job.get("ownedPetId"),
        search_session_id=job.get("searchSessionId"),
        errors=[ImageError(**e) for e in errors],
    )

    if not decoded:
        return result

    pil_images = [d.image for d in decoded]
    stage_results: dict[str, Any] = {}

    moderation_tasks: dict[str, Any] = {}
    if "quality" in stages:
        moderation_tasks["quality"] = run_in_executor(_run_quality, decoded)
    if "safety" in stages and cfg.safety_enabled:
        moderation_tasks["safety"] = safety_stage(pil_images, cfg)

    stage_results.update(await _run_stage_batch(moderation_tasks))

    matching_tasks: dict[str, Any] = {}
    if "fingerprint" in stages:
        matching_tasks["fingerprint"] = run_in_executor(_run_fingerprint, decoded)
    if "embed" in stages and cfg.embed_enabled:
        matching_tasks["match"] = match_stage(pil_images, pet_type, cfg)

    stage_results.update(await _run_stage_batch(matching_tasks))

    fingerprints = stage_results.get("fingerprint", [("", "")] * len(decoded))
    qualities = stage_results.get("quality", [None] * len(decoded))
    safeties = stage_results.get("safety", [None] * len(decoded))
    matches: list[MatchStageResult] = stage_results.get(
        "match", [MatchStageResult(embedding=[]) for _ in decoded]
    )

    for i, item in enumerate(decoded):
        md5, phash = fingerprints[i] if i < len(fingerprints) else ("", "")
        match = matches[i] if i < len(matches) else MatchStageResult(embedding=[])
        img_result = ProcessedImageResult(
            url=item.url,
            s3Key=item.url,
            md5=md5,
            phash=phash,
            embedding=match.embedding,
        )
        if i < len(qualities) and qualities[i]:
            img_result.quality = qualities[i]
        if i < len(safeties) and safeties[i]:
            img_result.safety = safeties[i]
        if match.relevance:
            img_result.relevance = match.relevance
        result.images.append(img_result)

    logger.info(
        "Pipeline finished in %.2fs — processed %d images, encountered %d errors",
        time.perf_counter() - t0,
        len(result.images),
        len(result.errors),
    )
    return result


def build_callback_payload(
    result: JobResult, cfg: Settings = settings
) -> CallbackPayload:
    """Serialize job results for the Next.js callback API."""
    images_out = []
    for img in result.images:
        row: dict = {
            "url": img.url,
            "s3Key": img.s3Key or img.url,
            "md5": img.md5,
            "phash": img.phash,
            "embedding": img.embedding,
        }
        if img.safety:
            row["safety"] = img.safety.model_dump()
        if img.relevance:
            row["relevance"] = img.relevance.model_dump()
        if img.quality:
            row["quality"] = img.quality.model_dump()
        images_out.append(row)

    model_health = health_models(cfg)
    payload = CallbackPayload(
        jobType=result.job_type,
        workerVersion=cfg.worker_version,
        matchModel=cfg.match_model or "",
        safetyModel=cfg.safety_model or "",
        embeddingModel=cfg.match_model or "",
        runtime=model_health.get("runtime", cfg.runtime),
        executionProvider=model_health.get("executionProvider", ""),
        modelPrecision=model_health.get("precision", cfg.precision),
        images=images_out,
        errors=[e.model_dump() for e in result.errors],
    )
    if result.listing_id:
        payload.listingId = result.listing_id
    if result.owned_pet_id:
        payload.ownedPetId = result.owned_pet_id
    if result.search_session_id:
        payload.searchSessionId = result.search_session_id
    return payload
