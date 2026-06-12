"""
Pipeline Orchestrator — downloads images once, then coordinates execution stages.

This module acts as the pipeline coordinator. It manages job steps in a specific
moderation-first sequence to protect computational resources (e.g., CPU/GPU memory):
  Phase 1. Moderation Signals (resolution, blur, NSFW)
  Phase 2. Deduplication Fingerprinting & Matching (MD5, pHash, embeddings, relevance)
"""

import asyncio
import logging
import time
from typing import Any

from app.config.settings import Settings, settings
from app.models.registry import run_in_executor
from app.pipeline.download import DecodedImage, download_all
from app.pipeline.quality import assess_quality
from app.pipeline.stages.embed import embed_stage
from app.pipeline.stages.fingerprint import fingerprint_image
from app.pipeline.stages.relevance import relevance_stage
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
    """
    Decides which execution stages to execute.
    If the job payload has a custom "pipeline" override array (e.g., ["quality"]),
    we use that; otherwise, we default to the active profile's stages.
    """
    override = job.get("pipeline")
    if override:
        return tuple(override)
    return cfg.profile.stages


def _run_fingerprint(decoded: list[DecodedImage]) -> list[tuple[str, str]]:
    """
    Utility helper to compute MD5 and pHash. Run in background executors.
    """
    return [fingerprint_image(d.raw_bytes, d.image) for d in decoded]


def _run_quality(decoded: list[DecodedImage]) -> list[QualityResult]:
    """
    Utility helper to run image resolution and blur checks. Run in background executors.
    """
    return [QualityResult(**assess_quality(d.image)) for d in decoded]


async def _run_stage_batch(
    stage_tasks: dict[str, Any],
) -> dict[str, Any]:
    """
    Utility that maps and awaits multiple asynchronous tasks concurrently.
    Matches the results dictionary back to their original stage keys.
    """
    if not stage_tasks:
        return {}
    keys = list(stage_tasks.keys())
    values = await asyncio.gather(*stage_tasks.values())
    return dict(zip(keys, values))


async def run_pipeline(job: dict, cfg: Settings = settings) -> JobResult:
    """
    Main pipeline executor. Downloads URLs, coordinates phases, and builds results.
    """
    job_type = job.get("jobType", "listing")
    image_urls = job.get("imageUrls", [])
    pet_type = job.get("petType", "")
    stages = resolve_stages(job, cfg)

    # 1. Image Download Phase
    t0 = time.perf_counter()
    decoded, errors = await download_all(image_urls)
    logger.info(
        "Downloaded %d/%d images in %.2fs. Active stages: %s",
        len(decoded),
        len(image_urls),
        time.perf_counter() - t0,
        stages,
    )

    # Instantiate the JobResult accumulator container
    result = JobResult(
        job_type=job_type,
        listing_id=job.get("listingId"),
        owned_pet_id=job.get("ownedPetId"),
        search_session_id=job.get("searchSessionId"),
        errors=[ImageError(**e) for e in errors],
    )

    # If no images downloaded successfully, return early
    if not decoded:
        return result

    # Extract PIL Image objects from the decoded wrappers
    pil_images = [d.image for d in decoded]
    stage_results: dict[str, Any] = {}

    # --------------------------------------------------------------------------
    # Phase 1 — Moderation Signals (Pre-deduplication checks)
    # --------------------------------------------------------------------------
    # Performs checks like resolution bounds and NSFW safety before doing
    # resource-intensive matching operations.
    # Quality and Safety are computed concurrently.
    # --------------------------------------------------------------------------
    moderation_tasks: dict[str, Any] = {}
    if "quality" in stages:
        # Offload OpenCV Laplacian calculations to ThreadPoolExecutor
        moderation_tasks["quality"] = run_in_executor(_run_quality, decoded)
    if "safety" in stages and cfg.safety_enabled:
        # Launch safety model classifier tasks
        moderation_tasks["safety"] = safety_stage(pil_images, cfg)

    # Await Phase 1 completion and update results store
    stage_results.update(await _run_stage_batch(moderation_tasks))

    # --------------------------------------------------------------------------
    # Phase 2 — Fingerprints and Match Inference
    # --------------------------------------------------------------------------
    # Generates unique identifiers (hashes) and visual embeddings.
    # --------------------------------------------------------------------------
    matching_tasks: dict[str, Any] = {}
    if "fingerprint" in stages:
        # Offload hashing calculations to ThreadPoolExecutor
        matching_tasks["fingerprint"] = run_in_executor(_run_fingerprint, decoded)
    if "embed" in stages and cfg.embed_enabled:
        # Run SigLIP2 vision encoder models
        matching_tasks["embed"] = embed_stage(pil_images, cfg)
    if "relevance" in stages and cfg.relevance_enabled and cfg.embed_enabled:
        # Run zero-shot pet verification matching
        matching_tasks["relevance"] = relevance_stage(pil_images, pet_type, cfg)

    # Await Phase 2 completion
    stage_results.update(await _run_stage_batch(matching_tasks))

    # --------------------------------------------------------------------------
    # Phase 3 — Result Merging
    # --------------------------------------------------------------------------
    # Maps results from separate stages back to each decoded image item.
    # --------------------------------------------------------------------------
    fingerprints = stage_results.get("fingerprint", [("", "")] * len(decoded))
    qualities = stage_results.get("quality", [None] * len(decoded))
    safeties = stage_results.get("safety", [None] * len(decoded))
    embeddings = stage_results.get("embed", [[] for _ in decoded])
    relevances = stage_results.get("relevance", [None] * len(decoded))

    for i, item in enumerate(decoded):
        md5, phash = fingerprints[i] if i < len(fingerprints) else ("", "")
        img_result = ProcessedImageResult(
            url=item.url,
            s3Key=item.url,
            md5=md5,
            phash=phash,
            embedding=embeddings[i] if i < len(embeddings) else [],
        )
        # Verify lists indices before appending optional stage results
        if i < len(qualities) and qualities[i]:
            img_result.quality = qualities[i]
        if i < len(safeties) and safeties[i]:
            img_result.safety = safeties[i]
        if i < len(relevances) and relevances[i]:
            img_result.relevance = relevances[i]
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
    """
    Serializes internal job result models into the POST payload expected by Next.js API.
    """
    images_out = []
    for img in result.images:
        row: dict = {
            "url": img.url,
            "s3Key": img.s3Key or img.url,
            "md5": img.md5,
            "phash": img.phash,
            "embedding": img.embedding,
        }
        # Dump sub-stages results if present
        if img.safety:
            row["safety"] = img.safety.model_dump()
        if img.relevance:
            row["relevance"] = img.relevance.model_dump()
        if img.quality:
            row["quality"] = img.quality.model_dump()
        images_out.append(row)

    payload = CallbackPayload(
        jobType=result.job_type,
        workerVersion=cfg.worker_version,
        matchModel=cfg.match_model or "",
        safetyModel=cfg.safety_model or "",
        embeddingModel=cfg.match_model or "",
        images=images_out,
        errors=[e.model_dump() for e in result.errors],
    )
    # Add optional context IDs to the callback
    if result.listing_id:
        payload.listingId = result.listing_id
    if result.owned_pet_id:
        payload.ownedPetId = result.owned_pet_id
    if result.search_session_id:
        payload.searchSessionId = result.search_session_id
    return payload
