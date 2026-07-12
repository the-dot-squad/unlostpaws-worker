"""
Vision profiles — capability tiers for the worker pipeline.

``VISION_PROFILE`` selects *what* runs (dedup / standard / quality) and which
SigLIP checkpoint to use. *Where* it runs (torch vs ONNX, CPU vs GPU vs CoreML)
is configured via ``INFERENCE_RUNTIME``, ``DEVICE``, and ``ORT_EXECUTION_PROVIDER``
in :mod:`app.config.settings`.

Both ``standard`` and ``quality`` run relevance scoring (zero-shot pet check).
The default difference is the SigLIP checkpoint:

  - standard → base @ 224px (768-d, fast default)
  - quality  → base @ 384px (768-d, higher resolution)

Stage order (orchestrator):

  quality → safety → fingerprint → embed → relevance
"""

from __future__ import annotations

from dataclasses import dataclass

SIGLIP2_BASE = "google/siglip2-base-patch16-224"
SIGLIP2_QUALITY = "google/siglip2-base-patch16-384"
FALCONSAI_NSFW = "Falconsai/nsfw_image_detection"
STRANGERGUARD_NSFW = "strangerguardhf/nsfw-image-detection"


@dataclass(frozen=True)
class VisionProfile:
    """Capability preset selected by VISION_PROFILE."""

    name: str
    stages: tuple[str, ...]
    match_model: str | None
    safety_model: str | None
    batch_size: int
    embed_enabled: bool
    safety_enabled: bool
    relevance_enabled: bool
    min_ram_mb: int
    min_vram_mb: int
    description: str
    default_torch_compile: bool = False


PRESETS: dict[str, VisionProfile] = {
    "dedup-only": VisionProfile(
        name="dedup-only",
        stages=("quality", "fingerprint"),
        match_model=None,
        safety_model=None,
        batch_size=0,
        embed_enabled=False,
        safety_enabled=False,
        relevance_enabled=False,
        min_ram_mb=512,
        min_vram_mb=0,
        description="MD5 + pHash + quality only — no ML models loaded",
    ),
    "standard": VisionProfile(
        name="standard",
        stages=("quality", "safety", "fingerprint", "embed", "relevance"),
        match_model=SIGLIP2_BASE,
        safety_model=FALCONSAI_NSFW,
        batch_size=1,
        embed_enabled=True,
        safety_enabled=True,
        relevance_enabled=True,
        min_ram_mb=3072,
        min_vram_mb=2048,
        description="SigLIP2 base @ 224px — embed + relevance + NSFW (Qdrant indexing)",
    ),
    "quality": VisionProfile(
        name="quality",
        stages=("quality", "safety", "fingerprint", "embed", "relevance"),
        match_model=SIGLIP2_QUALITY,
        safety_model=FALCONSAI_NSFW,
        batch_size=1,
        embed_enabled=True,
        safety_enabled=True,
        relevance_enabled=True,
        min_ram_mb=4096,
        min_vram_mb=4096,
        description="SigLIP2 base @ 384px — higher-res embed + relevance + NSFW",
        default_torch_compile=False,
    ),
}


def get_preset(name: str) -> VisionProfile:
    if name in PRESETS:
        return PRESETS[name]
    known = ", ".join(PRESETS.keys())
    raise ValueError(f"Unknown VISION_PROFILE '{name}'. Available options: {known}")
