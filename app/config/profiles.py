"""
Hardware profiles — model stack and pipeline stages via VISION_PROFILE.

Defines the available hardware execution profiles. These configurations map out
resource sizing guidelines, hardware constraints (CPU/GPU), stage sequences,
and model selection.

Stage order (enforced by the orchestrator):
  quality → safety → fingerprint → embed → relevance (optional, SigLIP2 only)
"""

from dataclasses import dataclass

# Hugging Face Model Identifier Constants:
# - SIGLIP2_BASE: Google's SigLIP2 image-text model. Produces 768-dimensional visual vectors
#                 and powers zero-shot image classification for pet relevance.
# - FALCONSAI_NSFW: A lightweight binary image classification model for SFW vs NSFW content.
# - STRANGERGUARD_NSFW: A larger, high-accuracy multi-class NSFW classification model.
SIGLIP2_BASE = "google/siglip2-base-patch16-224"
FALCONSAI_NSFW = "Falconsai/nsfw_image_detection"
STRANGERGUARD_NSFW = "strangerguardhf/nsfw-image-detection"


@dataclass(frozen=True)
class VisionProfile:
    """
    Resolved inference profile representing resource limits and pipeline stages
    for specific hosting configurations.
    """

    # Unique profile name string matching VISION_PROFILE settings value
    name: str

    # Ordered list of execution pipeline stages to run for this profile
    stages: tuple[str, ...]

    # Model ID to use for visual embedding vectors
    match_model: str | None

    # Model ID to use for NSFW safety checks
    safety_model: str | None

    # Target execution hardware ('cpu' or 'cuda')
    device: str

    # Inference batch size for PyTorch. Higher numbers speed up GPU execution but use more VRAM
    batch_size: int

    # Toggle to load the matching/embedding model
    embed_enabled: bool

    # Toggle to load the safety classifier model
    safety_enabled: bool

    # Toggle to load zero-shot classification tokenizers and prompts
    relevance_enabled: bool

    # System RAM threshold required to run this profile without risking system OOM crashes
    min_ram_mb: int

    # VRAM threshold required to run this profile on an NVIDIA GPU (0 if CPU profile)
    min_vram_mb: int

    # Descriptive summary outlining the intended deployment environment
    description: str


# Active execution profile presets.
# System administrators set the active configuration using the VISION_PROFILE env variable.
PRESETS: dict[str, VisionProfile] = {
    # --------------------------------------------------------------------------
    # CPU Profiles
    # --------------------------------------------------------------------------
    "dedup-only": VisionProfile(
        name="dedup-only",
        stages=("quality", "fingerprint"),
        match_model=None,
        safety_model=None,
        device="cpu",
        batch_size=0,
        embed_enabled=False,
        safety_enabled=False,
        relevance_enabled=False,
        min_ram_mb=512,
        min_vram_mb=0,
        description="MD5 + pHash + quality only — no torch or transformers loaded (minimal footprint)",
    ),
    "cpu-light": VisionProfile(
        name="cpu-light",
        stages=("quality", "safety", "fingerprint"),
        match_model=None,
        safety_model=FALCONSAI_NSFW,
        device="cpu",
        batch_size=0,
        embed_enabled=False,
        safety_enabled=True,
        relevance_enabled=False,
        min_ram_mb=1536,
        min_vram_mb=0,
        description="NSFW safety checks on CPU — no heavy matching vectors or vector search",
    ),
    "cpu-standard": VisionProfile(
        name="cpu-standard",
        stages=("quality", "safety", "fingerprint", "embed"),
        match_model=SIGLIP2_BASE,
        safety_model=FALCONSAI_NSFW,
        device="cpu",
        batch_size=1,
        embed_enabled=True,
        safety_enabled=True,
        relevance_enabled=False,
        min_ram_mb=3072,
        min_vram_mb=0,
        description="SigLIP2 embeddings + safety on CPU — omits the animal verification step",
    ),
    "cpu-quality": VisionProfile(
        name="cpu-quality",
        stages=("quality", "safety", "fingerprint", "embed", "relevance"),
        match_model=SIGLIP2_BASE,
        safety_model=FALCONSAI_NSFW,
        device="cpu",
        batch_size=1,
        embed_enabled=True,
        safety_enabled=True,
        relevance_enabled=True,
        min_ram_mb=4096,
        min_vram_mb=0,
        description="Full feature pipeline on CPU. Uses SigLIP2 + NSFW + relevance checking (Default Dev)",
    ),
    # --------------------------------------------------------------------------
    # GPU Profiles (CUDA required)
    # --------------------------------------------------------------------------
    "gpu-standard": VisionProfile(
        name="gpu-standard",
        stages=("quality", "safety", "fingerprint", "embed", "relevance"),
        match_model=SIGLIP2_BASE,
        safety_model=FALCONSAI_NSFW,
        device="cuda",
        batch_size=4,
        embed_enabled=True,
        safety_enabled=True,
        relevance_enabled=True,
        min_ram_mb=4096,
        min_vram_mb=4096,
        description="Full feature pipeline on GPU. Fast concurrent inference (Default Production)",
    ),
    "gpu-quality": VisionProfile(
        name="gpu-quality",
        stages=("quality", "safety", "fingerprint", "embed", "relevance"),
        match_model=SIGLIP2_BASE,
        safety_model=STRANGERGUARD_NSFW,
        device="cuda",
        batch_size=8,
        embed_enabled=True,
        safety_enabled=True,
        relevance_enabled=True,
        min_ram_mb=4096,
        min_vram_mb=6144,
        description="GPU profile using heavy multi-class safety moderation (StrangerGuard)",
    ),
}


def get_preset(name: str) -> VisionProfile:
    """
    Fetches the profile configuration corresponding to the requested profile name.
    Raises ValueError if the profile name is unknown.
    """
    if name not in PRESETS:
        known = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown VISION_PROFILE '{name}'. Available options: {known}")
    return PRESETS[name]
