"""
Vision profiles — one VISION_PROFILE preset controls the whole worker.

Terminology
-----------
- **Pipeline stage**: what runs per image (quality, safety, fingerprint, embed, relevance).
- **Vision profile**: this module's presets — stages, HF models, batch size, RAM hints,
  runtime (torch/onnx), execution provider, and precision in one name.
- **Runtime**: ``torch`` = PyTorch + Transformers; ``onnx`` = ONNX Runtime graphs.
- **Execution provider (EP)**: ONNX-only hardware driver (CPU, CUDA, TensorRT, …).

Operators set ``VISION_PROFILE`` in the environment; see ``PRESETS`` below and
docs/INFERENCE_BACKENDS.md. Stage order (enforced by the orchestrator):

  quality → safety → fingerprint → embed → relevance (optional, SigLIP2 only)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SIGLIP2_BASE = "google/siglip2-base-patch16-224"
FALCONSAI_NSFW = "Falconsai/nsfw_image_detection"
STRANGERGUARD_NSFW = "strangerguardhf/nsfw-image-detection"

RuntimeKind = Literal["torch", "onnx"]
PrecisionKind = Literal["fp32", "fp16", "int8"]


@dataclass(frozen=True)
class VisionProfile:
    """
    Named deployment preset selected by VISION_PROFILE.

    Most fields are consumed automatically — you rarely set them individually.
    ``runtime`` chooses torch vs onnx; ``execution_provider`` and ``precision``
    apply only when runtime is onnx.
    """

    name: str
    stages: tuple[str, ...]
    match_model: str | None
    safety_model: str | None
    device: str
    batch_size: int
    embed_enabled: bool
    safety_enabled: bool
    relevance_enabled: bool
    min_ram_mb: int
    min_vram_mb: int
    description: str
    runtime: RuntimeKind = "torch"
    execution_provider: str = "auto"
    precision: PrecisionKind = "fp32"
    torch_compile: bool = False

    @property
    def requires_cuda(self) -> bool:
        """True when this profile must run on NVIDIA CUDA (torch or ONNX)."""
        return self.device == "cuda" or self.execution_provider in ("cuda", "tensorrt")


@dataclass(frozen=True)
class _PresetSpec:
    """Shared pipeline/resource fields for building a VisionProfile preset."""

    stages: tuple[str, ...]
    match_model: str | None
    safety_model: str | None
    device: str
    batch_size: int
    embed_enabled: bool
    safety_enabled: bool
    relevance_enabled: bool
    min_ram_mb: int
    min_vram_mb: int
    description: str


def _torch_preset(
    name: str,
    spec: _PresetSpec,
    *,
    torch_compile: bool = False,
) -> VisionProfile:
    return VisionProfile(
        name=name,
        stages=spec.stages,
        match_model=spec.match_model,
        safety_model=spec.safety_model,
        device=spec.device,
        batch_size=spec.batch_size,
        embed_enabled=spec.embed_enabled,
        safety_enabled=spec.safety_enabled,
        relevance_enabled=spec.relevance_enabled,
        min_ram_mb=spec.min_ram_mb,
        min_vram_mb=spec.min_vram_mb,
        description=spec.description,
        runtime="torch",
        execution_provider="auto",
        precision="fp32",
        torch_compile=torch_compile,
    )


def _onnx_preset(
    name: str,
    spec: _PresetSpec,
    *,
    execution_provider: str,
    precision: PrecisionKind,
) -> VisionProfile:
    return VisionProfile(
        name=name,
        stages=spec.stages,
        match_model=spec.match_model,
        safety_model=spec.safety_model,
        device=spec.device,
        batch_size=spec.batch_size,
        embed_enabled=spec.embed_enabled,
        safety_enabled=spec.safety_enabled,
        relevance_enabled=spec.relevance_enabled,
        min_ram_mb=spec.min_ram_mb,
        min_vram_mb=spec.min_vram_mb,
        description=spec.description,
        runtime="onnx",
        execution_provider=execution_provider,
        precision=precision,
        torch_compile=False,
    )


PRESETS: dict[str, VisionProfile] = {
    "dedup-only": _torch_preset(
        "dedup-only",
        _PresetSpec(
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
            description="MD5 + pHash + quality only — no ML models loaded",
        ),
    ),
    "cpu-light": _torch_preset(
        "cpu-light",
        _PresetSpec(
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
            description="NSFW safety on CPU — no embeddings",
        ),
    ),
    "cpu-standard": _torch_preset(
        "cpu-standard",
        _PresetSpec(
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
            description="SigLIP2 embeddings + safety on CPU",
        ),
    ),
    "cpu-quality": _torch_preset(
        "cpu-quality",
        _PresetSpec(
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
            description="Full pipeline on CPU (default dev)",
        ),
    ),
    "gpu-standard": _torch_preset(
        "gpu-standard",
        _PresetSpec(
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
            description="Full pipeline on GPU with torch.compile (default prod)",
        ),
        torch_compile=True,
    ),
    "gpu-quality": _torch_preset(
        "gpu-quality",
        _PresetSpec(
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
            description="GPU + StrangerGuard safety with torch.compile",
        ),
        torch_compile=True,
    ),
    "onnx-cpu-standard": _onnx_preset(
        "onnx-cpu-standard",
        _PresetSpec(
            stages=("quality", "safety", "fingerprint", "embed"),
            match_model=SIGLIP2_BASE,
            safety_model=FALCONSAI_NSFW,
            device="cpu",
            batch_size=2,
            embed_enabled=True,
            safety_enabled=True,
            relevance_enabled=False,
            min_ram_mb=2048,
            min_vram_mb=0,
            description="ONNX INT8 SigLIP2 + safety on CPU",
        ),
        execution_provider="cpu",
        precision="int8",
    ),
    "onnx-cpu-quality": _onnx_preset(
        "onnx-cpu-quality",
        _PresetSpec(
            stages=("quality", "safety", "fingerprint", "embed", "relevance"),
            match_model=SIGLIP2_BASE,
            safety_model=FALCONSAI_NSFW,
            device="cpu",
            batch_size=2,
            embed_enabled=True,
            safety_enabled=True,
            relevance_enabled=True,
            min_ram_mb=3072,
            min_vram_mb=0,
            description="Full ONNX INT8 pipeline on CPU",
        ),
        execution_provider="cpu",
        precision="int8",
    ),
    "onnx-gpu-standard": _onnx_preset(
        "onnx-gpu-standard",
        _PresetSpec(
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
            description="ONNX FP16 full pipeline on NVIDIA GPU",
        ),
        execution_provider="cuda",
        precision="fp16",
    ),
    "onnx-gpu-quality": _onnx_preset(
        "onnx-gpu-quality",
        _PresetSpec(
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
            description="ONNX FP16 GPU + StrangerGuard safety",
        ),
        execution_provider="cuda",
        precision="fp16",
    ),
    "onnx-trt-standard": _onnx_preset(
        "onnx-trt-standard",
        _PresetSpec(
            stages=("quality", "safety", "fingerprint", "embed", "relevance"),
            match_model=SIGLIP2_BASE,
            safety_model=FALCONSAI_NSFW,
            device="cuda",
            batch_size=4,
            embed_enabled=True,
            safety_enabled=True,
            relevance_enabled=True,
            min_ram_mb=4096,
            min_vram_mb=6144,
            description="ONNX TensorRT EP — max NVIDIA throughput",
        ),
        execution_provider="tensorrt",
        precision="fp16",
    ),
    "onnx-trt-quality": _onnx_preset(
        "onnx-trt-quality",
        _PresetSpec(
            stages=("quality", "safety", "fingerprint", "embed", "relevance"),
            match_model=SIGLIP2_BASE,
            safety_model=STRANGERGUARD_NSFW,
            device="cuda",
            batch_size=8,
            embed_enabled=True,
            safety_enabled=True,
            relevance_enabled=True,
            min_ram_mb=4096,
            min_vram_mb=8192,
            description="ONNX TensorRT EP + StrangerGuard safety",
        ),
        execution_provider="tensorrt",
        precision="fp16",
    ),
    "onnx-intel": _onnx_preset(
        "onnx-intel",
        _PresetSpec(
            stages=("quality", "safety", "fingerprint", "embed", "relevance"),
            match_model=SIGLIP2_BASE,
            safety_model=FALCONSAI_NSFW,
            device="cpu",
            batch_size=2,
            embed_enabled=True,
            safety_enabled=True,
            relevance_enabled=True,
            min_ram_mb=3072,
            min_vram_mb=0,
            description="ONNX OpenVINO EP — Intel CPU/iGPU/NPU",
        ),
        execution_provider="openvino",
        precision="int8",
    ),
    "onnx-apple": _onnx_preset(
        "onnx-apple",
        _PresetSpec(
            stages=("quality", "safety", "fingerprint", "embed", "relevance"),
            match_model=SIGLIP2_BASE,
            safety_model=FALCONSAI_NSFW,
            device="cpu",
            batch_size=2,
            embed_enabled=True,
            safety_enabled=True,
            relevance_enabled=True,
            min_ram_mb=3072,
            min_vram_mb=0,
            description="ONNX CoreML EP — Apple Neural Engine",
        ),
        execution_provider="coreml",
        precision="fp16",
    ),
    "onnx-qualcomm": _onnx_preset(
        "onnx-qualcomm",
        _PresetSpec(
            stages=("quality", "safety", "fingerprint", "embed", "relevance"),
            match_model=SIGLIP2_BASE,
            safety_model=FALCONSAI_NSFW,
            device="cpu",
            batch_size=1,
            embed_enabled=True,
            safety_enabled=True,
            relevance_enabled=True,
            min_ram_mb=3072,
            min_vram_mb=0,
            description="ONNX QNN EP — Qualcomm Hexagon NPU",
        ),
        execution_provider="qnn",
        precision="int8",
    ),
}


def get_preset(name: str) -> VisionProfile:
    if name not in PRESETS:
        known = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown VISION_PROFILE '{name}'. Available options: {known}")
    return PRESETS[name]
