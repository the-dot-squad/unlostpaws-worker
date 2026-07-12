"""
Backend factory — builds torch or ONNX embedder/classifier from Settings.

Selection flow:
  1. ``Settings.profile`` (from VISION_PROFILE) sets ``runtime``, models, batch size.
  2. If ``runtime == "onnx"`` → :class:`OnnxSiglipEmbedder` / :class:`OnnxNsfwClassifier`.
  3. If ``runtime == "torch"`` → :class:`TorchSiglipEmbedder` / :class:`TorchNsfwClassifier`.

Pipeline stages call :func:`app.models.registry.get_match_embedder` — they never
import torch or onnx directly.
"""

from __future__ import annotations

import torch

from app.config.runtime_validation import RuntimeValidationError
from app.config.settings import Settings, settings
from app.models.onnx_nsfw import OnnxNsfwClassifier
from app.models.onnx_siglip import OnnxSiglipEmbedder
from app.models.protocols import ClassifierBackend, EmbedderBackend
from app.models.torch_nsfw import TorchNsfwClassifier
from app.models.torch_siglip import TorchSiglipEmbedder


def resolve_torch_device(preferred: str) -> str:
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeValidationError(
                "DEVICE=cuda requested but torch.cuda.is_available() is False.\n"
                "Fix: use docker-compose.gpu.yml on an NVIDIA GPU host, "
                "or set VISION_PROFILE to a CPU profile."
            )
        return "cuda"
    return "cpu"


def _openvino_device_for_settings(cfg: Settings) -> str:
    return cfg.openvino_device


def create_embedder(cfg: Settings = settings) -> EmbedderBackend | None:
    if not cfg.embed_enabled or not cfg.match_model:
        return None

    if cfg.runtime == "onnx":
        return OnnxSiglipEmbedder(
            cfg.match_model,
            cfg.batch_size,
            precision=cfg.precision,
            execution_provider=cfg.execution_provider,
            model_cache_dir=cfg.model_cache_dir,
            tensorrt_cache_dir=cfg.tensorrt_cache_dir,
            openvino_device=_openvino_device_for_settings(cfg),
        )

    return TorchSiglipEmbedder(
        cfg.match_model,
        resolve_torch_device(cfg.device),
        cfg.batch_size,
        torch_compile=cfg.torch_compile,
    )


def create_classifier(cfg: Settings = settings) -> ClassifierBackend | None:
    if not cfg.safety_enabled or not cfg.safety_model:
        return None

    if cfg.runtime == "onnx":
        return OnnxNsfwClassifier(
            cfg.safety_model,
            precision=cfg.precision,
            execution_provider=cfg.execution_provider,
            model_cache_dir=cfg.model_cache_dir,
            tensorrt_cache_dir=cfg.tensorrt_cache_dir,
            openvino_device=_openvino_device_for_settings(cfg),
        )

    return TorchNsfwClassifier(
        cfg.safety_model,
        resolve_torch_device(cfg.device),
        torch_compile=cfg.torch_compile,
    )
