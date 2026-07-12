"""
PyTorch SigLIP2 embedder — embeddings and zero-shot pet relevance.

Uses Hugging Face Transformers with optional torch.compile. Text prompt
embeddings are precomputed at load time so relevance inference only runs
the vision encoder per image batch.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForZeroShotImageClassification,
    AutoTokenizer,
)

from app.models.relevance import (
    NEGATIVE_PROMPTS,
    PET_PROMPTS,
    compute_relevance_from_logits,
)
from app.models.types import EmbedPrediction, MatchPrediction, RelevancePrediction

logger = logging.getLogger(__name__)

DEFAULT_MATCH_MODEL = "google/siglip2-base-patch16-224"


class TorchSiglipEmbedder:
    """SigLIP2 wrapper using PyTorch / Transformers."""

    supports_relevance = True
    runtime = "torch"

    def __init__(
        self,
        model_id: str,
        device: str,
        batch_size: int,
        *,
        torch_compile: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.batch_size = max(1, batch_size)
        self.torch_compile = torch_compile

        self._model = None
        self._image_processor = None
        self._tokenizer = None
        self._text_features: torch.Tensor | None = None
        self._logit_scale: torch.Tensor | None = None
        self._logit_bias: torch.Tensor | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        logger.info(
            "Loading SigLIP2 (torch) %s on device=%s compile=%s",
            self.model_id,
            self.device,
            self.torch_compile,
        )
        self._image_processor = AutoImageProcessor.from_pretrained(
            self.model_id, use_fast=False
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForZeroShotImageClassification.from_pretrained(
            self.model_id
        )
        self._model.to(self.device)
        self._model.eval()

        if self.torch_compile:
            self._model = torch.compile(self._model, mode="reduce-overhead")

        self._precompute_text_features()

    def _precompute_text_features(self) -> None:
        """Encode fixed pet + negative prompts once for fast relevance scoring."""
        assert self._model is not None and self._tokenizer is not None

        all_prompts = list(PET_PROMPTS.values()) + NEGATIVE_PROMPTS
        text_inputs = self._tokenizer(
            all_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        with torch.no_grad():
            text_features = self._model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(
                p=2, dim=-1, keepdim=True
            )

        self._text_features = text_features

        # SigLIP applies learned scale/bias to dot products for logits_per_image.
        inner = self._model.model if hasattr(self._model, "model") else self._model
        if hasattr(inner, "logit_scale"):
            self._logit_scale = inner.logit_scale.exp()
        if hasattr(inner, "logit_bias"):
            self._logit_bias = inner.logit_bias

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _image_features(self, images: list[Image.Image]) -> torch.Tensor:
        assert self._model is not None and self._image_processor is not None

        inputs = self._image_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            features = self._model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features

    def _logits_from_features(self, image_features: torch.Tensor) -> torch.Tensor:
        """Compute logits_per_image from normalized image and text features."""
        assert self._text_features is not None
        logits = image_features @ self._text_features.T
        if self._logit_scale is not None:
            logits = logits * self._logit_scale
        if self._logit_bias is not None:
            logits = logits + self._logit_bias
        return logits

    def embed_batch(self, images: list[Image.Image]) -> list[EmbedPrediction]:
        self.load()
        results: list[EmbedPrediction] = []
        for start in range(0, len(images), self.batch_size):
            chunk = images[start : start + self.batch_size]
            features = self._image_features(chunk)
            for vec in features.cpu().tolist():
                results.append(EmbedPrediction(embedding=vec))
        return results

    def relevance_batch(
        self, images: list[Image.Image], pet_type: str = ""
    ) -> list[RelevancePrediction]:
        return [
            pred.relevance
            for pred in self.match_batch(images, pet_type, include_relevance=True)
            if pred.relevance is not None
        ]

    def match_batch(
        self,
        images: list[Image.Image],
        pet_type: str = "",
        *,
        include_relevance: bool = True,
    ) -> list[MatchPrediction]:
        """Single vision forward pass returning embedding and optional relevance."""
        self.load()
        results: list[MatchPrediction] = []
        for start in range(0, len(images), self.batch_size):
            chunk = images[start : start + self.batch_size]
            image_features = self._image_features(chunk)
            logits_batch = (
                self._logits_from_features(image_features)
                if include_relevance and self.supports_relevance
                else None
            )

            for row_idx, features in enumerate(image_features):
                embedding = features.cpu().tolist()
                relevance = None
                if logits_batch is not None:
                    logits = logits_batch[row_idx]
                    relevance = compute_relevance_from_logits(
                        np.asarray(logits.cpu().tolist(), dtype=np.float64),
                        pet_type=pet_type,
                    )
                results.append(
                    MatchPrediction(embedding=embedding, relevance=relevance)
                )
        return results
