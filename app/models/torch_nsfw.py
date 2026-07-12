"""
PyTorch NSFW classifier — Falconsai and StrangerGuard safety models.
"""

from __future__ import annotations

import logging

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from app.models.safety_scoring import nsfw_score_from_probs
from app.models.types import NsfwPrediction

logger = logging.getLogger(__name__)


class TorchNsfwClassifier:
    """Image classification wrapper using PyTorch / Transformers."""

    runtime = "torch"

    def __init__(
        self,
        model_id: str,
        device: str,
        *,
        torch_compile: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.torch_compile = torch_compile
        self._model = None
        self._processor = None
        self._id2label: dict[int, str] = {}

    def load(self) -> None:
        if self._model is not None:
            return
        logger.info(
            "Loading safety model (torch) %s on device=%s compile=%s",
            self.model_id,
            self.device,
            self.torch_compile,
        )
        self._processor = AutoImageProcessor.from_pretrained(
            self.model_id, use_fast=False
        )
        self._model = AutoModelForImageClassification.from_pretrained(self.model_id)
        self._model.to(self.device)
        self._model.eval()
        self._id2label = self._model.config.id2label or {}

        if self.torch_compile:
            self._model = torch.compile(self._model, mode="reduce-overhead")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, images: list[Image.Image]) -> list[NsfwPrediction]:
        self.load()
        assert self._model is not None and self._processor is not None

        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

        out: list[NsfwPrediction] = []
        for row in probs:
            score, label = nsfw_score_from_probs(row.cpu().tolist(), self._id2label)
            out.append(NsfwPrediction(nsfw_score=round(score, 4), label=label))
        return out
