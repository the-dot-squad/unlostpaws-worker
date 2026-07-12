"""
ONNX Runtime NSFW classifier — Falconsai and StrangerGuard safety models.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image
from transformers import AutoImageProcessor

from app.models.artifacts import resolve_onnx_artifacts
from app.models.onnx_session import OnnxSession
from app.models.safety_scoring import nsfw_score_from_probs
from app.models.types import NsfwPrediction

logger = logging.getLogger(__name__)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


class OnnxNsfwClassifier:
    """Image classifier using ONNX Runtime."""

    runtime = "onnx"

    def __init__(
        self,
        model_id: str,
        *,
        precision: str = "fp32",
        execution_provider: str = "auto",
        model_cache_dir: str = "/app/.cache/huggingface/onnx",
        tensorrt_cache_dir: str = "/app/.cache/tensorrt",
        openvino_device: str = "CPU",
    ) -> None:
        self.model_id = model_id
        self.precision = precision
        self.execution_provider = execution_provider
        self.model_cache_dir = model_cache_dir
        self.tensorrt_cache_dir = tensorrt_cache_dir
        self.openvino_device = openvino_device

        self._session: OnnxSession | None = None
        self._processor = None
        self._id2label: dict[int, str] = {}
        self.active_provider: str = ""

    def load(self) -> None:
        if self._session is not None:
            return

        artifacts = resolve_onnx_artifacts(
            self.model_id, self.precision, self.model_cache_dir
        )
        logger.info(
            "Loading safety model (onnx) %s precision=%s from %s",
            self.model_id,
            self.precision,
            artifacts.onnx_file,
        )

        self._session = OnnxSession(
            artifacts.onnx_file,
            execution_provider=self.execution_provider,
            tensorrt_cache_dir=self.tensorrt_cache_dir,
            openvino_device=self.openvino_device,
        )
        self.active_provider = self._session.active_provider

        processor_source = (
            artifacts.model_dir
            if (artifacts.model_dir / "preprocessor_config.json").exists()
            else self.model_id
        )
        self._processor = AutoImageProcessor.from_pretrained(
            processor_source, use_fast=False
        )

        config_path = artifacts.model_dir / "config.json"
        if config_path.is_file():
            import json

            config = json.loads(config_path.read_text(encoding="utf-8"))
            raw = config.get("id2label", {})
            self._id2label = {int(k): v for k, v in raw.items()}

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    def predict(self, images: list[Image.Image]) -> list[NsfwPrediction]:
        self.load()
        assert self._session is not None and self._processor is not None

        pixel_values = self._processor(images=images, return_tensors="np")[
            "pixel_values"
        ].astype(np.float32)

        logits = self._session.run_output({"pixel_values": pixel_values}, "logits")
        probs = _softmax(logits.astype(np.float64))

        out: list[NsfwPrediction] = []
        for row in probs:
            score, label = nsfw_score_from_probs(row.tolist(), self._id2label)
            out.append(NsfwPrediction(nsfw_score=round(score, 4), label=label))
        return out
