"""
ONNX Runtime SigLIP2 embedder — embeddings and zero-shot pet relevance.

Uses precomputed text embeddings (same strategy as the torch backend) so
per-image inference only runs the vision encoder graph.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

from app.models.artifacts import resolve_onnx_artifacts
from app.models.onnx_session import OnnxSession
from app.models.relevance import (
    NEGATIVE_PROMPTS,
    PET_PROMPTS,
    compute_relevance_from_logits,
)
from app.models.types import EmbedPrediction, RelevancePrediction

logger = logging.getLogger(__name__)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


class OnnxSiglipEmbedder:
    """SigLIP2 wrapper using ONNX Runtime."""

    supports_relevance = True
    runtime = "onnx"

    def __init__(
        self,
        model_id: str,
        batch_size: int,
        *,
        precision: str = "fp32",
        execution_provider: str = "auto",
        model_cache_dir: str = "/app/.cache/huggingface/onnx",
        tensorrt_cache_dir: str = "/app/.cache/tensorrt",
        openvino_device: str = "CPU",
    ) -> None:
        self.model_id = model_id
        self.batch_size = max(1, batch_size)
        self.precision = precision
        self.execution_provider = execution_provider
        self.model_cache_dir = model_cache_dir
        self.tensorrt_cache_dir = tensorrt_cache_dir
        self.openvino_device = openvino_device

        self._session: OnnxSession | None = None
        self._image_processor = None
        self._tokenizer = None
        self._prompt_input_ids: np.ndarray | None = None
        self._prompt_attention_mask: np.ndarray | None = None
        self.active_provider: str = ""

    def load(self) -> None:
        if self._session is not None:
            return

        artifacts = resolve_onnx_artifacts(
            self.model_id, self.precision, self.model_cache_dir
        )
        logger.info(
            "Loading SigLIP2 (onnx) %s precision=%s from %s",
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
            artifacts.model_dir if artifacts.model_dir.is_dir() else self.model_id
        )
        tokenizer_source = (
            artifacts.model_dir
            if (artifacts.model_dir / "tokenizer.json").exists()
            else self.model_id
        )

        self._image_processor = AutoImageProcessor.from_pretrained(processor_source)
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        self._precompute_prompt_tokens()

    def _precompute_prompt_tokens(self) -> None:
        """Tokenize fixed pet + negative prompts once for relevance inference."""
        assert self._tokenizer is not None

        all_prompts = list(PET_PROMPTS.values()) + NEGATIVE_PROMPTS
        text_inputs = self._tokenizer(
            all_prompts, return_tensors="np", padding=True, truncation=True
        )
        self._prompt_input_ids = text_inputs["input_ids"].astype(np.int64)
        if "attention_mask" in text_inputs:
            self._prompt_attention_mask = text_inputs["attention_mask"].astype(np.int64)

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    def _image_features(self, images: list[Image.Image]) -> np.ndarray:
        assert self._session is not None and self._image_processor is not None

        pixel_values = self._image_processor(images=images, return_tensors="np")[
            "pixel_values"
        ].astype(np.float32)

        feeds: dict[str, np.ndarray] = {"pixel_values": pixel_values}
        if "input_ids" in self._session.input_names and self._tokenizer is not None:
            dummy = self._tokenizer(
                [""], return_tensors="np", padding=True, truncation=True
            )
            feeds["input_ids"] = np.tile(
                dummy["input_ids"].astype(np.int64), (len(images), 1)
            )
            if "attention_mask" in self._session.input_names:
                feeds["attention_mask"] = np.tile(
                    dummy["attention_mask"].astype(np.int64), (len(images), 1)
                )

        image_embeds = self._session.run_output(feeds, "image_embeds")
        return _l2_normalize(image_embeds.astype(np.float64))

    def _logits_per_image(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Run the ONNX graph and return native logits_per_image rows."""
        assert (
            self._session is not None
            and self._image_processor is not None
            and self._prompt_input_ids is not None
        )

        pixel_values = self._image_processor(images=images, return_tensors="np")[
            "pixel_values"
        ].astype(np.float32)

        logits_rows: list[np.ndarray] = []
        for row in pixel_values:
            feeds: dict[str, np.ndarray] = {
                "input_ids": self._prompt_input_ids,
                "pixel_values": np.expand_dims(row, axis=0),
            }
            if (
                self._prompt_attention_mask is not None
                and "attention_mask" in self._session.input_names
            ):
                feeds["attention_mask"] = self._prompt_attention_mask

            logits = self._session.run_output(feeds, "logits_per_image")
            logits_rows.append(np.asarray(logits[0], dtype=np.float64))
        return logits_rows

    def embed_batch(self, images: list[Image.Image]) -> list[EmbedPrediction]:
        self.load()
        results: list[EmbedPrediction] = []
        for start in range(0, len(images), self.batch_size):
            chunk = images[start : start + self.batch_size]
            features = self._image_features(chunk)
            for vec in features.tolist():
                results.append(EmbedPrediction(embedding=vec))
        return results

    def relevance_batch(
        self, images: list[Image.Image], pet_type: str = ""
    ) -> list[RelevancePrediction]:
        self.load()
        results: list[RelevancePrediction] = []
        for start in range(0, len(images), self.batch_size):
            chunk = images[start : start + self.batch_size]
            for logits in self._logits_per_image(chunk):
                results.append(compute_relevance_from_logits(logits, pet_type=pet_type))
        return results
