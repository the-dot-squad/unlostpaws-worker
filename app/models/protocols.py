"""
Backend protocols for inference runtimes.

Pipeline stages depend on these structural contracts, not concrete torch/onnx
implementations. Both backends expose identical method signatures.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from PIL import Image

from app.models.types import EmbedPrediction, NsfwPrediction, RelevancePrediction


@runtime_checkable
class EmbedderBackend(Protocol):
    """Contract for image embedding and optional relevance scoring."""

    supports_relevance: bool
    model_id: str
    runtime: str

    def load(self) -> None: ...

    @property
    def is_loaded(self) -> bool: ...

    def embed_batch(self, images: list[Image.Image]) -> list[EmbedPrediction]: ...

    def relevance_batch(
        self, images: list[Image.Image], pet_type: str = ""
    ) -> list[RelevancePrediction]: ...


@runtime_checkable
class ClassifierBackend(Protocol):
    """Contract for NSFW / safety image classifiers."""

    model_id: str
    runtime: str

    def load(self) -> None: ...

    @property
    def is_loaded(self) -> bool: ...

    def predict(self, images: list[Image.Image]) -> list[NsfwPrediction]: ...
