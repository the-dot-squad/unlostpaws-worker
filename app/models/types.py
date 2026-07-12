"""
Shared prediction datatypes used by all inference backends.

These types are backend-agnostic: torch, ONNX, and future runtimes all return
the same structures so pipeline stages never depend on a specific ML framework.
"""

from dataclasses import dataclass


@dataclass
class EmbedPrediction:
    """Result wrapper for visual vector embeddings."""

    embedding: list[float]


@dataclass
class RelevancePrediction:
    """Result wrapper for pet verification relevance."""

    pet_likelihood: float
    top_label: str


@dataclass
class MatchPrediction:
    """Combined embedding and optional relevance from one vision forward pass."""

    embedding: list[float]
    relevance: RelevancePrediction | None = None


@dataclass
class NsfwPrediction:
    """Result wrapper for NSFW safety classification."""

    nsfw_score: float
    label: str
