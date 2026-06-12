"""
NSFW Classifiers — FalconSFW and multi-class safety evaluation models.

This module contains classification wrappers to run visual content moderation
checking. It supports binary classification (e.g. Falconsai/nsfw_image_detection)
and multi-class models (e.g. strangerguardhf/nsfw-image-detection).
"""

import logging
from dataclasses import dataclass

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

logger = logging.getLogger(__name__)

# Keywords representing non-SFW content across various model output labels.
# If these keywords match the model heads, their probability score is treated
# as a safety violation signal.
_NSFW_LABEL_KEYWORDS = frozenset(
    {
        "nsfw",
        "porn",
        "pornography",
        "hentai",
        "nude",
        "explicit",
        "enticing",
        "sensual",
    }
)


@dataclass
class NsfwPrediction:
    """
    Data model representing safety assessment results.
    """

    # Floating point confidence score representing the likelihood of NSFW contents [0, 1]
    nsfw_score: float

    # Text label matching the dominant classification (e.g., 'normal', 'nsfw', 'porn')
    label: str


class NsfwClassifier:
    """
    Wraps AutoModelForImageClassification for thread-safe lazy-loading and inference.
    """

    def __init__(self, model_id: str, device: str) -> None:
        """
        Initializes settings without loading heavy model parameters into memory.
        """
        self.model_id = model_id
        self.device = device

        # Lazy-loaded model instances
        self._model = None
        self._processor = None

        # Maps model logits indices to text labels (e.g., {0: "normal", 1: "nsfw"})
        self._id2label: dict[int, str] = {}

    def load(self) -> None:
        """
        Loads the image processor and model parameters onto the designated hardware.
        Called lazily on the first prediction request or during pre-warmup.
        """
        if self._model is not None:
            return
        logger.info("Loading safety model %s on device: %s", self.model_id, self.device)
        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageClassification.from_pretrained(self.model_id)
        self._model.to(self.device)
        self._model.eval()  # Put model in evaluation mode to turn off dropout/batchnorm
        self._id2label = self._model.config.id2label or {}

    @property
    def is_loaded(self) -> bool:
        """
        Returns true if the model has been loaded.
        """
        return self._model is not None

    def _nsfw_score_from_probs(self, probs: torch.Tensor) -> tuple[float, str]:
        """
        Determines the total NSFW score and best category label from category probabilities.
        """
        # Argmax returns the index of the highest probability category
        best_idx = int(probs.argmax().item())
        best_label = self._id2label.get(best_idx, str(best_idx)).lower()

        nsfw_score = 0.0
        top_label = best_label

        # Sum or max the probability scores of all labels matching our keyword filters
        for idx, prob in enumerate(probs.tolist()):
            label = self._id2label.get(idx, str(idx)).lower()
            if label in ("normal", "sfw", "safe"):
                continue
            if any(kw in label for kw in _NSFW_LABEL_KEYWORDS):
                nsfw_score = max(nsfw_score, prob)

        # Binary fallback: If nsfw_score is 0 but the dominant label is classified
        # as non-safe, default the score to the dominant probability.
        if nsfw_score == 0.0 and best_label not in ("normal", "sfw", "safe"):
            nsfw_score = float(probs[best_idx].item())
            top_label = best_label
        elif nsfw_score == 0.0:
            top_label = "normal"

        return nsfw_score, top_label

    def predict(self, images: list[Image.Image]) -> list[NsfwPrediction]:
        """
        Executes inference on a batch of PIL images.
        """
        # Ensure the model is loaded in memory/GPU
        self.load()
        assert self._model is not None and self._processor is not None

        # Preprocess images to convert them to model inputs (e.g. resizing, normalizations)
        inputs = self._processor(images=images, return_tensors="pt")
        # Move tensors to CPU or CUDA device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Disable gradient tracking for faster inference and lower memory usage
        with torch.no_grad():
            logits = self._model(**inputs).logits
            # Convert raw model output logits to [0, 1] probability distribution
            probs = torch.softmax(logits, dim=-1)

        # Resolve probability distribution back to SFW/NSFW categories
        out: list[NsfwPrediction] = []
        for row in probs:
            score, label = self._nsfw_score_from_probs(row)
            out.append(NsfwPrediction(nsfw_score=round(score, 4), label=label))
        return out
