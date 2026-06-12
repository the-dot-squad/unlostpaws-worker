"""
SigLIP2 Embeddings and Zero-Shot Relevance Verification.

This module houses the SigLIP2 model wrapper. It performs dual roles:
1. Extracting high-quality 768-dimensional normalized image embedding vectors.
2. Running zero-shot image classification to verify pet relevance using mathematical
   margins between pet prompts and negative distractors (mostly documents, advertisements).
"""

import logging
from dataclasses import dataclass

import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForZeroShotImageClassification,
    AutoTokenizer,
)

logger = logging.getLogger(__name__)

# Default Hugging Face repository for Google's SigLIP2 model
DEFAULT_MATCH_MODEL = "google/siglip2-base-patch16-224"

# Positive classification prompts mapping pet classes to descriptions.
PET_PROMPTS: dict[str, str] = {
    "dog": "a clear photo of a dog",
    "cat": "a clear photo of a cat",
    "bird": "a clear photo of a bird",
    "rabbit": "a clear photo of a rabbit",
    "hamster": "a clear photo of a hamster",
    "fish": "a clear photo of a fish",
    "reptile": "a clear photo of a reptile",
    "horse": "a clear photo of a horse",
    "other": "a clear photo of a pet animal",
}

# Distractor prompts. Used to compute margins against positive prompts.
# Helps detect invalid uploads like document scans, advertisements, or landscapes.
NEGATIVE_PROMPTS = [
    "a landscape or scenery photo without animals",
    "a photo with mostly text or documents",
    "a product advertisement or packaging",
]


@dataclass
class EmbedPrediction:
    """
    Result wrapper for visual vector embeddings.
    """

    # 768-dimensional visual representation vector
    embedding: list[float]


@dataclass
class RelevancePrediction:
    """
    Result wrapper for pet verification relevance.
    """

    # Relevance rating [0.0, 1.0] indicating confidence that the image contains a pet
    pet_likelihood: float

    # Classification label representing the detected pet type (e.g. 'dog', 'cat')
    top_label: str


class SiglipEmbedder:
    """
    Wraps SigLIP2 model for parallel batch feature encoding and zero-shot reasoning.
    """

    # Gating flag checked by the pipeline orchestrator
    supports_relevance = True

    def __init__(self, model_id: str, device: str, batch_size: int) -> None:
        """
        Initializes configuration variables. Model loading is deferred (lazy loading).
        """
        self.model_id = model_id
        self.device = device
        self.batch_size = max(1, batch_size)

        # Lazy-loaded model instances
        self._model = None
        self._image_processor = None
        self._tokenizer = None

    def load(self) -> None:
        """
        Pre-loads SigLIP2 model, tokenizer, and image processor onto the target hardware device.
        """
        if self._model is not None:
            return
        logger.info(
            "Loading SigLIP2 model %s on device: %s", self.model_id, self.device
        )
        self._image_processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForZeroShotImageClassification.from_pretrained(
            self.model_id
        )
        self._model.to(self.device)
        self._model.eval()

    @property
    def is_loaded(self) -> bool:
        """
        Checks if the model is currently active in memory.
        """
        return self._model is not None

    def _image_features(self, images: list[Image.Image]) -> torch.Tensor:
        """
        Extracts 768-dimensional vision features from a list of PIL images.
        Enforces L2 normalization so that dot products yield standard cosine similarity.
        """
        assert self._model is not None and self._image_processor is not None

        # Preprocess images to match model dimensions and values
        inputs = self._image_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # Extract raw vision representation features
            features = self._model.get_image_features(**inputs)
            # Perform L2 normalization on features: features / ||features||
            features = features / features.norm(p=2, dim=-1, keepdim=True)

        return features

    def embed_batch(self, images: list[Image.Image]) -> list[EmbedPrediction]:
        """
        Generates vector embeddings for a list of images, executing in chunks of self.batch_size.
        """
        self.load()
        results: list[EmbedPrediction] = []
        for start in range(0, len(images), self.batch_size):
            chunk = images[start : start + self.batch_size]
            features = self._image_features(chunk)
            # Convert GPU/CPU tensors to standard python list of floats
            for vec in features.cpu().tolist():
                results.append(EmbedPrediction(embedding=vec))
        return results

    def relevance_batch(
        self, images: list[Image.Image], pet_type: str = ""
    ) -> list[RelevancePrediction]:
        """
        Evaluates pet relevance using zero-shot image classification and margin math.

        Mathematical concept:
        We compare the similarity of the image against a list of positive prompts
        (e.g., "a photo of a dog") and negative prompts (scenery, text, advertisements).
        We calculate:
        - The margin: Similarity difference between the best positive label and the best negative label.
        - Sigmoid margin score: Maps this margin difference to [0.0, 1.0].
        - Softmax confidence: Probability distribution among pet categories.
        - The final score is a weighted mean of the margin sigmoid and the pet softmax confidence.
        """
        self.load()
        assert self._model is not None and self._tokenizer is not None

        pet_keys = list(PET_PROMPTS.keys())
        pet_prompts = list(PET_PROMPTS.values())
        all_prompts = pet_prompts + NEGATIVE_PROMPTS
        n_pet = len(pet_prompts)

        results: list[RelevancePrediction] = []
        for image in images:
            # 1. Preprocess the single image
            image_inputs = self._image_processor(images=[image], return_tensors="pt")

            # 2. Tokenize positive and negative prompts
            text_inputs = self._tokenizer(
                all_prompts, return_tensors="pt", padding=True, truncation=True
            )

            # Merge image and text features into a single batch payload
            batch = {
                k: v.to(self.device) for k, v in {**image_inputs, **text_inputs}.items()
            }

            with torch.no_grad():
                # Get raw logits (scale and bias applied similarities) for the image against text prompts
                logits = self._model(**batch).logits_per_image[0]

            # 3. Separate pet (positive) logits and distractor (negative) logits
            pet_logits = logits[:n_pet]
            neg_logits = logits[n_pet:]

            # Find the best pet match index and score
            best_pet_idx = int(pet_logits.argmax().item())
            pet_max = pet_logits.max()
            neg_max = neg_logits.max()

            # If user provided a specific target pet class (e.g. dog), compare against that type
            compare_logit = pet_max
            if pet_type and pet_type in PET_PROMPTS:
                typed_idx = pet_keys.index(pet_type)
                # Take the max logit between the best general pet and the specific requested pet type
                compare_logit = torch.max(pet_max, pet_logits[typed_idx])

            # 4. Math Logic: Compute Sigmoid margin and Softmax pet distribution
            # margin_score maps the distance (compare_logit - neg_max) via a sigmoid function.
            # Large positive margin -> score near 1.0. Large negative margin -> score near 0.0.
            margin_score = torch.sigmoid(compare_logit - neg_max)

            # pet_confidence computes softmax probabilities restricted to pet labels only
            pet_confidence = torch.softmax(pet_logits, dim=0).max()

            # Combine margin confidence and label classification confidence (50% weight each)
            likelihood = float((0.5 * margin_score + 0.5 * pet_confidence).item())

            # 5. Determine the resulting top pet label
            top_label = pet_keys[best_pet_idx]
            if pet_type and pet_type in PET_PROMPTS:
                typed_idx = pet_keys.index(pet_type)
                # If specific pet type has a comparable logit value, prefer it to prevent false overrides
                if pet_logits[typed_idx] >= pet_logits[best_pet_idx]:
                    top_label = pet_type

            results.append(
                RelevancePrediction(
                    pet_likelihood=round(likelihood, 4),
                    top_label=top_label,
                )
            )
        return results
