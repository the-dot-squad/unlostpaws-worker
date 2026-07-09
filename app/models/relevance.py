"""
Shared pet-relevance scoring logic for SigLIP-based zero-shot classification.

Both PyTorch and ONNX backends produce per-prompt logits (image-text similarity
scores). This module converts those logits into a single pet_likelihood score
and top_label using margin-based sigmoid blending — the same algorithm used
since the original siglip.py implementation.
"""

import numpy as np

from app.models.types import RelevancePrediction

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

# Distractor prompts compared against positive pet prompts.
NEGATIVE_PROMPTS: list[str] = [
    "a landscape or scenery photo without animals",
    "a photo with mostly text or documents",
    "a product advertisement or packaging",
]

PET_KEYS: list[str] = list(PET_PROMPTS.keys())
N_PET: int = len(PET_PROMPTS)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _softmax_max(values: np.ndarray) -> float:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp)
    return float(np.max(probs))


def compute_relevance_from_logits(
    logits: np.ndarray,
    pet_type: str = "",
    pet_keys: list[str] | None = None,
) -> RelevancePrediction:
    """
    Convert a vector of per-prompt logits into a relevance score and label.

    Args:
        logits: 1-D array of length ``len(PET_PROMPTS) + len(NEGATIVE_PROMPTS)``.
        pet_type: Optional requested pet class (e.g. ``"dog"``) to bias label selection.
        pet_keys: Ordered pet label keys matching the first ``N_PET`` logits entries.

    Returns:
        RelevancePrediction with rounded pet_likelihood and resolved top_label.
    """
    keys = pet_keys or PET_KEYS
    n_pet = N_PET

    pet_logits = np.asarray(logits[:n_pet], dtype=np.float64)
    neg_logits = np.asarray(logits[n_pet:], dtype=np.float64)

    best_pet_idx = int(np.argmax(pet_logits))
    pet_max = float(np.max(pet_logits))
    neg_max = float(np.max(neg_logits))

    compare_logit = pet_max
    if pet_type and pet_type in PET_PROMPTS:
        typed_idx = keys.index(pet_type)
        compare_logit = max(pet_max, float(pet_logits[typed_idx]))

    margin_score = _sigmoid(compare_logit - neg_max)
    pet_confidence = _softmax_max(pet_logits)
    likelihood = 0.5 * margin_score + 0.5 * pet_confidence

    top_label = keys[best_pet_idx]
    if pet_type and pet_type in PET_PROMPTS:
        typed_idx = keys.index(pet_type)
        if pet_logits[typed_idx] >= pet_logits[best_pet_idx]:
            top_label = pet_type

    return RelevancePrediction(
        pet_likelihood=round(likelihood, 4),
        top_label=top_label,
    )
