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
PROMPT_ENSEMBLES: dict[str, list[str]] = {
    "dog": [
        "a clear photo of a dog",
        "a close-up portrait of a pet dog's face",
        "a photo of a dog or puppy",
    ],
    "cat": [
        "a clear photo of a cat",
        "a close-up portrait of a pet cat's face",
        "a photo of a cat or kitten",
    ],
    "bird": [
        "a photo of a bird, parrot, budgie, or pet bird",
        "a close-up photo of a pet bird on a branch or in a cage",
    ],
    "rabbit": [
        "a clear photo of a rabbit",
        "a cute pet rabbit in a home or garden",
    ],
    "hamster": [
        "a clear photo of a hamster",
        "a small pet hamster, gerbil, or mouse in a cage",
    ],
    "fish": [
        "a photo of a fish, goldfish, or aquarium fish",
        "a pet fish swimming in an aquarium or fishbowl",
    ],
    "reptile": [
        "a photo of a reptile, lizard, snake, turtle, tortoise, or amphibian",
        "a pet lizard, snake, or turtle in a terrarium",
    ],
    "horse": [
        "a clear photo of a horse",
        "a domestic horse in a stable or pasture field",
    ],
    "other": [
        "a photo of a pet animal, domestic animal, monkey, ferret, guinea pig, or other exotic pet",
        "a close-up photo of a ferret, guinea pig, or small domestic mammal",
    ],
}

PET_KEYS: list[str] = list(PROMPT_ENSEMBLES.keys())
N_PET: int = len(PET_KEYS)

# To keep backward compatibility and allow torch/onnx models to load the full flat list of prompts,
# we map each individual prompt in the ensemble to a unique key.
PET_PROMPTS: dict[str, str] = {}
for category, prompts in PROMPT_ENSEMBLES.items():
    for i, prompt in enumerate(prompts):
        PET_PROMPTS[f"{category}_{i}"] = prompt

# Distractor prompts compared against positive pet prompts.
NEGATIVE_PROMPTS: list[str] = [
    "a landscape or scenery photo without animals",
    "a photo with mostly text or documents",
    "a product advertisement or packaging",
    "a photo of shoes, clothing, or fashion items",
    "a photo of a car, truck, bicycle, or vehicle",
    "a photo of food, drinks, or kitchen items",
    "a photo of furniture, house interior, or building",
    "a photo of a tree, plant, flower, or forest without animals",
    "a photo of a human, person, face, or selfie without pets",
    "a screenshot of a phone screen, app UI, text message, or document",
    "a cartoon, drawing, illustration, or graphic without real animals",
]


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
    formulation: str | None = None,
    temp_scale: float | None = None,
    margin_threshold: float | None = None,
) -> RelevancePrediction:
    """
    Convert a vector of per-prompt logits into a relevance score and label.

    When ``pet_type`` is set (e.g. ``"dog"`` from the job payload), it:
    - boosts the likelihood score toward that class when computing margin vs distractors
    - resolves ``topLabel`` to the hint when logit margin between top pet classes is < margin_threshold
    - does not override the label when the hint contradicts a strong model signal

    Args:
        logits: 1-D array containing the raw similarity logits for the ensembled prompts.
        pet_type: Optional requested pet class to bias label selection.
        pet_keys: Ordered pet label keys.
        formulation: Optional formulation override ("baseline" or "unified_softmax").
        temp_scale: Optional temperature scaling override.
        margin_threshold: Optional specific label decision margin threshold override.

    Returns:
        RelevancePrediction with rounded pet_likelihood and resolved top_label.
    """
    from app.config.settings import settings

    use_formulation = formulation or settings.relevance_formulation
    use_temp = temp_scale if temp_scale is not None else settings.relevance_temp_scale
    use_margin = margin_threshold if margin_threshold is not None else settings.relevance_margin_threshold

    keys = pet_keys or PET_KEYS

    # Map the individual prompt logits back to their categories and average them
    averaged_pet_logits = []
    current_idx = 0
    for category in keys:
        n_prompts = len(PROMPT_ENSEMBLES[category])
        class_logits = logits[current_idx : current_idx + n_prompts]
        averaged_pet_logits.append(float(np.mean(class_logits)))
        current_idx += n_prompts

    pet_logits = np.array(averaged_pet_logits, dtype=np.float64)
    neg_logits = np.asarray(logits[current_idx:], dtype=np.float64)

    best_pet_idx = int(np.argmax(pet_logits))
    pet_max = float(np.max(pet_logits))
    neg_max = float(np.max(neg_logits))

    if use_formulation == "unified_softmax":
        # Concatenate averaged pet logits and individual negative logits
        combined = np.concatenate([pet_logits, neg_logits]) / use_temp
        # Shift for numerical stability in softmax
        shifted = combined - np.max(combined)
        exp_vals = np.exp(shifted)
        probs = exp_vals / np.sum(exp_vals)
        likelihood = float(np.sum(probs[0 : len(keys)]))
    else:
        # Baseline margin-based sigmoid blending formulation
        compare_logit = pet_max
        if pet_type and pet_type in keys:
            typed_idx = keys.index(pet_type)
            compare_logit = max(pet_max, float(pet_logits[typed_idx]))

        margin_score = _sigmoid(compare_logit - neg_max)
        pet_confidence = _softmax_max(pet_logits)
        likelihood = 0.5 * margin_score + 0.5 * pet_confidence

    # Safe fine-grained fallback: if the model is uncertain between top pet categories,
    # default to "other" (generic pet) unless the user provided a stabilizing pet_type hint.
    sorted_pet_logits = np.sort(pet_logits)
    top1 = sorted_pet_logits[-1]
    top2 = sorted_pet_logits[-2]

    if (top1 - top2) < use_margin:
        top_label = "other"
    else:
        top_label = keys[best_pet_idx]

    if pet_type and pet_type in keys:
        typed_idx = keys.index(pet_type)
        if pet_logits[typed_idx] >= (top1 - use_margin):
            top_label = pet_type

    return RelevancePrediction(
        pet_likelihood=round(likelihood, 4),
        top_label=top_label,
    )
