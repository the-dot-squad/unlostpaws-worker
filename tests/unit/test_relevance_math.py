"""Unit tests for shared pet-relevance scoring math."""

import numpy as np
import pytest

from app.models.relevance import (
    N_PET,
    NEGATIVE_PROMPTS,
    PET_KEYS,
    PROMPT_ENSEMBLES,
    compute_relevance_from_logits,
)


def _make_logits(pet_values: list[float], neg_values: list[float]) -> np.ndarray:
    # Build ensembled pet logits by duplicating value for each category's prompt count
    ensembled_pet = []
    for idx, category in enumerate(PET_KEYS):
        val = pet_values[idx]
        n_prompts = len(PROMPT_ENSEMBLES[category])
        ensembled_pet.extend([val] * n_prompts)

    # Pad or truncate neg_values to match actual NEGATIVE_PROMPTS length dynamically
    n_neg = len(NEGATIVE_PROMPTS)
    padded_neg = list(neg_values)
    if len(padded_neg) < n_neg:
        padded_neg.extend([-5.0] * (n_neg - len(padded_neg)))
    elif len(padded_neg) > n_neg:
        padded_neg = padded_neg[:n_neg]
    return np.array(ensembled_pet + padded_neg, dtype=np.float64)


def test_strong_pet_signal():
    logits = _make_logits(
        [5.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [-1.0, -2.0, -3.0]
    )
    result = compute_relevance_from_logits(logits)
    assert result.top_label == "dog"
    assert result.pet_likelihood > 0.7


def test_weak_pet_signal():
    logits = _make_logits([0.1] * N_PET, [0.2, 0.3, 0.4])
    result = compute_relevance_from_logits(logits)
    assert 0.0 <= result.pet_likelihood <= 1.0


def test_pet_type_override_label():
    logits = _make_logits(
        [1.0, 5.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [-1.0, -2.0, -3.0]
    )
    result = compute_relevance_from_logits(logits, pet_type="cat")
    assert result.top_label == "cat"


def test_pet_type_does_not_override_when_lower():
    logits = _make_logits(
        [5.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [-1.0, -2.0, -3.0]
    )
    result = compute_relevance_from_logits(logits, pet_type="cat")
    assert result.top_label == "dog"


def test_output_rounded_to_four_decimals():
    logits = _make_logits(
        [2.5, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [-0.5, -1.0, -1.5]
    )
    result = compute_relevance_from_logits(logits)
    assert result.pet_likelihood == round(result.pet_likelihood, 4)


def test_prompt_constants_length():
    assert len(PROMPT_ENSEMBLES) == N_PET
    assert len(NEGATIVE_PROMPTS) == 8
    assert len(PET_KEYS) == N_PET


@pytest.mark.parametrize("pet_type", ["", "dog", "rabbit", "other"])
def test_all_pet_types_accepted(pet_type: str):
    logits = _make_logits([1.0] * N_PET, [0.0, 0.0, 0.0])
    result = compute_relevance_from_logits(logits, pet_type=pet_type)
    assert result.top_label in PET_KEYS


def test_unrelated_object_rejection():
    # Simulated shoe/landscape image where negative distractors are high (-5.0)
    # and pet logits are low (-10.0)
    logits = _make_logits(
        [-10.0, -11.0, -11.0, -11.0, -11.0, -11.0, -11.0, -11.0, -10.5],
        [-5.0, -11.0, -11.0],
    )
    result = compute_relevance_from_logits(logits)
    # Margin score = sigmoid(-10.0 - (-5.0)) = sigmoid(-5.0) = ~0.0067
    # Confidence score (softmax max of pet logits) = ~0.38
    # Weighted likelihood = 0.5 * 0.0067 + 0.5 * 0.38 = ~0.193 (which is < 0.25)
    assert result.pet_likelihood < 0.25


def test_rare_or_poor_pet_retention():
    # Simulated poor quality photo or rare pet where best pet logit is low (-2.0)
    # but still much higher than distractor logits (-5.0)
    logits = _make_logits(
        [-2.0, -4.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -3.0], [-5.0, -6.0, -6.0]
    )
    result = compute_relevance_from_logits(logits)
    # A positive margin (+3.0) keeps the likelihood high (> 0.5) to avoid false negatives.
    assert result.pet_likelihood > 0.50
