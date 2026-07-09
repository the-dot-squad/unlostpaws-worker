"""Unit tests for shared pet-relevance scoring math."""

import numpy as np
import pytest

from app.models.relevance import (
    N_PET,
    NEGATIVE_PROMPTS,
    PET_KEYS,
    PET_PROMPTS,
    compute_relevance_from_logits,
)


def _make_logits(pet_values: list[float], neg_values: list[float]) -> np.ndarray:
    return np.array(pet_values + neg_values, dtype=np.float64)


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
    assert len(PET_PROMPTS) == N_PET
    assert len(NEGATIVE_PROMPTS) == 3
    assert len(PET_KEYS) == N_PET


@pytest.mark.parametrize("pet_type", ["", "dog", "rabbit", "other"])
def test_all_pet_types_accepted(pet_type: str):
    logits = _make_logits([1.0] * N_PET, [0.0, 0.0, 0.0])
    result = compute_relevance_from_logits(logits, pet_type=pet_type)
    assert result.top_label in PET_KEYS
