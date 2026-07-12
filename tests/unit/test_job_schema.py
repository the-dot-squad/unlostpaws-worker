"""Tests for Redis job payload validation."""

import pytest
from pydantic import ValidationError

from app.schemas.job import JobPayload, parse_job


def test_pet_type_valid_normalized():
    job = JobPayload(imageUrls=["https://example.com/a.jpg"], petType="Dog")
    assert job.petType == "dog"


def test_pet_type_empty():
    job = JobPayload(imageUrls=["https://example.com/a.jpg"], petType="")
    assert job.petType == ""


def test_pet_type_unknown_becomes_empty():
    job = JobPayload(imageUrls=["https://example.com/a.jpg"], petType="dragon")
    assert job.petType == ""


def test_pet_type_whitespace_stripped():
    job = JobPayload(imageUrls=["https://example.com/a.jpg"], petType="  cat  ")
    assert job.petType == "cat"


@pytest.mark.parametrize(
    "pet_type",
    ["dog", "cat", "bird", "rabbit", "hamster", "fish", "reptile", "horse", "other"],
)
def test_all_valid_pet_types(pet_type: str):
    job = JobPayload(imageUrls=["https://example.com/a.jpg"], petType=pet_type)
    assert job.petType == pet_type


def test_parse_job_from_dict():
    raw = {"imageUrls": ["https://example.com/a.jpg"], "petType": "hamster"}
    job = parse_job(raw)
    assert job.petType == "hamster"


def test_parse_job_requires_image_urls():
    with pytest.raises(ValidationError):
        parse_job({"petType": "dog"})
