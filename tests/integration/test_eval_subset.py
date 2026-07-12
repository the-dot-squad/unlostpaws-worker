"""Integration test for in-repo eval fixture subset (mocked inference)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from PIL import Image


@pytest.mark.integration
@pytest.mark.asyncio
async def test_eval_subset_quality_beats_standard(monkeypatch, tmp_path):
    """Quality tier should score >= standard on the same mocked animal fixture."""
    from app.pipeline.stages.match import MatchStageResult
    from app.schemas.result import RelevanceResult

    fixtures = tmp_path / "eval"
    fixtures.mkdir()
    Image.new("RGB", (384, 384), color=(180, 140, 90)).save(
        fixtures / "animal_warm.png"
    )

    standard_match = MatchStageResult(
        embedding=[0.1] * 768,
        relevance=RelevanceResult(petLikelihood=0.55, topLabel="dog"),
    )
    quality_match = MatchStageResult(
        embedding=[0.2] * 768,
        relevance=RelevanceResult(petLikelihood=0.72, topLabel="dog"),
    )

    async def fake_match(images, pet_type, cfg):
        if cfg.profile.name == "quality":
            return [quality_match]
        return [standard_match]

    with patch("app.pipeline.stages.match.match_stage", new=fake_match):
        from tools.eval import evaluate_profile

        standard = await evaluate_profile("standard", fixtures)
        quality = await evaluate_profile("quality", fixtures)

    assert standard["ok"] and quality["ok"]
    assert quality["meanPetLikelihood"] >= standard["meanPetLikelihood"]
    assert quality["fixtures"][0]["embeddingDim"] == 768
    assert standard["fixtures"][0]["embeddingDim"] == 768
