"""Tests that optional petType from the job payload reaches match_stage."""

from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from app.pipeline.stages.match import MatchStageResult
from app.schemas.result import RelevanceResult


@pytest.mark.asyncio
async def test_orchestrator_forwards_pet_type_hint(monkeypatch):
    monkeypatch.setenv("VISION_PROFILE", "standard")
    monkeypatch.setenv("INFERENCE_RUNTIME", "torch")

    from importlib import reload

    import app.config.settings as settings_mod

    reload(settings_mod)

    from app.config.settings import load_settings
    from app.pipeline.orchestrator import run_pipeline

    cfg = load_settings()
    captured_pet_type: list[str] = []

    async def fake_match(images, pet_type, config):
        captured_pet_type.append(pet_type)
        return [
            MatchStageResult(
                embedding=[0.1] * 768,
                relevance=RelevanceResult(
                    petLikelihood=0.8, topLabel=pet_type or "dog"
                ),
            )
        ]

    async def fake_download(urls):
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        return [type("D", (), {"url": urls[0], "image": img, "raw_bytes": b"x"})()], []

    with (
        patch("app.pipeline.orchestrator.download_all", new=fake_download),
        patch(
            "app.pipeline.orchestrator.safety_stage", new=AsyncMock(return_value=[None])
        ),
        patch("app.pipeline.orchestrator.match_stage", new=fake_match),
    ):
        job = {
            "jobType": "listing",
            "listingId": "test_1",
            "imageUrls": ["http://127.0.0.1/test.jpg"],
            "petType": "cat",
        }
        result = await run_pipeline(job, cfg)

    assert captured_pet_type == ["cat"]
    assert result.images[0].relevance.topLabel == "cat"
