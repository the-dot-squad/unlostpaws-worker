"""End-to-end pipeline smoke test for a single VISION_PROFILE."""

from __future__ import annotations

import os
from importlib import reload
from io import BytesIO
from unittest.mock import patch

from PIL import Image

from tools._paths import ROOT, ensure_import_path


def _fixture_image() -> Image.Image:
    """Prefer a real fixture; fall back to a synthetic RGB image."""
    fixtures = ROOT / "tests" / "fixtures" / "images"
    if fixtures.is_dir():
        for path in sorted(fixtures.glob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                return Image.open(path).convert("RGB")
    return Image.new("RGB", (400, 300), color=(90, 140, 200))


async def run_smoke(profile: str) -> dict:
    """
    Warm up models for ``profile``, run one mocked pipeline pass, return a JSON-serializable report.
    """
    ensure_import_path()
    os.environ["VISION_PROFILE"] = profile
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    os.environ.setdefault(
        "MODEL_CACHE_DIR", str(ROOT / ".cache" / "huggingface" / "onnx")
    )

    import app.config.settings as settings_mod
    import app.models.registry as registry_mod

    reload(settings_mod)
    reload(registry_mod)

    from app.config.settings import load_settings
    from app.models.registry import health_models, warmup
    from app.pipeline.download import DecodedImage
    from app.pipeline.orchestrator import run_pipeline

    cfg = load_settings()
    image = _fixture_image()
    buf = BytesIO()
    image.save(buf, format="JPEG")
    decoded = DecodedImage(
        url="test://local.jpg", raw_bytes=buf.getvalue(), image=image
    )

    await warmup(cfg)
    health = health_models(cfg)

    report: dict = {
        "profile": profile,
        "runtime": cfg.runtime,
        "executionProvider": health.get("executionProvider", ""),
        "device": health.get("device", ""),
        "matchLoaded": health.get("matchLoaded"),
        "safetyLoaded": health.get("safetyLoaded"),
        "stages": list(cfg.profile.stages),
        "ok": True,
        "error": None,
    }

    job = {
        "jobType": "listing",
        "listingId": "smoke-test",
        "imageUrls": ["test://local.jpg"],
        "petType": "dog",
        "webhookUrl": "http://localhost:9999/callback",
    }

    try:
        with patch(
            "app.pipeline.orchestrator.download_all",
            return_value=([decoded], []),
        ):
            result = await run_pipeline(job, cfg)

        if not result.images:
            raise RuntimeError(
                result.errors[0].error if result.errors else "no images processed"
            )

        img = result.images[0]
        report["output"] = {
            "qualityOk": img.quality.ok if img.quality else None,
            "safetyLabel": img.safety.label if img.safety else None,
            "embeddingDim": len(img.embedding) if img.embedding else 0,
            "relevanceScore": round(img.relevance.petLikelihood, 4)
            if img.relevance
            else None,
            "md5Len": len(img.md5) if img.md5 else 0,
        }
    except Exception as exc:
        report["ok"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"

    return report
