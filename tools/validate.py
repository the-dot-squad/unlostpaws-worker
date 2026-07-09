"""Compare ONNX model outputs against PyTorch baselines (maintainer / CI)."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from tools._paths import ensure_import_path

logger = logging.getLogger(__name__)

EMBED_THRESHOLD = 0.99
RELEVANCE_THRESHOLD = 0.05
NSFW_THRESHOLD = 0.02


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(va, vb) / denom)


def load_fixture_images(directory: Path) -> list[Image.Image]:
    if not directory.is_dir():
        logger.warning(
            "Fixture directory missing: %s — using synthetic image", directory
        )
        return [Image.new("RGB", (224, 224), color=(128, 64, 32))]
    images: list[Image.Image] = []
    for path in sorted(directory.glob("*")):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            images.append(Image.open(path).convert("RGB"))
    if not images:
        images.append(Image.new("RGB", (224, 224), color=(128, 64, 32)))
    return images


def stage_export_to_cache(
    export_dir: Path, cache_dir: Path, model_id: str, folder: str
) -> None:
    safe = model_id.replace("/", "__")
    target = cache_dir / safe / "fp32"
    target.mkdir(parents=True, exist_ok=True)
    src = export_dir / folder
    if not src.is_dir():
        raise FileNotFoundError(f"Missing export folder: {src}")
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, target / item.name)


def validate_siglip(images: list[Image.Image], models_dir: Path) -> bool:
    from app.models.onnx_siglip import OnnxSiglipEmbedder
    from app.models.torch_siglip import TorchSiglipEmbedder

    model_id = "google/siglip2-base-patch16-224"
    torch_model = TorchSiglipEmbedder(model_id, "cpu", 1)
    onnx_model = OnnxSiglipEmbedder(
        model_id,
        1,
        precision="fp32",
        execution_provider="cpu",
        model_cache_dir=str(models_dir),
    )
    torch_model.load()
    onnx_model.load()

    ok = True
    for i, (te, oe) in enumerate(
        zip(torch_model.embed_batch(images), onnx_model.embed_batch(images))
    ):
        sim = cosine_similarity(te.embedding, oe.embedding)
        if sim < EMBED_THRESHOLD:
            logger.error(
                "Embed cosine %.4f < %.2f for image %d", sim, EMBED_THRESHOLD, i
            )
            ok = False

    for i, (tr, or_) in enumerate(
        zip(torch_model.relevance_batch(images), onnx_model.relevance_batch(images))
    ):
        delta = abs(tr.pet_likelihood - or_.pet_likelihood)
        if delta > RELEVANCE_THRESHOLD:
            logger.error(
                "Relevance delta %.4f > %.2f for image %d",
                delta,
                RELEVANCE_THRESHOLD,
                i,
            )
            ok = False
    return ok


def validate_nsfw(
    images: list[Image.Image], models_dir: Path, model_id: str, folder: str
) -> bool:
    from app.models.onnx_nsfw import OnnxNsfwClassifier
    from app.models.torch_nsfw import TorchNsfwClassifier

    torch_model = TorchNsfwClassifier(model_id, "cpu")
    onnx_model = OnnxNsfwClassifier(
        model_id,
        precision="fp32",
        execution_provider="cpu",
        model_cache_dir=str(models_dir),
    )
    torch_model.load()
    onnx_model.load()

    ok = True
    for i, (tp, op) in enumerate(
        zip(torch_model.predict(images), onnx_model.predict(images))
    ):
        delta = abs(tp.nsfw_score - op.nsfw_score)
        if delta > NSFW_THRESHOLD:
            logger.error(
                "NSFW delta %.4f > %.2f for image %d", delta, NSFW_THRESHOLD, i
            )
            ok = False
    return ok


def run_validate(models_dir: Path, fixtures: Path) -> int:
    ensure_import_path()
    cache_dir = models_dir / "cache"
    images = load_fixture_images(fixtures)

    stage_export_to_cache(
        models_dir, cache_dir, "google/siglip2-base-patch16-224", "siglip2"
    )
    stage_export_to_cache(
        models_dir, cache_dir, "Falconsai/nsfw_image_detection", "nsfw-falconsai"
    )

    results = [
        ("siglip2", validate_siglip(images, cache_dir)),
        (
            "nsfw-falconsai",
            validate_nsfw(
                images, cache_dir, "Falconsai/nsfw_image_detection", "nsfw-falconsai"
            ),
        ),
    ]

    if (models_dir / "nsfw-strangerguard").is_dir():
        stage_export_to_cache(
            models_dir,
            cache_dir,
            "strangerguardhf/nsfw-image-detection",
            "nsfw-strangerguard",
        )
        results.append(
            (
                "nsfw-strangerguard",
                validate_nsfw(
                    images,
                    cache_dir,
                    "strangerguardhf/nsfw-image-detection",
                    "nsfw-strangerguard",
                ),
            )
        )

    failed = [name for name, passed in results if not passed]
    if failed:
        logger.error("Validation failed for: %s", ", ".join(failed))
        return 1
    logger.info("All validations passed (%d model groups)", len(results))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Validate ONNX vs PyTorch accuracy")
    parser.add_argument("--models-dir", default="output/onnx")
    parser.add_argument("--fixtures", default="tests/fixtures/images")
    args = parser.parse_args(argv)
    return run_validate(Path(args.models_dir), Path(args.fixtures))
