"""
In-repo relevance evaluation on a committed fixture subset.

These solid-color PNGs are CI smoke fixtures only — they are not real pet photos.
For accuracy on the 305-image dev set, use:

  python dev_benchmarks/evaluate_workflow.py --profile standard
"""

from __future__ import annotations

import argparse
import json
import os
from importlib import reload
from pathlib import Path

from PIL import Image

from tools._console import out
from tools._paths import ROOT, ensure_import_path

EVAL_FIXTURES = ROOT / "tests" / "fixtures" / "eval"


def _load_eval_images(directory: Path) -> list[tuple[str, Image.Image]]:
    images: list[tuple[str, Image.Image]] = []
    for path in sorted(directory.glob("*.png")):
        images.append((path.stem, Image.open(path).convert("RGB")))
    return images


def _ensure_fixtures(directory: Path) -> None:
    """Create minimal synthetic fixtures when none are committed."""
    if any(directory.glob("*.png")):
        return
    directory.mkdir(parents=True, exist_ok=True)
    # Warm pet-like colors vs neutral scene — scoring is heuristic, not ground truth.
    Image.new("RGB", (384, 384), color=(180, 140, 90)).save(
        directory / "animal_warm.png"
    )
    Image.new("RGB", (384, 384), color=(40, 120, 200)).save(
        directory / "scene_cool.png"
    )


async def evaluate_profile(profile: str, fixtures_dir: Path) -> dict:
    ensure_import_path()
    os.environ["VISION_PROFILE"] = profile
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

    import app.config.settings as settings_mod
    import app.models.registry as registry_mod

    reload(settings_mod)
    reload(registry_mod)

    from app.config.settings import load_settings
    from app.models.registry import warmup
    from app.pipeline.stages.match import match_stage

    cfg = load_settings()
    _ensure_fixtures(fixtures_dir)
    items = _load_eval_images(fixtures_dir)
    if not items:
        return {"profile": profile, "ok": False, "error": "no fixtures found"}

    await warmup(cfg)
    names, images = zip(*items, strict=True)
    matches = await match_stage(list(images), "dog", cfg)

    rows = []
    scores = []
    for name, match in zip(names, matches, strict=True):
        score = match.relevance.petLikelihood if match.relevance else None
        if score is not None:
            scores.append(score)
        rows.append(
            {
                "fixture": name,
                "petLikelihood": score,
                "topLabel": match.relevance.topLabel if match.relevance else None,
                "embeddingDim": len(match.embedding),
            }
        )

    return {
        "profile": profile,
        "runtime": cfg.runtime,
        "matchModel": cfg.match_model,
        "relevanceEnabled": cfg.relevance_enabled,
        "fixtures": rows,
        "meanPetLikelihood": round(sum(scores) / len(scores), 4) if scores else None,
        "ok": True,
    }


def run_eval(profile: str, fixtures_dir: Path) -> int:
    import asyncio

    result = asyncio.run(evaluate_profile(profile, fixtures_dir))
    out(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate relevance on fixture subset")
    parser.add_argument("--profile", default="quality")
    parser.add_argument("--fixtures", default=str(EVAL_FIXTURES))
    args = parser.parse_args(argv)
    return run_eval(args.profile, Path(args.fixtures))
