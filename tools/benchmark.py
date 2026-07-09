"""Inference latency benchmarks — single profile or batch JSON output."""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from importlib import reload
from pathlib import Path

from PIL import Image

from tools._console import out
from tools._paths import ROOT, ensure_import_path

DEFAULT_PROFILES = [
    "cpu-quality",
    "cpu-light",
    "cpu-standard",
    "onnx-cpu-quality",
    "onnx-apple",
]


def load_images(directory: Path) -> list[Image.Image]:
    if not directory.is_dir():
        return [Image.new("RGB", (224, 224), color=(100, 150, 200))]
    images: list[Image.Image] = []
    for path in sorted(directory.glob("*")):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            images.append(Image.open(path).convert("RGB"))
    return images or [Image.new("RGB", (224, 224), color=(100, 150, 200))]


def _stage_stats(timings: list[float]) -> dict[str, float] | None:
    if not timings:
        return None
    ordered = sorted(timings)
    return {
        "p50_ms": round(statistics.median(timings) * 1000, 1),
        "p95_ms": round(ordered[max(0, int(len(ordered) * 0.95) - 1)] * 1000, 1),
    }


async def benchmark_profile(
    profile: str,
    images_dir: Path,
    runs: int,
) -> dict:
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

    from app.config.profiles import get_preset
    from app.config.settings import load_settings
    from app.models.registry import warmup
    from app.pipeline.stages.embed import embed_stage
    from app.pipeline.stages.relevance import relevance_stage
    from app.pipeline.stages.safety import safety_stage

    preset = get_preset(profile)
    cfg = load_settings()
    images = load_images(images_dir)

    await warmup(cfg)

    stages: dict[str, list[float]] = {"safety": [], "embed": [], "relevance": []}
    for _ in range(runs):
        if cfg.safety_enabled:
            t0 = time.perf_counter()
            await safety_stage(images, cfg)
            stages["safety"].append(time.perf_counter() - t0)
        if cfg.embed_enabled:
            t0 = time.perf_counter()
            await embed_stage(images, cfg)
            stages["embed"].append(time.perf_counter() - t0)
        if cfg.relevance_enabled:
            t0 = time.perf_counter()
            await relevance_stage(images, "dog", cfg)
            stages["relevance"].append(time.perf_counter() - t0)

    return {
        "profile": profile,
        "runtime": preset.runtime,
        "execution_provider": preset.execution_provider,
        "precision": preset.precision,
        "images": len(images),
        "runs": runs,
        "stages": {name: _stage_stats(vals) for name, vals in stages.items() if vals},
    }


def print_benchmark(result: dict) -> None:
    out(
        f"\nProfile: {result['profile']} "
        f"(runtime={result['runtime']}, ep={result['execution_provider']})"
    )
    out(f"Images: {result['images']}  Runs: {result['runs']}")
    for stage, stats in result.get("stages", {}).items():
        if stats:
            out(
                f"  {stage:10s}  p50={stats['p50_ms']:7.1f}ms  "
                f"p95={stats['p95_ms']:7.1f}ms"
            )


async def run_benchmarks(
    profiles: list[str],
    images_dir: Path,
    runs: int,
    output: Path | None,
) -> int:
    results: list[dict] = []
    for profile in profiles:
        try:
            entry = await benchmark_profile(profile, images_dir, runs)
            results.append(entry)
            if len(profiles) == 1:
                print_benchmark(entry)
        except Exception as exc:
            results.append({"profile": profile, "ok": False, "error": str(exc)})

    if len(profiles) > 1:
        payload = {
            "meta": {"runs": runs, "images_dir": str(images_dir)},
            "benchmarks": results,
        }
        text = json.dumps(payload, indent=2)
        if output:
            output.write_text(text, encoding="utf-8")
        out(text)

    failed = [r for r in results if r.get("error") or not r.get("stages")]
    return 1 if failed else 0


def run_single(profile: str, images_dir: Path, runs: int) -> int:
    return asyncio.run(run_benchmarks([profile], images_dir, runs, None))
