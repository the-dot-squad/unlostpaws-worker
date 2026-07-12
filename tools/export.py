"""Export Hugging Face vision models to ONNX (maintainer / CI workflow)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from app.models.onnx_export import export_model_to_onnx
from tools._paths import ROOT, ensure_import_path

logger = logging.getLogger(__name__)

EXPORTS = [
    (
        "google/siglip2-base-patch16-224",
        "zero-shot-image-classification",
        "siglip2-base",
    ),
    (
        "google/siglip2-base-patch16-384",
        "zero-shot-image-classification",
        "siglip2-quality",
    ),
    ("Falconsai/nsfw_image_detection", "image-classification", "nsfw-falconsai"),
    (
        "strangerguardhf/nsfw-image-detection",
        "image-classification",
        "nsfw-strangerguard",
    ),
]


def export_model(model_id: str, task: str, output_dir: Path) -> None:
    export_model_to_onnx(model_id, task, output_dir)


def quantize_model(model_dir: Path) -> None:
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    quantizer = ORTQuantizer.from_pretrained(model_dir)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=model_dir, quantization_config=qconfig)
    logger.info("Quantized model saved under %s", model_dir)


def run_export(output: Path, skip_quantize: bool) -> None:
    ensure_import_path()
    manifest_path = ROOT / "app" / "models" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for model_id, task, folder in EXPORTS:
        target = output / folder
        export_model(model_id, task, target)
        if not skip_quantize:
            quantize_model(target)
        manifest[model_id] = manifest.get(model_id, {})
        manifest[model_id]["local_export"] = str(target)

    summary = output / "export_summary.json"
    summary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Export complete. Summary: %s", summary)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Export vision models to ONNX")
    parser.add_argument("--output", default="output/onnx")
    parser.add_argument("--skip-quantize", action="store_true")
    args = parser.parse_args(argv)
    run_export(Path(args.output), args.skip_quantize)
    return 0
