"""Shared Hugging Face → ONNX export helpers.

Uses Optimum's exporter API for supported model/task pairs. SigLIP image
classifiers fall back to ``torch.onnx.export`` because Optimum only registers
SigLIP for feature-extraction and zero-shot tasks.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_ONNX_OPSET = 18


def _is_siglip_image_classifier(model_id: str) -> bool:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id)
    architectures = config.architectures or []
    return (
        config.model_type == "siglip"
        and "SiglipForImageClassification" in architectures
    )


def _export_siglip_image_classifier(model_id: str, output_dir: Path) -> None:
    """Export a fine-tuned SigLIP classifier to a single-input ONNX graph."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    output_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.eval()
    processor = AutoImageProcessor.from_pretrained(model_id)
    processor.save_pretrained(output_dir)
    model.config.save_pretrained(output_dir)

    class _VisionClassifier(torch.nn.Module):
        """Wrap the HF model so ONNX only exposes pixel_values → logits."""

        def __init__(self, backbone: torch.nn.Module) -> None:
            super().__init__()
            self.backbone = backbone

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            return self.backbone(pixel_values=pixel_values).logits

    wrapper = _VisionClassifier(model)
    sample = processor(
        images=[Image.new("RGB", (224, 224))],
        return_tensors="pt",
    )["pixel_values"]

    onnx_path = output_dir / "model.onnx"
    torch.onnx.export(
        wrapper,
        (sample,),
        onnx_path.as_posix(),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=DEFAULT_ONNX_OPSET,
    )


def export_model_to_onnx(model_id: str, task: str, output_dir: Path) -> None:
    """Export a Hugging Face model to ONNX under ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Exporting %s (%s) -> %s", model_id, task, output_dir)

    if task == "image-classification" and _is_siglip_image_classifier(model_id):
        _export_siglip_image_classifier(model_id, output_dir)
        return

    from optimum.exporters.onnx import main_export

    main_export(
        model_id,
        output_dir,
        task=task,
        opset=DEFAULT_ONNX_OPSET,
    )

    # Optimum does not always persist processor/tokenizer files alongside ONNX.
    if task == "zero-shot-image-classification":
        from transformers import AutoImageProcessor, AutoTokenizer

        AutoImageProcessor.from_pretrained(model_id).save_pretrained(output_dir)
        AutoTokenizer.from_pretrained(model_id).save_pretrained(output_dir)
    elif task == "image-classification":
        from transformers import AutoImageProcessor

        AutoImageProcessor.from_pretrained(model_id).save_pretrained(output_dir)
    else:
        raise ValueError(f"Unsupported ONNX export task: {task}")
