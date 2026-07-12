"""Tests for ONNX backend post-processing with mocked ORT sessions."""

from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from app.models.onnx_nsfw import OnnxNsfwClassifier
from app.models.onnx_siglip import OnnxSiglipEmbedder


def _mock_session(input_names, output_map):
    session = MagicMock()
    session.input_names = input_names
    session.output_names = list(output_map.keys())
    session.active_provider = "CPUExecutionProvider"

    def run_output(feeds, output_name):
        handler = output_map[output_name]
        return handler(feeds)

    def run(feeds):
        return {name: handler(feeds) for name, handler in output_map.items()}

    session.run_output = run_output
    session.run = run
    return session


@patch("app.models.onnx_siglip.resolve_onnx_artifacts")
@patch("app.models.onnx_siglip.AutoImageProcessor")
@patch("app.models.onnx_siglip.AutoTokenizer")
@patch("app.models.onnx_siglip.OnnxSession")
def test_onnx_siglip_embed_batch(mock_session_cls, mock_tok, mock_proc, mock_artifacts):
    mock_artifacts.return_value = MagicMock(
        onnx_file=MagicMock(), model_dir=MagicMock(is_dir=lambda: False)
    )
    processor = MagicMock()
    processor.return_value = {
        "pixel_values": np.zeros((1, 3, 224, 224), dtype=np.float32)
    }
    mock_proc.from_pretrained.return_value = processor

    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": np.zeros((15, 16), dtype=np.int64),
        "attention_mask": np.ones((15, 16), dtype=np.int64),
    }
    mock_tok.from_pretrained.return_value = tokenizer

    def text_embeds(_feeds):
        return np.random.randn(15, 768).astype(np.float32)

    def image_embeds(_feeds):
        batch = _feeds["pixel_values"].shape[0]
        return np.random.randn(batch, 768).astype(np.float32)

    mock_session_cls.return_value = _mock_session(
        ["pixel_values", "input_ids", "attention_mask"],
        {"text_embeds": text_embeds, "image_embeds": image_embeds},
    )

    embedder = OnnxSiglipEmbedder("google/siglip2-base-patch16-224", 1)
    image = Image.new("RGB", (224, 224))
    results = embedder.embed_batch([image])
    assert len(results) == 1
    assert len(results[0].embedding) == 768


@patch("app.models.onnx_siglip.resolve_onnx_artifacts")
@patch("app.models.onnx_siglip.AutoImageProcessor")
@patch("app.models.onnx_siglip.AutoTokenizer")
@patch("app.models.onnx_siglip.OnnxSession")
def test_onnx_siglip_relevance_batch(
    mock_session_cls, mock_tok, mock_proc, mock_artifacts
):
    mock_artifacts.return_value = MagicMock(
        onnx_file=MagicMock(), model_dir=MagicMock(is_dir=lambda: False)
    )
    processor = MagicMock()
    processor.return_value = {
        "pixel_values": np.zeros((1, 3, 224, 224), dtype=np.float32)
    }
    mock_proc.from_pretrained.return_value = processor

    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": np.zeros((29, 8), dtype=np.int64),
        "attention_mask": np.ones((29, 8), dtype=np.int64),
    }
    mock_tok.from_pretrained.return_value = tokenizer

    def logits_per_image(_feeds):
        # 21 ensembled pet logits (dog=5.0, cat=1.0, others=0.5) + 8 negative logits
        pet_part = (
            [5.0] * 3
            + [1.0] * 3
            + [0.5] * 2
            + [0.5] * 2
            + [0.5] * 2
            + [0.5] * 2
            + [0.5] * 2
            + [0.5] * 2
            + [0.5] * 3
        )
        neg_part = [-1.0, -2.0, -3.0, -5.0, -5.0, -5.0, -5.0, -5.0]
        return np.array([pet_part + neg_part], dtype=np.float32)

    def image_embeds(_feeds):
        return np.random.randn(1, 768).astype(np.float32)

    mock_session_cls.return_value = _mock_session(
        ["pixel_values", "input_ids", "attention_mask"],
        {
            "logits_per_image": logits_per_image,
            "image_embeds": image_embeds,
        },
    )

    embedder = OnnxSiglipEmbedder("google/siglip2-base-patch16-224", 1)
    results = embedder.relevance_batch([Image.new("RGB", (224, 224))], pet_type="dog")
    assert len(results) == 1
    assert results[0].top_label == "dog"
    assert results[0].pet_likelihood > 0.7


@patch("app.models.onnx_nsfw.resolve_onnx_artifacts")
@patch("app.models.onnx_nsfw.AutoImageProcessor")
@patch("app.models.onnx_nsfw.OnnxSession")
def test_onnx_nsfw_predict(mock_session_cls, mock_proc, mock_artifacts, tmp_path):
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        '{"id2label": {"0": "normal", "1": "nsfw"}}', encoding="utf-8"
    )
    mock_artifacts.return_value = MagicMock(
        onnx_file=tmp_path / "model.onnx",
        model_dir=config_dir,
    )
    (tmp_path / "model.onnx").write_bytes(b"x")

    processor = MagicMock()
    processor.return_value = {
        "pixel_values": np.zeros((1, 3, 224, 224), dtype=np.float32)
    }
    mock_proc.from_pretrained.return_value = processor

    def logits(_feeds):
        return np.array([[2.0, 0.1]], dtype=np.float32)

    mock_session_cls.return_value = _mock_session(
        ["pixel_values"],
        {"logits": logits},
    )

    classifier = OnnxNsfwClassifier("Falconsai/nsfw_image_detection")
    preds = classifier.predict([Image.new("RGB", (224, 224))])
    assert len(preds) == 1
    assert preds[0].label in ("normal", "nsfw")
