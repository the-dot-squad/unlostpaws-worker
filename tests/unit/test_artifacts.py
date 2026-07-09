"""Tests for ONNX artifact manifest and cache resolution."""

from unittest.mock import MagicMock

import pytest

from app.models.artifacts import _load_manifest, resolve_onnx_artifacts


def test_manifest_contains_siglip():
    manifest = _load_manifest()
    assert "google/siglip2-base-patch16-224" in manifest
    entry = manifest["google/siglip2-base-patch16-224"]
    assert entry["export_task"] == "zero-shot-image-classification"


def test_resolve_cached_artifact(tmp_path):
    model_id = "Falconsai/nsfw_image_detection"
    precision = "fp32"
    safe = model_id.replace("/", "__")
    cache_root = tmp_path / safe / precision
    cache_root.mkdir(parents=True)
    onnx_file = cache_root / "model.onnx"
    onnx_file.write_bytes(b"fake-onnx")

    result = resolve_onnx_artifacts(model_id, precision, str(tmp_path))
    assert result.onnx_file == onnx_file
    assert result.model_dir == cache_root


def test_resolve_missing_without_export_raises(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        resolve_onnx_artifacts(
            "unknown/model-id",
            "fp32",
            str(tmp_path),
        )


def test_download_from_hub(tmp_path, monkeypatch):
    import sys

    model_id = "google/siglip2-base-patch16-224"
    fake_path = tmp_path / "downloaded.onnx"
    fake_path.write_bytes(b"onnx-data")

    hub_mock = MagicMock()
    hub_mock.hf_hub_download.return_value = str(fake_path)
    hub_mock.snapshot_download.return_value = str(tmp_path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_mock)

    result = resolve_onnx_artifacts(model_id, "fp32", str(tmp_path / "cache"))
    assert result.onnx_file.name.endswith(".onnx")
