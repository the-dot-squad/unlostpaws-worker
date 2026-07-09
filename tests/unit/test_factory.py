"""Tests for backend factory selection."""

from pathlib import Path

import pytest

from app.config.settings import _default_hf_home, load_settings
from app.models.factory import create_classifier, create_embedder
from app.models.onnx_nsfw import OnnxNsfwClassifier
from app.models.onnx_siglip import OnnxSiglipEmbedder
from app.models.torch_nsfw import TorchNsfwClassifier
from app.models.torch_siglip import TorchSiglipEmbedder


@pytest.fixture
def torch_quality_env(monkeypatch):
    monkeypatch.setenv("VISION_PROFILE", "cpu-quality")
    monkeypatch.setenv("INFERENCE_RUNTIME", "torch")
    return load_settings()


@pytest.fixture
def onnx_quality_env(monkeypatch):
    monkeypatch.setenv("VISION_PROFILE", "onnx-cpu-quality")
    monkeypatch.setenv("INFERENCE_RUNTIME", "onnx")
    return load_settings()


def test_create_torch_embedder(torch_quality_env):
    embedder = create_embedder(torch_quality_env)
    assert isinstance(embedder, TorchSiglipEmbedder)
    assert embedder.runtime == "torch"


def test_create_onnx_embedder(onnx_quality_env):
    embedder = create_embedder(onnx_quality_env)
    assert isinstance(embedder, OnnxSiglipEmbedder)
    assert embedder.runtime == "onnx"


def test_create_torch_classifier(torch_quality_env):
    classifier = create_classifier(torch_quality_env)
    assert isinstance(classifier, TorchNsfwClassifier)


def test_create_onnx_classifier(onnx_quality_env):
    classifier = create_classifier(onnx_quality_env)
    assert isinstance(classifier, OnnxNsfwClassifier)


def test_embedder_none_when_disabled(monkeypatch):
    monkeypatch.setenv("VISION_PROFILE", "dedup-only")
    cfg = load_settings()
    assert create_embedder(cfg) is None
    assert create_classifier(cfg) is None


def test_runtime_override(monkeypatch):
    monkeypatch.setenv("VISION_PROFILE", "cpu-quality")
    monkeypatch.setenv("INFERENCE_RUNTIME", "onnx")
    cfg = load_settings()
    assert cfg.runtime == "onnx"


def test_default_hf_home_local(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("RUNNING_IN_DOCKER", "false")
    monkeypatch.setattr(Path, "exists", lambda self: False)
    assert _default_hf_home() == str(Path.home() / ".cache" / "huggingface")


def test_default_hf_home_docker(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("RUNNING_IN_DOCKER", "true")
    assert _default_hf_home() == "/app/.cache/huggingface"
