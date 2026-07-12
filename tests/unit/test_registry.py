"""Unit tests: Model registry."""

import pytest

from app.models.registry import resolve_torch_device, health_models, warmup

# ==============================================================================
# 6. Model Registry tests
# ==============================================================================


def test_resolve_torch_device():
    import torch

    from app.config.runtime_validation import RuntimeValidationError

    torch.cuda.is_available.return_value = False
    with pytest.raises(RuntimeValidationError):
        resolve_torch_device("cuda")
    assert resolve_torch_device("cpu") == "cpu"

    torch.cuda.is_available.return_value = True
    assert resolve_torch_device("cuda") == "cuda"
    assert resolve_torch_device("cpu") == "cpu"


def test_health_models():
    from app.config.settings import settings

    res = health_models(settings)
    assert "device" in res
    assert "runtime" in res
    assert "executionProvider" in res
    assert "precision" in res
    assert "matchModel" in res
    assert "safetyLoaded" in res
    assert "relevanceEnabled" in res


@pytest.mark.asyncio
async def test_warmup_none_enabled():
    from app.config.settings import settings

    orig_embed_enabled = settings.embed_enabled
    orig_safety_enabled = settings.safety_enabled

    object.__setattr__(settings, "embed_enabled", False)
    object.__setattr__(settings, "safety_enabled", False)
    try:
        await warmup(settings)
    finally:
        object.__setattr__(settings, "embed_enabled", orig_embed_enabled)
        object.__setattr__(settings, "safety_enabled", orig_safety_enabled)
