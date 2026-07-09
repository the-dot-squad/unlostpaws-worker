"""Unit-test fixtures: mock heavy ML deps so tests stay fast and offline."""

import sys
from unittest.mock import MagicMock


class MockTensor:
    def __init__(self, *args, **kwargs):
        pass

    def to(self, *args, **kwargs):
        return self

    def norm(self, *args, **kwargs):
        return self

    def cpu(self, *args, **kwargs):
        return self

    def tolist(self, *args, **kwargs):
        return [[0.0] * 768]

    def argmax(self, *args, **kwargs):
        item_mock = MagicMock()
        item_mock.item.return_value = 0
        return item_mock

    def item(self, *args, **kwargs):
        return 0.0

    def max(self, *args, **kwargs):
        return self

    def __getitem__(self, idx):
        return self

    def __truediv__(self, other):
        return self


# Mock torch before app code imports it.
mock_torch = MagicMock()
mock_torch.Tensor = MockTensor
mock_torch.no_grad = MagicMock()


class MockNoGrad:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


mock_torch.no_grad.return_value = MockNoGrad()
mock_torch.cuda = MagicMock()
mock_torch.cuda.is_available.return_value = False

sys.modules["torch"] = mock_torch

mock_ort = MagicMock()
mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
sys.modules["onnxruntime"] = mock_ort
sys.modules["optimum"] = MagicMock()
sys.modules["optimum.onnxruntime"] = MagicMock()

sys.modules["transformers"] = MagicMock()
sys.modules["sentencepiece"] = MagicMock()
sys.modules["protobuf"] = MagicMock()


class AsyncMock(MagicMock):
    """Async-compatible mock for awaited Redis/HTTP calls."""

    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
