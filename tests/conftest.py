import sys
from unittest.mock import MagicMock

# Prevent ModuleNotFoundError for torch and transformers by mocking them in sys.modules
# before any of the application code is imported.


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


# Create a mock torch module
mock_torch = MagicMock()
mock_torch.Tensor = MockTensor
mock_torch.no_grad = MagicMock()


# Implement mock context manager for torch.no_grad
class MockNoGrad:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


mock_torch.no_grad.return_value = MockNoGrad()
mock_torch.cuda = MagicMock()
mock_torch.cuda.is_available.return_value = False

sys.modules["torch"] = mock_torch

# Create mock transformers modules
sys.modules["transformers"] = MagicMock()
sys.modules["sentencepiece"] = MagicMock()
sys.modules["protobuf"] = MagicMock()
