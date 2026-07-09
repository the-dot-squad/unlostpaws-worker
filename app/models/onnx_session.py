"""
Shared ONNX Runtime session wrapper.

ORT InferenceSession objects are thread-safe for concurrent ``run()`` calls,
which matches our ThreadPoolExecutor inference pattern.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.models.execution_providers import resolve_ort_providers

logger = logging.getLogger(__name__)


class OnnxSession:
    """Thin wrapper around onnxruntime.InferenceSession."""

    def __init__(
        self,
        model_path: Path,
        *,
        execution_provider: str = "auto",
        tensorrt_cache_dir: str = "/app/.cache/tensorrt",
        openvino_device: str = "CPU",
    ) -> None:
        import onnxruntime as ort

        providers = resolve_ort_providers(
            execution_provider,
            tensorrt_cache_dir=tensorrt_cache_dir,
            openvino_device=openvino_device,
        )
        self._session = ort.InferenceSession(
            str(model_path),
            providers=[name for name, _ in providers],
            provider_options=[opts for _, opts in providers],
        )
        self.model_path = model_path
        self.providers = [name for name, _ in providers]
        self.active_provider = (
            self.providers[0] if self.providers else "CPUExecutionProvider"
        )

        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        logger.info(
            "ONNX session loaded path=%s provider=%s inputs=%s outputs=%s",
            model_path.name,
            self.active_provider,
            [i.name for i in inputs],
            [o.name for o in outputs],
        )

        self._input_names = [i.name for i in inputs]
        self._output_names = [o.name for o in outputs]
        self._input_shapes = {i.name: i.shape for i in inputs}

    @property
    def input_names(self) -> list[str]:
        return list(self._input_names)

    @property
    def output_names(self) -> list[str]:
        return list(self._output_names)

    def run(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        outputs = self._session.run(self._output_names, feeds)
        return dict(zip(self._output_names, outputs))

    def run_output(self, feeds: dict[str, np.ndarray], output_name: str) -> np.ndarray:
        result = self.run(feeds)
        if output_name not in result:
            raise KeyError(
                f"Output '{output_name}' not in model outputs: {list(result.keys())}"
            )
        return result[output_name]
