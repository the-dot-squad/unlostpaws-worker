"""
ONNX artifact resolution — downloads or exports models at runtime.

Artifacts are cached under ``{MODEL_CACHE_DIR}/{model_id}/{precision}/`` so
subsequent worker restarts skip re-download. When a manifest entry has
``export_on_miss: true``, missing artifacts are exported via Hugging Face Optimum.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from app.models.onnx_export import export_model_to_onnx

logger = logging.getLogger(__name__)

_MANIFEST_PATH = Path(__file__).with_name("manifest.json")


@dataclass(frozen=True)
class ArtifactPaths:
    """Resolved ONNX model directory and primary graph file."""

    model_dir: Path
    onnx_file: Path
    precision: str


def _load_manifest() -> dict:
    with _MANIFEST_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _safe_dir_name(model_id: str) -> str:
    return model_id.replace("/", "__")


def _cache_root(cache_dir: str, model_id: str, precision: str) -> Path:
    return Path(cache_dir) / _safe_dir_name(model_id) / precision


def _find_onnx_file(directory: Path, precision: str) -> Path | None:
    if precision == "int8":
        names = ("model_quantized.onnx", "model.onnx", "encoder_model.onnx")
    else:
        names = ("model.onnx", "encoder_model.onnx", "model_quantized.onnx")
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    onnx_files = sorted(directory.rglob("*.onnx"))
    return onnx_files[0] if onnx_files else None


def _export_model(model_id: str, task: str, target_dir: Path) -> None:
    export_model_to_onnx(model_id, task, target_dir)


def _download_from_hub(onnx_repo: str, relative_file: str, target_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    target_dir.mkdir(parents=True, exist_ok=True)
    local_path = hf_hub_download(repo_id=onnx_repo, filename=relative_file)
    downloaded = Path(local_path)
    link_target = target_dir / downloaded.name
    if not link_target.exists():
        link_target.write_bytes(downloaded.read_bytes())
    return link_target


def resolve_onnx_artifacts(
    model_id: str,
    precision: str,
    cache_dir: str,
) -> ArtifactPaths:
    """
    Resolve ONNX artifacts for a model, downloading or exporting as needed.

    Returns:
        ArtifactPaths with model directory and primary ``.onnx`` file path.
    """
    manifest = _load_manifest()
    entry = manifest.get(model_id)
    if entry is None:
        raise ValueError(f"No ONNX manifest entry for model: {model_id}")

    cache_root = _cache_root(cache_dir, model_id, precision)
    cache_root.mkdir(parents=True, exist_ok=True)

    onnx_file = _find_onnx_file(cache_root, precision)
    if onnx_file is not None:
        logger.debug("Using cached ONNX artifact: %s", onnx_file)
        return ArtifactPaths(
            model_dir=cache_root, onnx_file=onnx_file, precision=precision
        )

    files = entry.get("files", {})
    relative = files.get(precision) or files.get("fp32")
    onnx_repo = entry.get("onnx_repo")

    if onnx_repo and relative:
        downloaded = _download_from_hub(onnx_repo, relative, cache_root)
        # Also fetch tokenizer/processor configs from the ONNX repo root.
        try:
            from huggingface_hub import snapshot_download

            snapshot_dir = Path(
                snapshot_download(
                    repo_id=onnx_repo,
                    allow_patterns=[
                        "*.json",
                        "*.txt",
                        "tokenizer*",
                        "preprocessor*",
                        "spiece.model",
                    ],
                )
            )
            for item in snapshot_dir.iterdir():
                dest = cache_root / item.name
                if item.is_file() and not dest.exists():
                    dest.write_bytes(item.read_bytes())
        except Exception as exc:
            logger.warning("Could not download ONNX repo configs: %s", exc)

        return ArtifactPaths(
            model_dir=cache_root, onnx_file=downloaded, precision=precision
        )

    if entry.get("export_on_miss"):
        _export_model(model_id, entry["export_task"], cache_root)
        onnx_file = _find_onnx_file(cache_root, precision)
        if onnx_file is None:
            raise FileNotFoundError(
                f"ONNX export completed but no .onnx file found in {cache_root}"
            )
        return ArtifactPaths(
            model_dir=cache_root, onnx_file=onnx_file, precision=precision
        )

    raise FileNotFoundError(
        f"No ONNX artifacts for {model_id} (precision={precision}) and export disabled"
    )
