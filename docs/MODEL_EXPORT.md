# Model Export Guide

ONNX artifacts are downloaded from Hugging Face at runtime by default. For concepts (vision profile, runtime, execution provider), see [GUIDE.md](GUIDE.md).

This guide covers manual export, quantization, and CI validation.

## Manifest

Model → ONNX mapping lives in [`app/models/manifest.json`](../app/models/manifest.json).

Each entry specifies:

- `onnx_repo` — pre-exported Hugging Face repo (if available)
- `files` — relative ONNX path per precision (`fp32`, `fp16`, `int8`)
- `export_task` — Optimum export task name
- `export_on_miss` — auto-export via Optimum when no cached artifact exists

## Manual export

```bash
pip install -r requirements-torch.txt
python -m tools export --output output/onnx
```

This exports:

| Model | Task | Output folder |
|---|---|---|
| `google/siglip2-base-patch16-224` | zero-shot-image-classification | `output/onnx/siglip2-base/` |
| `google/siglip2-base-patch16-384` | zero-shot-image-classification | `output/onnx/siglip2-quality/` |
| `Falconsai/nsfw_image_detection` | image-classification | `output/onnx/nsfw-falconsai/` |
| `strangerguardhf/nsfw-image-detection` | image-classification | `output/onnx/nsfw-strangerguard/` |

INT8 dynamic quantization runs automatically. Skip with `--skip-quantize`.

## Optimum CLI (individual models)

```bash
optimum-cli export onnx \
  --model google/siglip2-base-patch16-224 \
  --task zero-shot-image-classification \
  --opset 18 \
  ./models/siglip2/

optimum-cli export onnx \
  --model Falconsai/nsfw_image_detection \
  --task image-classification \
  ./models/nsfw-falconsai/
```

## Accuracy validation

Compare ONNX outputs against PyTorch baselines:

```bash
python -m tools export --output /tmp/onnx-export
python -m tools validate --models-dir /tmp/onnx-export
```

Thresholds:

| Metric | Threshold |
|---|---|
| Embedding cosine similarity | >= 0.99 |
| Relevance `pet_likelihood` delta | <= 0.05 |
| NSFW score delta | <= 0.02 |

## CI workflow

[`.github/workflows/export-models.yml`](../.github/workflows/export-models.yml) runs on:

- Manual dispatch
- Changes to `manifest.json` or export scripts

Artifacts are uploaded to GitHub Actions (reference only — workers still download from HF at runtime).

## Cache layout

Runtime cache path: `{MODEL_CACHE_DIR}/{model_id_with_slashes_replaced}/{precision}/`

Example:

```
/app/.cache/huggingface/onnx/
  google__siglip2-base-patch16-224/
    int8/
      model_quantized.onnx
      tokenizer.json
      preprocessor_config.json
```

## Pre-exported ONNX on Hugging Face

SigLIP2: [`onnx-community/siglip2-base-patch16-224-ONNX`](https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX)

NSFW models are exported on first run (`export_on_miss: true`) or via `python -m tools export`.
