# Performance Benchmarks

Latency and accuracy for the vision worker. Re-run after dependency or model updates.

---

## Model comparison (305 images, M4 Mac, torch CPU)

Zero-shot eval (`petType=""`):

```bash
python dev_benchmarks/download_test_images.py   # first time only
python dev_benchmarks/evaluate_workflow.py --profile standard
python dev_benchmarks/evaluate_workflow.py --profile quality
```

| Profile | Relevance Formulation | Safety Model | Embed Dim | Pet Relevance | Subclass Match | Avg Latency |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `standard` (baseline) | Baseline Blending | Falconsai NSFW | 768 | 86.6% | 51.0% | 0.089s |
| `standard` (default) | Unified Softmax | Falconsai NSFW | 768 | **90.8%** | **60.0%** | **0.078s** |
| `standard` (custom safety)| Unified Softmax | AdamCodd NSFW | 768 | **90.8%** | **60.0%** | 0.133s |
| `quality` (baseline) | Baseline Blending | Falconsai NSFW | 768 | 84.9% | 64.1% | 0.131s |
| `quality` (default) | Unified Softmax | Falconsai NSFW | 768 | **90.5%** | **69.4%** | 0.176s |
| `quality` (custom safety) | Unified Softmax | AdamCodd NSFW | 768 | **90.5%** | **69.4%** | 0.228s |

### Safety Moderation Models Comparison
Evaluated on the 10 suggestive (unsafe) and 295 SFW (safe) images in the benchmark set:

| Safety Model | Accuracy | Precision | Recall (Sensitivity) | F1 Score | Conf. Matrix (TP/FN/TN/FP) | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Falconsai/nsfw_image_detection`** *(Default)* | 98.0% | **100.0%** | 40.0% | 57.1% | 4 / 6 / 295 / 0 | **~35ms** |
| **`strangerguardhf/nsfw-image-detection`** | 93.1% | 31.0% | 90.0% | 46.2% | 9 / 1 / 275 / 20 | ~45ms |
| **`AdamCodd/vit-base-nsfw-detector`** *(Custom)* | **98.7%** | 71.4% | **100.0%** | **83.3%** | **10 / 0 / 291 / 4** | ~60ms |

- **Default Falconsai model** has very low recall (40%), missing 6 out of 10 suggestive images, presenting a safety risk.
- **AdamCodd model** achieves **100% recall** (0 suggestive images missed) with a minimal false positive rate of **1.35%** (only 4 false flags). It is highly recommended for production security.

Reports: `dev_benchmarks/evaluation_report_{standard,quality}_default.md` and `_safety.md`.

### Hint-stabilized subclass (standard profile)

With correct `petType` hints: subclass match **51.0% → 77.1%**. Reproduce:

```bash
python dev_benchmarks/simulate_hints.py
```

### Rejected experiments (archived)

| Model | Pet relevance | Notes |
|-------|---------------|-------|
| SigLIP2-large @ 384px + adaptive DETR | 45.6% | DETR crop harmed cats/dogs |
| SigLIP2 SO400M @ 384px | 58.0% | Tested July 2026; removed from profiles |

---

## Latency methodology

| Setting | Value |
|---------|-------|
| Host | Apple M4, macOS, Python 3.12 |
| Warmup | Full `warmup()` before timing |
| Runs | 5 per stage |
| Script | `python -m tools benchmark --profile <name> --runs 5` |

Stages timed: **safety** (NSFW) and **match** (fused SigLIP embed + relevance).

Historical fused-match latencies (torch CPU, 1 image):

| Profile | safety p50 | match p50 |
|---------|------------|-----------|
| standard (base-224) | ~36ms | ~41ms |

---

## Reproduce

```bash
pip install -e ".[dev]"
python -m tools doctor --profile quality
python -m tools smoke --profile standard
python -m tools smoke --profile quality
python -m tools benchmark --profile standard --runs 5
python -m tools benchmark --profile quality --runs 5
```

---

## Related docs

- [GUIDE.md](GUIDE.md) — profile and env configuration
- [MODEL_EXPORT.md](MODEL_EXPORT.md) — ONNX export
