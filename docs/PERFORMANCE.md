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

| Profile | Match model | Embed dim | Pet relevance | Subclass match | Avg latency |
|---------|-------------|-----------|---------------|----------------|-------------|
| `standard` | base @ 224px | 768 | **86.6%** | 51.0% | **0.089s** |
| `quality` | base @ 384px | 768 | **84.9%** | **64.1%** | 0.131s |

Reports: `dev_benchmarks/evaluation_report_{standard,quality}.md`.

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
