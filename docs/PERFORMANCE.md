# Performance Benchmarks

Latency measurements for vision worker inference stages on a single 400×300 synthetic test image (or fixture from `tests/fixtures/images/` when present). Numbers are **per-image, per-stage** medians after model warmup.

> Re-run benchmarks after dependency or model updates. Raw JSON artifacts live in this directory.

---

## Methodology

| Setting | Value |
| :--- | :--- |
| Host (native) | Apple M4, macOS, Python 3.12, arm64 |
| Host (Docker) | Linux arm64 container via Docker Desktop on the same M4 |
| Warmup | Full `warmup()` before timing |
| Runs per stage | 5 |
| Script | `python -m tools benchmark --all --runs 5` |

Stages timed independently:

- **safety** — NSFW classification (`Falconsai/nsfw_image_detection`)
- **embed** — SigLIP2 768-dim vector (`google/siglip2-base-patch16-224`)
- **relevance** — zero-shot pet likelihood (SigLIP2 prompts)

First production job on a cold host is slower — models download during the first `warmup()`.

---

## Native macOS (Apple M4) — July 2026

| Profile | Runtime | EP | Safety p50 | Embed p50 | Relevance p50 | Total ML p50* |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| `cpu-quality` | torch | — | 30 ms | 34 ms | 33 ms | **~97 ms** |
| `cpu-standard` | torch | — | 32 ms | 35 ms | — | **~67 ms** |
| `cpu-light` | torch | — | 32 ms | — | — | **~32 ms** |
| `onnx-cpu-quality` | onnx | CPU | 36 ms | 28 ms | 24 ms | **~88 ms** |
| `onnx-apple` | onnx | CoreML | 87 ms | 98 ms | 99 ms | **~284 ms** |

\*Sum of enabled ML stages; quality + fingerprint are sub-millisecond and omitted.

**Takeaways (M4 native):**

- **`cpu-quality`** is the best default for Apple Silicon dev — simple, fast, full pipeline.
- **`onnx-cpu-quality`** trades a small accuracy delta (INT8) for slightly faster embed/relevance on CPU.
- **`onnx-apple`** uses the Neural Engine but adds ORT/CoreML partition overhead on these models; not faster than torch CPU on M4 for single-image latency. Re-evaluate on larger batches or after CoreML engine caching.

Raw data: [`benchmarks-native-m4.json`](benchmarks-native-m4.json)

---

## Docker Linux container (arm64 on M4) — July 2026

Same hardware, worker running inside the CPU Docker image (`unlostpaws-worker:local-test`):

| Profile | Runtime | EP | Safety p50 | Embed p50 | Relevance p50 | Total ML p50* |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| `cpu-quality` | torch | — | 92 ms | 108 ms | 106 ms | **~306 ms** |
| `onnx-cpu-quality` | onnx | CPU | 82 ms | 43 ms | 40 ms | **~165 ms** |

\*Sum of enabled ML stages.

**Takeaways (Docker on Mac):**

- Container overhead roughly **3×** torch CPU latency vs native on this host.
- **`onnx-cpu-quality`** in Docker is **~2× faster** than torch for the full ML stack — prefer ONNX CPU profiles in Linux containers on resource-constrained hosts.
- For local M4 development, **run natively** (`onnx-apple` or `cpu-quality`) rather than Docker when latency matters.

Raw data: [`benchmarks-docker-m4.json`](benchmarks-docker-m4.json)

---

## Smoke test matrix (verified)

All profiles below passed end-to-end pipeline smoke tests (`python -m tools smoke`) — warmup, inference, 768-dim embeddings where applicable.

| Profile | Native M4 | Docker (CPU image) | Notes |
| :--- | :---: | :---: | :--- |
| `dedup-only` | Pass | Pass | No ML loaded |
| `cpu-light` | Pass | Pass | Safety only |
| `cpu-standard` | Pass | Pass | No relevance |
| `cpu-quality` | Pass | Pass | Default dev |
| `onnx-cpu-standard` | Pass | Pass | No relevance |
| `onnx-cpu-quality` | Pass | Pass | INT8 ONNX |
| `onnx-apple` | Pass | **Blocked** | CoreML requires native macOS |
| `gpu-standard` | **Blocked** | **Blocked** | Requires NVIDIA CUDA + GPU image |

Fail-fast validation prevents silent misconfiguration (e.g. `gpu-standard` on the CPU image, `onnx-apple` inside Linux Docker).

---

## Reproduce

```bash
# Native (macOS / Linux bare metal)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m tools doctor
python -m tools doctor --smoke --profile cpu-quality

python -m tools benchmark --all --runs 5 --output docs/benchmarks-native.json

# Docker
docker build -t unlostpaws-worker:local-test -f Dockerfile .
docker run --rm \
  -e VISION_PROFILE=cpu-quality \
  -e WORKER_IMAGE_VARIANT=cpu \
  -e HF_HOME=/app/.cache/huggingface \
  -v "$(pwd)/.cache/huggingface:/app/.cache/huggingface" \
  unlostpaws-worker:local-test \
  python -m tools benchmark --profile cpu-quality --runs 5
```

Single-profile benchmark:

```bash
python -m tools benchmark --profile cpu-quality --runs 10
```

---

## Related docs

- [GUIDE.md](GUIDE.md) — profile selection and hardware mapping
- [MODEL_EXPORT.md](MODEL_EXPORT.md) — ONNX export and accuracy validation
