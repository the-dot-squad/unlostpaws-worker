# UnLostPaws Vision Worker

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![CI](https://github.com/the-dot-squad/unlostpaws-worker/actions/workflows/ci.yml/badge.svg)](https://github.com/the-dot-squad/unlostpaws-worker/actions/workflows/ci.yml)

Background async ML worker for **[UnLostPaws](https://github.com/the-dot-squad/unlostpaws)** — an open-source lost-and-found pet platform. This repo is **not** the website. It is the Python service that runs heavy vision inference off the request path and POSTs results back via webhook.

When a user uploads pet photos, the Next.js app enqueues a job on a **Redis Stream** (`unlostpaws:stream:vision-processing`). This worker pulls jobs with consumer groups, runs quality / safety / fingerprint / match stages, and callbacks to `/api/webhooks/vision` so the app can approve listings, deduplicate abuse, and index SigLIP2 vectors in Qdrant.

---

## Architecture

```mermaid
flowchart TB
    subgraph Web["Web Platform (Next.js)"]
        UI["Browser / API Client"]
        Enqueue["enqueueImageJob()"]
        Webhook["/api/webhooks/vision"]
        Mongo[("MongoDB")]
        Qdrant[("Qdrant DB")]
    end

    subgraph Infra["Shared Infrastructure"]
        Redis[("Redis Streams")]
        S3[("S3 / R2 Storage")]
    end

    subgraph Worker["Vision Worker"]
        start["app/main.py"]
        preflight{"Runtime Preflight"}
        warmup["Model Warmup"]
        consumer["queue/consumer.py"]
        orch["pipeline/orchestrator.py"]
        dl["pipeline/download.py"]
        quality["quality"]
        safety["safety"]
        fp["fingerprint"]
        match["match (embed + relevance)"]
        cb["callback/client.py"]
    end

    UI --> Enqueue --> Redis
    start --> preflight --> warmup --> consumer
    Redis --> consumer --> orch --> dl --> S3
    dl --> quality & safety & fp & match
    quality & safety & fp & match --> cb --> Webhook
    Webhook --> Mongo & Qdrant
```

---

## Quick start

You can set up and run the UnLostPaws Vision Worker using our interactive setup script.

### Option 1: Quick Web-based Setup (No Clone Required)
If you just want to run the pre-built worker in Docker without cloning this repository, execute this command in an empty directory:
```bash
curl -fsSL https://raw.githubusercontent.com/the-dot-squad/unlostpaws-worker/main/tools/run.sh | bash
```

### Option 2: Clone & Local Setup
If you have cloned the repository, simply run the setup script:
```bash
./tools/run.sh setup
```
This interactive script will:
1. Detect your system specs (CPU, GPU, RAM, Disk).
2. Help you choose between CPU/GPU and PyTorch/ONNX runtimes.
3. Automatically download/copy compose files, verify hardware limits, and configure your `.env` file.
4. Prompt you to launch the container immediately.

For specific environment combinations, see the table below:

| I have… | Suggested env | How to run |
| :--- | :--- | :--- |
| Dev laptop / Linux CPU | `VISION_PROFILE=quality` `INFERENCE_RUNTIME=torch` `DEVICE=cpu` | `docker compose up -d` |
| ARM64 SBC / Graviton | `VISION_PROFILE=quality` `INFERENCE_RUNTIME=onnx` `ORT_EXECUTION_PROVIDER=cpu` | `docker compose up -d` |
| NVIDIA GPU | `VISION_PROFILE=quality` `DEVICE=cuda` | `docker compose up -d` (after running setup) |
| Apple Silicon | `VISION_PROFILE=quality` `ORT_EXECUTION_PROVIDER=coreml` | Native Python on macOS (CoreML not available in Linux Docker) |
| Hashing only | `VISION_PROFILE=dedup-only` | Any CPU path |

**CLI Subcommands:**
Once configured, you can run other local development subcommands inside the cloned repository:
```bash
./tools/run.sh doctor       # Hardware preflight and configuration doctor
./tools/run.sh smoke        # Run a quick smoke test on the worker pipeline
./tools/run.sh benchmark    # Profile worker latency and throughput
./tools/run.sh export       # Export PyTorch models to ONNX Runtime
./tools/run.sh validate     # Validate database or stream schema messages
```

**Docs:** [Guide](docs/GUIDE.md) · [Performance](docs/PERFORMANCE.md) · [ONNX export (maintainers)](docs/MODEL_EXPORT.md)

---

## Profiles and models

Three capability tiers. Hardware (CPU / GPU / CoreML) is configured separately via env vars.

| Profile | SigLIP Model (Hugging Face) | Embed Dim | Relevance | Typical Use |
| :--- | :--- | :---: | :---: | :--- |
| `dedup-only` | — | — | off | MD5 + pHash + quality only |
| `standard` | [`google/siglip2-base-patch16-224`](https://huggingface.co/google/siglip2-base-patch16-224) | 768 | on | Fast indexing, lower latency |
| `quality` | [`google/siglip2-base-patch16-384`](https://huggingface.co/google/siglip2-base-patch16-384) | 768 | on | Default production (higher resolution) |

### Safety Moderation Models
Safety classification runs in the pipeline before image relevance verification. You can configure which safety model is loaded by setting `SAFETY_MODEL` in `.env`:
* **[`Falconsai/nsfw_image_detection`](https://huggingface.co/Falconsai/nsfw_image_detection)** *(Default profile safety model)*: Extremely lightweight (~80MB), low latency (~35ms), but lower sensitivity on suggestive content (40% recall).
* **[`AdamCodd/vit-base-nsfw-detector`](https://huggingface.co/AdamCodd/vit-base-nsfw-detector)** *(Custom safety model option)*: Outstanding classification coverage (~343MB), catching **100%** of suggestive/NSFW content with a minimal 1.35% false-positive rate (~60ms latency).
* **[`strangerguardhf/nsfw-image-detection`](https://huggingface.co/strangerguardhf/nsfw-image-detection)** *(Multi-class option)*: SigLIP2-based multi-class classifier categorizing content into SFW, Anime, Hentai, Pornography, and Sensual. High recall (90%) but prone to high false-positives (6.78%) on organic pet shapes.

Both ML profiles run the same fused pipeline on the full image:

**quality → safety → fingerprint → match** (SigLIP embed + relevance in one forward pass)

Default for new deploys: **`VISION_PROFILE=quality`**. Use **`standard`** when you want the fastest path with the same 768-d Qdrant vectors.

---

## Hardware and minimum requirements

| Profile | Min RAM | HF cache (first run) | GPU VRAM (optional) |
| :--- | :--- | :--- | :--- |
| `dedup-only` | 512 MB | negligible | — |
| `standard` | 3 GB | ~1 GB | 2 GB+ with `DEVICE=cuda` |
| `quality` | 4 GB | ~1.5 GB | 4 GB+ with `DEVICE=cuda` |

Also required:

- **Python 3.12** (pinned in Docker; 3.14+ not supported for torch/onnx wheels)
- **Redis** reachable at `REDIS_URL`
- **Apple Silicon CoreML** — run Python natively on macOS; CoreML is not available inside Linux Docker containers

Run `./tools/run.sh doctor --profile quality` to print resource hints for your host.

---

## Configuration

| Variable | Purpose |
| :--- | :--- |
| `VISION_PROFILE` | `dedup-only` \| `standard` \| `quality` |
| `INFERENCE_RUNTIME` | `torch` (default) or `onnx` |
| `DEVICE` | Torch: `cpu` or `cuda` |
| `ORT_EXECUTION_PROVIDER` | ONNX: `cpu`, `cuda`, `tensorrt`, `coreml`, `openvino`, `qnn` |

Optional overrides: `MATCH_MODEL`, `SAFETY_MODEL`, `MODEL_PRECISION`, `BATCH_SIZE`.

Full env reference: [`.env.example`](.env.example) and [docs/GUIDE.md](docs/GUIDE.md).

---

## Job contract and optional `petType`

**Enqueue** (Redis `XADD`, field `payload`) — produced by [unlostpaws `enqueueImageJob`](https://github.com/the-dot-squad/unlostpaws):

```json
{
  "jobType": "listing",
  "listingId": "listing_123",
  "imageUrls": ["https://example.com/pet.jpg"],
  "petType": "dog",
  "webhookUrl": "https://myapp.com/api/internal/ml-callback"
}
```

### Optional `petType` hint

| | |
| :--- | :--- |
| **When to set** | The listing already knows the species (user selected it during upload) |
| **When to omit** | Zero-shot mode — omit the field or pass `""` |
| **Valid values** | `dog`, `cat`, `bird`, `rabbit`, `hamster`, `fish`, `reptile`, `horse`, `other` |
| **Unknown values** | Normalized to `""` at validation (graceful no-op) |

**What it does** (does not change the embedding):

1. Boosts `petLikelihood` scoring toward the hinted class when comparing pet vs distractor logits
2. Resolves `topLabel` to the hint when the model is uncertain between species (logit margin &lt; 0.75)
3. Does **not** force the label when the hint contradicts a strong model signal

Flow: `job.petType` → [`orchestrator.py`](app/pipeline/orchestrator.py) → [`match_stage`](app/pipeline/stages/match.py) → [`compute_relevance_from_logits`](app/models/relevance.py).

**Success callback** includes per-image `embedding`, `safety`, `relevance`, `quality`, `md5`, `phash`. **Failure** after max retries → DLQ + failure webhook.

Payload shapes live in `app/schemas/` and must stay compatible with the web app's webhook handler.

---

## Benchmark results

Evaluated on **305 images** (M4 Mac, torch CPU, zero-shot with `petType=""`). Reproduce:

```bash
python dev_benchmarks/download_test_images.py   # first time only
python dev_benchmarks/evaluate_workflow.py --profile standard
python dev_benchmarks/evaluate_workflow.py --profile quality
```

### Profile comparison (zero-shot)

| Profile | Relevance Formulation | Safety Model | Pet Relevance | Subclass Match | Avg Latency |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `standard` (baseline) | Baseline Blending | Falconsai NSFW | 86.6% | 51.0% | 0.089s |
| `standard` (default) | Unified Softmax | Falconsai NSFW | **90.8%** | **60.0%** | **0.078s** |
| `standard` (custom safety)| Unified Softmax | AdamCodd NSFW | **90.8%** | **60.0%** | 0.133s |
| `quality` (baseline) | Baseline Blending | Falconsai NSFW | 84.9% | 64.1% | 0.131s |
| `quality` (default) | Unified Softmax | Falconsai NSFW | **90.5%** | **69.4%** | 0.176s |
| `quality` (custom safety) | Unified Softmax | AdamCodd NSFW | **90.5%** | **69.4%** | 0.228s |

- **Unified Softmax Relevance:** Improves binary relevance accuracy to **~90.8%** and specific subclass matching to **~69.4%** by suppressing distractor false positives.
- **Custom Safety Model:** Setting `SAFETY_MODEL=AdamCodd/vit-base-nsfw-detector` provides **100% recall** on suggestive images (up from 40%), keeping the platform fully secure.

See [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for detailed methodology.

### Effect of `petType` hints (standard profile)

When the correct species hint is provided, subclass match improves from **51.0% → 77.1%** on the same image set. Reproduce hint simulation:

```bash
python dev_benchmarks/simulate_hints.py
```

Relevance scoring defaults to a unified softmax over all prompts, with a **0.32** likelihood threshold and **0.40** logit-margin fallback to `"other"` when uncertain. Custom calibration can be tuned via environment variables. Algorithm details: [docs/GUIDE.md](docs/GUIDE.md#relevance-scoring).

CI smoke fixtures (`python -m tools eval`) use synthetic PNGs — not for accuracy measurement.

---

## Operator tools

Python implements all logic (`python -m tools`). On servers, use **`./tools/run.sh`** — it picks `.venv/bin/python` when present.

```bash
./tools/run.sh doctor --profile quality     # preflight + resource hints
./tools/run.sh smoke --profile quality      # full pipeline test
./tools/run.sh benchmark --profile quality --runs 5
./tools/run.sh eval --profile quality       # CI smoke fixtures only
./tools/run.sh export --output output/onnx  # maintainers
```

Equivalent without bash:

```bash
python -m tools doctor --profile quality
python -m tools smoke --profile standard
```

Bare metal worker (Python 3.12):

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python app/main.py
```

---

## Development

### Prerequisites

- Python 3.12, Redis at `REDIS_URL` for integration tests
- ~2 GB disk for Hugging Face model cache on first smoke/integration run

### Local setup

```bash
git clone https://github.com/the-dot-squad/unlostpaws-worker.git
cd unlostpaws-worker
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
python app/main.py
```

Run against the full UnLostPaws stack: clone [unlostpaws](https://github.com/the-dot-squad/unlostpaws), start Redis/Mongo/Qdrant, point both `.env` files at the same `REDIS_URL`.

### Testing

```bash
pytest                         # unit tests (mocked ML) — same as CI
pytest -m integration -v         # slow; real model warmup
pytest tests/unit/test_job_schema.py -v
```

| Marker | What runs | When |
| :--- | :--- | :--- |
| default | `tests/unit/` mocked | Every PR |
| `integration` | real warmup + pipeline | Before release |

### Local accuracy benchmarks

The gitignored `dev_benchmarks/` folder holds the 305-image eval set and reports. See [Benchmark results](#benchmark-results) above.

### Lint

```bash
ruff check app tests tools
ruff format app tests tools
```

CI runs ruff + unit pytest on Python 3.12. Docker images publish to GHCR on `v*` tags.

### Project conventions

1. **Profile-first config** — stages and models come from `VISION_PROFILE` ([`app/config/profiles.py`](app/config/profiles.py))
2. **Fail fast** — wrong hardware exits at startup ([`app/config/runtime_validation.py`](app/config/runtime_validation.py))
3. **Job boundary validation** — Redis payloads parsed with [`app/schemas/job.py`](app/schemas/job.py)
4. **Moderation-first** — quality and safety before embeddings ([`app/pipeline/orchestrator.py`](app/pipeline/orchestrator.py))
5. **Torch + ONNX parity** — new models need [`manifest.json`](app/models/manifest.json) + factory wiring

| Change type | Touch |
| :--- | :--- |
| Profile / model | `profiles.py`, `docs/GUIDE.md`, smoke/benchmark |
| Pipeline stage | `app/pipeline/stages/`, orchestrator, schemas, tests |
| Webhook field | `app/schemas/result.py` + coordinated change in **unlostpaws** |

Pull requests should include unit tests. Run `./tools/run.sh smoke --profile quality` when touching inference.

---

## License

This project is [AGPL-3.0](LICENSE). If you modify the worker and run it as a network service, you must make corresponding source available to users.
