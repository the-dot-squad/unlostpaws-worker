# Vision Worker Guide

Deploy, configure, and troubleshoot the UnLostPaws vision worker.

**Related:** [PERFORMANCE.md](PERFORMANCE.md) · [MODEL_EXPORT.md](MODEL_EXPORT.md)

---

## Concepts

| Term | Meaning |
| :--- | :--- |
| **Capability** (`VISION_PROFILE`) | `dedup-only`, `standard`, or `quality` |
| **Runtime** (`INFERENCE_RUNTIME`) | `torch` or `onnx` |
| **Hardware** | `DEVICE` (torch) or `ORT_EXECUTION_PROVIDER` (onnx) |
| **Pipeline** | quality → safety → fingerprint → fused match (embed + relevance) |

The worker validates hardware at startup and **exits** on mismatch (no silent GPU→CPU fallback).

---

## Quick start

```bash
cp .env.example .env
./tools/run doctor --profile quality
docker compose up -d
python -m tools smoke --profile quality
```

| Target | Command |
| :--- | :--- |
| CPU / dev | `docker compose up -d` |
| NVIDIA GPU | `docker compose -f docker-compose.gpu.yml up -d` |
| Apple Silicon | `./tools/run doctor` → run `python app/main.py` natively |

---

## Capability tiers

| Profile | SigLIP | Relevance | Embed dim | Min RAM | Use when |
|---------|--------|-----------|-----------|---------|----------|
| `dedup-only` | — | off | — | 512 MB | Hashing only |
| `standard` | base @ 224px | on | 768 | 3 GB | Fast indexing |
| `quality` | base @ 384px | on | 768 | 4 GB | Default production |

Both ML profiles run identical stages. Default difference: SigLIP resolution (224px vs 384px).

---

## Environment variables

| Variable | Required | Default | Description |
| :--- | :---: | :--- | :--- |
| `REDIS_URL` | Yes | — | Use `rediss://` for TLS (Upstash) |
| `VISION_PROFILE` | No | `quality` | Capability tier |
| `INFERENCE_RUNTIME` | No | `torch` | `torch` or `onnx` |
| `DEVICE` | No | `auto` | Torch: `cpu` or `cuda` |
| `ORT_EXECUTION_PROVIDER` | No | `cpu` | ONNX EP selection |
| `CONSUMER_NAME` | No | `worker-1` | Worker instance id |

Optional: `MATCH_MODEL`, `SAFETY_MODEL`, `MODEL_PRECISION`, `BATCH_SIZE`, `TORCH_COMPILE`, `OPENVINO_DEVICE`.

### Relevance & Safety Custom Overrides
* `SAFETY_MODEL`: Override safety classifier (e.g. `AdamCodd/vit-base-nsfw-detector` for high sensitivity, `Falconsai/nsfw_image_detection` for speed, or `strangerguardhf/nsfw-image-detection` for multi-class).
* `RELEVANCE_FORMULATION`: `unified_softmax` (default) or `baseline` (sigmoid margin blending).
* `RELEVANCE_TEMP_SCALE`: Temperature scaling for unified softmax logits (default: `1.5`).
* `RELEVANCE_THRESHOLD`: Decision gate for `petLikelihood` (default: `0.32` for `unified_softmax`, `0.30` for `baseline`).
* `RELEVANCE_MARGIN_THRESHOLD`: Confidence margin between top-two classes before defaulting to `"other"` (default: `0.40` for `unified_softmax`, `0.75` for `baseline`).

Full template: [`.env.example`](../.env.example).

---

## Example deployments

| Host | VISION_PROFILE | INFERENCE_RUNTIME | DEVICE / ORT_EXECUTION_PROVIDER |
| :--- | :--- | :--- | :--- |
| Dev laptop CPU | `quality` | `torch` | `DEVICE=cpu` |
| Linux Docker ARM | `quality` | `onnx` | `cpu` |
| Apple Silicon native | `quality` | `onnx` | `coreml` |
| NVIDIA GPU prod | `quality` | `torch` | `DEVICE=cuda` |
| NVIDIA max throughput | `quality` | `onnx` | `tensorrt` |

GPU compose bakes in `VISION_PROFILE=quality`, `INFERENCE_RUNTIME=torch`, `DEVICE=cuda`.

---

## Job payload and `petType`

Jobs are validated at the Redis consumer boundary ([`app/schemas/job.py`](../app/schemas/job.py)).

```json
{
  "jobType": "listing",
  "listingId": "listing_123",
  "imageUrls": ["https://example.com/pet.jpg"],
  "petType": "dog"
}
```

| Field | Required | Notes |
| :--- | :---: | :--- |
| `imageUrls` | Yes | At least one HTTPS URL |
| `petType` | No | Species hint; omit or `""` for zero-shot |

**Valid `petType` values:** `dog`, `cat`, `bird`, `rabbit`, `hamster`, `fish`, `reptile`, `horse`, `other`.

Unknown values are normalized to `""` during validation. The hint affects relevance scoring and `topLabel` only — not the embedding vector.

---

## Relevance scoring

The relevance stage converts SigLIP zero-shot logits into `petLikelihood` (0–1) and `topLabel`. By default, it uses a **Unified Softmax** over all positive pet prompts and negative distractors to naturally suppress false positive scores for non-pet images.

1. **Unified Softmax (Default):** Runs softmax over concatenated pet + distractor prompt logits with a temperature scale `RELEVANCE_TEMP_SCALE=1.5`. The `petLikelihood` is the sum of all pet category probabilities.
   - **Threshold:** `0.32` decision gate (above this is considered a pet).
   - **Margin Fallback:** `0.40`. If the probability difference between the top-two species classes is less than `0.40`, the label falls back to `"other"` (generic pet).
2. **Baseline Blending (Legacy):** Calculates binary margin score via sigmoid blending: `0.5 * _sigmoid(pet_max - neg_max) + 0.5 * _softmax_max(pet_logits)`.
   - **Threshold:** `0.30` decision gate.
   - **Margin Fallback:** `0.75`.
3. **`petType` hint** — when provided and within margin of the top prediction, stabilizes `topLabel` to the hinted species; boosts likelihood toward that class.

Hint-stabilized subclass accuracy on the 305-image eval: **51.0% → 77.1%** (standard profile). Reproduce: `python dev_benchmarks/simulate_hints.py`.

---

## Operator CLI

```bash
./tools/run doctor --profile quality
./tools/run smoke --profile quality
./tools/run benchmark --profile quality --runs 5
python dev_benchmarks/evaluate_workflow.py --profile standard   # 305-image accuracy
python dev_benchmarks/evaluate_workflow.py --profile quality
```

Maintainers:

```bash
./tools/run export --output output/onnx
./tools/run validate --models-dir output/onnx
```

---

## Startup validation

Common fatal misconfigurations:

- `DEVICE=cuda` on CPU Docker image → use `docker-compose.gpu.yml`
- `ORT_EXECUTION_PROVIDER=coreml` inside Linux Docker → run natively on macOS
- CUDA requested but `torch.cuda.is_available()` is false

---

## Troubleshooting

| Symptom | Fix |
| :--- | :--- |
| Worker exits on start | `docker compose logs` + `python -m tools doctor` |
| Upstash connection failed | Use `rediss://` not `redis://` |
| Slow first job | Normal — models download on first warmup |
| INT8 relevance drift | `python -m tools validate` or use torch runtime |

---

## Webhook metadata

Callbacks include runtime info:

```json
{
  "runtime": "onnx",
  "executionProvider": "CPUExecutionProvider",
  "modelPrecision": "int8",
  "matchModel": "google/siglip2-base-patch16-384"
}
```
