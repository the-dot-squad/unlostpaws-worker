# UnLostPaws Vision Worker

[![Python Version](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![CI/CD Pipeline](https://github.com/the-dot-squad/unlostpaws-worker/actions/workflows/ci.yml/badge.svg)](https://github.com/the-dot-squad/unlostpaws-worker/actions/workflows/ci.yml)
[![Docker Support](https://img.shields.io/badge/docker-ready-blue.svg?logo=docker)](https://www.docker.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/the-dot-squad/unlostpaws-worker/pulls)

A modular, multi-stage background image intelligence worker built with **Python**, **PyTorch**, and **Hugging Face Transformers**. It runs as a lightweight, independent backend service that consumes resource-intensive machine learning jobs asynchronously from **Redis Streams** and posts results back via HTTP webhook callbacks.

---

## Architecture Overview

The communication between the Next.js frontend application and the Python ML worker is completely decoupled using **Redis Streams**, allowing for efficient queue management, horizontal scaling, and load-leveling.

```
┌─────────────────────────────────┐
│        Next.js Frontend         │◄───────────────────────────────────────────────┐
│      (App Router, Vercel)       │                                                │
└────────────────┬────────────────┘                                                │
                 │                                                                 │
                 │ 1. Enqueue Job (XADD)                                           │ 4. Webhook Callback
                 ▼                                                                 │    (HTTP POST)
┌─────────────────────────────────┐                                                │
│          Redis Streams          │                                                │
│ (unlostpaws:stream:processing)  │                                                │
└────────────────┬────────────────┘                                                │
                 │                                                                 │
                 │ 2. Fetch Job (XREADGROUP)                                       │
                 ▼                                                                 │
┌─────────────────────────────────┐                                                │
│      Python Vision Worker       │                                                │
└────────────────┬────────────────┘                                                │
                 │                                                                 │
                 │ 3. Execute Pipeline Stages                                      │
                 ▼                                                                 │
┌────────────────────────────────────────────────────────────────────────┐         │
│  ┌───────────────────────┐  ┌───────────────────────┐  ┌─────────────┐ │         │
│  │   Moderation/Safety   │  │   Image Fingerprint   │  │  Embedding  │ │─────────┘
│  │   - Resolution        │  │   - MD5 Checksum      │  │  - SigLIP2  │ │
│  │   - Laplacian Blur    │  │   - Perceptual Hash   │  │  - Relevance│ │
│  │   - NSFW Detection    │  │     (pHash)           │  │             │ │
│  └───────────────────────┘  └───────────────────────┘  └─────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Lifecycle

1. **Job Enqueueing:** Next.js uploads pet images to cloud storage (e.g., Cloudflare R2 or AWS S3), persists the initial database entry in MongoDB, and pushes a processing job payload containing the image URLs, metadata, and a webhook callback URL to the Redis stream using `XADD`.
2. **Job Acquisition:** The Python worker polls the Redis stream via `XREADGROUP`. Once a job is obtained, the worker updates its local heartbeat file (`/tmp/worker-heartbeat`) to signal health and proceeds to process the payload.
3. **Image Acquisition & Decoding:** The worker downloads the specified images concurrently using `httpx` with a configurable semaphore limit to avoid overloading network gateways. Images are decoded into standard RGB PIL arrays.
4. **Pipeline Processing:** Images proceed sequentially through the enabled execution stages (quality checks, safety filtering, cryptographic/perceptual fingerprinting, vector embedding, and zero-shot relevance classification).
5. **Success Callback:** The worker serializes the processed results (safety flags, quality status, MD5s, pHashes, and SigLIP2 vector embeddings) and issues an HTTP POST back to the Next.js frontend using the provided `webhookUrl`. Next.js stores the results in MongoDB and indices the vectors in Qdrant.
6. **Failure/DLQ Recovery:** If a job fails (e.g., due to persistent download failures or model crashes) and exceeds the retry threshold, it is automatically forwarded to a Dead Letter Queue (`DLQ`) Redis Stream. The worker then fires a failure webhook callback to Next.js so the system can gracefully flag the issue.

---

## Pipeline Stages

Each processed image goes through the following pipeline stages:

| Stage | Mechanism / Model | Description |
| :--- | :--- | :--- |
| **`quality`** | OpenCV Laplacian | Computes image resolution and blur score. Low variance of Laplacian indicates a blurry image. |
| **`safety`** | `Falconsai/nsfw_image_detection` | Classifies whether content is safe or contains adult material. |
| **`fingerprint`**| MD5 + Perceptual Hashing (pHash) | Generates a cryptographic MD5 (for exact duplicates) and a perceptual hash (for resized/reformatted near-duplicates). |
| **`embed`** | `google/siglip2-base-patch16-224` | Generates a 768-dimensional vector embedding optimized for visual similarity and search. |
| **`relevance`** | SigLIP2 Zero-Shot | Compares the image embedding against pet-specific classification prompts to verify that the uploaded image contains an actual animal matching the requested pet type. |

---

## Sizing & Execution Profiles

Choose the active profile based on your host server's CPU or GPU resources using the `VISION_PROFILE` environment variable. This configuration optimizes memory usage and prevents Out-Of-Memory (OOM) crashes:

| Profile Name | Device | Min RAM | Min VRAM | Description |
| :--- | :---: | :---: | :---: | :--- |
| **`dedup-only`** | CPU | 512 MB | 0 MB | Quality + fingerprinting checks only (No PyTorch/Transformers loaded). |
| **`cpu-light`** | CPU | 1.5 GB | 0 MB | NSFW safety only (No animal matching embeddings). |
| **`cpu-standard`** | CPU | 2.5 GB | 0 MB | Uses lightweight `DINO-v2-small` matching + NSFW safety. |
| **`cpu-quality`** | CPU | 4.0 GB | 0 MB | Uses `SigLIP2` matching + NSFW safety (**Default Development Profile**). |
| **`gpu-standard`** | GPU | 4.0 GB | 4.0 GB | Uses `SigLIP2` + NSFW safety on GPU (**Default Production Profile**). |
| **`gpu-quality`** | GPU | 4.0 GB | 6.0 GB | Uses `SigLIP2` + multi-class `strangerguardhf` NSFW on GPU. |

---

## Configuration & Environment Variables

Create a `.env` file at the root of the project. The following settings are supported:

| Variable | Required | Default | Description |
| :--- | :---: | :--- | :--- |
| `REDIS_URL` | Yes | *None* | Shared Redis instance URI (e.g., `redis://default:pwd@host:6379`). Can also use `UPSTASH_REDIS_URL`. |
| `STREAM_KEY` | No | `unlostpaws:stream:vision-processing` | Redis Stream key where the worker consumes job payloads. |
| `DLQ_STREAM_KEY`| No | `unlostpaws:stream:vision-processing:dlq` | Redis Stream key where failed jobs are sent after maximum retries. |
| `CONSUMER_GROUP`| No | `vision-worker` | The name of the Redis consumer group. |
| `CONSUMER_NAME` | No | `worker-1` | The unique identifier of this specific worker instance. |
| `MAX_JOB_ATTEMPTS` | No | `3` | Maximum processing attempts before a job is moved to the DLQ. |
| `VISION_PROFILE`| No | `cpu-quality` | Sizing profile to load (see table above). |
| `DEVICE` | No | `auto` | Force hardware execution device (`cpu` or `cuda`). |
| `BATCH_SIZE` | No | *Profile default*| Number of images processed in parallel during GPU inference. |
| `HF_HOME` | No | `/app/.cache/huggingface` | Hugging Face local cache directory for downloading models. |
| `DOWNLOAD_TIMEOUT` | No | `30` | Timeout in seconds for downloading each image. |
| `CALLBACK_TIMEOUT` | No | `60` | Timeout in seconds when calling the webhook callback URL. |
| `MAX_CONCURRENT_DOWNLOADS` | No | `4` | Maximum number of concurrent HTTP image downloads. |

---

## How to Run

### 1. Run Pre-built Image from GHCR (Recommended for Deployment)
Since the worker is published on the GitHub Container Registry (GHCR), you can pull and run the container directly without cloning the codebase or building it locally:

> [!NOTE]
> The published GHCR image is a multi-platform build supporting both **`linux/amd64` (x86_64)** and **`linux/arm64`** (e.g., Raspberry Pi 4, Apple Silicon/AWS Graviton).

1. **Configure Environment:** Create a local `.env` file containing your configurations (e.g., `REDIS_URL`).
2. **Execute Container:** Pull and launch the daemon (mounting a persistent volume for the model cache):
   ```bash
   docker run -d \
     --name unlostpaws-worker \
     --env-file .env \
     -v unlostpaws-hf-cache:/app/.cache/huggingface \
     ghcr.io/the-dot-squad/unlostpaws-worker:latest
   ```

### 2. Run Locally with Docker Compose (For Local Development)
If you want to run the worker locally and build it from the source code:
1. Create your local config:
   ```bash
   cp .env.example .env
   # Edit your .env with local Redis URL (e.g. redis://host.docker.internal:6379)
   ```
2. Start the services:
   ```bash
   docker compose up --build
   ```

### 3. Locally on Bare Metal (No Docker)
Ensure you have Python 3.12+ installed (we recommend using [uv](https://github.com/astral-sh/uv) for fast installs):
```bash
cp .env.example .env
# Install dependencies
pip install -r requirements.txt
# Run the worker process
export PYTHONPATH=.
python app/main.py
```

### 4. Production VM Deployment (Self-Hosted GPU/CPU VM)
Because the worker consumes and produces data asynchronously via Redis streams and callbacks, you do **not** need to expose any incoming ports on the container.

You can deploy directly using the existing [docker-compose.yml](./docker-compose.yml) configuration in this repository:

1. **Configure Environment:** Create a production `.env` file pointing to your shared cloud Redis instance (e.g. Upstash or AWS ElastiCache) and select your target `VISION_PROFILE`.
2. **GPU Optimization (Optional):** If deploying on an NVIDIA GPU VM, ensure the host has the NVIDIA Container Toolkit installed, and uncomment the `deploy` block in [docker-compose.yml](./docker-compose.yml).
3. **Start the Service:** Build and launch the container as a background daemon:
   ```bash
   docker compose up -d --build
   ```

---

## Redis Stream Integration Specification

### 1. Input Job Payload Format (Redis Stream entry)
Add a job to the Redis Stream defined by `STREAM_KEY` using `XADD` with a single `payload` field containing a serialized JSON string:

```json
{
  "jobType": "listing",
  "listingId": "listing_123",
  "ownedPetId": "pet_123",            // Optional
  "searchSessionId": "session_123",    // Optional
  "imageUrls": [
    "https://example.com/pet-image-1.jpg"
  ],
  "petType": "dog",                    // Used for relevance classification (e.g., cat, dog, rabbit)
  "webhookUrl": "https://myapp.com/api/internal/ml-callback?secret=token",
  "pipeline": ["quality", "safety", "fingerprint", "embed"] // Optional stage override
}
```

### 2. Success Callback Payload Format
When the worker successfully completes processing, it sends an HTTP POST request to the specified `webhookUrl` with the following body:

```json
{
  "jobType": "listing",
  "workerVersion": "0.1.2",
  "matchModel": "google/siglip2-base-patch16-224",
  "safetyModel": "Falconsai/nsfw_image_detection",
  "embeddingModel": "google/siglip2-base-patch16-224",
  "listingId": "listing_123",
  "ownedPetId": "pet_123",             // Echoed back if sent
  "searchSessionId": "session_123",     // Echoed back if sent
  "images": [
    {
      "url": "https://example.com/pet-image-1.jpg",
      "s3Key": "https://example.com/pet-image-1.jpg",
      "md5": "b10a8db164e0754105b7a99be72e3fe5",
      "phash": "8f1a3e2d6b5c7a90",
      "embedding": [0.0123, -0.456, 0.789], // Float array of size 768 (SigLIP2)
      "safety": {
        "nsfwScore": 0.002,
        "label": "normal",
        "model": "Falconsai/nsfw_image_detection"
      },
      "relevance": {
        "petLikelihood": 0.985,
        "topLabel": "dog"
      },
      "quality": {
        "width": 1200,
        "height": 900,
        "blurScore": 45.2,
        "ok": true
      }
    }
  ],
  "errors": []
}
```

### 3. Failure Callback Payload Format
If the job fails completely (e.g., timeout, invalid image formats, pipeline errors) and exceeds `MAX_JOB_ATTEMPTS`, it will be moved to the Dead Letter Queue (`DLQ_STREAM_KEY`), and a failure webhook is sent to notify the caller:

```json
{
  "jobType": "listing",
  "error": "Failed to download any images; connection timed out.",
  "listingId": "listing_123",
  "ownedPetId": "pet_123",
  "searchSessionId": "session_123"
}
```

---

## Contributing

We welcome contributions from the community to make the UnLostPaws Vision Worker better! To maintain a high level of code quality and consistency, please follow these guidelines:

1. **Fork the Repository:** Create a personal fork of the repository on GitHub.
2. **Create a Branch:** Create a descriptive feature or bugfix branch:
   ```bash
   git checkout -b feature/your-amazing-feature
   ```
3. **Write and Comment Code:** Ensure your code is thoroughly documented. Use docstrings for all modules, classes, and methods, and add explanatory comments for complex logic.
4. **Run Verification:** Compile your code using `py_compile` to ensure there are no syntax or indentation errors:
   ```bash
   python3 -m py_compile app/main.py app/healthcheck.py app/config/*.py app/models/*.py app/schemas/*.py app/queue/*.py app/callback/*.py app/pipeline/*.py app/pipeline/stages/*.py
   ```
5. **Commit and Push:** Write clean, descriptive commit messages, and push your changes to your fork.
6. **Open a Pull Request:** Submit a PR back to the main repository. Provide a detailed summary of the changes and what has been verified.

---

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. 

### What this means:
* **Strong Copyleft:** Anyone who modifies this code and runs it as a service over a network (e.g. hosting it as a SaaS backend worker) **must make their modified source code publicly available** under the same AGPL-3.0 license.
* **No Closed-Source Commercial Forks:** You cannot integrate this software into proprietary closed-source applications or services without open-sourcing the modifications.
* **Attribution:** Any usage, distribution, or execution of the software must retain the original copyright notice and license.

If you wish to use this software in a closed-source commercial product or need custom terms, please contact the maintainers to discuss custom licensing agreements. For full license details, see the [LICENSE](./LICENSE) file in the root of the repository.
