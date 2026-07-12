# ==============================================================================
# UnLostPaws Vision Worker — CPU production image
# ==============================================================================
# Python 3.12 slim + PyTorch (CPU) + ONNX Runtime (CPU) + Hugging Face Transformers.
#
# Default VISION_PROFILE=quality (full pipeline on CPU).
# Use with: docker compose up -d  (see docker-compose.yml)
#
# For NVIDIA GPU use Dockerfile.gpu and docker-compose.gpu.yml instead.
# ==============================================================================

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements-torch.txt ./

RUN pip install --no-cache-dir -r requirements-torch.txt

COPY app/ ./app/
COPY tools/ ./tools/

RUN useradd --create-home --uid 1001 appuser \
    && mkdir -p /app/.cache/huggingface \
    && chown -R appuser:appuser /app

USER appuser

# Hugging Face model download cache.
ENV HF_HOME=/app/.cache/huggingface
# ONNX artifact cache (used when INFERENCE_RUNTIME=onnx).
ENV MODEL_CACHE_DIR=/app/.cache/huggingface/onnx
ENV VISION_PROFILE=quality
ENV WORKER_IMAGE_VARIANT=cpu
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV RUNNING_IN_DOCKER=true

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python app/healthcheck.py

CMD ["python", "app/main.py"]
