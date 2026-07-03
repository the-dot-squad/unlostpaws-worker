# ==============================================================================
# UnLostPaws Vision Worker - Production Dockerfile
# ==============================================================================
# Multi-stage/modular image setup for Python machine learning inference.
# ==============================================================================

# Use Python 3.12 slim-debian image for a lightweight, secure, and production-ready foundation.
FROM python:3.12-slim

# Establish the working directory in the container.
WORKDIR /app

# Copy the dependencies file first to leverage Docker layer caching.
# Re-running pip install only occurs if requirements.txt changes.
COPY requirements.txt .

# Install Python dependencies.
# - --no-cache-dir: Prevents writing cache files, further reducing container size.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the core application source code.
COPY app/ ./app/

# Create a non-root system user for security compliance.
# - useradd -u 1001: Creates user 'appuser' with UID 1001.
# - mkdir -p: Generates Hugging Face cache directory so it can be mounted or persisted.
# - chown -R: Restricts files within /app to the 'appuser' permission boundaries.
RUN useradd --create-home --uid 1001 appuser \
    && mkdir -p /app/.cache/huggingface \
    && chown -R appuser:appuser /app

# Switch executing user context to the non-privileged system user.
USER appuser

# Configure runtime environment variables:
# - HF_HOME: Tells Hugging Face transformers where to read and write downloaded model weights.
# - VISION_PROFILE: Sets default fallback hardware execution preset to cpu-quality.
# - PYTHONPATH: Appends /app to python module resolution path so imports function correctly.
ENV HF_HOME=/app/.cache/huggingface
ENV VISION_PROFILE=cpu-quality
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV RUNNING_IN_DOCKER=true

# Container Healthcheck Configuration:
# - interval=30s: Run healthchecks every 30 seconds.
# - timeout=10s: If the check takes longer than 10 seconds, count as a failure.
# - start-period=180s: Gives the worker 3 minutes to download models during startup before failing.
# - retries=3: Fail three times consecutively before marking the container as unhealthy.
# - CMD: Invokes the heartbeat monitoring script app/healthcheck.py.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python app/healthcheck.py

# Default entrypoint CMD. Launches the asyncio daemon background consumer.
CMD ["python", "app/main.py"]
