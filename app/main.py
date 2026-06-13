"""
UnLostPaws Vision Worker — background consumer daemon.

This is the main entry point for the worker process. It performs initialization tasks,
pre-warms the machine learning models based on the active VISION_PROFILE, handles OS-level
termination signals (SIGINT/SIGTERM) to support graceful container shutdowns, and orchestrates
the asynchronous consumer event loop for Redis Streams.
"""

import asyncio
import logging
import signal
import sys

from app.config.settings import settings
from app.models.registry import warmup
from app.queue.consumer import create_redis_client, run_consumer

# Configure structured system logging format and level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Event signal utilized to broadcast shutdown requests to concurrent async tasks
stop_event = asyncio.Event()


def handle_exit_signal(sig, frame):
    """
    Callback handler for OS-level interrupts.
    Flags the stop_event so the worker can cleanly complete current tasks,
    close external network requests, and exit.
    """
    logger.info("Termination signal %s received. Shutting down worker...", sig)
    stop_event.set()


async def main():
    """
    Main asynchronous executor. Initializes settings, pre-warms models,
    creates connection clients, and keeps the stream consumer running.
    """
    logger.info(
        "Starting UnLostPaws Vision Worker (version=%s)...", settings.worker_version
    )

    # Register exit handlers for clean shutdown in orchestrations like Kubernetes or Docker
    # Catches SIGINT (Ctrl+C) and SIGTERM (Docker/K8s stop request)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handle_exit_signal)
        except ValueError:
            # signal registration is only allowed in the main thread.
            # Catching ValueError prevents crashes if run inside sub-threads.
            pass

    # Model Warmup Stage:
    # Pre-loads the ML weights (NSFW, SigLIP2) into memory (and GPU if enabled)
    # during start-up to avoid latency spike on the first incoming job.
    if settings.embed_enabled or settings.safety_enabled:
        try:
            await warmup()
        except Exception:
            # Log warmup issues, but allow execution to continue.
            # The models will lazy-load on-demand when the first job arrives.
            logger.exception(
                "Model warmup failed — loading models on demand during first job"
            )

    # Verify that the broker URL is configured before spinning up the consumer loop.
    if not settings.redis_url:
        logger.error("REDIS_URL not set — stream consumer disabled. Exiting.")
        sys.exit(1)

    # Instantiate the asynchronous Redis client using settings broker URL
    redis_client = create_redis_client()

    # Verify connection to the Redis server before starting the consumer loop
    try:
        logger.info("Verifying connection to Redis...")
        await redis_client.ping()
        logger.info("Successfully connected to Redis.")
    except Exception as exc:
        logger.error("Failed to connect to Redis: %s", exc)
        if "upstash.io" in settings.redis_url and settings.redis_url.startswith(
            "redis://"
        ):
            logger.error(
                "CRITICAL ERROR: Detected Upstash Redis URL using 'redis://' (non-SSL) protocol. "
                "Upstash typically requires SSL/TLS. Please configure your REDIS_URL to use the 'rediss://' protocol "
                "instead of 'redis://' to enable SSL."
            )
        else:
            logger.error(
                "CRITICAL ERROR: Could not establish a connection to the Redis server. "
                "Please verify your REDIS_URL configuration and network connectivity."
            )
        await redis_client.aclose()
        sys.exit(1)

    # Schedule the stream consumer task and the shutdown monitor task concurrently
    consumer_task = asyncio.create_task(run_consumer(redis_client))
    stop_task = asyncio.create_task(stop_event.wait())

    # Maintain execution until either the consumer crashes or a termination signal fires.
    try:
        done, pending = await asyncio.wait(
            [consumer_task, stop_task], return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        # Graceful cleanup procedure:
        # Cancel the shutdown monitor task.
        stop_task.cancel()

        # Shut down the background stream consumer task.
        logger.info("Cancelling stream consumer task...")
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            # Handled expected CancelledError upon shutdown.
            pass

        # Safely close the Redis client connection pool.
        logger.info("Closing Redis client connection...")
        await redis_client.aclose()
        logger.info("Worker stopped successfully.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Catch fallback Ctrl+C keyboard interrupt exceptions.
        logger.info("Worker stopped via KeyboardInterrupt.")
