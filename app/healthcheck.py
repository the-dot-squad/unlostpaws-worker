"""
Lightweight health check script.

This script runs as a side-car command or periodic Docker HEALTHCHECK invocation.
It verifies that the background worker is actively polling Redis and processing
jobs. It accomplishes this by reading the modified time (mtime) of a local
heartbeat file (/tmp/worker-heartbeat) updated by the consumer loop.
"""

import os
import time
import sys

# Path to the temporary heartbeat file updated by the Redis consumer loop
HEARTBEAT_FILE = "/tmp/worker-heartbeat"

# Maximum threshold in seconds since the last heartbeat update.
# If the file hasn't been updated in 60 seconds, the worker is considered stalled.
MAX_AGE_SECONDS = 60


def main() -> None:
    """
    Evaluates the status of the worker heartbeat file.
    Exits with code 0 if healthy, or 1 if unhealthy or missing.
    """
    # 1. Check if the heartbeat file has been created at all.
    if not os.path.exists(HEARTBEAT_FILE):
        print(
            f"Error: Heartbeat file {HEARTBEAT_FILE} does not exist. Worker has not started polling yet.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 2. Retrieve the file's last modified timestamp.
    try:
        mtime = os.path.getmtime(HEARTBEAT_FILE)
    except Exception as exc:
        print(
            f"Error: Failed to read heartbeat file modified time: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    # 3. Compute the time elapsed since the last update.
    age = time.time() - mtime

    # 4. If the age exceeds our maximum threshold, flag the worker as unhealthy.
    if age > MAX_AGE_SECONDS:
        print(
            f"Error: Heartbeat file is too old ({age:.1f}s > {MAX_AGE_SECONDS}s). Consumer loop might be deadlocked or blocked.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 5. Output success and exit with 0 to indicate a healthy status to Docker.
    print(f"Healthy: Worker heartbeat is up-to-date (age: {age:.1f}s).")
    sys.exit(0)


if __name__ == "__main__":
    main()
