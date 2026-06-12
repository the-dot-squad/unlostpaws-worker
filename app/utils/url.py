import os


def rewrite_local_url(url: str) -> str:
    """
    Rewrites localhost/127.0.0.1 in URLs to host.docker.internal if running
    inside a Docker container to allow reaching the host services.
    """
    if os.path.exists("/.dockerenv") or os.environ.get("RUNNING_IN_DOCKER") == "true":
        if "localhost" in url:
            return url.replace("localhost", "host.docker.internal", 1)
        if "127.0.0.1" in url:
            return url.replace("127.0.0.1", "host.docker.internal", 1)
    return url
