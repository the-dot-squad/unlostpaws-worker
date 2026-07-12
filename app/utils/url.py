import os
from urllib.parse import urlparse, urlunparse


def rewrite_local_url(url: str) -> str:
    """
    Rewrites localhost/127.0.0.1 in URLs to host.docker.internal if running
    inside a Docker container to allow reaching the host services.
    """
    if os.path.exists("/.dockerenv") or os.environ.get("RUNNING_IN_DOCKER") == "true":
        try:
            parsed = urlparse(url)
            netloc = parsed.netloc
            hostname = parsed.hostname
            if hostname == "localhost":
                new_netloc = netloc.replace("localhost", "host.docker.internal", 1)
                return urlunparse(parsed._replace(netloc=new_netloc))
            elif hostname == "127.0.0.1":
                new_netloc = netloc.replace("127.0.0.1", "host.docker.internal", 1)
                return urlunparse(parsed._replace(netloc=new_netloc))
        except Exception:
            pass
    return url
