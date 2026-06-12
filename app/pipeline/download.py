"""
Parallel image download and single-pass PIL decoder module.

Provides functions to fetch image assets concurrently from public URLs or cloud storage
objects (S3/R2) using async HTTP, decode the raw byte responses, and convert
them into standardized RGB PIL Images for model compatibility.
"""

import asyncio
import logging
from dataclasses import dataclass
from io import BytesIO

import httpx
from PIL import Image

from app.config.settings import settings
from app.utils.url import rewrite_local_url

logger = logging.getLogger(__name__)


@dataclass
class DecodedImage:
    """
    Wrapper carrying downloaded image details and decoded representations.
    """

    # Original URL of the image
    url: str

    # Raw uncompressed download bytes
    raw_bytes: bytes

    # Pillow Image representation, converted to RGB mode
    image: Image.Image


async def download_bytes(client: httpx.AsyncClient, url: str) -> bytes:
    """
    Fetches raw bytes of a single asset using an active HTTP client connection.
    Implements retries with exponential backoff for transient HTTP statuses (429, 500, 502, 503, 504)
    and network connection failures.
    """
    request_url = rewrite_local_url(url)
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            response = await client.get(request_url)
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as exc:
            # Check if status code is transient (429 or 5xx)
            status_code = getattr(exc.response, "status_code", None)
            is_transient = status_code in (429, 500, 502, 503, 504)
            if not is_transient or attempt == max_retries:
                raise
            delay = 2**attempt
            logger.warning(
                "Transient HTTP status %s downloading %s. Retrying in %ds (Attempt %d/%d)...",
                str(status_code),
                url,
                delay,
                attempt + 1,
                max_retries,
            )
            await asyncio.sleep(delay)
        except httpx.RequestError as exc:
            if attempt == max_retries:
                raise
            delay = 2**attempt
            logger.warning(
                "Network error '%s' downloading %s. Retrying in %ds (Attempt %d/%d)...",
                exc,
                url,
                delay,
                attempt + 1,
                max_retries,
            )
            await asyncio.sleep(delay)


def decode_image(url: str, data: bytes) -> DecodedImage:
    """
    Decodes raw bytes using Pillow.
    Forces conversion to 'RGB' color space to prevent channel mismatch errors
    when feeding images to PyTorch/Hugging Face classifiers (e.g. discarding CMYK
    or alpha channels).
    """
    img = Image.open(BytesIO(data)).convert("RGB")
    return DecodedImage(url=url, raw_bytes=data, image=img)


async def download_all(urls: list[str]) -> tuple[list[DecodedImage], list[dict]]:
    """
    Fetches and decodes all specified image URLs concurrently.

    Uses:
      - asyncio.Semaphore to throttle concurrency and prevent file descriptor / connection exhaustion.
      - httpx.AsyncClient connection pooling for HTTP request reuse.

    Returns a tuple:
      - list[DecodedImage]: Successfully downloaded and decoded items.
      - list[dict]: Encountered error details (URL + error message string).
    """
    # Throttling gate to limit max concurrent open network requests
    sem = asyncio.Semaphore(settings.max_concurrent_downloads)
    results: list[DecodedImage | tuple[str, Exception]] = []

    # Configure the client connection pool
    async with httpx.AsyncClient(
        timeout=settings.download_timeout,
        follow_redirects=True,  # Support CDN redirects automatically
    ) as client:

        async def fetch(url: str):
            async with sem:
                try:
                    data = await download_bytes(client, url)
                    return decode_image(url, data)
                except Exception as exc:
                    # Capture, log, and return the exception instead of crash-terminating
                    # the entire parallel download set.
                    logger.exception("Download failed for URL: %s", url)
                    return url, exc

        # Fire concurrent requests and gather results
        gathered = await asyncio.gather(*[fetch(u) for u in urls])
        results = list(gathered)

    # Sort results into success list vs failure logs
    decoded: list[DecodedImage] = []
    errors: list[dict] = []
    for item in results:
        if isinstance(item, DecodedImage):
            decoded.append(item)
        else:
            url, exc = item
            errors.append({"url": url, "error": str(exc)})

    return decoded, errors
