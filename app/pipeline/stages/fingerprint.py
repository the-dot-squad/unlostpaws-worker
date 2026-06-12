"""
MD5 Cryptographic and Perceptual Hashing (pHash) Fingerprinting.

Generates stable identifiers for duplicate and near-duplicate detection:
1. MD5: Cryptographic checksum of raw bytes. Detects exact duplicates.
2. pHash: Perceptual hash of the visual contents. Detects near-duplicates
   that have been resized, compressed, or slightly modified.
"""

import hashlib
from io import BytesIO

import imagehash
from PIL import Image


def compute_md5(image_bytes: bytes) -> str:
    """
    Computes hex digest MD5 hash of raw image data.
    """
    return hashlib.md5(image_bytes).hexdigest()


def compute_phash(image: Image.Image) -> str:
    """
    Computes a 64-bit perceptual hash (pHash) of the PIL image.
    Uses discrete cosine transform (DCT) under the hood to capture structure.
    """
    return str(imagehash.phash(image))


def fingerprint_image(
    image_bytes: bytes, pil_image: Image.Image | None = None
) -> tuple[str, str]:
    """
    Consolidated helper. Resolves (md5, phash) for a single image raw data.
    If pil_image is provided, it avoids decoding the raw bytes again.
    """
    md5 = compute_md5(image_bytes)
    if pil_image is None:
        # Decode the raw bytes to a PIL image to compute the perceptual hash
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
    else:
        img = pil_image
    phash = compute_phash(img)
    return md5, phash
