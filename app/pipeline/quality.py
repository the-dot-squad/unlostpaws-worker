"""
Image Quality Heuristics — Blur calculation and minimum dimension evaluations.

Provides utility functions to evaluate image blur scores and verify that
the dimensions satisfy minimum resolution requirements.
"""

import numpy as np
from PIL import Image


def laplacian_blur_score(image: Image.Image) -> float:
    """
    Computes a normalized blur score in the range [0.0, 1.0].
      - 0.0 represents a sharp, high-contrast image.
      - 1.0 represents a completely blurry or solid image.

    Execution Steps:
      1. Converts the PIL Image to grayscale ('L') and downscales it to 256x256
         using Bilinear interpolation. This standardizes the image scale and speeds up math.
      2. Converts the image to a NumPy float32 array.
      3. Applies a discrete Laplacian operator (convolution kernel) to calculate second derivatives:
           laplacian = -4 * center + top + bottom + left + right
      4. Calculates the variance of the resulting Laplacian array. High contrast edges
         produce high variance, whereas blurry or smooth images produce very low variance.
      5. Normalizes the variance using a sigmoid-like curve: 1.0 / (1.0 + variance / 100.0).
         This maps low variance (blurry) to scores near 1.0, and high variance (sharp) to scores near 0.0.
    """
    # Downscale and convert to grayscale array
    gray = np.array(
        image.convert("L").resize((256, 256), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )

    # Apply discrete Laplacian convolution using array slicing
    lap = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    # Compute contrast variance
    variance = float(lap.var())

    # Normalize and return score
    return float(1.0 / (1.0 + variance / 100.0))


def assess_quality(
    image: Image.Image, min_width: int = 0, min_height: int = 0, max_blur: float = 1.0
) -> dict:
    """
    Evaluates image specifications against size and blur thresholds.

    Returns a dictionary:
      - width: decoded width in pixels.
      - height: decoded height in pixels.
      - blurScore: normalized float score in [0, 1].
      - ok: Boolean flag indicating if image passes criteria.
    """
    width, height = image.size
    blur_score = laplacian_blur_score(image)

    # Verify image dimensions and blur limits
    ok = width >= min_width and height >= min_height and blur_score <= max_blur

    return {
        "width": width,
        "height": height,
        "blurScore": round(blur_score, 4),
        "ok": ok,
    }
