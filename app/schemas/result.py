"""
Pydantic Schemas for pipeline results and ML callback payloads.

This module houses the data structures that hold execution results for
each stage of the pipeline, as well as the composite callback payload schema
posted back to Next.js webhook handlers.
"""

from pydantic import BaseModel, Field


class SafetyResult(BaseModel):
    """
    NSFW safety classification result details.
    """

    # Probability confidence score that the content is unsafe [0, 1]
    nsfwScore: float = 0.0

    # Classification result label (e.g. 'normal', 'nsfw')
    label: str = "unknown"

    # Model identifier string used to run the safety classification
    model: str = ""


class RelevanceResult(BaseModel):
    """
    Zero-shot pet relevance verification result details.
    """

    # Relevance rating [0, 1] representing the confidence that the image contains a pet
    petLikelihood: float = 0.0

    # Classification label representing the detected pet type (e.g. 'dog', 'cat')
    topLabel: str = ""


class QualityResult(BaseModel):
    """
    Image resolution and quality assessment details.
    """

    # Width of the decoded image in pixels
    width: int = 0

    # Height of the decoded image in pixels
    height: int = 0

    # Laplacian blur rating in [0, 1] (lower values mean sharper, higher values mean blurrier)
    blurScore: float = 0.0

    # True if the image satisfies minimum size constraints and maximum blur constraints
    ok: bool = True


class ProcessedImageResult(BaseModel):
    """
    Consolidated processing results for a single image.
    """

    # Original image URL downloaded by the worker
    url: str

    # Storage key where the image is persisted (copied from url by default)
    s3Key: str = ""

    # Cryptographic MD5 checksum of the raw download bytes (exact-match tracking)
    md5: str = ""

    # Perceptual hash string of the visual contents (near-duplicate tracking)
    phash: str = ""

    # SigLIP embedding vector (768-d for standard and quality profiles)
    embedding: list[float] = Field(default_factory=list)

    # NSFW classification details if the safety stage was executed
    safety: SafetyResult | None = None

    # Pet validation details if the relevance stage was executed
    relevance: RelevanceResult | None = None

    # Image resolution and blur details if the quality stage was executed
    quality: QualityResult | None = None


class ImageError(BaseModel):
    """
    Error tracking container for a single failed image URL.
    """

    # The URL that failed to process
    url: str

    # Error message detailing why downloading or processing failed
    error: str


class JobResult(BaseModel):
    """
    Internal accumulator holding outputs for all image items in a single job.
    """

    # The type category of the processing job (listing, search, etc.)
    job_type: str

    # Listing database identifier (optional)
    listing_id: str | None = None

    # Registered pet identifier (optional)
    owned_pet_id: str | None = None

    # Search session identifier (optional)
    search_session_id: str | None = None

    # Collection of results for successfully processed images
    images: list[ProcessedImageResult] = Field(default_factory=list)

    # Collection of failures tracked during execution
    errors: list[ImageError] = Field(default_factory=list)


class CallbackPayload(BaseModel):
    """
    Data validation schema representing the payload POSTed back to the Next.js API.
    """

    # Category representing the job classification
    jobType: str

    # Semantic version string of the active worker instance
    workerVersion: str

    # Model ID used to extract search vector representations
    matchModel: str = ""

    # Model ID used to perform safety classification Checks
    safetyModel: str = ""

    # Model ID used to compute vector embeddings (same as matchModel)
    embeddingModel: str = ""

    # Active inference runtime (torch or onnx)
    runtime: str = ""

    # Active ONNX Execution Provider name (empty for torch)
    executionProvider: str = ""

    # Model precision used for inference (fp32, fp16, int8)
    modelPrecision: str = ""

    # Listing database identifier (echoed back if present)
    listingId: str | None = None

    # Registered pet identifier (echoed back if present)
    ownedPetId: str | None = None

    # Search session identifier (echoed back if present)
    searchSessionId: str | None = None

    # List of serialized ProcessedImageResult dictionaries
    images: list[dict] = Field(default_factory=list)

    # List of serialized ImageError dictionaries
    errors: list[dict] = Field(default_factory=list)
