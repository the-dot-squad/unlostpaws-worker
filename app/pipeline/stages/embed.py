"""
SigLIP / DINO Vector Embedding Stage.

Dispatches a batch of decoded PIL images to the active feature extractor
model to generate visual representations.
"""

from PIL import Image

from app.config.settings import Settings, settings
from app.models.registry import get_match_embedder, run_in_executor


async def embed_stage(
    images: list[Image.Image], cfg: Settings = settings
) -> list[list[float]]:
    """
    Asynchronous entrypoint to generate image vector embeddings.

    If embedding feature is disabled, returns a list of empty lists matching
    the input images size.
    """
    if not cfg.embed_enabled:
        return [[] for _ in images]

    # Retrieve matching model embedder singleton
    embedder = get_match_embedder(cfg)
    if not embedder:
        return [[] for _ in images]

    def _run():
        preds = embedder.embed_batch(images)
        return [p.embedding for p in preds]

    # Offload PyTorch vector generation to background executor
    return await run_in_executor(_run)
