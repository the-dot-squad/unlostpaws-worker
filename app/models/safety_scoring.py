"""
Shared NSFW probability aggregation for multi-class safety models.

Keyword-based label matching is identical across torch and ONNX backends so
threshold tuning stays consistent regardless of runtime.
"""

_NSFW_LABEL_KEYWORDS = frozenset(
    {
        "nsfw",
        "porn",
        "pornography",
        "hentai",
        "nude",
        "explicit",
        "enticing",
        "sensual",
    }
)


def nsfw_score_from_probs(
    probs: list[float],
    id2label: dict[int, str],
) -> tuple[float, str]:
    """
    Determine NSFW score and dominant label from a probability vector.

    Returns:
        Tuple of (nsfw_score, label) where nsfw_score is in [0, 1].
    """
    best_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    best_label = id2label.get(best_idx, str(best_idx)).lower()

    nsfw_score = 0.0
    top_label = best_label

    for idx, prob in enumerate(probs):
        label = id2label.get(idx, str(idx)).lower()
        if label in ("normal", "sfw", "safe"):
            continue
        if any(kw in label for kw in _NSFW_LABEL_KEYWORDS):
            nsfw_score = max(nsfw_score, prob)

    if nsfw_score == 0.0 and best_label not in ("normal", "sfw", "safe"):
        nsfw_score = float(probs[best_idx])
        top_label = best_label
    elif nsfw_score == 0.0:
        top_label = "normal"

    return nsfw_score, top_label
