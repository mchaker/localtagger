"""Tag post-processing and tag-string formatting.

Extracted from the original ``run_wd14`` so every backend formats its output
identically. Takes a ``{tag: score}`` dict and produces the sorted dict plus a
ready-to-display tag string honoring the caller's formatting options.
"""

import random
import re
from typing import Dict, List, Tuple

RE_SPECIAL = re.compile(r"([\\()])")

# Tags known to hallucinate due to color bleeding from clothing/backgrounds.
# We enforce a stricter minimum confidence for these regardless of the
# user-provided threshold.
HIGH_HALLUCINATION_TAGS = {
    "blue_skin",
    "green_skin",
    "red_skin",
    "colored_skin",
    "pale_skin",
}
HALLUCINATION_MIN_CONFIDENCE = 0.85


def filter_hallucinations(tags: Dict[str, float]) -> Dict[str, float]:
    """Drop notorious color-bleed tags that fall below a strict confidence."""
    filtered = {}
    for tag, score in tags.items():
        tag_norm = tag.lower().replace(" ", "_")
        if tag_norm in HIGH_HALLUCINATION_TAGS and score < HALLUCINATION_MIN_CONFIDENCE:
            continue
        filtered[tag] = score
    return filtered


def format_tags(
    tags: Dict[str, float],
    *,
    use_spaces: bool = False,
    use_escape: bool = True,
    include_ranks: bool = False,
    score_descend: bool = True,
    trigger_word: str = "",
    random_order: bool = False,
    drop_hallucinations: bool = True,
) -> Tuple[Dict[str, float], str]:
    """Return ``(sorted_tags, tag_string)`` for a ``{tag: score}`` mapping."""
    if drop_hallucinations:
        tags = filter_hallucinations(tags)

    text_items: List[str] = []
    tags_pairs = list(tags.items())

    if random_order:
        random.shuffle(tags_pairs)
    elif score_descend:
        tags_pairs = sorted(tags_pairs, key=lambda x: (-x[1], x[0]))

    for tag, score in tags_pairs:
        tag_outformat = tag
        if use_spaces:
            tag_outformat = tag_outformat.replace("_", "-")
        else:
            tag_outformat = tag_outformat.replace(" ", ", ")
            tag_outformat = tag_outformat.replace("_", " ")
        if use_escape:
            tag_outformat = re.sub(RE_SPECIAL, r"\\\1", tag_outformat)
        if include_ranks:
            tag_outformat = f"({tag_outformat}:{score:.3f})"
        text_items.append(tag_outformat)

    if trigger_word:
        text_items.insert(0, trigger_word)

    output_text = " ".join(text_items) if use_spaces else ", ".join(text_items)

    sorted_tags = dict(sorted(tags.items(), key=lambda item: item[1], reverse=True))
    return sorted_tags, output_text
