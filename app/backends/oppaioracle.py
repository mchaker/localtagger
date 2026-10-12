"""OppaiOracle taggers (onnxruntime; no PyTorch or remote code).

Grio43's OppaiOracle releases ship ``model.onnx`` (sigmoid already applied) and
``selected_tags.csv``, which maps each output index (``tag_id``) to a tag. Tag
names carry a category prefix (``gen:``, ``char:``, ``copyright:``, ``artist:``,
``meta:``, ``rating:``) that is stripped here; index 0 and 1 are PAD and UNK.

The graph takes two inputs: the image and a padding mask. Preprocessing mirrors
the release's ``infer.py``: flatten alpha onto grey 114, downscale (never
upscale) with LANCZOS to fit the input size, centre on grey 114 padding, scale
to [-1, 1], and mark the padded pixels True in the mask.
"""

import csv

import numpy as np
from PIL import Image

from app.config import settings

from .base import TagResult, Tagger

IMAGE_SIZE = 448
PAD_COLOR = (114, 114, 114)
_CATEGORY_FIELDS = {
    "gen": "general",
    "char": "character",
    "copyright": "copyright",
    "artist": "artist",
    "meta": "meta",
    "rating": "rating",
}


def _prepare_image(image: Image.Image, size: int = IMAGE_SIZE):
    """Return ``(pixel_values, padding_mask)`` batches of one."""
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGB", image.size, PAD_COLOR)
        background.paste(rgba.convert("RGB"), mask=rgba.getchannel("A"))
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    scale = min(1.0, size / image.width, size / image.height)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    if (width, height) != image.size:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), PAD_COLOR)
    left, top = (size - width) // 2, (size - height) // 2
    canvas.paste(image, (left, top))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    mask = np.ones((size, size), dtype=bool)
    mask[top:top + height, left:left + width] = False
    return np.ascontiguousarray(arr.transpose(2, 0, 1))[None, ...], mask[None, ...]


def _parse_tags(rows):
    """Group ``selected_tags.csv`` rows into ``(field, indices, names)`` blocks.

    Rows without a known prefix (PAD, UNK) are dropped.
    """
    grouped = {}
    for row in rows:
        prefix, _, name = row["name"].partition(":")
        field_name = _CATEGORY_FIELDS.get(prefix)
        if field_name:
            indices, names = grouped.setdefault(field_name, ([], []))
            indices.append(int(row["tag_id"]))
            names.append(name)
    return [(f, np.array(i), n) for f, (i, n) in grouped.items()]


class OppaiOracleTagger(Tagger):
    def __init__(self, spec):
        super().__init__(spec)
        self._session = None
        self._categories = []

    def load(self) -> None:
        if self._loaded:
            return

        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        token = settings.hf_token
        model_path = hf_hub_download(repo_id=self.spec.repo, filename="model.onnx", token=token)
        tags_path = hf_hub_download(repo_id=self.spec.repo, filename="selected_tags.csv", token=token)

        with open(tags_path, encoding="utf-8", newline="") as f:
            self._categories = _parse_tags(csv.DictReader(f))

        providers = ["CPUExecutionProvider"]
        if (
            settings.resolve_device().startswith("cuda")
            and "CUDAExecutionProvider" in ort.get_available_providers()
        ):
            providers.insert(0, "CUDAExecutionProvider")
        self._session = ort.InferenceSession(model_path, providers=providers)
        self._loaded = True

    def _probabilities(self, image: Image.Image) -> np.ndarray:
        pixels, mask = _prepare_image(image)
        return self._session.run(
            ["probabilities"], {"pixel_values": pixels, "padding_mask": mask}
        )[0][0]

    def _tag_one(
        self, image: Image.Image, general_threshold: float, character_threshold: float
    ) -> TagResult:
        probs = self._probabilities(image)

        # Return the other categories unfiltered; Tagger.tag() applies their
        # cutoffs so callers can lower any of them below the recommended value.
        cutoffs = {"general": general_threshold, "character": character_threshold}
        result = TagResult()
        for field_name, indices, names in self._categories:
            scores = probs[indices]
            selected = np.flatnonzero(scores > cutoffs.get(field_name, 0.0))
            setattr(result, field_name, {names[i]: float(scores[i]) for i in selected})
        return result

    def unload(self) -> None:
        self._session = None
        self._loaded = False
