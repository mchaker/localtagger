"""PixAI v1.0 via an ONNX export (onnxruntime; no PyTorch or remote code).

The export ships ``model.onnx`` with its weights in ``model.onnx.data`` (the
two must sit side by side, which hf_hub_download's snapshot directory
guarantees) and ``tags.json``, which maps the 30,877 logits to categories by
offset. Preprocessing mirrors the upstream ``RescalePadProcessor``: flatten
alpha onto white, fit inside 1008 x 1008, centre on black padding, then scale
to [-1, 1] so the padding becomes -1.
"""

import json

import numpy as np
from PIL import Image

from app.config import settings

from .base import TagResult, Tagger

IMAGE_SIZE = 1008
# PixAI calls its Danbooru artist labels "style".
_CATEGORY_FIELDS = {
    "general": "general",
    "character": "character",
    "copyright": "copyright",
    "style": "artist",
    "meta": "meta",
    "rating": "rating",
}


def _prepare_image(image: Image.Image, size: int = IMAGE_SIZE) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGBA")
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

    width, height = image.size
    if width != size or height != size:
        scale = min(size / width, size / height)
        new_width, new_height = int(width * scale), int(height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        padded = Image.new("RGB", (size, size), (0, 0, 0))
        padded.paste(image, ((size - new_width) // 2, (size - new_height) // 2))
        image = padded

    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return arr.transpose(2, 0, 1)[None, ...]


class PixaiOnnxTagger(Tagger):
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
        hf_hub_download(repo_id=self.spec.repo, filename="model.onnx.data", token=token)
        tags_path = hf_hub_download(repo_id=self.spec.repo, filename="tags.json", token=token)

        with open(tags_path, encoding="utf-8") as f:
            tag_map = json.load(f)
        self._categories = [
            (_CATEGORY_FIELDS[c["name"]], int(c["offset"]), c["tags"])
            for c in tag_map["categories"]
            if c["name"] in _CATEGORY_FIELDS
        ]

        providers = ["CPUExecutionProvider"]
        if (
            settings.resolve_device().startswith("cuda")
            and "CUDAExecutionProvider" in ort.get_available_providers()
        ):
            providers.insert(0, "CUDAExecutionProvider")
        self._session = ort.InferenceSession(model_path, providers=providers)
        self._loaded = True

    def _probabilities(self, image: Image.Image) -> np.ndarray:
        input_name = self._session.get_inputs()[0].name
        logits = self._session.run(None, {input_name: _prepare_image(image)})[0][0]
        return np.exp(-np.logaddexp(0.0, -logits.astype(np.float32)))  # stable sigmoid

    def _tag_one(
        self, image: Image.Image, general_threshold: float, character_threshold: float
    ) -> TagResult:
        probs = self._probabilities(image)

        # Return the other categories unfiltered; Tagger.tag() applies their
        # cutoffs so callers can lower any of them below the recommended value.
        cutoffs = {"general": general_threshold, "character": character_threshold}
        result = TagResult()
        for field_name, offset, tags in self._categories:
            scores = probs[offset:offset + len(tags)]
            selected = np.flatnonzero(scores > cutoffs.get(field_name, 0.0))
            setattr(result, field_name, {tags[i]: float(scores[i]) for i in selected})
        return result

    def unload(self) -> None:
        self._session = None
        self._loaded = False
