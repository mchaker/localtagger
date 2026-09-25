"""PixAI v1.0 via its native Transformers pipeline (no ONNX export needed)."""

from PIL import Image

from app.config import settings

from .base import TagResult, Tagger


class PixaiTransformersTagger(Tagger):
    def __init__(self, spec):
        super().__init__(spec)
        self._pipeline = None

    def load(self) -> None:
        if self._loaded:
            return

        import torch
        from transformers import pipeline

        self._pipeline = pipeline(
            model=self.spec.repo,
            image_processor=self.spec.repo,
            trust_remote_code=True,
            token=settings.hf_token,
            device=settings.resolve_device(),
            dtype=torch.float32,
            use_fast=False,
        )
        self._loaded = True

    def _tag_one(
        self, image: Image.Image, general_threshold: float, character_threshold: float
    ) -> TagResult:
        categories = self._pipeline(image, threshold={
            "general": general_threshold,
            "character": character_threshold,
            # Return these scores before filtering in Tagger.tag(), allowing
            # callers to lower any category below the recommended cutoff.
            "copyright": 0.0,
            "style": 0.0,
            "meta": 0.0,
            "rating": 0.0,
        })["results"]
        return TagResult(
            general=categories.get("general", {}),
            character=categories.get("character", {}),
            copyright=categories.get("copyright", {}),
            # PixAI calls its Danbooru artist labels "style".
            artist=categories.get("style", {}),
            meta=categories.get("meta", {}),
            rating=categories.get("rating", {}),
        )

    def unload(self) -> None:
        self._pipeline = None
        self._loaded = False
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
