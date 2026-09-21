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

        device = settings.resolve_device()
        # Keep CPU inference in FP32. The optional mixed-BF16 checkpoint is
        # useful on supported GPUs, and keeps its classification head in FP32.
        use_bf16 = (
            self.spec.dtype == "bfloat16"
            and device.startswith("cuda")
            and torch.cuda.is_bf16_supported()
        )
        tagger = pipeline(
            model=self.spec.repo,
            image_processor=self.spec.repo,
            trust_remote_code=True,
            token=settings.hf_token,
            device=device,
            dtype=torch.float32,
            use_fast=False,
        )
        if use_bf16:
            # Loading the whole checkpoint as BF16 would round the FP32 head
            # before we could restore its dtype. Preserve its original values.
            head_state = {name: value.clone() for name, value in tagger.model.head.state_dict().items()}
            tagger.model.bfloat16()
            tagger.model.head.float()
            tagger.model.head.load_state_dict(head_state)
        self._pipeline = tagger
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
