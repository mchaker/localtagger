"""Pixai v0.9 backend via imgutils (ONNX, auto-downloaded).

Pixai is an EVA02-based danbooru tagger ("eva02 but broader"). imgutils serves
it from its own ONNX mirror, so no HF token is needed despite the source repo
being gated. Unlike WD14 it produces general + character tags only (no rating),
and takes per-category thresholds via a dict keyed by category name.
"""

from PIL import Image

from .base import TagResult, Tagger


class PixaiTagger(Tagger):
    def load(self) -> None:
        if self._loaded:
            return
        # Warm the import so the first request isn't paying the download cost.
        from imgutils.tagging import get_pixai_tags  # noqa: F401

        self._loaded = True

    def _tag_one(
        self, image: Image.Image, general_threshold: float, character_threshold: float
    ) -> TagResult:
        from imgutils.tagging import get_pixai_tags

        general, character = get_pixai_tags(
            image,
            model_name=self.spec.model_name or "v0.9",
            thresholds={"general": general_threshold, "character": character_threshold},
            fmt=("general", "character"),
        )
        return TagResult(
            general={k: float(v) for k, v in general.items()},
            character={k: float(v) for k, v in character.items()},
        )
