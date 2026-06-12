"""Camie Tagger v2 backend via imgutils (ONNX, auto-downloaded).

Artist guessing is intentionally omitted (issue #2, note3: Camie is bad at it).
"""

from PIL import Image

from .base import TagResult, Tagger


class CamieTagger(Tagger):
    def load(self) -> None:
        if self._loaded:
            return
        from imgutils.tagging import get_camie_tags  # noqa: F401

        self._loaded = True

    def _tag_one(
        self, image: Image.Image, general_threshold: float, character_threshold: float
    ) -> TagResult:
        from imgutils.tagging import get_camie_tags

        # Camie applies its own thresholds; pass the general threshold through.
        rating, general, character = get_camie_tags(
            image,
            model_name=self.spec.model_name or "initial",
            thresholds=general_threshold,
            fmt=("rating", "general", "character"),
        )
        return TagResult(
            general={k: float(v) for k, v in general.items()},
            character={k: float(v) for k, v in character.items()},
            rating={k: float(v) for k, v in rating.items()},
        )
