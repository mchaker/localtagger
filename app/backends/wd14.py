"""WD14 v3 backend via imgutils (ONNX, auto-downloaded)."""

from PIL import Image

from .base import TagResult, Tagger


class WD14Tagger(Tagger):
    def load(self) -> None:
        if self._loaded:
            return
        # Warm the model so the first request isn't paying the download cost.
        from imgutils.tagging import get_wd14_tags  # noqa: F401

        self._loaded = True

    def _tag_one(
        self, image: Image.Image, general_threshold: float, character_threshold: float
    ) -> TagResult:
        from imgutils.tagging import get_wd14_tags

        rating, general, character = get_wd14_tags(
            image,
            model_name=self.spec.model_name,
            general_threshold=general_threshold,
            character_threshold=character_threshold,
            fmt=("rating", "general", "character"),
        )
        return TagResult(
            general={k: float(v) for k, v in general.items()},
            character={k: float(v) for k, v in character.items()},
            rating={k: float(v) for k, v in rating.items()},
        )
