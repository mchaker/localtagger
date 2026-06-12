"""Common tagger interface.

A :class:`Tagger` wraps one model. ``tag()`` returns one ``TagResult`` per input
image, each holding the rating / general / character tag dicts. Routers merge
general+character into the flat ``tags`` map and run :func:`app.formatting.format_tags`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from PIL import Image

from app.catalog import ModelSpec


@dataclass
class TagResult:
    general: Dict[str, float] = field(default_factory=dict)
    character: Dict[str, float] = field(default_factory=dict)
    rating: Dict[str, float] = field(default_factory=dict)

    def merged(self) -> Dict[str, float]:
        """General + character tags as a single map (backward-compat shape)."""
        merged = dict(self.general)
        merged.update(self.character)
        return merged


class Tagger(ABC):
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self._loaded = False

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        """Download weights / initialize the model. Idempotent."""

    @abstractmethod
    def _tag_one(
        self, image: Image.Image, general_threshold: float, character_threshold: float
    ) -> TagResult:
        ...

    def tag(
        self,
        images: List[Image.Image],
        general_threshold: Optional[float] = None,
        character_threshold: Optional[float] = None,
    ) -> List[TagResult]:
        if not self._loaded:
            self.load()
        gt = self.spec.default_threshold if general_threshold is None else general_threshold
        ct = (
            self.spec.default_character_threshold
            if character_threshold is None
            else character_threshold
        )
        return [self._tag_one(img, gt, ct) for img in images]

    def unload(self) -> None:
        """Release model resources. Override where it matters (e.g. torch/VRAM)."""
        self._loaded = False
