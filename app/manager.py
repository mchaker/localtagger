"""ModelManager: lazy construction/loading of taggers keyed by model id.

Mirrors the original on-demand loading approach: a tagger is built and its
weights loaded on first use, then cached. Unknown/disabled ids raise an HTTP 404
so routers get a clean error.
"""

from typing import Dict

from fastapi import HTTPException

from app.backends import Tagger, build_tagger
from app.backends.kaloscope import KaloscopeClassifier
from app.catalog import Catalog


class ModelManager:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self._taggers: Dict[str, Tagger] = {}
        self.kaloscope = KaloscopeClassifier()

    def resolve(self, model_id: str) -> str:
        """Validate a model id against the enabled catalog."""
        if model_id not in self.catalog:
            available = ", ".join(s.id for s in self.catalog.specs)
            raise HTTPException(
                status_code=404,
                detail=f"Unknown or disabled model '{model_id}'. Available: {available}",
            )
        return model_id

    def get(self, model_id: str) -> Tagger:
        self.resolve(model_id)
        tagger = self._taggers.get(model_id)
        if tagger is None:
            spec = self.catalog.get(model_id)
            tagger = build_tagger(spec)
            self._taggers[model_id] = tagger
        return tagger

    def is_loaded(self, model_id: str) -> bool:
        tagger = self._taggers.get(model_id)
        return bool(tagger and tagger.loaded)

    def unload(self, model_id: str) -> None:
        tagger = self._taggers.get(model_id)
        if tagger is not None:
            tagger.unload()
