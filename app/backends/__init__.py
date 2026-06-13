"""Tagger backends.

Each backend implements :class:`~app.backends.base.Tagger` and is constructed
from a :class:`~app.catalog.ModelSpec`. The factory below maps a spec's
``family`` to the concrete backend class.
"""

from app.catalog import ModelSpec

from .animetimm import AnimetimmTagger
from .base import Tagger
from .camie import CamieTagger
from .pixai import PixaiTagger
from .wd14 import WD14Tagger

_FAMILY_BACKENDS = {
    "wd14": WD14Tagger,
    "pixai": PixaiTagger,
    "camie": CamieTagger,
    "animetimm": AnimetimmTagger,
}


def build_tagger(spec: ModelSpec) -> Tagger:
    try:
        backend_cls = _FAMILY_BACKENDS[spec.family]
    except KeyError:
        raise ValueError(f"No backend for family '{spec.family}'")
    return backend_cls(spec)


__all__ = ["Tagger", "build_tagger"]
