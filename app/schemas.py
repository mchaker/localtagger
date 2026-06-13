"""Pydantic response models."""

from typing import Dict, List

from pydantic import BaseModel


class InterrogateResult(BaseModel):
    # Flat {tag: score} map (general + character merged) for frontend compat.
    tags: Dict[str, float]
    tag_string: str
    # Richer breakdown for newer clients; safe to ignore.
    rating: Dict[str, float] = {}
    character: Dict[str, float] = {}
    model: str = ""


class ModelInfo(BaseModel):
    id: str
    label: str
    description: str
    # Display heading to group models under in the picker (e.g. "WD Tagger v3").
    group: str
    family: str
    recommended: bool
    gated: bool
    loaded: bool
    default_threshold: float
    default_character_threshold: float


class ModelsResponse(BaseModel):
    models: List[ModelInfo]


class ArtistMatch(BaseModel):
    name: str
    score: float


class KaloscopeResponse(BaseModel):
    artists: List[ArtistMatch]
    model: str = "kaloscope-2.0"
