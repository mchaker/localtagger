"""Pydantic response models."""

from typing import Dict, List

from pydantic import BaseModel


class InterrogateResult(BaseModel):
    # Flat {tag: score} map (all non-rating categories) for frontend compat.
    tags: Dict[str, float]
    tag_string: str
    # Richer breakdown for newer clients; safe to ignore.
    rating: Dict[str, float] = {}
    character: Dict[str, float] = {}
    copyright: Dict[str, float] = {}
    artist: Dict[str, float] = {}
    meta: Dict[str, float] = {}
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
    default_thresholds: Dict[str, float] = {}


class ModelsResponse(BaseModel):
    models: List[ModelInfo]


class ArtistMatch(BaseModel):
    name: str
    score: float


class KaloscopeResponse(BaseModel):
    artists: List[ArtistMatch]
    model: str = "kaloscope-2.0"
