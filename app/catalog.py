"""Model catalog: parses models.yaml into ModelSpec objects.

The catalog is the single source of truth for which models exist and how to run
them. It is also what the ``GET /models`` endpoint exposes to the frontend, so
new models can be surfaced without a frontend redeploy.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

VALID_FAMILIES = {"wd14", "pixai", "camie", "animetimm", "wd14_st"}


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    description: str
    # Display heading the frontend groups this model under (e.g. "WD Tagger v3").
    # Falls back to ``label`` when omitted so the picker is never headingless.
    group: str
    family: str
    repo: str
    model_name: str
    gated: bool
    recommended: bool
    default: bool
    default_threshold: float
    default_character_threshold: float


class Catalog:
    """Holds the enabled models keyed by id."""

    def __init__(self, specs: List[ModelSpec]):
        self._specs: Dict[str, ModelSpec] = {spec.id: spec for spec in specs}

    @property
    def specs(self) -> List[ModelSpec]:
        return list(self._specs.values())

    def get(self, model_id: str) -> Optional[ModelSpec]:
        return self._specs.get(model_id)

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._specs

    def default_model_id(self, family: Optional[str] = None) -> Optional[str]:
        """First enabled model, optionally restricted to a family."""
        for spec in self._specs.values():
            if family is None or spec.family == family:
                return spec.id
        return None


def _parse_spec(raw: dict) -> ModelSpec:
    family = raw["family"]
    if family not in VALID_FAMILIES:
        raise ValueError(
            f"Model '{raw.get('id')}' has unknown family '{family}' "
            f"(expected one of {sorted(VALID_FAMILIES)})"
        )
    return ModelSpec(
        id=raw["id"],
        label=raw.get("label", raw["id"]),
        description=raw.get("description", ""),
        group=raw.get("group") or raw.get("label", raw["id"]),
        family=family,
        repo=raw["repo"],
        model_name=raw.get("model_name", "") or "",
        gated=bool(raw.get("gated", False)),
        recommended=bool(raw.get("recommended", False)),
        default=bool(raw.get("default", False)),
        default_threshold=float(raw.get("default_threshold", 0.35)),
        default_character_threshold=float(raw.get("default_character_threshold", 0.85)),
    )


def load_catalog(path: Path, enabled_ids: Optional[List[str]] = None) -> Catalog:
    """Load models.yaml and filter to the enabled set.

    If ``enabled_ids`` is empty/None, models with ``default: true`` are enabled.
    """
    data = yaml.safe_load(Path(path).read_text())
    all_specs = [_parse_spec(raw) for raw in data.get("models", [])]

    if enabled_ids:
        wanted = set(enabled_ids)
        unknown = wanted - {s.id for s in all_specs}
        if unknown:
            raise ValueError(f"ENABLED_MODELS references unknown model ids: {sorted(unknown)}")
        selected = [s for s in all_specs if s.id in wanted]
    else:
        selected = [s for s in all_specs if s.default]

    if not selected:
        raise ValueError("No models enabled — check models.yaml and ENABLED_MODELS")
    return Catalog(selected)
