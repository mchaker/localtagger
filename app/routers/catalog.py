"""Model discovery endpoint.

Exposes the enabled model catalog so the frontend can populate its model picker
without a redeploy. Strings are English keys/labels; the frontend owns i18n,
translating by the stable ``id``.
"""

from fastapi import APIRouter, Depends

from app.catalog import Catalog
from app.deps import get_catalog, get_manager
from app.manager import ModelManager
from app.schemas import ModelInfo, ModelsResponse

router = APIRouter()


@router.get("/models", response_model=ModelsResponse)
async def list_models(
    catalog: Catalog = Depends(get_catalog),
    manager: ModelManager = Depends(get_manager),
):
    models = [
        ModelInfo(
            id=spec.id,
            label=spec.label,
            description=spec.description,
            family=spec.family,
            recommended=spec.recommended,
            gated=spec.gated,
            loaded=manager.is_loaded(spec.id),
            default_threshold=spec.default_threshold,
            default_character_threshold=spec.default_character_threshold,
        )
        for spec in catalog.specs
    ]
    return ModelsResponse(models=models)
