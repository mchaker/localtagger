"""Image interrogation endpoints.

Unified ``POST /interrogate?model=<id>`` plus the legacy per-model alias routes
(kept for backward compatibility until the frontend migrates) and a
``GET /interrogate`` that takes image URLs.
"""

import os
import tempfile
import zipfile
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from starlette.background import BackgroundTask

from app.deps import get_manager
from app.formatting import format_tags
from app.image_utils import download_image, load_image_from_bytes
from app.manager import ModelManager

router = APIRouter()

# Legacy route suffix -> preferred model id. Resolved against the enabled
# catalog at request time, falling back to the default model when disabled.
ALIASES = {
    "eva": "wd-eva02-large-v3",
    "pixai": "pixai-v0.9",
    "camie": "camie-v2",
    "taggerine": "animetimm-caformer-b36",
}

BATCH_SIZE = 8


def _resolve_model(manager: ModelManager, model: Optional[str]) -> str:
    """Pick a concrete, enabled model id from a request's model/alias."""
    if model and model in manager.catalog:
        return model
    if model in ALIASES and ALIASES[model] in manager.catalog:
        return ALIASES[model]
    # Unknown but explicitly provided -> let resolve() raise a 404.
    if model:
        manager.resolve(model)
    return manager.catalog.default_model_id()


def _tag_images(
    manager: ModelManager,
    model_id: str,
    images: List[Image.Image],
    *,
    threshold: float,
    character_threshold: Optional[float],
    use_spaces: bool,
    use_escape: bool,
    include_ranks: bool,
    score_descend: bool,
    trigger_word: str,
    random_order: bool,
) -> List[dict]:
    tagger = manager.get(model_id)
    results = tagger.tag(images, general_threshold=threshold, character_threshold=character_threshold)

    formatted = []
    for res in results:
        sorted_tags, tag_string = format_tags(
            res.merged(),
            use_spaces=use_spaces,
            use_escape=use_escape,
            include_ranks=include_ranks,
            score_descend=score_descend,
            trigger_word=trigger_word,
            random_order=random_order,
        )
        formatted.append(
            {
                "tags": sorted_tags,
                "tag_string": tag_string,
                "rating": res.rating,
                "character": res.character,
                "model": model_id,
            }
        )
    return formatted


@router.post("/interrogate/{alias}")
@router.post("/interrogate")
async def interrogate_post(
    alias: Optional[str] = None,
    file: List[UploadFile] = File(...),
    model: Optional[str] = Query(None, description="Model id from /models"),
    threshold: float = 0.35,
    character_threshold: Optional[float] = Query(None),
    use_spaces: bool = False,
    use_escape: bool = True,
    include_ranks: bool = False,
    score_descend: bool = True,
    output_format: str = Query("json", enum=["json", "zip"]),
    trigger_word: str = Query("", description="Optional trigger word to prepend to tags"),
    random_order: bool = Query(False, description="Randomize tag order (useful for training)"),
    manager: ModelManager = Depends(get_manager),
):
    model_id = _resolve_model(manager, model or alias)
    print(
        f"POST /interrogate: model={model_id} files={len(file)} "
        f"threshold={threshold} format={output_format}"
    )

    if output_format == "zip":
        return await _zip_response(
            manager,
            model_id,
            file,
            threshold=threshold,
            character_threshold=character_threshold,
            use_spaces=use_spaces,
            use_escape=use_escape,
            include_ranks=include_ranks,
            score_descend=score_descend,
            trigger_word=trigger_word,
            random_order=random_order,
        )

    all_results = []
    for i in range(0, len(file), BATCH_SIZE):
        batch_files = file[i : i + BATCH_SIZE]
        batch_images = [load_image_from_bytes(await f.read()) for f in batch_files]
        all_results.extend(
            _tag_images(
                manager,
                model_id,
                batch_images,
                threshold=threshold,
                character_threshold=character_threshold,
                use_spaces=use_spaces,
                use_escape=use_escape,
                include_ranks=include_ranks,
                score_descend=score_descend,
                trigger_word=trigger_word,
                random_order=random_order,
            )
        )
        del batch_images
    return all_results


async def _zip_response(manager, model_id, file, **fmt):
    """Stream a ZIP of original images + .txt tag files."""
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            for i in range(0, len(file), BATCH_SIZE):
                batch_files = file[i : i + BATCH_SIZE]
                batch_bytes, batch_images, batch_names = [], [], []
                for f in batch_files:
                    data = await f.read()
                    batch_bytes.append(data)
                    batch_images.append(load_image_from_bytes(data))
                    batch_names.append(f.filename)

                results = _tag_images(manager, model_id, batch_images, **fmt)
                for j, res in enumerate(results):
                    zf.writestr(batch_names[j], batch_bytes[j])
                    base_name = batch_names[j].rsplit(".", 1)[0]
                    zf.writestr(f"{base_name}.txt", res["tag_string"])
                del batch_images, batch_bytes, results
        tmp.close()
        return FileResponse(
            tmp.name,
            media_type="application/zip",
            filename="dataset.zip",
            background=BackgroundTask(os.remove, tmp.name),
        )
    except Exception:
        tmp.close()
        if os.path.exists(tmp.name):
            os.remove(tmp.name)
        raise


@router.get("/interrogate")
async def interrogate_get(
    url: List[str] = Query(...),
    model: Optional[str] = Query(None, description="Model id from /models"),
    threshold: float = 0.35,
    character_threshold: Optional[float] = Query(None),
    use_spaces: bool = False,
    use_escape: bool = True,
    include_ranks: bool = False,
    score_descend: bool = True,
    trigger_word: str = Query("", description="Optional trigger word to prepend to tags"),
    random_order: bool = Query(False, description="Randomize tag order (useful for training)"),
    manager: ModelManager = Depends(get_manager),
):
    model_id = _resolve_model(manager, model)
    images = [download_image(u) for u in url]
    return _tag_images(
        manager,
        model_id,
        images,
        threshold=threshold,
        character_threshold=character_threshold,
        use_spaces=use_spaces,
        use_escape=use_escape,
        include_ranks=include_ranks,
        score_descend=score_descend,
        trigger_word=trigger_word,
        random_order=random_order,
    )
