"""Kaloscope artist-style classification endpoint (unchanged behavior)."""

from fastapi import APIRouter, Depends, File, Query, UploadFile

from app.deps import get_manager
from app.image_utils import load_image_from_bytes
from app.manager import ModelManager
from app.schemas import KaloscopeResponse

router = APIRouter()


@router.post("/kaloscope/infer", response_model=KaloscopeResponse)
async def kaloscope_infer(
    file: UploadFile = File(...),
    top_k: int = Query(10, ge=1, le=50, description="Number of top artist matches to return"),
    manager: ModelManager = Depends(get_manager),
):
    image = load_image_from_bytes(await file.read())
    artists = manager.kaloscope.infer(image, top_k=top_k)
    return KaloscopeResponse(artists=artists, model="kaloscope-2.0")
