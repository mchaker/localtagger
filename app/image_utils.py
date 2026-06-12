"""Image loading helpers shared across backends and routers."""

import io

import requests
from fastapi import HTTPException
from PIL import Image

_DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
}


def download_image(url: str, timeout: int = 10) -> Image.Image:
    """Fetch an image from a URL and return it as an RGB PIL image."""
    try:
        response = requests.get(url, headers=_DOWNLOAD_HEADERS, timeout=timeout)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {e}")


def load_image_from_bytes(image_data: bytes) -> Image.Image:
    """Decode raw image bytes into an RGB PIL image."""
    try:
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
