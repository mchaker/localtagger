"""FastAPI dependencies for accessing shared app state."""

from fastapi import Request

from app.catalog import Catalog
from app.manager import ModelManager


def get_manager(request: Request) -> ModelManager:
    return request.app.state.manager


def get_catalog(request: Request) -> Catalog:
    return request.app.state.catalog
