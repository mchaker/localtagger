"""FastAPI dependencies for accessing shared app state.

These are ``async def`` on purpose: FastAPI runs plain ``def`` dependencies on
the worker thread pool, so when every worker is busy (or stuck) in inference,
even ``GET /models`` would queue behind them and hang. Reading app state never
blocks, so it stays on the event loop.
"""

from fastapi import Request

from app.catalog import Catalog
from app.manager import ModelManager


async def get_manager(request: Request) -> ModelManager:
    return request.app.state.manager


async def get_catalog(request: Request) -> Catalog:
    return request.app.state.catalog
