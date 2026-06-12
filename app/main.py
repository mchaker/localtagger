"""FastAPI application factory and entrypoint."""

from fastapi import FastAPI

from app import __version__
from app.catalog import load_catalog
from app.config import settings
from app.manager import ModelManager
from app.routers import catalog as catalog_router
from app.routers import interrogate as interrogate_router
from app.routers import kaloscope as kaloscope_router


def create_app() -> FastAPI:
    # Make HF hosting/cache/token settings visible to imgutils, timm and hf_hub.
    settings.apply_hf_env()

    catalog = load_catalog(settings.catalog_path, settings.enabled_models)
    manager = ModelManager(catalog)

    app = FastAPI(title="LAN Image Interrogator", version=__version__)
    app.state.settings = settings
    app.state.catalog = catalog
    app.state.manager = manager

    app.include_router(interrogate_router.router)
    app.include_router(catalog_router.router)
    app.include_router(kaloscope_router.router)

    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    print(
        "localtagger ready. Enabled models: "
        + ", ".join(spec.id for spec in catalog.specs)
    )
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
