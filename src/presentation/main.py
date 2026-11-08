"""FastAPI application entrypoint and lifespan manager (Phase 5 - Task 5.1)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.infrastructure.config import get_settings
from src.presentation.api.dependencies import ServiceContainer
from src.presentation.api.error_handlers import register_error_handlers
from src.presentation.api.v1.ingest_routes import router as ingest_router
from src.presentation.api.v1.search_routes import router as search_router


def create_app(container: ServiceContainer | None = None) -> FastAPI:
    """Application factory creating and configuring the FastAPI instance."""
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI) -> AsyncIterator[None]:
        active_container = container or getattr(app_instance.state, "container", None)
        if active_container is None:
            active_container = ServiceContainer(settings=settings)
        app_instance.state.container = active_container
        try:
            yield
        finally:
            active_container.close()

    app_instance = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description=(
            "Multi-Modal Financial Parsing, Deterministic Hybrid Retrieval "
            "& Numeric Fact-Checking System"
        ),
        lifespan=lifespan,
    )

    if container is not None:
        app_instance.state.container = container

    app_instance.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_error_handlers(app_instance)
    app_instance.include_router(ingest_router, prefix="/api/v1")
    app_instance.include_router(search_router, prefix="/api/v1")

    @app_instance.get("/health", tags=["Health"])
    def health_check() -> dict[str, str]:
        """Return service readiness and version information."""
        return {
            "status": "ok",
            "app": settings.app_name,
            "version": "0.1.0",
        }

    return app_instance


app = create_app()
