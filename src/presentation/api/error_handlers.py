"""Global FastAPI exception handlers mapping domain exceptions to HTTP responses (Task 5.1)."""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from src.domain.exceptions import (
    ChunkingError,
    DomainValidationError,
    EmbeddingError,
    FinDocError,
    HallucinationError,
    ParsingError,
    RepositoryError,
    RerankingError,
)


def register_error_handlers(app: FastAPI) -> None:
    """Register domain exception handlers on a FastAPI application instance."""

    @app.exception_handler(DomainValidationError)
    async def handle_domain_validation_error(
        _request: Request, exc: DomainValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error_type": "DomainValidationError",
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(ParsingError)
    async def handle_parsing_error(_request: Request, exc: ParsingError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error_type": type(exc).__name__,
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(ChunkingError)
    async def handle_chunking_error(_request: Request, exc: ChunkingError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error_type": "ChunkingError",
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(HallucinationError)
    async def handle_hallucination_error(
        _request: Request, exc: HallucinationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error_type": "HallucinationError",
                "message": exc.message,
                "hallucinated_numbers": exc.unverified_numbers,
                "details": exc.details,
            },
        )

    @app.exception_handler(EmbeddingError)
    @app.exception_handler(RepositoryError)
    @app.exception_handler(RerankingError)
    async def handle_infrastructure_error(_request: Request, exc: FinDocError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error_type": type(exc).__name__,
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(FinDocError)
    async def handle_generic_findoc_error(_request: Request, exc: FinDocError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error_type": type(exc).__name__,
                "message": exc.message,
                "details": exc.details,
            },
        )
