"""Dependency Injection container wiring for FastAPI (Phase 5 - Task 5.1)."""

from __future__ import annotations

from typing import cast

from fastapi import Request

from src.application.services.financial_chunker import FinancialChunker
from src.application.services.fusion_service import FusionService
from src.application.use_cases.answer_with_verify import (
    AnswerWithVerifyUseCase,
    GeminiLLMClient,
    LLMClientProtocol,
)
from src.application.use_cases.hybrid_search import HybridSearchUseCase
from src.application.use_cases.ingest_document import IngestDocumentUseCase
from src.domain.interfaces.i_embedder import IEmbeddingModel
from src.domain.interfaces.i_parser import IDocumentParser
from src.domain.interfaces.i_reranker import IReranker
from src.domain.interfaces.i_sparse_repo import ISparseRepository
from src.domain.interfaces.i_validator import INumericValidator
from src.domain.interfaces.i_vector_repo import IVectorRepository
from src.infrastructure.config import Settings, get_settings
from src.infrastructure.ml_models.embedding_adapter import EmbeddingAdapter
from src.infrastructure.ml_models.reranker_adapter import RerankerAdapter
from src.infrastructure.parsers.pdfplumber_adapter import PdfPlumberAdapter
from src.infrastructure.persistence.bm25_repo import BM25SparseRepository
from src.infrastructure.persistence.qdrant_repo import QdrantVectorRepository
from src.infrastructure.validators.numeric_guardrail import NumericGuardrail


class ServiceContainer:
    """IoC Container managing singleton lifecycles of repositories, ML adapters, and use cases."""

    def __init__(
        self,
        settings: Settings | None = None,
        parser: IDocumentParser | None = None,
        chunker: FinancialChunker | None = None,
        embedder: IEmbeddingModel | None = None,
        vector_repo: IVectorRepository | None = None,
        sparse_repo: ISparseRepository | None = None,
        fusion_service: FusionService | None = None,
        reranker: IReranker | None = None,
        validator: INumericValidator | None = None,
        llm_client: LLMClientProtocol | None = None,
    ) -> None:
        self.settings: Settings = settings or get_settings()
        self.parser: IDocumentParser = parser or PdfPlumberAdapter()
        self.chunker: FinancialChunker = chunker or FinancialChunker(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self.embedder: IEmbeddingModel = embedder or EmbeddingAdapter(settings=self.settings)
        self.vector_repo: IVectorRepository = vector_repo or QdrantVectorRepository(
            settings=self.settings
        )
        self.sparse_repo: ISparseRepository = sparse_repo or BM25SparseRepository(
            settings=self.settings
        )
        self.fusion_service: FusionService = fusion_service or FusionService(
            k=self.settings.rrf_k
        )
        self.reranker: IReranker = reranker or RerankerAdapter(settings=self.settings)
        self.validator: INumericValidator = validator or NumericGuardrail(
            settings=self.settings
        )
        self.llm_client: LLMClientProtocol = llm_client or GeminiLLMClient(
            api_key=self.settings.gemini_api_key,
            model_name=self.settings.gemini_model_name,
        )

        self.ingest_use_case = IngestDocumentUseCase(
            parser=self.parser,
            chunker=self.chunker,
            embedder=self.embedder,
            vector_repo=self.vector_repo,
            sparse_repo=self.sparse_repo,
        )
        self.hybrid_search_use_case = HybridSearchUseCase(
            embedder=self.embedder,
            vector_repo=self.vector_repo,
            sparse_repo=self.sparse_repo,
            fusion_service=self.fusion_service,
            reranker=self.reranker,
        )
        self.answer_use_case = AnswerWithVerifyUseCase(
            hybrid_search=self.hybrid_search_use_case,
            validator=self.validator,
            llm_client=self.llm_client,
        )

    def close(self) -> None:
        """Release underlying database/file locks held by singleton repositories."""
        if isinstance(self.vector_repo, QdrantVectorRepository):
            self.vector_repo.close()


def get_container(request: Request) -> ServiceContainer:
    """Retrieve the active ServiceContainer attached to `request.app.state`."""
    container = getattr(request.app.state, "container", None)
    if container is None:
        container = ServiceContainer()
        request.app.state.container = container
    return cast(ServiceContainer, container)


def provide_ingest_use_case(request: Request) -> IngestDocumentUseCase:
    """FastAPI dependency returning IngestDocumentUseCase."""
    return get_container(request).ingest_use_case


def provide_hybrid_search_use_case(request: Request) -> HybridSearchUseCase:
    """FastAPI dependency returning HybridSearchUseCase."""
    return get_container(request).hybrid_search_use_case


def provide_answer_use_case(request: Request) -> AnswerWithVerifyUseCase:
    """FastAPI dependency returning AnswerWithVerifyUseCase."""
    return get_container(request).answer_use_case
