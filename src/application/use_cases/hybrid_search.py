"""Hybrid search pipeline: Dense + Sparse -> RRF -> Rerank (Phase 4 - Task 4.4)."""

from __future__ import annotations

from src.application.dto.search_dto import SearchRequestDTO, SearchResultDTO
from src.application.services.fusion_service import FusionService
from src.domain.entities.query import SearchQuery
from src.domain.entities.response import Citation, SearchResult
from src.domain.interfaces.i_embedder import IEmbeddingModel
from src.domain.interfaces.i_reranker import IReranker
from src.domain.interfaces.i_sparse_repo import ISparseRepository
from src.domain.interfaces.i_vector_repo import IVectorRepository


class HybridSearchUseCase:
    """Orchestrates Dense + Sparse retrieval, RRF fusion, and Cross-Encoder reranking."""

    def __init__(
        self,
        embedder: IEmbeddingModel,
        vector_repo: IVectorRepository,
        sparse_repo: ISparseRepository,
        fusion_service: FusionService | None = None,
        reranker: IReranker | None = None,
    ) -> None:
        self._embedder = embedder
        self._vector_repo = vector_repo
        self._sparse_repo = sparse_repo
        self._fusion_service = fusion_service or FusionService(k=60)
        self._reranker = reranker

    def execute(self, query: SearchQuery) -> list[SearchResult]:
        """Execute hybrid retrieval, RRF fusion, and optional semantic reranking."""
        # Step 1: Dense vector search
        query_vector = self._embedder.embed_query(query.query_text)
        dense_results = self._vector_repo.search(
            query_vector=query_vector,
            top_k=query.candidate_pool_size,
            filter=query.query_filter,
        )

        # Step 2: Sparse BM25 lexical search
        sparse_results = self._sparse_repo.search_bm25(
            query=query.query_text,
            top_k=query.candidate_pool_size,
            filter=query.query_filter,
        )

        # Step 3: Reciprocal Rank Fusion (RRF)
        fused_results = self._fusion_service.fuse(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=query.candidate_pool_size,
        )
        if not fused_results:
            return []

        # Step 4: Cross-Encoder Reranking (if enabled and reranker is configured)
        if query.enable_reranking and self._reranker is not None:
            return self._reranker.rerank_results(
                query=query.query_text,
                candidates=fused_results,
                top_k=query.top_k,
            )

        return fused_results[: query.top_k]

    def execute_dto(self, request: SearchRequestDTO) -> SearchResultDTO:
        """Execute hybrid search from a SearchRequestDTO and return a SearchResultDTO."""
        domain_query = request.to_domain_query()
        results = self.execute(domain_query)
        citations = [
            res.citation if res.citation is not None else Citation.from_chunk(res.chunk)
            for res in results
        ]
        return SearchResultDTO(
            query=domain_query.query_text,
            results=results,
            citations=citations,
        )
