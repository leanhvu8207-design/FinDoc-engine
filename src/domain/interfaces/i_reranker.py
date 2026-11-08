"""Interface contract for Cross-Encoder reranking models (Phase 1 & Phase 4)."""

from abc import ABC, abstractmethod

from src.domain.entities.document import Chunk
from src.domain.entities.response import Citation, SearchResult


class IReranker(ABC):
    """Port interface for Cross-Encoder semantic reranking adapters."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[Chunk],
        top_k: int = 5,
    ) -> list[Chunk]:
        """Score (query, chunk) pairs using a Cross-Encoder and return the top_k chunks.

        Args:
            query: User search query string.
            candidates: Candidate chunks produced by Reciprocal Rank Fusion (RRF).
            top_k: Number of highest-scoring chunks to retain.

        Returns:
            Top-k Chunk entities sorted by descending Cross-Encoder relevance score.

        Raises:
            RerankingError: If Cross-Encoder inference fails.
        """

    def rerank_results(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Score SearchResult candidates and return top_k SearchResults with rerank_score.

        Default implementation delegates to `rerank()` while preserving existing scores.
        Concrete adapters may override this to attach exact raw Cross-Encoder logits/scores.
        """
        if not candidates:
            return []
        result_by_id = {res.chunk.chunk_id: res for res in candidates}
        top_chunks = self.rerank(
            query=query,
            candidates=[res.chunk for res in candidates],
            top_k=top_k,
        )
        reranked: list[SearchResult] = []
        for new_rank, chunk in enumerate(top_chunks, start=1):
            orig = result_by_id[chunk.chunk_id]
            reranked.append(
                SearchResult(
                    chunk=chunk,
                    score=orig.score,
                    rank=new_rank,
                    dense_score=orig.dense_score,
                    sparse_score=orig.sparse_score,
                    rrf_score=orig.rrf_score,
                    rerank_score=orig.rerank_score,
                    citation=orig.citation or Citation.from_chunk(chunk),
                )
            )
        return reranked
