"""Interface contract for Cross-Encoder reranking models (Phase 1 - Task 1.2)."""

from abc import ABC, abstractmethod

from src.domain.entities.document import Chunk


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
