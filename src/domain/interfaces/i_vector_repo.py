"""Interface contract for vector database repositories (Phase 1 - Task 1.2)."""

from abc import ABC, abstractmethod

from src.domain.entities.document import Chunk
from src.domain.entities.query import QueryFilter
from src.domain.entities.response import SearchResult


class IVectorRepository(ABC):
    """Port interface for dense vector storage and similarity search (e.g., Qdrant)."""

    @abstractmethod
    def upsert(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]] | None = None,
    ) -> int:
        """Insert or update a batch of document chunks and their dense vectors.

        Args:
            chunks: List of domain Chunk entities to store.
            vectors: Corresponding dense embedding vectors for each chunk.

        Returns:
            Number of chunks successfully upserted.

        Raises:
            VectorStoreError: If the database upsert operation fails.
        """

    @abstractmethod
    def search(
        self,
        vector: list[float],
        top_k: int = 10,
        filter: QueryFilter | None = None,
    ) -> list[SearchResult]:
        """Perform dense cosine similarity search with optional metadata filtering.

        Args:
            vector: Query dense embedding vector.
            top_k: Maximum number of top results to return.
            filter: Optional QueryFilter for ticker, year, quarter, or report_type.

        Returns:
            Ranked list of SearchResult entities ordered by descending similarity score.

        Raises:
            VectorStoreError: If the vector query fails.
        """
