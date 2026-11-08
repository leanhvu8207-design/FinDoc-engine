"""Interface contract for sparse BM25 keyword repositories (Phase 1 - Task 1.2)."""

from abc import ABC, abstractmethod

from src.domain.entities.document import Chunk
from src.domain.entities.query import QueryFilter
from src.domain.entities.response import SearchResult


class ISparseRepository(ABC):
    """Port interface for lexical/sparse keyword indexing and BM25 search."""

    @abstractmethod
    def index(self, chunks: list[Chunk]) -> int:
        """Build or update the inverted index from a collection of document chunks.

        Args:
            chunks: List of domain Chunk entities to index.

        Returns:
            Total number of chunks indexed.

        Raises:
            SparseIndexError: If indexing or tokenization fails.
        """

    @abstractmethod
    def search_bm25(
        self,
        query: str,
        top_k: int = 10,
        filter: QueryFilter | None = None,
    ) -> list[SearchResult]:
        """Search the sparse index using BM25 lexical matching.

        Args:
            query: Raw search query string containing financial terms or metrics.
            top_k: Maximum number of top results to return.
            filter: Optional QueryFilter to restrict candidate chunks by metadata.

        Returns:
            Ranked list of SearchResult entities ordered by descending BM25 score.

        Raises:
            SparseIndexError: If lexical search fails.
        """
