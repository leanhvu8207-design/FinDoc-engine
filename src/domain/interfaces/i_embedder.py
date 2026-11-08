"""Interface contract for dense vector embedding models (Phase 1 - Task 1.2)."""

from abc import ABC, abstractmethod


class IEmbeddingModel(ABC):
    """Port interface for dense text and table embedding adapters."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the output embedding vectors."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of document chunk texts into normalized dense vectors.

        Args:
            texts: Non-empty list of document chunk strings to embed.

        Returns:
            2D list of floats with shape (len(texts), dimension).

        Raises:
            EmbeddingError: If vector generation fails.
        """

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Encode a single search query string into a normalized dense vector.

        Args:
            text: Search query string.

        Returns:
            1D list of floats with length equal to `dimension`.

        Raises:
            EmbeddingError: If vector generation fails.
        """
