"""Domain-level custom exceptions for FinDoc-Engine (Phase 1 - Task 1.1 & 1.2)."""

from typing import Any


class FinDocError(Exception):
    """Base exception for all domain and application errors in FinDoc-Engine."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message: str = message
        self.details: dict[str, Any] = details if details is not None else {}

    def __str__(self) -> str:
        if not self.details:
            return self.message
        return f"{self.message} | details={self.details}"


class DomainValidationError(FinDocError):
    """Raised when a domain entity or value object violates business invariants."""


class ParsingError(FinDocError):
    """Raised when a PDF document cannot be read or parsed into structural blocks."""


class TableExtractionError(ParsingError):
    """Raised when a financial table structure or header matrix cannot be extracted."""


class ChunkingError(FinDocError):
    """Raised when context-aware financial chunking fails."""


class EmbeddingError(FinDocError):
    """Raised when dense vector generation fails."""


class RepositoryError(FinDocError):
    """Base exception for persistence and storage operations."""


class VectorStoreError(RepositoryError):
    """Raised when Qdrant vector database operations fail."""


class SparseIndexError(RepositoryError):
    """Raised when BM25 sparse keyword index operations fail."""


class RerankingError(FinDocError):
    """Raised when Cross-Encoder reranking fails."""


class HallucinationError(FinDocError):
    """Raised when deterministic numeric validation detects unverified numbers in an answer."""

    def __init__(
        self,
        message: str,
        unverified_numbers: list[float] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged_details: dict[str, Any] = dict(details) if details else {}
        if unverified_numbers is not None:
            merged_details["unverified_numbers"] = unverified_numbers
        super().__init__(message=message, details=merged_details)
        self.unverified_numbers: list[float] = (
            list(unverified_numbers) if unverified_numbers is not None else []
        )
