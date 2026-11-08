"""SearchQuery and QueryFilter entities (Phase 1 - Task 1.1)."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.domain.entities.document import ChunkType, ReportType


class QueryFilter(BaseModel):
    """Structured metadata filter criteria for vector and sparse search."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str | None = Field(
        default=None,
        min_length=2,
        max_length=15,
        description="Filter by stock ticker symbol (e.g., 'FPT')",
    )
    year: int | None = Field(
        default=None,
        ge=1990,
        le=2100,
        description="Filter by fiscal year",
    )
    quarter: int | None = Field(
        default=None,
        ge=1,
        le=4,
        description="Filter by fiscal quarter (1-4)",
    )
    report_type: ReportType | None = Field(
        default=None,
        description="Filter by financial statement report type",
    )
    chunk_type: ChunkType | None = Field(
        default=None,
        description="Filter specifically for TEXT or TABLE chunks",
    )

    @field_validator("ticker")
    @classmethod
    def normalize_optional_ticker(cls, value: str | None) -> str | None:
        """Normalize ticker filter to uppercase if provided."""
        if value is None:
            return None
        cleaned = value.strip().upper()
        if not cleaned.isalnum():
            raise ValueError(f"Ticker filter must be alphanumeric, got '{value}'")
        return cleaned

    @property
    def is_empty(self) -> bool:
        """Return True if no metadata filter condition is specified."""
        return (
            self.ticker is None
            and self.year is None
            and self.quarter is None
            and self.report_type is None
            and self.chunk_type is None
        )


class SearchQuery(BaseModel):
    """Represents a user retrieval or question-answering request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_text: str = Field(
        ...,
        min_length=1,
        description="Natural language question or financial search keywords",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of final top chunks to retrieve",
    )
    candidate_pool_size: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Number of candidates retrieved from each index prior to RRF & reranking",
    )
    query_filter: QueryFilter | None = Field(
        default=None,
        description="Optional metadata filter constraints",
    )
    enable_reranking: bool = Field(
        default=True,
        description="Whether to run Cross-Encoder reranking on RRF candidates",
    )

    @field_validator("query_text")
    @classmethod
    def validate_non_blank_query(cls, value: str) -> str:
        """Ensure query string is not solely whitespace."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query_text cannot be empty or whitespace only")
        return cleaned
