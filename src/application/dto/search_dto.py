"""Search and Answer Data Transfer Objects (Phase 4 - Task 4.4)."""

from pydantic import BaseModel, ConfigDict, Field

from src.domain.entities.document import ChunkType, ReportType
from src.domain.entities.query import QueryFilter, SearchQuery
from src.domain.entities.response import Citation, SearchResult, ValidationReport


class SearchRequestDTO(BaseModel):
    """Input DTO for executing a hybrid dense + sparse + RRF + rerank search."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_text: str = Field(..., min_length=1, description="User search query or question")
    top_k: int = Field(default=5, ge=1, le=100, description="Final number of chunks to return")
    candidate_pool_size: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Number of candidates retrieved from each index before RRF & reranking",
    )
    ticker: str | None = Field(default=None, description="Optional stock ticker filter")
    year: int | None = Field(default=None, ge=1990, le=2100, description="Optional fiscal year")
    quarter: int | None = Field(default=None, ge=1, le=4, description="Optional fiscal quarter")
    report_type: ReportType | None = Field(default=None, description="Optional report type filter")
    chunk_type: ChunkType | None = Field(default=None, description="Optional TEXT/TABLE filter")
    enable_reranking: bool = Field(
        default=True,
        description="Whether to run Cross-Encoder reranking on RRF candidates",
    )

    def to_domain_query(self) -> SearchQuery:
        """Convert DTO into a validated domain SearchQuery entity."""
        q_filter = QueryFilter(
            ticker=self.ticker,
            year=self.year,
            quarter=self.quarter,
            report_type=self.report_type,
            chunk_type=self.chunk_type,
        )
        return SearchQuery(
            query_text=self.query_text,
            top_k=self.top_k,
            candidate_pool_size=self.candidate_pool_size,
            query_filter=None if q_filter.is_empty else q_filter,
            enable_reranking=self.enable_reranking,
        )


class SearchResultDTO(BaseModel):
    """Output DTO wrapping ranked hybrid search results and their citations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(..., description="Executed search query")
    results: list[SearchResult] = Field(
        default_factory=list,
        description="Ranked SearchResult list after RRF and optional Cross-Encoder reranking",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Provenance citations extracted from top results",
    )


class AnswerRequestDTO(BaseModel):
    """Input DTO for answering a financial question with deterministic numeric verification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_text: str = Field(..., min_length=1, description="Financial question to answer")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of context chunks to retrieve")
    candidate_pool_size: int = Field(default=20, ge=1, le=200)
    ticker: str | None = Field(default=None)
    year: int | None = Field(default=None, ge=1990, le=2100)
    quarter: int | None = Field(default=None, ge=1, le=4)
    report_type: ReportType | None = Field(default=None)
    enable_reranking: bool = Field(default=True)
    strict_guardrail: bool = Field(
        default=False,
        description="If True, raise HallucinationError when numeric validation fails",
    )

    def to_search_query(self) -> SearchQuery:
        """Convert AnswerRequestDTO into a domain SearchQuery."""
        q_filter = QueryFilter(
            ticker=self.ticker,
            year=self.year,
            quarter=self.quarter,
            report_type=self.report_type,
        )
        return SearchQuery(
            query_text=self.query_text,
            top_k=self.top_k,
            candidate_pool_size=self.candidate_pool_size,
            query_filter=None if q_filter.is_empty else q_filter,
            enable_reranking=self.enable_reranking,
        )


class AnswerResponseDTO(BaseModel):
    """Output DTO containing the synthesized answer, citations, and numeric validation report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Synthesized answer based strictly on retrieved context")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Source page and table citations supporting the answer",
    )
    results: list[SearchResult] = Field(
        default_factory=list,
        description="Retrieved context chunks used to generate the answer",
    )
    validation_report: ValidationReport = Field(
        ...,
        description="Deterministic numeric guardrail audit report",
    )
