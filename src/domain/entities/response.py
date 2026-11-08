"""SearchResult, Citation, ValidationReport, and ValidationStatus entities (Phase 1 - Task 1.1)."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from src.domain.entities.document import Chunk, ChunkType
from src.domain.entities.page import BoundingBox


class ValidationStatus(StrEnum):
    """Outcome status of deterministic numeric fact-checking."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    WARNING = "WARNING"


class Citation(BaseModel):
    """Exact source provenance pointing to the page and table/block in a PDF report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(..., min_length=1, description="Source chunk identifier")
    document_id: str = Field(..., min_length=1, description="Source document identifier")
    ticker: str = Field(..., min_length=2, description="Company stock ticker")
    year: int = Field(..., ge=1990, le=2100, description="Fiscal year")
    quarter: int | None = Field(default=None, ge=1, le=4, description="Fiscal quarter")
    page_number: int = Field(..., ge=1, description="1-based page number in the PDF")
    chunk_type: ChunkType = Field(..., description="TEXT or TABLE source type")
    section_title: str | None = Field(
        default=None,
        description="Section heading or table title in the financial statement",
    )
    bounding_box: BoundingBox | None = Field(
        default=None,
        description="Spatial bounding box of the cited block/table",
    )
    excerpt: str = Field(
        default="",
        description="Short snippet from the source chunk supporting the citation",
    )

    @classmethod
    def from_chunk(cls, chunk: Chunk, max_excerpt_length: int = 200) -> "Citation":
        """Construct a Citation directly from a domain Chunk entity."""
        snippet = chunk.content[:max_excerpt_length].strip()
        return cls(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            ticker=chunk.metadata.ticker,
            year=chunk.metadata.year,
            quarter=chunk.metadata.quarter,
            page_number=chunk.page_number,
            chunk_type=chunk.chunk_type,
            section_title=chunk.section_title,
            bounding_box=chunk.bounding_box,
            excerpt=snippet,
        )


class SearchResult(BaseModel):
    """A ranked chunk candidate returned by Dense, Sparse, RRF, or Cross-Encoder stages."""

    model_config = ConfigDict(extra="forbid")

    chunk: Chunk = Field(..., description="Retrieved domain chunk")
    score: float = Field(..., description="Primary relevance score for current ranking stage")
    rank: int = Field(default=1, ge=1, description="1-based rank position in result list")
    dense_score: float | None = Field(
        default=None,
        description="Cosine similarity score from Qdrant dense vector retrieval",
    )
    sparse_score: float | None = Field(
        default=None,
        description="BM25 relevance score from sparse keyword retrieval",
    )
    rrf_score: float | None = Field(
        default=None,
        description="Combined Reciprocal Rank Fusion score",
    )
    rerank_score: float | None = Field(
        default=None,
        description="Cross-Encoder semantic reranking score",
    )
    citation: Citation | None = Field(
        default=None,
        description="Source citation generated from the retrieved chunk",
    )


class ValidationReport(BaseModel):
    """Detailed audit report produced by the deterministic numeric guardrail."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: ValidationStatus = Field(
        ...,
        description="Overall verification status (PASSED, FAILED, SKIPPED, WARNING)",
    )
    extracted_numbers: list[float] = Field(
        default_factory=list,
        description="All numeric values extracted from the generated answer",
    )
    verified_numbers: list[float] = Field(
        default_factory=list,
        description="Numbers confirmed to exist in the source tables/chunks",
    )
    unverified_numbers: list[float] = Field(
        default_factory=list,
        description="Hallucinated or mismatched numbers absent from source tables",
    )
    message: str = Field(
        default="",
        description="Human-readable explanation of the verification outcome",
    )

    @property
    def is_valid(self) -> bool:
        """Return True if validation passed or was skipped (no numeric claims made)."""
        return self.status in (ValidationStatus.PASSED, ValidationStatus.SKIPPED)
