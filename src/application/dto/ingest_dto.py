"""IngestRequest and IngestResponse DTOs (Phase 3 - Task 3.4)."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.domain.entities.document import DocumentMetadata


class IngestRequest(BaseModel):
    """Input Data Transfer Object for the document ingestion use case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_path: Path = Field(
        ...,
        description="Filesystem path to the target PDF financial statement",
    )
    metadata: DocumentMetadata = Field(
        ...,
        description="Corporate metadata (ticker, year, quarter, report_type)",
    )


class IngestResponse(BaseModel):
    """Output Data Transfer Object summarizing a completed document ingestion pipeline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str = Field(..., min_length=1, description="Generated FinancialDocument ID")
    ticker: str = Field(..., min_length=2, description="Normalized stock ticker")
    year: int = Field(..., ge=1990, le=2100, description="Fiscal year")
    quarter: int | None = Field(default=None, ge=1, le=4, description="Fiscal quarter")
    total_pages: int = Field(..., ge=1, description="Total number of pages parsed")
    total_tables: int = Field(..., ge=0, description="Total structured tables extracted")
    total_chunks: int = Field(..., ge=0, description="Total chunks created")
    text_chunks_count: int = Field(..., ge=0, description="Number of narrative TEXT chunks")
    table_chunks_count: int = Field(..., ge=0, description="Number of intact TABLE chunks")
    vector_upserted_count: int = Field(
        ...,
        ge=0,
        description="Number of chunks indexed into dense vector repository",
    )
    sparse_indexed_count: int = Field(
        ...,
        ge=0,
        description="Number of chunks indexed into sparse BM25 repository",
    )
