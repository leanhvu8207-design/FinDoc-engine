"""FinancialDocument, DocumentMetadata, and Chunk entities (Phase 1 - Task 1.1)."""

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.domain.entities.page import BoundingBox, PageBlock
from src.domain.entities.table import TableBlock


class ReportType(StrEnum):
    """Classification of Vietnamese corporate financial statements."""

    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"
    SEMI_ANNUAL = "SEMI_ANNUAL"
    ANNUAL = "ANNUAL"
    AUDITED = "AUDITED"
    CONSOLIDATED = "CONSOLIDATED"
    SEPARATE = "SEPARATE"


class ChunkType(StrEnum):
    """Classification of indexed document chunks."""

    TEXT = "TEXT"
    TABLE = "TABLE"


class DocumentMetadata(BaseModel):
    """Core business metadata identifying a listed company's financial report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(
        ...,
        min_length=2,
        max_length=15,
        description="Stock ticker symbol (e.g., 'FPT', 'VNM', 'HPG')",
    )
    year: int = Field(
        ...,
        ge=1990,
        le=2100,
        description="Fiscal year of the financial statement",
    )
    quarter: int | None = Field(
        default=None,
        ge=1,
        le=4,
        description="Fiscal quarter (1-4), or None for full-year annual reports",
    )
    report_type: ReportType = Field(
        default=ReportType.CONSOLIDATED,
        description="Type of financial report",
    )
    company_name: str | None = Field(
        default=None,
        description="Full legal name of the listed enterprise",
    )
    currency_unit: str = Field(
        default="VND",
        description="Default currency unit used across the statement",
    )
    source_filename: str | None = Field(
        default=None,
        description="Original PDF file name",
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        """Strip whitespace and convert ticker to uppercase."""
        cleaned = value.strip().upper()
        if not cleaned.isalnum():
            raise ValueError(f"Ticker must be alphanumeric, got '{value}'")
        return cleaned

    def format_context_header(self, section_title: str | None = None) -> str:
        """Build a deterministic metadata prefix header for context-aware chunking."""
        period = f"Q{self.quarter}/{self.year}" if self.quarter is not None else f"FY{self.year}"
        parts = [f"Ticker: {self.ticker}", f"Period: {period}", f"Type: {self.report_type.value}"]
        if section_title:
            parts.append(f"Section: {section_title}")
        return "[" + " | ".join(parts) + "]"


class Chunk(BaseModel):
    """Context-preserving indexable unit of text or full financial table."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
        description="Unique identifier for the chunk",
    )
    document_id: str = Field(
        ...,
        min_length=1,
        description="Parent FinancialDocument ID",
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Full text or Markdown content (including context header)",
    )
    chunk_type: ChunkType = Field(
        default=ChunkType.TEXT,
        description="Whether chunk contains narrative TEXT or structured TABLE",
    )
    metadata: DocumentMetadata = Field(
        ...,
        description="Corporate financial metadata inherited from parent document",
    )
    page_number: int = Field(
        ...,
        ge=1,
        description="1-based page number where the chunk starts",
    )
    section_title: str | None = Field(
        default=None,
        description="Heading or financial note title associated with this chunk",
    )
    bounding_box: BoundingBox | None = Field(
        default=None,
        description="Spatial bounding box on the source PDF page",
    )
    table_block: TableBlock | None = Field(
        default=None,
        description="Attached TableBlock when chunk_type == ChunkType.TABLE",
    )

    @model_validator(mode="after")
    def validate_table_chunk_consistency(self) -> "Chunk":
        """Ensure TABLE chunks and table_block attachment remain consistent."""
        if self.chunk_type == ChunkType.TEXT and self.table_block is not None:
            raise ValueError("A chunk with chunk_type=TEXT cannot hold a table_block")
        return self


class FinancialDocument(BaseModel):
    """Aggregate root representing an ingested PDF financial statement."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
        description="Unique identifier for the financial document",
    )
    metadata: DocumentMetadata = Field(
        ...,
        description="Company, period, and report classification metadata",
    )
    file_path: Path = Field(
        ...,
        description="Filesystem path to the raw PDF document",
    )
    total_pages: int = Field(
        default=1,
        ge=1,
        description="Total number of pages in the PDF document",
    )
    blocks: list[PageBlock] = Field(
        default_factory=list,
        description="Ordered sequence of parsed page blocks",
    )
    tables: list[TableBlock] = Field(
        default_factory=list,
        description="Extracted structured accounting tables",
    )
    chunks: list[Chunk] = Field(
        default_factory=list,
        description="Context-aware chunks generated from blocks and tables",
    )
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when the document was parsed/ingested",
    )
