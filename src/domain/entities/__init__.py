"""Domain entities package exporting core business models (Phase 1 - Task 1.1)."""

from src.domain.entities.document import (
    Chunk,
    ChunkType,
    DocumentMetadata,
    FinancialDocument,
    ReportType,
)
from src.domain.entities.page import BlockType, BoundingBox, DocumentBlock, PageBlock
from src.domain.entities.query import QueryFilter, SearchQuery
from src.domain.entities.response import (
    Citation,
    SearchResult,
    ValidationReport,
    ValidationStatus,
)
from src.domain.entities.table import FinancialTable, TableBlock, TableCell

__all__ = [
    "BlockType",
    "BoundingBox",
    "Chunk",
    "ChunkType",
    "Citation",
    "DocumentBlock",
    "DocumentMetadata",
    "FinancialDocument",
    "FinancialTable",
    "PageBlock",
    "QueryFilter",
    "ReportType",
    "SearchQuery",
    "SearchResult",
    "TableBlock",
    "TableCell",
    "ValidationReport",
    "ValidationStatus",
]
