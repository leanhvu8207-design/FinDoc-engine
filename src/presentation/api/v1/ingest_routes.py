"""POST /api/v1/documents/ingest endpoint (Phase 5 - Task 5.1)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from src.application.dto.ingest_dto import IngestRequest, IngestResponse
from src.application.use_cases.ingest_document import IngestDocumentUseCase
from src.domain.entities.document import DocumentMetadata, ReportType
from src.presentation.api.dependencies import provide_ingest_use_case

router = APIRouter(prefix="/documents", tags=["Ingestion"])


class IngestDocumentPayload(BaseModel):
    """JSON request body for ingesting a PDF financial statement from disk."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_path: str = Field(..., min_length=1, description="Path to the PDF file on server/local")
    ticker: str = Field(..., min_length=2, max_length=15, description="Stock ticker (e.g., FPT)")
    year: int = Field(..., ge=1990, le=2100, description="Fiscal year")
    quarter: int | None = Field(default=None, ge=1, le=4, description="Fiscal quarter (1-4)")
    report_type: ReportType = Field(
        default=ReportType.CONSOLIDATED,
        description="Financial report type",
    )
    company_name: str | None = Field(default=None, description="Optional company legal name")

    def to_ingest_request(self) -> IngestRequest:
        """Convert API payload into an application-layer IngestRequest DTO."""
        metadata = DocumentMetadata(
            ticker=self.ticker,
            year=self.year,
            quarter=self.quarter,
            report_type=self.report_type,
            company_name=self.company_name,
            source_filename=Path(self.file_path).name,
        )
        return IngestRequest(file_path=Path(self.file_path), metadata=metadata)


@router.post("/ingest", response_model=IngestResponse)
def ingest_document(
    payload: IngestDocumentPayload,
    use_case: Annotated[IngestDocumentUseCase, Depends(provide_ingest_use_case)],
) -> IngestResponse:
    """Parse, chunk, embed, and index a corporate financial statement PDF."""
    return use_case.execute_request(payload.to_ingest_request())
