"""POST /api/v1/search and POST /api/v1/query endpoints (Phase 5 - Task 5.1)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from src.application.dto.search_dto import (
    AnswerRequestDTO,
    AnswerResponseDTO,
    SearchRequestDTO,
    SearchResultDTO,
)
from src.application.use_cases.answer_with_verify import AnswerWithVerifyUseCase
from src.application.use_cases.hybrid_search import HybridSearchUseCase
from src.presentation.api.dependencies import (
    provide_answer_use_case,
    provide_hybrid_search_use_case,
)

router = APIRouter(tags=["Retrieval & Fact-Checked QA"])


@router.post("/search", response_model=SearchResultDTO)
def hybrid_search(
    request: SearchRequestDTO,
    use_case: Annotated[HybridSearchUseCase, Depends(provide_hybrid_search_use_case)],
) -> SearchResultDTO:
    """Execute Hybrid Search (Dense Qdrant + Sparse BM25 + RRF + Cross-Encoder Rerank)."""
    return use_case.execute_dto(request)


@router.post("/query", response_model=AnswerResponseDTO)
@router.post("/ask", response_model=AnswerResponseDTO)
def answer_with_verification(
    request: AnswerRequestDTO,
    use_case: Annotated[AnswerWithVerifyUseCase, Depends(provide_answer_use_case)],
) -> AnswerResponseDTO:
    """Synthesize a financial answer and verify all numeric claims via NumericGuardrail."""
    return use_case.execute_dto(request)
