"""Integration tests for HybridSearchUseCase and AnswerWithVerifyUseCase (Phase 4 - Task 4.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.download_sample_bctc import ensure_sample_bctc_pdfs
from src.application.dto.search_dto import AnswerRequestDTO, SearchRequestDTO
from src.application.services.financial_chunker import FinancialChunker
from src.application.services.fusion_service import FusionService
from src.application.use_cases.answer_with_verify import AnswerWithVerifyUseCase
from src.application.use_cases.hybrid_search import HybridSearchUseCase
from src.application.use_cases.ingest_document import IngestDocumentUseCase
from src.domain.entities.document import ChunkType, DocumentMetadata, ReportType
from src.domain.entities.response import ValidationStatus
from src.domain.exceptions import HallucinationError
from src.infrastructure.ml_models.embedding_adapter import EmbeddingAdapter
from src.infrastructure.ml_models.reranker_adapter import RerankerAdapter
from src.infrastructure.parsers.pdfplumber_adapter import PdfPlumberAdapter
from src.infrastructure.persistence.bm25_repo import BM25SparseRepository
from src.infrastructure.persistence.qdrant_repo import QdrantVectorRepository
from src.infrastructure.validators.numeric_guardrail import NumericGuardrail


class MockSentenceEncoder:
    """Deterministic sentence encoder for hybrid retrieval integration tests."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        """Produce deterministic vectors favoring income statement content."""
        _ = (normalize_embeddings, convert_to_numpy, show_progress_bar)
        out: list[list[float]] = []
        for text in sentences:
            lower = text.lower()
            if "doanh thu" in lower or "62.849.000" in lower or "9.420.800" in lower:
                out.append([1.0, 0.5] + [0.1] * (self.dim - 2))
            else:
                out.append([0.2, 0.9] + [0.1] * (self.dim - 2))
        return out


class MockCrossEncoderModel:
    """Deterministic Cross-Encoder boosting chunks that contain the queried metric."""

    def predict(
        self,
        sentences: list[tuple[str, str]],
        *,
        show_progress_bar: bool = False,
    ) -> list[float]:
        """Score (query, passage) pairs deterministically."""
        _ = show_progress_bar
        scores: list[float] = []
        for _query, passage in sentences:
            if "62.849.000" in passage and "9.420.800" in passage:
                scores.append(5.25)
            elif "38.450.200" in passage:
                scores.append(2.10)
            else:
                scores.append(0.15)
        return scores


class MockLLMClient:
    """Mock LLM generator returning configurable factual or hallucinated answers."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.last_query: str = ""
        self.last_context: str = ""

    def generate_answer(self, query: str, context: str) -> str:
        """Record inputs and return configured answer."""
        self.last_query = query
        self.last_context = context
        return self.response_text


def test_hybrid_search_and_answer_with_verify_end_to_end(tmp_path: Path) -> None:
    """Verify full flow: Ingest PDF -> Hybrid Search (RRF+Rerank) -> Answer & Guardrail."""
    pdf_dir = tmp_path / "raw_pdfs"
    ensure_sample_bctc_pdfs(output_dir=pdf_dir)
    fpt_pdf = pdf_dir / "FPT_2025_Q4_Consolidated.pdf"

    dim = 16
    parser = PdfPlumberAdapter()
    chunker = FinancialChunker(chunk_size=512, chunk_overlap=64)
    embedder = EmbeddingAdapter(
        model_name="intfloat/multilingual-e5-base",
        vector_dim=dim,
        encoder=MockSentenceEncoder(dim=dim),
    )
    qdrant_repo = QdrantVectorRepository(
        collection_name="hybrid_search_test",
        vector_dim=dim,
        storage_path=tmp_path / "qdrant_storage",
    )
    bm25_repo = BM25SparseRepository(
        index_path=tmp_path / "bm25_index.pkl",
        auto_persist=True,
    )
    fusion_service = FusionService(k=60)
    reranker = RerankerAdapter(cross_encoder=MockCrossEncoderModel())
    guardrail = NumericGuardrail()

    try:
        # 1. Ingest sample FPT financial statement
        ingest_uc = IngestDocumentUseCase(
            parser=parser,
            chunker=chunker,
            embedder=embedder,
            vector_repo=qdrant_repo,
            sparse_repo=bm25_repo,
        )
        meta = DocumentMetadata(
            ticker="FPT",
            year=2025,
            quarter=4,
            report_type=ReportType.CONSOLIDATED,
        )
        ingest_uc.execute(file_path=fpt_pdf, metadata=meta)

        # 2. Run HybridSearchUseCase
        hybrid_search_uc = HybridSearchUseCase(
            embedder=embedder,
            vector_repo=qdrant_repo,
            sparse_repo=bm25_repo,
            fusion_service=fusion_service,
            reranker=reranker,
        )
        search_dto = SearchRequestDTO(
            query_text="Doanh thu ban hang va Loi nhuan sau thue cua FPT",
            top_k=3,
            candidate_pool_size=10,
            ticker="FPT",
            year=2025,
            enable_reranking=True,
        )
        search_response = hybrid_search_uc.execute_dto(search_dto)

        assert len(search_response.results) == 3
        top_result = search_response.results[0]
        assert top_result.rank == 1
        assert top_result.rerank_score == 5.25
        assert top_result.rrf_score is not None
        assert top_result.chunk.chunk_type == ChunkType.TABLE
        assert "62.849.000" in top_result.chunk.content
        assert len(search_response.citations) == 3

        # 3. Run AnswerWithVerifyUseCase with a faithful mock LLM response (PASSED)
        faithful_llm = MockLLMClient(
            response_text=(
                "Theo Báo cáo kết quả hoạt động kinh doanh Quý 4/2025 (Trang 2), "
                "Doanh thu thuần của FPT đạt 62.724.500 triệu VND và "
                "Lợi nhuận sau thuế đạt 9.420.800 triệu VND."
            )
        )
        answer_uc = AnswerWithVerifyUseCase(
            hybrid_search=hybrid_search_uc,
            validator=guardrail,
            llm_client=faithful_llm,
        )
        ans_req = AnswerRequestDTO(
            query_text="Doanh thu thuan va loi nhuan sau thue cua FPT la bao nhieu?",
            top_k=3,
            ticker="FPT",
            strict_guardrail=True,
        )
        ans_resp = answer_uc.execute_dto(ans_req)

        assert ans_resp.validation_report.status == ValidationStatus.PASSED
        assert ans_resp.validation_report.is_valid is True
        assert ans_resp.validation_report.verified_numbers == [62724500.0, 9420800.0]
        assert ans_resp.validation_report.hallucinated_numbers == []
        assert len(ans_resp.citations) == 3

        # 4. Run AnswerWithVerifyUseCase with a hallucinating mock LLM response (FAILED)
        hallucinating_llm = MockLLMClient(
            response_text="Lợi nhuận sau thuế của FPT Quý 4/2025 đạt 77.777.000 triệu VND."
        )
        bad_answer_uc = AnswerWithVerifyUseCase(
            hybrid_search=hybrid_search_uc,
            validator=guardrail,
            llm_client=hallucinating_llm,
        )
        non_strict_req = AnswerRequestDTO(
            query_text="Loi nhuan sau thue cua FPT?",
            top_k=3,
            ticker="FPT",
            strict_guardrail=False,
        )
        bad_resp = bad_answer_uc.execute_dto(non_strict_req)
        assert bad_resp.validation_report.status == ValidationStatus.FAILED
        assert bad_resp.validation_report.hallucinated_numbers == [77777000.0]

        strict_req = AnswerRequestDTO(
            query_text="Loi nhuan sau thue cua FPT?",
            top_k=3,
            ticker="FPT",
            strict_guardrail=True,
        )
        with pytest.raises(HallucinationError):
            bad_answer_uc.execute_dto(strict_req)
    finally:
        qdrant_repo.close()
