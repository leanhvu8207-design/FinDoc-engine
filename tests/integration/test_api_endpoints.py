"""Integration tests for FastAPI REST endpoints (Phase 5 - Task 5.1)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from scripts.download_sample_bctc import ensure_sample_bctc_pdfs
from src.application.services.financial_chunker import FinancialChunker
from src.application.services.fusion_service import FusionService
from src.infrastructure.ml_models.embedding_adapter import EmbeddingAdapter
from src.infrastructure.ml_models.reranker_adapter import RerankerAdapter
from src.infrastructure.parsers.pdfplumber_adapter import PdfPlumberAdapter
from src.infrastructure.persistence.bm25_repo import BM25SparseRepository
from src.infrastructure.persistence.qdrant_repo import QdrantVectorRepository
from src.infrastructure.validators.numeric_guardrail import NumericGuardrail
from src.presentation.api.dependencies import ServiceContainer
from src.presentation.main import create_app


class APIMockSentenceEncoder:
    """Deterministic 16-dim sentence encoder for API endpoint testing."""

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
        """Encode sentences deterministically."""
        _ = (normalize_embeddings, convert_to_numpy, show_progress_bar)
        out: list[list[float]] = []
        for text in sentences:
            lower = text.lower()
            if "62.849.000" in lower or "doanh thu" in lower:
                out.append([1.0, 0.4] + [0.1] * (self.dim - 2))
            else:
                out.append([0.3, 0.9] + [0.1] * (self.dim - 2))
        return out


class APIMockCrossEncoder:
    """Deterministic Cross-Encoder for API endpoint testing."""

    def predict(
        self,
        sentences: list[tuple[str, str]],
        *,
        show_progress_bar: bool = False,
    ) -> list[float]:
        """Score candidate pairs."""
        _ = show_progress_bar
        return [
            4.90 if ("62.849.000" in passage or "9.420.800" in passage) else 0.50
            for _, passage in sentences
        ]


class ConfigurableLLMClient:
    """Configurable mock LLM client for testing both faithful and hallucinated responses."""

    def __init__(self) -> None:
        self.answer_to_return = (
            "Doanh thu thuần Quý 4/2025 của FPT đạt 62.724.500 triệu VND "
            "và Lợi nhuận sau thuế đạt 9.420.800 triệu VND."
        )

    def generate_answer(self, query: str, context: str) -> str:
        """Return configured answer string."""
        _ = (query, context)
        return self.answer_to_return


def test_fastapi_endpoints_ingest_search_query_and_error_handlers(tmp_path: Path) -> None:
    """Verify /health, /ingest, /search, /query, and domain error handlers via TestClient."""
    pdf_dir = tmp_path / "raw_pdfs"
    ensure_sample_bctc_pdfs(output_dir=pdf_dir)
    fpt_pdf = pdf_dir / "FPT_2025_Q4_Consolidated.pdf"

    dim = 16
    mock_llm = ConfigurableLLMClient()
    container = ServiceContainer(
        parser=PdfPlumberAdapter(),
        chunker=FinancialChunker(chunk_size=512, chunk_overlap=64),
        embedder=EmbeddingAdapter(vector_dim=dim, encoder=APIMockSentenceEncoder(dim=dim)),
        vector_repo=QdrantVectorRepository(
            collection_name="api_test_collection",
            vector_dim=dim,
            storage_path=tmp_path / "qdrant_api",
        ),
        sparse_repo=BM25SparseRepository(
            index_path=tmp_path / "bm25_api.pkl",
            auto_persist=True,
        ),
        fusion_service=FusionService(k=60),
        reranker=RerankerAdapter(cross_encoder=APIMockCrossEncoder()),
        validator=NumericGuardrail(),
        llm_client=mock_llm,
    )

    app = create_app(container=container)

    with TestClient(app) as client:
        # 1. GET /health
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "ok"

        # 2. POST /api/v1/documents/ingest
        ingest_resp = client.post(
            "/api/v1/documents/ingest",
            json={
                "file_path": str(fpt_pdf),
                "ticker": "FPT",
                "year": 2025,
                "quarter": 4,
                "report_type": "CONSOLIDATED",
                "company_name": "CTCP FPT",
            },
        )
        assert ingest_resp.status_code == 200
        ingest_data = ingest_resp.json()
        assert ingest_data["ticker"] == "FPT"
        assert ingest_data["total_tables"] == 2
        assert ingest_data["total_chunks"] >= 4

        # 3. POST /api/v1/documents/ingest with non-existent PDF -> 400 Bad Request (ParsingError)
        bad_ingest_resp = client.post(
            "/api/v1/documents/ingest",
            json={
                "file_path": str(tmp_path / "missing_file.pdf"),
                "ticker": "FPT",
                "year": 2025,
            },
        )
        assert bad_ingest_resp.status_code == 400
        assert bad_ingest_resp.json()["error_type"] == "ParsingError"

        # 4. POST /api/v1/search
        search_resp = client.post(
            "/api/v1/search",
            json={
                "query_text": "Doanh thu thuan va loi nhuan sau thue FPT",
                "top_k": 3,
                "ticker": "FPT",
                "year": 2025,
                "enable_reranking": True,
            },
        )
        assert search_resp.status_code == 200
        search_data = search_resp.json()
        assert len(search_data["results"]) == 3
        assert len(search_data["citations"]) == 3

        # 5. POST /api/v1/query with faithful answer -> 200 OK & ValidationStatus.PASSED
        query_resp = client.post(
            "/api/v1/query",
            json={
                "query_text": "Ket qua kinh doanh FPT Quy 4/2025?",
                "top_k": 3,
                "ticker": "FPT",
                "strict_guardrail": True,
            },
        )
        assert query_resp.status_code == 200
        query_data = query_resp.json()
        assert query_data["validation_report"]["status"] == "PASSED"
        assert query_data["validation_report"]["verified_numbers"] == [62724500.0, 9420800.0]

        # 6. POST /api/v1/query with hallucinated number & strict_guardrail=True -> 422
        mock_llm.answer_to_return = "Lợi nhuận sau thuế FPT đạt 123.456.789 triệu VND."
        hallucinated_resp = client.post(
            "/api/v1/query",
            json={
                "query_text": "Loi nhuan sau thue FPT?",
                "top_k": 3,
                "ticker": "FPT",
                "strict_guardrail": True,
            },
        )
        assert hallucinated_resp.status_code == 422
        err_body = hallucinated_resp.json()
        assert err_body["error_type"] == "HallucinationError"
        assert err_body["hallucinated_numbers"] == [123456789.0]
