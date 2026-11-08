"""End-to-end integration tests for IngestDocumentUseCase (Phase 3 - Task 3.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.download_sample_bctc import ensure_sample_bctc_pdfs
from src.application.dto.ingest_dto import IngestRequest
from src.application.services.financial_chunker import FinancialChunker
from src.application.use_cases.ingest_document import IngestDocumentUseCase
from src.domain.entities.document import ChunkType, DocumentMetadata, ReportType
from src.domain.entities.query import QueryFilter
from src.domain.exceptions import EmbeddingError
from src.infrastructure.ml_models.embedding_adapter import EmbeddingAdapter
from src.infrastructure.parsers.pdfplumber_adapter import PdfPlumberAdapter
from src.infrastructure.persistence.bm25_repo import BM25SparseRepository
from src.infrastructure.persistence.qdrant_repo import QdrantVectorRepository


class DeterministicIntegrationEncoder:
    """Deterministic encoder producing 16-dim vectors for end-to-end ingestion tests."""

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
        """Encode sentences into deterministic vectors."""
        _ = (normalize_embeddings, convert_to_numpy, show_progress_bar)
        vectors: list[list[float]] = []
        for idx, text in enumerate(sentences, start=1):
            vec = [float((len(text) + col * 3 + idx) % 17 + 1) for col in range(self.dim)]
            vectors.append(vec)
        return vectors


def test_ingest_document_use_case_end_to_end(tmp_path: Path) -> None:
    """Verify full PDF ingestion pipeline into both Qdrant Local Mode and BM25 Sparse Index."""
    pdf_dir = tmp_path / "raw_pdfs"
    ensure_sample_bctc_pdfs(output_dir=pdf_dir)
    fpt_pdf = pdf_dir / "FPT_2025_Q4_Consolidated.pdf"
    assert fpt_pdf.exists()

    vector_dim = 16
    parser = PdfPlumberAdapter()
    chunker = FinancialChunker(chunk_size=512, chunk_overlap=64)
    embedder = EmbeddingAdapter(
        model_name="intfloat/multilingual-e5-base",
        vector_dim=vector_dim,
        encoder=DeterministicIntegrationEncoder(dim=vector_dim),
    )
    qdrant_repo = QdrantVectorRepository(
        collection_name="integration_financial_docs",
        vector_dim=vector_dim,
        storage_path=tmp_path / "qdrant_storage",
    )
    bm25_repo = BM25SparseRepository(
        index_path=tmp_path / "bm25_index" / "index.pkl",
        auto_persist=True,
    )

    try:
        use_case = IngestDocumentUseCase(
            parser=parser,
            chunker=chunker,
            embedder=embedder,
            vector_repo=qdrant_repo,
            sparse_repo=bm25_repo,
        )

        metadata = DocumentMetadata(
            ticker="FPT",
            year=2025,
            quarter=4,
            report_type=ReportType.CONSOLIDATED,
            company_name="CTCP FPT",
        )
        request = IngestRequest(file_path=fpt_pdf, metadata=metadata)
        response = use_case.execute_request(request)

        assert response.ticker == "FPT"
        assert response.year == 2025
        assert response.quarter == 4
        assert response.total_pages == 2
        assert response.total_tables == 2
        assert response.table_chunks_count == 2
        assert response.text_chunks_count >= 2
        assert response.total_chunks == response.table_chunks_count + response.text_chunks_count
        assert response.vector_upserted_count == response.total_chunks
        assert response.sparse_indexed_count == response.total_chunks

        # Verify Qdrant Local Mode retrieval
        query_vec = embedder.embed_query("Doanh thu bán hàng và cung cấp dịch vụ của FPT")
        dense_results = qdrant_repo.search(
            query_vector=query_vec,
            top_k=5,
            filter=QueryFilter(ticker="FPT", chunk_type=ChunkType.TABLE),
        )
        assert len(dense_results) == 2
        assert all(r.chunk.table_block is not None for r in dense_results)

        # Verify BM25 Sparse Index retrieval for accounting terms
        sparse_results = bm25_repo.search_bm25(
            query="Doanh thu ban hang va cung cap dich vu 62.849.000",
            top_k=5,
            filter=QueryFilter(ticker="FPT"),
        )
        assert len(sparse_results) >= 1
        assert "62.849.000" in sparse_results[0].chunk.content
    finally:
        qdrant_repo.close()


def test_ingest_use_case_detects_td01_vector_dimension_mismatch(tmp_path: Path) -> None:
    """Verify [TD-01] safeguard raises EmbeddingError when encoder output dim mismatches config."""
    pdf_dir = tmp_path / "raw_pdfs"
    ensure_sample_bctc_pdfs(output_dir=pdf_dir)
    fpt_pdf = pdf_dir / "FPT_2025_Q4_Consolidated.pdf"

    parser = PdfPlumberAdapter()
    chunker = FinancialChunker()
    # Configure embedder expecting 16-dim but inject encoder returning 8-dim vectors
    mismatched_embedder = EmbeddingAdapter(
        vector_dim=16,
        encoder=DeterministicIntegrationEncoder(dim=8),
    )
    qdrant_repo = QdrantVectorRepository(
        collection_name="test_td01",
        vector_dim=16,
        in_memory=True,
    )
    bm25_repo = BM25SparseRepository(
        index_path=tmp_path / "bm25_td01.pkl",
        auto_persist=False,
    )

    try:
        use_case = IngestDocumentUseCase(
            parser=parser,
            chunker=chunker,
            embedder=mismatched_embedder,
            vector_repo=qdrant_repo,
            sparse_repo=bm25_repo,
        )
        meta = DocumentMetadata(ticker="FPT", year=2025, quarter=4)
        with pytest.raises(EmbeddingError, match="dimensionality mismatch"):
            use_case.execute(file_path=fpt_pdf, metadata=meta)
    finally:
        qdrant_repo.close()
