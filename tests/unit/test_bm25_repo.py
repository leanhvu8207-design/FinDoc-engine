"""Unit tests for BM25SparseRepository (Phase 3 - Task 3.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.domain.entities.document import Chunk, ChunkType, DocumentMetadata, ReportType
from src.domain.entities.query import QueryFilter
from src.domain.exceptions import SparseIndexError
from src.infrastructure.persistence.bm25_repo import BM25SparseRepository


def _build_financial_corpus() -> list[Chunk]:
    """Create a realistic corpus of Vietnamese financial statement chunks for BM25 testing."""
    fpt_meta = DocumentMetadata(
        ticker="FPT",
        year=2025,
        quarter=4,
        report_type=ReportType.CONSOLIDATED,
    )
    vnm_meta = DocumentMetadata(
        ticker="VNM",
        year=2025,
        quarter=4,
        report_type=ReportType.CONSOLIDATED,
    )
    return [
        Chunk(
            chunk_id="chunk-rev-fpt",
            document_id="doc-fpt",
            content=(
                "[MÃ CP: FPT | NĂM: 2025 | KỲ: Q4]\n"
                "Chỉ tiêu Doanh thu thuần về bán hàng và cung cấp dịch vụ "
                "đạt 62.724.500 triệu đồng."
            ),
            chunk_type=ChunkType.TEXT,
            metadata=fpt_meta,
            page_number=2,
            section_title="Báo cáo kết quả hoạt động kinh doanh",
        ),
        Chunk(
            chunk_id="chunk-profit-fpt",
            document_id="doc-fpt",
            content=(
                "[MÃ CP: FPT | NĂM: 2025 | KỲ: Q4]\n"
                "Lợi nhuận sau thuế thu nhập doanh nghiệp đạt 9.420.800 triệu đồng. "
                "Chỉ số ROE đạt 28,4%, EPS đạt 6.450 đồng/cổ phiếu và hệ số P/E là 21,5x."
            ),
            chunk_type=ChunkType.TEXT,
            metadata=fpt_meta,
            page_number=2,
            section_title="Chỉ số tài chính cơ bản",
        ),
        Chunk(
            chunk_id="chunk-equity-vnm",
            document_id="doc-vnm",
            content=(
                "[MÃ CP: VNM | NĂM: 2025 | KỲ: Q4]\n"
                "Vốn chủ sở hữu hợp nhất đạt 35.400.000 triệu đồng. "
                "Biên EBITDA duy trì ở mức 26,2% trong năm 2025."
            ),
            chunk_type=ChunkType.TEXT,
            metadata=vnm_meta,
            page_number=1,
            section_title="Bảng cân đối kế toán",
        ),
    ]


def test_financial_tokenizer_preserves_metrics_and_numbers() -> None:
    """Verify tokenizer retains ROE, EPS, EBITDA, P/E, note codes, and formatted numbers."""
    text = (
        "Thuyết minh V.01: Doanh thu thuần đạt 62.724.500 triệu VND, "
        "biên EBITDA 26,2%, chỉ số ROE 28,4%, EPS và P/E tăng trưởng."
    )
    tokens = BM25SparseRepository.tokenize_financial_text(text)

    assert "v.01" in tokens
    assert "62.724.500" in tokens
    assert "62724500" in tokens
    assert "ebitda" in tokens
    assert "26,2%" in tokens
    assert "roe" in tokens
    assert "eps" in tokens
    assert "p/e" in tokens
    assert "doanh_thu" in tokens
    assert "thu_thuần" in tokens


def test_bm25_exact_financial_keyword_retrieval(tmp_path: Path) -> None:
    """Verify BM25 ranks exact Vietnamese financial phrases and ratios at Top 1."""
    index_file = tmp_path / "bm25_test.pkl"
    repo = BM25SparseRepository(index_path=index_file, auto_persist=True)

    chunks = _build_financial_corpus()
    indexed_count = repo.index(chunks)
    assert indexed_count == 3
    assert repo.total_chunks == 3
    assert index_file.exists()

    # 1. Query "doanh thu thuần" -> must hit chunk-rev-fpt at Rank 1
    rev_results = repo.search_bm25("doanh thu thuần", top_k=3)
    assert len(rev_results) >= 1
    assert rev_results[0].chunk.chunk_id == "chunk-rev-fpt"
    assert rev_results[0].rank == 1
    assert rev_results[0].sparse_score is not None
    assert rev_results[0].sparse_score > 0.0

    # 2. Query "lợi nhuận sau thuế ROE P/E" -> must hit chunk-profit-fpt at Rank 1
    profit_results = repo.search_bm25("lợi nhuận sau thuế ROE P/E", top_k=3)
    assert len(profit_results) >= 1
    assert profit_results[0].chunk.chunk_id == "chunk-profit-fpt"
    assert profit_results[0].rank == 1

    # 3. Query "Vốn chủ sở hữu EBITDA" -> must hit chunk-equity-vnm at Rank 1
    equity_results = repo.search_bm25("Vốn chủ sở hữu EBITDA", top_k=3)
    assert len(equity_results) >= 1
    assert equity_results[0].chunk.chunk_id == "chunk-equity-vnm"


def test_bm25_persistence_and_metadata_filtering(tmp_path: Path) -> None:
    """Verify BM25 index reloads from disk (JSON & Pickle) and filters by QueryFilter."""
    json_path = tmp_path / "bm25_index.json"
    repo = BM25SparseRepository(index_path=json_path, auto_persist=True)
    repo.index(_build_financial_corpus())

    # Reload into a fresh repository instance from JSON file
    reloaded_repo = BM25SparseRepository(index_path=json_path, auto_persist=True)
    assert reloaded_repo.total_chunks == 3

    # Filter specifically for VNM
    vnm_only = reloaded_repo.search_bm25(
        query="triệu đồng",
        top_k=5,
        filter=QueryFilter(ticker="VNM"),
    )
    assert len(vnm_only) == 1
    assert vnm_only[0].chunk.metadata.ticker == "VNM"

    with pytest.raises(SparseIndexError):
        reloaded_repo.search_bm25(query="   ", top_k=3)
