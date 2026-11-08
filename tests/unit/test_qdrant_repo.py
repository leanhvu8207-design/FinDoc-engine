"""Unit tests for QdrantVectorRepository in Local Mode (Phase 3 - Task 3.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.domain.entities.document import Chunk, ChunkType, DocumentMetadata, ReportType
from src.domain.entities.query import QueryFilter
from src.domain.entities.table import TableBlock
from src.domain.exceptions import VectorStoreError
from src.infrastructure.persistence.qdrant_repo import QdrantVectorRepository


def _make_sample_chunks() -> list[Chunk]:
    """Create sample financial chunks across multiple tickers and types for testing."""
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
    sample_table = TableBlock(
        page_number=2,
        title="Bảng cân đối kế toán FPT",
        headers=["Chỉ tiêu", "Số cuối kỳ"],
        rows=[["Tiền và tương đương tiền", "9.320.150"]],
        raw_markdown=(
            "| Chỉ tiêu | Số cuối kỳ |\n| --- | --- |\n| Tiền và tương đương tiền | 9.320.150 |"
        ),
    )

    return [
        Chunk(
            chunk_id="fpt-table-1",
            document_id="doc-fpt-2025",
            content="[MÃ CP: FPT | NĂM: 2025 | KỲ: Q4]\n" + sample_table.raw_markdown,
            chunk_type=ChunkType.TABLE,
            metadata=fpt_meta,
            page_number=2,
            section_title="Bảng cân đối kế toán FPT",
            table_block=sample_table,
        ),
        Chunk(
            chunk_id="fpt-text-1",
            document_id="doc-fpt-2025",
            content="[MÃ CP: FPT | NĂM: 2025 | KỲ: Q4]\nDoanh thu mảng công nghệ tăng trưởng 20%.",
            chunk_type=ChunkType.TEXT,
            metadata=fpt_meta,
            page_number=5,
            section_title="Thuyết minh doanh thu",
        ),
        Chunk(
            chunk_id="vnm-text-1",
            document_id="doc-vnm-2025",
            content="[MÃ CP: VNM | NĂM: 2025 | KỲ: Q4]\nDoanh thu xuất khẩu sữa tăng trưởng mạnh.",
            chunk_type=ChunkType.TEXT,
            metadata=vnm_meta,
            page_number=3,
            section_title="Thuyết minh bộ phận",
        ),
    ]


def test_qdrant_local_mode_upsert_and_filtered_search(tmp_path: Path) -> None:
    """Verify QdrantVectorRepository persists chunks to local storage and filters by payload."""
    storage_dir = tmp_path / "qdrant_local"
    repo = QdrantVectorRepository(
        collection_name="test_financial_chunks",
        vector_dim=4,
        storage_path=storage_dir,
    )
    try:
        chunks = _make_sample_chunks()
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],  # fpt-table-1
            [0.8, 0.6, 0.0, 0.0],  # fpt-text-1
            [0.9, 0.43589, 0.0, 0.0],  # vnm-text-1
        ]

        upserted_count = repo.upsert_chunks(chunks=chunks, embeddings=embeddings)
        assert upserted_count == 3

        # 1. Search with ticker="FPT" filter
        fpt_results = repo.search(
            query_vector=[1.0, 0.0, 0.0, 0.0],
            top_k=5,
            filter=QueryFilter(ticker="FPT"),
        )
        assert len(fpt_results) == 2
        assert all(r.chunk.metadata.ticker == "FPT" for r in fpt_results)
        assert fpt_results[0].chunk.chunk_id == "fpt-table-1"
        assert fpt_results[0].chunk.table_block is not None
        assert fpt_results[0].citation is not None
        assert fpt_results[0].citation.page_number == 2

        # 2. Search with ticker="FPT" and chunk_type=ChunkType.TABLE
        table_only = repo.search_chunks(
            query_vector=[1.0, 0.0, 0.0, 0.0],
            top_k=5,
            filter=QueryFilter(ticker="FPT", chunk_type=ChunkType.TABLE),
        )
        assert len(table_only) == 1
        assert table_only[0].chunk_id == "fpt-table-1"

        # 3. Search with ticker="VNM"
        vnm_results = repo.search(
            query_vector=[1.0, 0.0, 0.0, 0.0],
            top_k=5,
            filter=QueryFilter(ticker="VNM", year=2025, quarter=4),
        )
        assert len(vnm_results) == 1
        assert vnm_results[0].chunk.chunk_id == "vnm-text-1"
    finally:
        repo.close()


def test_qdrant_repo_validation_errors() -> None:
    """Verify dimension mismatch or count mismatch raises VectorStoreError."""
    repo = QdrantVectorRepository(
        collection_name="test_errors",
        vector_dim=4,
        in_memory=True,
    )
    try:
        chunks = _make_sample_chunks()[:1]

        # Count mismatch
        with pytest.raises(VectorStoreError):
            repo.upsert_chunks(
                chunks=chunks,
                embeddings=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            )

        # Vector dimension mismatch on upsert
        with pytest.raises(VectorStoreError):
            repo.upsert_chunks(chunks=chunks, embeddings=[[1.0, 0.0]])

        # Vector dimension mismatch on search
        with pytest.raises(VectorStoreError):
            repo.search(query_vector=[1.0, 0.0], top_k=5)
    finally:
        repo.close()
