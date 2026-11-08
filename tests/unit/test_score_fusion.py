"""Unit tests for Reciprocal Rank Fusion (FusionService) (Phase 4 - Task 4.1)."""

from __future__ import annotations

import math

import pytest

from src.application.services.score_fusion import FusionService
from src.domain.entities.document import Chunk, ChunkType, DocumentMetadata, ReportType
from src.domain.entities.response import SearchResult
from src.domain.exceptions import DomainValidationError


def _make_chunk(chunk_id: str, text: str) -> Chunk:
    """Create a sample Chunk for RRF testing."""
    meta = DocumentMetadata(
        ticker="FPT",
        year=2025,
        quarter=4,
        report_type=ReportType.CONSOLIDATED,
    )
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-fpt",
        content=text,
        chunk_type=ChunkType.TEXT,
        metadata=meta,
        page_number=1,
    )


def test_rrf_promotes_overlapping_chunk_to_top_1() -> None:
    """Verify chunk appearing in both Dense and Sparse streams is promoted to Rank 1."""
    chunk_a = _make_chunk("chunk-A", "Chỉ xuất hiện ở Dense hạng 1")
    chunk_b = _make_chunk("chunk-B", "Xuất hiện ở cả Dense hạng 2 và Sparse hạng 2")
    chunk_c = _make_chunk("chunk-C", "Chỉ xuất hiện ở Sparse hạng 1")

    dense_results = [
        SearchResult(chunk=chunk_a, score=0.95, rank=1, dense_score=0.95),
        SearchResult(chunk=chunk_b, score=0.88, rank=2, dense_score=0.88),
    ]
    sparse_results = [
        SearchResult(chunk=chunk_c, score=18.5, rank=1, sparse_score=18.5),
        SearchResult(chunk=chunk_b, score=15.2, rank=2, sparse_score=15.2),
    ]

    service = FusionService(k=60)
    fused = service.fuse(dense_results=dense_results, sparse_results=sparse_results)

    assert len(fused) == 3
    # chunk-B has RRF = 1/(60+2) + 1/(60+2) = 2/62 = 0.032258...
    # chunk-A and chunk-C each have RRF = 1/(60+1) = 1/61 = 0.016393...
    assert fused[0].chunk.chunk_id == "chunk-B"
    assert fused[0].rank == 1
    expected_b_rrf = (1.0 / 62.0) + (1.0 / 62.0)
    assert fused[0].rrf_score is not None
    assert math.isclose(fused[0].rrf_score, expected_b_rrf, rel_tol=1e-6)
    assert fused[0].dense_score == 0.88
    assert fused[0].sparse_score == 15.2

    expected_single_rrf = 1.0 / 61.0
    assert fused[1].rrf_score is not None
    assert math.isclose(fused[1].rrf_score, expected_single_rrf, rel_tol=1e-6)
    assert fused[1].rank == 2
    assert fused[2].rank == 3


def test_rrf_handles_single_empty_stream_and_both_empty() -> None:
    """Verify RRF works cleanly when one or both retrieval streams return empty lists."""
    chunk_1 = _make_chunk("chunk-1", "Kết quả duy nhất từ BM25")
    sparse_only = [SearchResult(chunk=chunk_1, score=12.0, rank=1, sparse_score=12.0)]

    service = FusionService(k=60)

    # Dense empty, Sparse non-empty
    res_sparse = service.fuse(dense_results=[], sparse_results=sparse_only)
    assert len(res_sparse) == 1
    assert res_sparse[0].chunk.chunk_id == "chunk-1"
    assert res_sparse[0].dense_score is None
    assert res_sparse[0].sparse_score == 12.0
    assert res_sparse[0].rrf_score is not None
    assert math.isclose(res_sparse[0].rrf_score, 1.0 / 61.0, rel_tol=1e-6)

    # Both empty
    assert service.fuse(dense_results=[], sparse_results=[]) == []


def test_rrf_top_k_and_validation() -> None:
    """Verify top_k truncation and invalid k parameter validation."""
    service = FusionService(k=60)
    dense = [
        SearchResult(chunk=_make_chunk(f"c-{i}", f"Text {i}"), score=1.0 / i, rank=i)
        for i in range(1, 6)
    ]
    fused_top2 = service.fuse(dense_results=dense, sparse_results=[], top_k=2)
    assert len(fused_top2) == 2
    assert [r.rank for r in fused_top2] == [1, 2]

    with pytest.raises(DomainValidationError):
        FusionService(k=0)

    with pytest.raises(DomainValidationError):
        service.fuse(dense_results=dense, sparse_results=[], k=-10)

    with pytest.raises(DomainValidationError):
        service.fuse(dense_results=dense, sparse_results=[], top_k=0)
