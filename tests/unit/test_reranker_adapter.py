"""Unit tests for Cross-Encoder RerankerAdapter (Phase 4 - Task 4.2)."""

from __future__ import annotations

import pytest

from src.domain.entities.document import Chunk, ChunkType, DocumentMetadata, ReportType
from src.domain.entities.response import SearchResult
from src.domain.exceptions import RerankingError
from src.infrastructure.ml_models.reranker_adapter import RerankerAdapter


class MockCrossEncoder:
    """Deterministic mock Cross-Encoder scoring pairs by keyword overlap and target match."""

    def __init__(self) -> None:
        self.recorded_pairs: list[tuple[str, str]] = []

    def predict(
        self,
        sentences: list[tuple[str, str]],
        *,
        show_progress_bar: bool = False,
    ) -> list[float]:
        """Return deterministic cross-attention scores based on passage content."""
        _ = show_progress_bar
        self.recorded_pairs.extend(sentences)
        scores: list[float] = []
        for query, passage in sentences:
            if "9.420.800" in passage and "lợi nhuận sau thuế" in passage.lower():
                scores.append(4.85)
            elif "doanh thu thuần" in passage.lower():
                scores.append(1.20)
            else:
                scores.append(-2.10 + 0.001 * len(query))
        return scores


def _build_candidates() -> list[Chunk]:
    """Create candidate chunks in suboptimal initial order for reranking test."""
    meta = DocumentMetadata(
        ticker="FPT",
        year=2025,
        quarter=4,
        report_type=ReportType.CONSOLIDATED,
    )
    return [
        Chunk(
            chunk_id="chunk-noise",
            document_id="doc-fpt",
            content="Các khoản phải thu ngắn hạn khác đạt 450.000 triệu đồng.",
            chunk_type=ChunkType.TEXT,
            metadata=meta,
            page_number=1,
        ),
        Chunk(
            chunk_id="chunk-rev",
            document_id="doc-fpt",
            content="Doanh thu thuần về bán hàng và cung cấp dịch vụ đạt 62.724.500 triệu đồng.",
            chunk_type=ChunkType.TEXT,
            metadata=meta,
            page_number=2,
        ),
        Chunk(
            chunk_id="chunk-target",
            document_id="doc-fpt",
            content="Lợi nhuận sau thuế thu nhập doanh nghiệp đạt 9.420.800 triệu đồng.",
            chunk_type=ChunkType.TEXT,
            metadata=meta,
            page_number=2,
        ),
    ]


def test_reranker_adapter_sorts_candidates_by_cross_attention_score() -> None:
    """Verify RerankerAdapter scores (query, chunk) pairs and returns top_k in descending order."""
    mock_ce = MockCrossEncoder()
    adapter = RerankerAdapter(
        model_name="BAAI/bge-reranker-base",
        cross_encoder=mock_ce,
    )
    assert adapter.model_name == "BAAI/bge-reranker-base"

    candidates = _build_candidates()
    top_chunks = adapter.rerank(
        query="Lợi nhuận sau thuế của FPT năm 2025 là bao nhiêu?",
        candidates=candidates,
        top_k=2,
    )

    assert len(top_chunks) == 2
    assert top_chunks[0].chunk_id == "chunk-target"
    assert top_chunks[1].chunk_id == "chunk-rev"
    assert len(mock_ce.recorded_pairs) == 3


def test_rerank_results_preserves_rrf_metadata_and_sets_rerank_score() -> None:
    """Verify rerank_results keeps dense/sparse/rrf scores and assigns new rank and rerank_score."""
    mock_ce = MockCrossEncoder()
    adapter = RerankerAdapter(cross_encoder=mock_ce)
    chunks = _build_candidates()

    rrf_candidates = [
        SearchResult(
            chunk=c,
            score=0.03 - idx * 0.005,
            rank=idx + 1,
            dense_score=0.80,
            sparse_score=10.5,
            rrf_score=0.03 - idx * 0.005,
        )
        for idx, c in enumerate(chunks)
    ]

    reranked = adapter.rerank_results(
        query="Lợi nhuận sau thuế FPT",
        candidates=rrf_candidates,
        top_k=2,
    )

    assert len(reranked) == 2
    assert reranked[0].chunk.chunk_id == "chunk-target"
    assert reranked[0].rank == 1
    assert reranked[0].rerank_score == 4.85
    assert reranked[0].score == 4.85
    assert reranked[0].dense_score == 0.80
    assert reranked[0].sparse_score == 10.5
    assert reranked[0].rrf_score is not None


def test_reranker_adapter_edge_cases_and_validation() -> None:
    """Verify empty candidate handling and RerankingError on invalid query or top_k."""
    mock_ce = MockCrossEncoder()
    adapter = RerankerAdapter(cross_encoder=mock_ce)
    candidates = _build_candidates()

    assert adapter.rerank(query="Hợp lệ", candidates=[], top_k=5) == []
    assert adapter.rerank_results(query="Hợp lệ", candidates=[], top_k=5) == []

    with pytest.raises(RerankingError):
        adapter.rerank(query="   ", candidates=candidates, top_k=2)

    with pytest.raises(RerankingError):
        adapter.rerank(query="Hợp lệ", candidates=candidates, top_k=0)
