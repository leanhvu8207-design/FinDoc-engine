"""Unit tests for EmbeddingAdapter (Phase 3 - Task 3.1)."""

from __future__ import annotations

import math

import pytest

from src.domain.exceptions import EmbeddingError
from src.infrastructure.ml_models.embedding_adapter import EmbeddingAdapter


class DeterministicMockEncoder:
    """Lightweight deterministic encoder for fast offline unit testing."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.recorded_sentences: list[str] = []

    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        """Generate unnormalized deterministic vectors based on string content."""
        _ = (normalize_embeddings, convert_to_numpy, show_progress_bar)
        self.recorded_sentences.extend(sentences)
        output: list[list[float]] = []
        for idx, text in enumerate(sentences, start=1):
            vec = [float((len(text) + i + idx) % 11 + 1) for i in range(self.dim)]
            output.append(vec)
        return output


def test_e5_prefix_injection_and_l2_normalization() -> None:
    """Verify E5 model automatically adds 'passage: ' and 'query: ' prefixes and L2 normalizes."""
    mock_encoder = DeterministicMockEncoder(dim=8)
    adapter = EmbeddingAdapter(
        model_name="intfloat/multilingual-e5-base",
        vector_dim=8,
        encoder=mock_encoder,
    )
    assert adapter.uses_e5_prefix is True
    assert adapter.dimension == 8

    sample_docs = [f"Báo cáo tài chính hợp nhất quý {i} năm 2025" for i in range(1, 11)]
    doc_vectors = adapter.embed_documents(sample_docs)

    assert len(doc_vectors) == 10
    for vec in doc_vectors:
        assert len(vec) == 8
        l2_norm = math.sqrt(sum(x * x for x in vec))
        assert math.isclose(l2_norm, 1.0, rel_tol=1e-6)

    assert all(s.startswith("passage: ") for s in mock_encoder.recorded_sentences[:10])

    query_vec = adapter.embed_query("Doanh thu thuần của FPT năm 2025 là bao nhiêu?")
    assert len(query_vec) == 8
    query_norm = math.sqrt(sum(x * x for x in query_vec))
    assert math.isclose(query_norm, 1.0, rel_tol=1e-6)
    assert mock_encoder.recorded_sentences[-1].startswith("query: ")


def test_non_e5_model_does_not_inject_prefix() -> None:
    """Verify BGE-M3 or non-E5 models encode raw text without 'passage: ' or 'query: ' prefixes."""
    mock_encoder = DeterministicMockEncoder(dim=4)
    adapter = EmbeddingAdapter(
        model_name="BAAI/bge-m3",
        vector_dim=4,
        encoder=mock_encoder,
    )
    assert adapter.uses_e5_prefix is False

    adapter.embed_documents(["Tài sản ngắn hạn đạt 38.450 tỷ đồng"])
    adapter.embed_query("Tài sản ngắn hạn")

    assert mock_encoder.recorded_sentences[0] == "Tài sản ngắn hạn đạt 38.450 tỷ đồng"
    assert mock_encoder.recorded_sentences[1] == "Tài sản ngắn hạn"


def test_embedding_adapter_rejects_empty_inputs() -> None:
    """Verify empty batch or whitespace strings raise domain EmbeddingError."""
    mock_encoder = DeterministicMockEncoder(dim=4)
    adapter = EmbeddingAdapter(vector_dim=4, encoder=mock_encoder)

    with pytest.raises(EmbeddingError):
        adapter.embed_documents([])

    with pytest.raises(EmbeddingError):
        adapter.embed_documents(["Hợp lệ", "   "])

    with pytest.raises(EmbeddingError):
        adapter.embed_query("   ")
