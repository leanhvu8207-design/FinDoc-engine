"""Cross-Encoder reranker adapter implementing IReranker (Phase 4 - Task 4.2)."""

from __future__ import annotations

from typing import Any, Protocol, cast

from src.domain.entities.document import Chunk
from src.domain.entities.response import Citation, SearchResult
from src.domain.exceptions import RerankingError
from src.domain.interfaces.i_reranker import IReranker
from src.infrastructure.config import Settings, get_settings


class CrossEncoderProtocol(Protocol):
    """Structural protocol for sentence-transformers CrossEncoder models."""

    def predict(
        self,
        sentences: list[tuple[str, str]],
        *,
        show_progress_bar: bool = False,
    ) -> Any:
        """Predict cross-attention relevance scores for (query, passage) pairs."""


class RerankerAdapter(IReranker):
    """Concrete adapter for semantic reranking using a Cross-Encoder model."""

    def __init__(
        self,
        settings: Settings | None = None,
        model_name: str | None = None,
        device: str | None = None,
        cross_encoder: CrossEncoderProtocol | None = None,
    ) -> None:
        cfg = settings or get_settings()
        self._model_name: str = model_name or cfg.reranker_model_name
        self._device: str = device or cfg.model_device
        self._cross_encoder: CrossEncoderProtocol | None = cross_encoder

    @property
    def model_name(self) -> str:
        """Return configured Cross-Encoder model name."""
        return self._model_name

    def _get_or_load_model(self) -> CrossEncoderProtocol:
        """Lazily initialize the CrossEncoder singleton on first inference."""
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder

                loaded = CrossEncoder(
                    self._model_name,
                    device=self._device,
                )
                self._cross_encoder = cast(CrossEncoderProtocol, loaded)
            except Exception as exc:
                raise RerankingError(
                    f"Failed to load CrossEncoder model '{self._model_name}'",
                    details={"model_name": self._model_name, "error": str(exc)},
                ) from exc
        return self._cross_encoder

    def _score_pairs(self, query: str, contents: list[str]) -> list[float]:
        """Compute cross-attention relevance scores for a query against candidate contents."""
        cleaned_query = query.strip()
        if not cleaned_query:
            raise RerankingError("Rerank query cannot be empty or whitespace")

        pairs: list[tuple[str, str]] = [(cleaned_query, text) for text in contents]
        try:
            model = self._get_or_load_model()
            raw_scores = model.predict(pairs, show_progress_bar=False)
            if hasattr(raw_scores, "tolist"):
                scores_list = [float(x) for x in raw_scores.tolist()]
            else:
                scores_list = [float(x) for x in raw_scores]

            if len(scores_list) != len(contents):
                raise RerankingError(
                    f"CrossEncoder returned {len(scores_list)} scores for "
                    f"{len(contents)} candidates"
                )
            return scores_list
        except RerankingError:
            raise
        except Exception as exc:
            raise RerankingError(
                "CrossEncoder inference failed during reranking",
                details={"candidate_count": len(contents), "error": str(exc)},
            ) from exc

    def rerank(
        self,
        query: str,
        candidates: list[Chunk],
        top_k: int = 5,
    ) -> list[Chunk]:
        """Score (query, chunk.content) pairs and return the top_k highest-scoring Chunks."""
        if top_k <= 0:
            raise RerankingError(f"top_k must be positive, got {top_k}")
        if not query or not query.strip():
            raise RerankingError("Rerank query cannot be empty or whitespace")
        if not candidates:
            return []

        scores = self._score_pairs(
            query=query,
            contents=[chunk.content for chunk in candidates],
        )
        scored_chunks = list(zip(scores, range(len(candidates)), candidates, strict=True))
        # Sort by descending cross-encoder score, preserving initial order on exact tie
        scored_chunks.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        return [chunk for _, _, chunk in scored_chunks[:top_k]]

    def rerank_results(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Score SearchResult candidates and return top_k SearchResults with rerank_score."""
        if top_k <= 0:
            raise RerankingError(f"top_k must be positive, got {top_k}")
        if not query or not query.strip():
            raise RerankingError("Rerank query cannot be empty or whitespace")
        if not candidates:
            return []

        scores = self._score_pairs(
            query=query,
            contents=[res.chunk.content for res in candidates],
        )
        scored_items = list(zip(scores, range(len(candidates)), candidates, strict=True))
        scored_items.sort(key=lambda item: (item[0], -item[1]), reverse=True)

        reranked_results: list[SearchResult] = []
        for new_rank, (rerank_score, _, orig_res) in enumerate(scored_items[:top_k], start=1):
            reranked_results.append(
                SearchResult(
                    chunk=orig_res.chunk,
                    score=rerank_score,
                    rank=new_rank,
                    dense_score=orig_res.dense_score,
                    sparse_score=orig_res.sparse_score,
                    rrf_score=orig_res.rrf_score,
                    rerank_score=rerank_score,
                    citation=orig_res.citation or Citation.from_chunk(orig_res.chunk),
                )
            )
        return reranked_results
