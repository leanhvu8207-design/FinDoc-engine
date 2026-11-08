"""Reciprocal Rank Fusion (RRF) scoring service (Phase 4 - Task 4.1)."""

from __future__ import annotations

from src.domain.entities.document import Chunk
from src.domain.entities.response import Citation, SearchResult
from src.domain.exceptions import DomainValidationError


class FusionService:
    """Combines Dense Vector and Sparse BM25 rankings using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_Score(d) = sum_{m in {dense, sparse}} 1 / (k + rank_m(d))
    where `k` is a smoothing constant (default 60) and `rank_m(d)` is the 1-based rank
    of chunk `d` in retrieval stream `m`.
    """

    def __init__(self, k: int = 60) -> None:
        if k <= 0:
            raise DomainValidationError(
                f"RRF smoothing constant k must be positive, got {k}",
                details={"k": k},
            )
        self._k = k

    @property
    def k(self) -> int:
        """Return default RRF smoothing constant k."""
        return self._k

    def fuse(
        self,
        dense_results: list[SearchResult],
        sparse_results: list[SearchResult],
        k: int | None = None,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Fuse dense and sparse SearchResult lists into a unified RRF-ranked list.

        Args:
            dense_results: Ranked results from vector similarity search.
            sparse_results: Ranked results from BM25 lexical search.
            k: Optional override for the smoothing constant `k` (defaults to `self.k`).
            top_k: Optional maximum number of fused results to return.

        Returns:
            List of deduplicated SearchResult objects sorted by descending `rrf_score`.
        """
        smoothing_k = k if k is not None else self._k
        if smoothing_k <= 0:
            raise DomainValidationError(
                f"RRF smoothing constant k must be positive, got {smoothing_k}",
                details={"k": smoothing_k},
            )
        if top_k is not None and top_k <= 0:
            raise DomainValidationError(
                f"top_k must be positive when specified, got {top_k}",
                details={"top_k": top_k},
            )

        if not dense_results and not sparse_results:
            return []

        chunks_by_id: dict[str, Chunk] = {}
        citations_by_id: dict[str, Citation] = {}
        dense_ranks: dict[str, int] = {}
        sparse_ranks: dict[str, int] = {}
        dense_scores: dict[str, float] = {}
        sparse_scores: dict[str, float] = {}

        # Process dense stream (1-based rank)
        for idx, res in enumerate(dense_results, start=1):
            cid = res.chunk.chunk_id
            chunks_by_id[cid] = res.chunk
            if res.citation is not None:
                citations_by_id[cid] = res.citation
            rank_val = res.rank if res.rank >= 1 else idx
            if cid not in dense_ranks or rank_val < dense_ranks[cid]:
                dense_ranks[cid] = rank_val
                dense_scores[cid] = (
                    res.dense_score if res.dense_score is not None else res.score
                )

        # Process sparse stream (1-based rank)
        for idx, res in enumerate(sparse_results, start=1):
            cid = res.chunk.chunk_id
            chunks_by_id.setdefault(cid, res.chunk)
            if cid not in citations_by_id and res.citation is not None:
                citations_by_id[cid] = res.citation
            rank_val = res.rank if res.rank >= 1 else idx
            if cid not in sparse_ranks or rank_val < sparse_ranks[cid]:
                sparse_ranks[cid] = rank_val
                sparse_scores[cid] = (
                    res.sparse_score if res.sparse_score is not None else res.score
                )

        # Compute RRF score for every unique candidate chunk
        fused_entries: list[tuple[float, float, str]] = []
        for cid in chunks_by_id:
            rrf_score = 0.0
            if cid in dense_ranks:
                rrf_score += 1.0 / (smoothing_k + dense_ranks[cid])
            if cid in sparse_ranks:
                rrf_score += 1.0 / (smoothing_k + sparse_ranks[cid])

            # Secondary tie-breaker: presence in both streams + normalized raw scores
            in_both_bonus = 1.0 if (cid in dense_ranks and cid in sparse_ranks) else 0.0
            tie_breaker = in_both_bonus + (dense_scores.get(cid, 0.0) * 1e-4)
            fused_entries.append((rrf_score, tie_breaker, cid))

        fused_entries.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if top_k is not None:
            fused_entries = fused_entries[:top_k]

        fused_results: list[SearchResult] = []
        for new_rank, (rrf_val, _, cid) in enumerate(fused_entries, start=1):
            chunk = chunks_by_id[cid]
            citation = citations_by_id.get(cid) or Citation.from_chunk(chunk)
            fused_results.append(
                SearchResult(
                    chunk=chunk,
                    score=rrf_val,
                    rank=new_rank,
                    dense_score=dense_scores.get(cid),
                    sparse_score=sparse_scores.get(cid),
                    rrf_score=rrf_val,
                    citation=citation,
                )
            )

        return fused_results
