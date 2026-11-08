"""BM25 sparse keyword index repository implementation (Phase 3 - Task 3.3)."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Literal

from rank_bm25 import BM25Okapi, BM25Plus

from src.domain.entities.document import Chunk
from src.domain.entities.query import QueryFilter
from src.domain.entities.response import Citation, SearchResult
from src.domain.exceptions import SparseIndexError
from src.domain.interfaces.i_sparse_repo import ISparseRepository
from src.infrastructure.config import Settings, get_settings

# Captures Vietnamese words, financial acronyms/ratios (P/E, ROE, EBITDA, V.01), and numbers
_TOKEN_RE = re.compile(
    r"[+-]?\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?%?|"
    r"[+-]?\d+(?:[.,]\d+)?%?|"
    r"[a-zA-ZÀ-ỹ0-9]+(?:[/._-][a-zA-ZÀ-ỹ0-9]+)*",
    re.UNICODE,
)

# Pure alphabetic Vietnamese word check for generating phrase bigrams
_ALPHA_WORD_RE = re.compile(r"^[a-zA-ZÀ-ỹ]+$", re.UNICODE)

# Formatted thousand-separated number pattern for dual-token indexing (e.g. 38.450.200 -> 38450200)
_THOUSAND_SEP_NUM_RE = re.compile(r"^([+-]?\d{1,3}(?:[.,]\d{3})+)$")


class BM25SparseRepository(ISparseRepository):
    """Concrete implementation of ISparseRepository using rank-bm25 and financial tokenization."""

    def __init__(
        self,
        settings: Settings | None = None,
        index_path: Path | None = None,
        algorithm: Literal["bm25_plus", "bm25_okapi"] = "bm25_plus",
        auto_persist: bool = True,
    ) -> None:
        cfg = settings or get_settings()
        self._index_path: Path = index_path or cfg.bm25_index_path
        self._algorithm: Literal["bm25_plus", "bm25_okapi"] = algorithm
        self._auto_persist: bool = auto_persist

        self._chunks_by_id: dict[str, Chunk] = {}
        self._chunk_order: list[str] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Plus | BM25Okapi | None = None

        if self._auto_persist and self._index_path.exists():
            self.load(self._index_path)

    @property
    def total_chunks(self) -> int:
        """Return total number of chunks currently stored in the sparse index."""
        return len(self._chunk_order)

    @property
    def index_path(self) -> Path:
        """Return configured persistence file path for the BM25 index."""
        return self._index_path

    @classmethod
    def tokenize_financial_text(cls, text: str) -> list[str]:
        """Tokenize Vietnamese financial text while preserving ratios, codes, and numbers.

        Preserves tokens such as `roe`, `eps`, `ebitda`, `p/e`, `v.01`, `-312400`, `38.450.200`,
        and adds adjacent alphabetic word bigrams (e.g., `doanh_thu`, `thu_thuần`) to boost
        exact Vietnamese financial terminology matching.
        """
        if not text or not text.strip():
            return []

        raw_matches = _TOKEN_RE.findall(text.lower())
        tokens: list[str] = []
        alpha_seq: list[str] = []

        for tok in raw_matches:
            cleaned = tok.strip(".,;:")
            if not cleaned:
                continue
            tokens.append(cleaned)

            # If token is a thousand-separated integer like 38.450.200, also index 38450200
            if _THOUSAND_SEP_NUM_RE.match(cleaned):
                compact_num = cleaned.replace(".", "").replace(",", "")
                if compact_num != cleaned:
                    tokens.append(compact_num)

            if _ALPHA_WORD_RE.match(cleaned):
                alpha_seq.append(cleaned)
            else:
                if len(alpha_seq) >= 2:
                    for i in range(len(alpha_seq) - 1):
                        tokens.append(f"{alpha_seq[i]}_{alpha_seq[i + 1]}")
                alpha_seq = []

        if len(alpha_seq) >= 2:
            for i in range(len(alpha_seq) - 1):
                tokens.append(f"{alpha_seq[i]}_{alpha_seq[i + 1]}")

        return tokens

    def _rebuild_bm25_engine(self) -> None:
        """Rebuild the in-memory rank-bm25 model from `_tokenized_corpus`."""
        if not self._tokenized_corpus:
            self._bm25 = None
            return
        if self._algorithm == "bm25_okapi":
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        else:
            self._bm25 = BM25Plus(self._tokenized_corpus)

    def index(self, chunks: list[Chunk]) -> int:
        """Build or update the BM25 inverted index from a collection of domain chunks."""
        if not chunks:
            return self.total_chunks

        try:
            for chunk in chunks:
                self._chunks_by_id[chunk.chunk_id] = chunk

            self._chunk_order = list(self._chunks_by_id.keys())
            self._tokenized_corpus = [
                self.tokenize_financial_text(self._chunks_by_id[cid].content)
                for cid in self._chunk_order
            ]
            self._rebuild_bm25_engine()

            if self._auto_persist:
                self.save(self._index_path)

            return len(chunks)
        except SparseIndexError:
            raise
        except Exception as exc:
            raise SparseIndexError(
                "Failed to build BM25 sparse index",
                details={"chunk_count": len(chunks), "error": str(exc)},
            ) from exc

    @staticmethod
    def _matches_filter(chunk: Chunk, query_filter: QueryFilter | None) -> bool:
        """Check whether a chunk satisfies all conditions in QueryFilter."""
        if query_filter is None or query_filter.is_empty:
            return True
        if query_filter.ticker is not None and chunk.metadata.ticker != query_filter.ticker:
            return False
        if query_filter.year is not None and chunk.metadata.year != query_filter.year:
            return False
        if query_filter.quarter is not None and chunk.metadata.quarter != query_filter.quarter:
            return False
        if (
            query_filter.report_type is not None
            and chunk.metadata.report_type != query_filter.report_type
        ):
            return False
        if query_filter.chunk_type is not None and chunk.chunk_type != query_filter.chunk_type:
            return False
        return True

    def search_bm25(
        self,
        query: str,
        top_k: int = 10,
        filter: QueryFilter | None = None,
    ) -> list[SearchResult]:
        """Search the sparse index using BM25 lexical matching and optional metadata filter."""
        if not query or not query.strip():
            raise SparseIndexError("BM25 search query cannot be empty or whitespace")
        if top_k <= 0:
            raise SparseIndexError(f"top_k must be positive, got {top_k}")

        if self._bm25 is None or not self._chunk_order:
            return []

        query_tokens = self.tokenize_financial_text(query)
        if not query_tokens:
            return []

        try:
            raw_scores = self._bm25.get_scores(query_tokens)
        except Exception as exc:
            raise SparseIndexError(
                "Failed to compute BM25 scores",
                details={"query": query, "error": str(exc)},
            ) from exc

        scored_candidates: list[tuple[float, Chunk]] = []
        query_token_set = set(query_tokens)

        for idx, chunk_id in enumerate(self._chunk_order):
            chunk = self._chunks_by_id[chunk_id]
            if not self._matches_filter(chunk, filter):
                continue
            doc_tokens = self._tokenized_corpus[idx]
            # Require at least one lexical token overlap so completely unrelated docs are excluded
            if not query_token_set.intersection(doc_tokens):
                continue
            score = float(raw_scores[idx])
            scored_candidates.append((score, chunk))

        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        top_candidates = scored_candidates[:top_k]

        results: list[SearchResult] = []
        for rank_idx, (score_val, chunk) in enumerate(top_candidates, start=1):
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score_val,
                    rank=rank_idx,
                    sparse_score=score_val,
                    citation=Citation.from_chunk(chunk),
                )
            )
        return results

    def save(self, target_path: Path | None = None) -> Path:
        """Persist the indexed chunks and tokenized corpus to disk (JSON or Pickle)."""
        path = target_path or self._index_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            serialized_chunks = {
                cid: chunk.model_dump(mode="json") for cid, chunk in self._chunks_by_id.items()
            }
            state_payload = {
                "algorithm": self._algorithm,
                "chunk_order": self._chunk_order,
                "tokenized_corpus": self._tokenized_corpus,
                "chunks": serialized_chunks,
            }
            if path.suffix.lower() == ".json":
                path.write_text(
                    json.dumps(state_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            else:
                with path.open("wb") as f:
                    pickle.dump(state_payload, f)
            return path
        except Exception as exc:
            raise SparseIndexError(
                f"Failed to persist BM25 index to {path}",
                details={"path": str(path), "error": str(exc)},
            ) from exc

    def load(self, source_path: Path | None = None) -> int:
        """Load persisted chunks and rebuild the BM25 index from disk."""
        path = source_path or self._index_path
        if not path.exists():
            raise SparseIndexError(
                f"BM25 index file not found: {path}",
                details={"path": str(path)},
            )
        try:
            if path.suffix.lower() == ".json":
                state_payload = json.loads(path.read_text(encoding="utf-8"))
            else:
                with path.open("rb") as f:
                    state_payload = pickle.load(f)

            raw_chunks = state_payload.get("chunks", {})
            self._chunks_by_id = {
                cid: Chunk.model_validate(cdata) for cid, cdata in raw_chunks.items()
            }
            self._chunk_order = list(state_payload.get("chunk_order", list(self._chunks_by_id)))
            self._tokenized_corpus = list(state_payload.get("tokenized_corpus", []))
            if len(self._tokenized_corpus) != len(self._chunk_order):
                self._tokenized_corpus = [
                    self.tokenize_financial_text(self._chunks_by_id[cid].content)
                    for cid in self._chunk_order
                ]
            self._rebuild_bm25_engine()
            return len(self._chunk_order)
        except SparseIndexError:
            raise
        except Exception as exc:
            raise SparseIndexError(
                f"Failed to load BM25 index from {path}",
                details={"path": str(path), "error": str(exc)},
            ) from exc
