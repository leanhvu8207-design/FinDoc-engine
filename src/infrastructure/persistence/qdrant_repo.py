"""Qdrant vector database repository implementation (Phase 3 - Task 3.2)."""

from __future__ import annotations

import uuid
import warnings
from pathlib import Path

from qdrant_client import QdrantClient, models

from src.domain.entities.document import Chunk
from src.domain.entities.query import QueryFilter
from src.domain.entities.response import Citation, SearchResult
from src.domain.exceptions import VectorStoreError
from src.domain.interfaces.i_vector_repo import IVectorRepository
from src.infrastructure.config import Settings, get_settings


class QdrantVectorRepository(IVectorRepository):
    """Concrete implementation of IVectorRepository backed by Qdrant (Local Mode or Server)."""

    PAYLOAD_INDEX_SCHEMAS: dict[str, models.PayloadSchemaType] = {
        "ticker": models.PayloadSchemaType.KEYWORD,
        "year": models.PayloadSchemaType.INTEGER,
        "quarter": models.PayloadSchemaType.INTEGER,
        "report_type": models.PayloadSchemaType.KEYWORD,
        "chunk_type": models.PayloadSchemaType.KEYWORD,
    }

    def __init__(
        self,
        settings: Settings | None = None,
        client: QdrantClient | None = None,
        collection_name: str | None = None,
        vector_dim: int | None = None,
        storage_path: Path | None = None,
        in_memory: bool = False,
    ) -> None:
        cfg = settings or get_settings()
        self._collection_name: str = collection_name or cfg.qdrant_collection_name
        self._vector_dim: int = vector_dim or cfg.vector_dim
        self._storage_path: Path = storage_path or cfg.qdrant_local_path

        try:
            if client is not None:
                self._client = client
            elif in_memory:
                self._client = QdrantClient(location=":memory:")
            elif cfg.qdrant_use_local or storage_path is not None:
                self._storage_path.mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=str(self._storage_path))
            else:
                self._client = QdrantClient(
                    host=cfg.qdrant_host,
                    port=cfg.qdrant_port,
                    grpc_port=cfg.qdrant_grpc_port,
                )

            self._ensure_collection_and_indexes()
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                "Failed to initialize QdrantVectorRepository",
                details={"collection": self._collection_name, "error": str(exc)},
            ) from exc

    @property
    def collection_name(self) -> str:
        """Return active Qdrant collection name."""
        return self._collection_name

    @property
    def client(self) -> QdrantClient:
        """Return underlying QdrantClient instance."""
        return self._client

    def close(self) -> None:
        """Close the underlying QdrantClient and release file locks in Local Mode."""
        self._client.close()

    def _ensure_collection_and_indexes(self) -> None:
        """Create the collection with Cosine metric and register payload indexes if missing."""
        if not self._client.collection_exists(collection_name=self._collection_name):
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=self._vector_dim,
                    distance=models.Distance.COSINE,
                ),
            )

        for field_name, field_schema in self.PAYLOAD_INDEX_SCHEMAS.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    self._client.create_payload_index(
                        collection_name=self._collection_name,
                        field_name=field_name,
                        field_schema=field_schema,
                    )
            except Exception:
                # Embedded local Qdrant indexes payload dynamically in SQLite/memory
                continue

    @staticmethod
    def _to_point_uuid(chunk_id: str) -> str:
        """Convert an arbitrary chunk_id string into a deterministic UUID string for Qdrant."""
        try:
            return str(uuid.UUID(chunk_id))
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    @staticmethod
    def _build_qdrant_filter(query_filter: QueryFilter | None) -> models.Filter | None:
        """Translate a domain QueryFilter into a Qdrant `models.Filter` object."""
        if query_filter is None or query_filter.is_empty:
            return None

        must_conditions: list[models.Condition] = []
        if query_filter.ticker is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="ticker",
                    match=models.MatchValue(value=query_filter.ticker),
                )
            )
        if query_filter.year is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="year",
                    match=models.MatchValue(value=query_filter.year),
                )
            )
        if query_filter.quarter is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="quarter",
                    match=models.MatchValue(value=query_filter.quarter),
                )
            )
        if query_filter.report_type is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="report_type",
                    match=models.MatchValue(value=query_filter.report_type.value),
                )
            )
        if query_filter.chunk_type is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="chunk_type",
                    match=models.MatchValue(value=query_filter.chunk_type.value),
                )
            )

        return models.Filter(must=must_conditions) if must_conditions else None

    def upsert(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> int:
        """Upsert a batch of chunks and their corresponding dense vectors into Qdrant."""
        if not chunks:
            return 0
        if len(chunks) != len(vectors):
            raise VectorStoreError(
                f"Length mismatch: {len(chunks)} chunks vs {len(vectors)} vectors",
                details={"chunks_count": len(chunks), "vectors_count": len(vectors)},
            )

        points: list[models.PointStruct] = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            if len(vector) != self._vector_dim:
                raise VectorStoreError(
                    f"Vector at index {idx} has dimension {len(vector)}, "
                    f"expected {self._vector_dim}",
                    details={"index": idx, "actual_dim": len(vector)},
                )
            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "ticker": chunk.metadata.ticker,
                "year": chunk.metadata.year,
                "quarter": chunk.metadata.quarter,
                "report_type": chunk.metadata.report_type.value,
                "chunk_type": chunk.chunk_type.value,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "content": chunk.content,
                "chunk_data": chunk.model_dump(mode="json"),
            }
            points.append(
                models.PointStruct(
                    id=self._to_point_uuid(chunk.chunk_id),
                    vector=vector,
                    payload=payload,
                )
            )

        try:
            self._client.upsert(
                collection_name=self._collection_name,
                points=points,
                wait=True,
            )
            return len(points)
        except Exception as exc:
            raise VectorStoreError(
                "Failed to upsert points into Qdrant",
                details={"count": len(points), "error": str(exc)},
            ) from exc

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter: QueryFilter | None = None,
    ) -> list[SearchResult]:
        """Search Qdrant for the top-k closest chunks using Cosine similarity and filter."""
        if len(query_vector) != self._vector_dim:
            raise VectorStoreError(
                f"Query vector dimension ({len(query_vector)}) does not match "
                f"collection dimension ({self._vector_dim})"
            )
        if top_k <= 0:
            raise VectorStoreError(f"top_k must be positive, got {top_k}")

        qdrant_filter = self._build_qdrant_filter(filter)
        try:
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreError(
                "Failed to execute vector similarity search in Qdrant",
                details={"top_k": top_k, "error": str(exc)},
            ) from exc

        results: list[SearchResult] = []
        for rank_idx, scored_point in enumerate(response.points, start=1):
            payload = scored_point.payload or {}
            chunk_data = payload.get("chunk_data")
            if not isinstance(chunk_data, dict):
                continue
            chunk = Chunk.model_validate(chunk_data)
            score_val = float(scored_point.score)
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score_val,
                    rank=rank_idx,
                    dense_score=score_val,
                    citation=Citation.from_chunk(chunk),
                )
            )

        return results

    def search_chunks(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter: QueryFilter | None = None,
    ) -> list[Chunk]:
        """Convenience method returning only the ranked domain Chunk entities."""
        results = self.search(query_vector=query_vector, top_k=top_k, filter=filter)
        return [res.chunk for res in results]
