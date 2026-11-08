"""Ingest document pipeline: Parse -> Chunk -> Embed -> Index (Phase 3 - Task 3.4)."""

from __future__ import annotations

from pathlib import Path

from src.application.dto.ingest_dto import IngestRequest, IngestResponse
from src.application.services.financial_chunker import FinancialChunker
from src.domain.entities.document import ChunkType, DocumentMetadata
from src.domain.exceptions import EmbeddingError
from src.domain.interfaces.i_embedder import IEmbeddingModel
from src.domain.interfaces.i_parser import IDocumentParser
from src.domain.interfaces.i_sparse_repo import ISparseRepository
from src.domain.interfaces.i_vector_repo import IVectorRepository


class IngestDocumentUseCase:
    """Orchestrates end-to-end ingestion of a PDF financial statement into Vector and BM25 stores.

    Addresses Technical Debts:
    - [TD-01]: Strictly verifies that generated embedding vectors match `embedder.dimension`
      and chunk batch size prior to calling `vector_repo.upsert`.
    - [TD-03]: Receives injected singleton repositories (`vector_repo`, `sparse_repo`) via
      Dependency Inversion, preventing duplicate Qdrant Local Mode file lock contention.
    """

    def __init__(
        self,
        parser: IDocumentParser,
        chunker: FinancialChunker,
        embedder: IEmbeddingModel,
        vector_repo: IVectorRepository,
        sparse_repo: ISparseRepository,
    ) -> None:
        if embedder.dimension <= 0:
            raise EmbeddingError(
                f"Invalid embedder dimension: {embedder.dimension}",
                details={"dimension": embedder.dimension},
            )
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._vector_repo = vector_repo
        self._sparse_repo = sparse_repo

    def _validate_embeddings(
        self,
        embeddings: list[list[float]],
        expected_count: int,
    ) -> None:
        """Verify vector batch count and dimensionality consistency ([TD-01] safeguard)."""
        if len(embeddings) != expected_count:
            raise EmbeddingError(
                f"Embedder returned {len(embeddings)} vectors for {expected_count} chunks",
                details={"expected_count": expected_count, "actual_count": len(embeddings)},
            )
        expected_dim = self._embedder.dimension
        for idx, vec in enumerate(embeddings):
            if len(vec) != expected_dim:
                raise EmbeddingError(
                    f"Embedding vector at index {idx} has dimension {len(vec)}, "
                    f"expected {expected_dim} ([TD-01] dimensionality mismatch)",
                    details={
                        "index": idx,
                        "expected_dim": expected_dim,
                        "actual_dim": len(vec),
                    },
                )

    def execute(
        self,
        file_path: Path,
        metadata: DocumentMetadata,
    ) -> IngestResponse:
        """Run the full ingestion pipeline for a single PDF document and metadata."""
        # Step 1: Parse PDF into blocks and wireframe tables
        document = self._parser.parse_document(file_path=file_path, metadata=metadata)

        # Step 2: Context-aware chunking (preserving intact tables + prepending context headers)
        chunks = self._chunker.chunk_document(document)
        document.chunks = chunks

        text_chunks_count = sum(1 for c in chunks if c.chunk_type == ChunkType.TEXT)
        table_chunks_count = sum(1 for c in chunks if c.chunk_type == ChunkType.TABLE)

        if not chunks:
            return IngestResponse(
                document_id=document.document_id,
                ticker=metadata.ticker,
                year=metadata.year,
                quarter=metadata.quarter,
                total_pages=document.total_pages,
                total_tables=len(document.tables),
                total_chunks=0,
                text_chunks_count=0,
                table_chunks_count=0,
                vector_upserted_count=0,
                sparse_indexed_count=0,
            )

        # Step 3: Dense vectorization & [TD-01] dimensionality verification
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = self._embedder.embed_documents(chunk_texts)
        self._validate_embeddings(embeddings=embeddings, expected_count=len(chunks))

        # Step 4: Dual persistence into Qdrant Vector DB & BM25 Sparse Index
        vector_upserted = self._vector_repo.upsert(chunks=chunks, vectors=embeddings)
        sparse_indexed = self._sparse_repo.index(chunks=chunks)

        return IngestResponse(
            document_id=document.document_id,
            ticker=metadata.ticker,
            year=metadata.year,
            quarter=metadata.quarter,
            total_pages=document.total_pages,
            total_tables=len(document.tables),
            total_chunks=len(chunks),
            text_chunks_count=text_chunks_count,
            table_chunks_count=table_chunks_count,
            vector_upserted_count=vector_upserted,
            sparse_indexed_count=sparse_indexed,
        )

    def execute_request(self, request: IngestRequest) -> IngestResponse:
        """Execute ingestion pipeline from an IngestRequest DTO."""
        return self.execute(file_path=request.file_path, metadata=request.metadata)
