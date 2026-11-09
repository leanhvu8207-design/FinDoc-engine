"""Application configuration management using Pydantic BaseSettings (Phase 0 - Task 0.3)."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized configuration for FinDoc-Engine loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application Metadata
    app_name: str = Field(default="FinDoc-Engine", description="Application name")
    app_env: Literal["development", "staging", "production", "testing"] = Field(
        default="development",
        description="Runtime environment",
    )
    debug: bool = Field(default=True, description="Enable debug mode")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )

    # AI Model Settings
    embedding_model_name: str = Field(
        default="intfloat/multilingual-e5-base",
        description="HuggingFace model name or local path for Dense Embeddings",
    )
    reranker_model_name: str = Field(
        default="BAAI/bge-reranker-base",
        description="HuggingFace model name or local path for Cross-Encoder Reranker",
    )
    vector_dim: int = Field(
        default=768,
        gt=0,
        description="Dimension of dense embedding vectors",
    )
    model_device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="Target hardware device for local AI models",
    )

    # Chunking & Parsing Settings
    chunk_size: int = Field(
        default=1024,
        gt=64,
        description="Maximum character/token target size per text chunk",
    )
    chunk_overlap: int = Field(
        default=128,
        ge=0,
        description="Overlap size between consecutive text chunks",
    )
    max_table_rows_per_chunk: int = Field(
        default=100,
        gt=0,
        description="Maximum rows allowed in a single table chunk before contextual splitting",
    )

    # Qdrant Vector Database Settings
    qdrant_host: str = Field(default="localhost", description="Qdrant server hostname")
    qdrant_port: int = Field(default=6333, gt=0, le=65535, description="Qdrant HTTP REST port")
    qdrant_grpc_port: int = Field(default=6334, gt=0, le=65535, description="Qdrant gRPC port")
    qdrant_collection_name: str = Field(
        default="financial_documents",
        description="Default Qdrant collection name for financial chunks",
    )
    qdrant_use_local: bool = Field(
        default=True,
        description="Use embedded local disk Qdrant instead of Docker server",
    )
    qdrant_local_path: Path = Field(
        default=Path("./data/qdrant_storage"),
        description="Filesystem directory for embedded Qdrant mode",
    )

    # LLM Settings (Gemini API)
    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API Key for answer synthesis",
    )
    gemini_model_name: str = Field(
        default="gemini-3.5-flash-lite",
        description="Google Gemini model identifier",
    )

    # Persistence Settings (DuckDB & Sparse BM25)
    duckdb_path: Path = Field(
        default=Path("./data/duckdb/findoc.duckdb"),
        description="Filesystem path to DuckDB database file",
    )
    bm25_index_path: Path = Field(
        default=Path("./data/processed/bm25_index.pkl"),
        description="Filesystem path to serialized BM25 sparse index",
    )

    # Retrieval & Numeric Guardrail Settings
    rrf_k: int = Field(
        default=60,
        gt=0,
        description="Smoothing constant k in Reciprocal Rank Fusion: 1 / (k + rank)",
    )
    default_top_k: int = Field(
        default=5,
        gt=0,
        description="Final number of top chunks returned after reranking",
    )
    rerank_top_k: int = Field(
        default=20,
        gt=0,
        description="Number of fused candidates passed into Cross-Encoder reranker",
    )
    numeric_tolerance: float = Field(
        default=1e-4,
        ge=0.0,
        description="Floating-point relative tolerance for deterministic numeric verification",
    )

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap_smaller_than_chunk_size(cls, v: int, info: object) -> int:
        """Ensure chunk_overlap is strictly smaller than chunk_size when both are present."""
        data = getattr(info, "data", {})
        chunk_size = data.get("chunk_size")
        if isinstance(chunk_size, int) and v >= chunk_size:
            raise ValueError(f"chunk_overlap ({v}) must be smaller than chunk_size ({chunk_size})")
        return v

    @property
    def qdrant_url(self) -> str:
        """Construct HTTP URL for Qdrant server connection."""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton instance of application Settings."""
    return Settings()
