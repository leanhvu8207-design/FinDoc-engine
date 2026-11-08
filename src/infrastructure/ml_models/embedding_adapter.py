"""Sentence-Transformers dense embedding adapter (Phase 3 - Task 3.1)."""

from __future__ import annotations

import math
from typing import Any, Protocol, cast

from src.domain.exceptions import EmbeddingError
from src.domain.interfaces.i_embedder import IEmbeddingModel
from src.infrastructure.config import Settings, get_settings


class SentenceEncoderProtocol(Protocol):
    """Structural protocol for SentenceTransformer-compatible encoders."""

    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> Any:
        """Encode a list of sentences into a 2D array or list of floats."""


class EmbeddingAdapter(IEmbeddingModel):
    """Concrete adapter for generating L2-normalized dense embeddings via sentence-transformers.

    Automatically injects `"passage: "` and `"query: "` semantic prefixes when using E5 family
    models (e.g., `intfloat/multilingual-e5-base`).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        model_name: str | None = None,
        vector_dim: int | None = None,
        device: str | None = None,
        encoder: SentenceEncoderProtocol | None = None,
    ) -> None:
        cfg = settings or get_settings()
        self._model_name: str = model_name or cfg.embedding_model_name
        self._vector_dim: int = vector_dim or cfg.vector_dim
        self._device: str = device or cfg.model_device
        self._encoder: SentenceEncoderProtocol | None = encoder
        self._use_e5_prefix: bool = "e5" in self._model_name.lower()

    @property
    def model_name(self) -> str:
        """Return configured HuggingFace model identifier."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Return output vector dimensionality."""
        return self._vector_dim

    @property
    def uses_e5_prefix(self) -> bool:
        """Return True if the configured model requires E5 'query: ' / 'passage: ' prefixes."""
        return self._use_e5_prefix

    def _get_or_load_encoder(self) -> SentenceEncoderProtocol:
        """Lazily initialize the SentenceTransformer model singleton on first inference."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer

                loaded_model = SentenceTransformer(
                    self._model_name,
                    device=self._device,
                )
                self._encoder = cast(SentenceEncoderProtocol, loaded_model)
            except Exception as exc:
                raise EmbeddingError(
                    f"Failed to load SentenceTransformer model '{self._model_name}'",
                    details={"model_name": self._model_name, "error": str(exc)},
                ) from exc
        return self._encoder

    @staticmethod
    def l2_normalize(vector: list[float]) -> list[float]:
        """Normalize a 1D vector to unit L2 norm for accurate Cosine Similarity."""
        norm = math.sqrt(sum(val * val for val in vector))
        if norm <= 1e-12:
            return list(vector)
        return [val / norm for val in vector]

    def _format_passage(self, text: str) -> str:
        """Apply E5 'passage: ' prefix if the active model belongs to the E5 family."""
        cleaned = text.strip()
        if not self._use_e5_prefix:
            return cleaned
        if cleaned.lower().startswith("passage:"):
            return cleaned
        return f"passage: {cleaned}"

    def _format_query(self, text: str) -> str:
        """Apply E5 'query: ' prefix if the active model belongs to the E5 family."""
        cleaned = text.strip()
        if not self._use_e5_prefix:
            return cleaned
        if cleaned.lower().startswith("query:"):
            return cleaned
        return f"query: {cleaned}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of document chunk texts into L2-normalized dense vectors."""
        if not texts:
            raise EmbeddingError("embed_documents requires a non-empty list of texts")

        formatted_texts: list[str] = []
        for idx, raw_text in enumerate(texts):
            if not raw_text or not raw_text.strip():
                raise EmbeddingError(
                    f"Document text at index {idx} cannot be empty or whitespace",
                    details={"index": idx},
                )
            formatted_texts.append(self._format_passage(raw_text))

        try:
            encoder = self._get_or_load_encoder()
            raw_output = encoder.encode(
                formatted_texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            if hasattr(raw_output, "tolist"):
                vectors_list: list[list[float]] = raw_output.tolist()
            else:
                vectors_list = [[float(x) for x in row] for row in raw_output]

            return [self.l2_normalize(vec) for vec in vectors_list]
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                "Failed to encode document batch",
                details={"batch_size": len(texts), "error": str(exc)},
            ) from exc

    def embed_query(self, text: str) -> list[float]:
        """Encode a single search query string into an L2-normalized dense vector."""
        if not text or not text.strip():
            raise EmbeddingError("Query text cannot be empty or whitespace")

        formatted_query = self._format_query(text)
        try:
            encoder = self._get_or_load_encoder()
            raw_output = encoder.encode(
                [formatted_query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            if hasattr(raw_output, "tolist"):
                vectors_list: list[list[float]] = raw_output.tolist()
            else:
                vectors_list = [[float(x) for x in row] for row in raw_output]

            if not vectors_list:
                raise EmbeddingError("Encoder returned empty output for query")
            return self.l2_normalize(vectors_list[0])
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                "Failed to encode search query",
                details={"query": text, "error": str(exc)},
            ) from exc
