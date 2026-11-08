"""Unit tests for application configuration loading (Phase 0 - Task 0.3)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.infrastructure.config import Settings, get_settings


def test_default_settings_load_successfully(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify default settings instantiate cleanly with expected baseline values."""
    monkeypatch.chdir(tmp_path)
    settings = Settings()
    assert settings.app_name == "FinDoc-Engine"
    assert settings.vector_dim == 768
    assert settings.chunk_size == 1024
    assert settings.chunk_overlap == 128
    assert settings.qdrant_host == "localhost"
    assert settings.qdrant_port == 6333
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.rrf_k == 60


def test_settings_load_from_dotenv_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify Settings accurately parses custom values from a .env file."""
    custom_env = tmp_path / ".env"
    custom_env.write_text(
        "\n".join(
            [
                "APP_NAME=FinDoc-Custom",
                "APP_ENV=testing",
                "LOG_LEVEL=DEBUG",
                "EMBEDDING_MODEL_NAME=BAAI/bge-m3",
                "VECTOR_DIM=1024",
                "CHUNK_SIZE=512",
                "CHUNK_OVERLAP=64",
                "QDRANT_HOST=127.0.0.1",
                "QDRANT_PORT=6335",
                "QDRANT_USE_LOCAL=true",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    settings = Settings()
    assert settings.app_name == "FinDoc-Custom"
    assert settings.app_env == "testing"
    assert settings.log_level == "DEBUG"
    assert settings.embedding_model_name == "BAAI/bge-m3"
    assert settings.vector_dim == 1024
    assert settings.chunk_size == 512
    assert settings.chunk_overlap == 64
    assert settings.qdrant_host == "127.0.0.1"
    assert settings.qdrant_port == 6335
    assert settings.qdrant_use_local is True
    assert settings.qdrant_url == "http://127.0.0.1:6335"


def test_invalid_chunk_overlap_raises_validation_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify chunk_overlap >= chunk_size raises a Pydantic ValidationError."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValidationError):
        Settings(chunk_size=256, chunk_overlap=256)


def test_get_settings_singleton_cache() -> None:
    """Verify get_settings returns a cached Settings instance."""
    get_settings.cache_clear()
    first = get_settings()
    second = get_settings()
    assert first is second
