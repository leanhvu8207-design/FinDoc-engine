# TIẾN ĐỘ TRIỂN KHAI DỰ ÁN FINDOC-ENGINE (PROJECT PROGRESS)

## Giai đoạn 0: Thiết lập Hạ tầng & Môi trường (Foundation & Scaffolding)
- [x] **Task 0.1**: Khởi tạo Project & Quản lý Dependencies (`pyproject.toml`, `.venv`, `uv`, `ruff`, `mypy`, `pytest`)
- [x] **Task 0.2**: Cấu hình Container Hạ tầng (`docker-compose.yml`, `Dockerfile` & chế độ `QDRANT_USE_LOCAL=true`)
- [x] **Task 0.3**: Xây dựng Cấu hình & Base Settings (`src/infrastructure/config.py`, `tests/unit/test_config.py`)

## Giai đoạn 1: Tầng Domain & Dữ liệu Cốt lõi (Domain Modeling)
- [x] **Task 1.1**: Định nghĩa các Thực thể Nghiệp vụ (`src/domain/entities/`) & Custom Exceptions (`src/domain/exceptions.py`)
- [x] **Task 1.2**: Định nghĩa các Giao diện Cốt lõi (`src/domain/interfaces/`) & Unit Tests (`tests/unit/test_domain_entities.py`)

## Giai đoạn 2: Parser Đa phương thức & Table Extraction (Ingestion Pipeline)
- [x] **Task 2.1**: Thu thập Dataset Báo cáo Tài chính Mẫu (`scripts/download_sample_bctc.py` -> `data/raw_pdfs/`)
- [x] **Task 2.2**: Module Nhận diện & Trích xuất Bảng Biểu (`src/infrastructure/parsers/table_extractor.py`, `src/infrastructure/parsers/pdfplumber_adapter.py`, `tests/unit/test_table_extractor.py`)
- [x] **Task 2.3**: Module Chunking Giữ Ngữ cảnh Tài chính (`src/application/services/financial_chunker.py`, `tests/unit/test_financial_chunker.py`)

## Giai đoạn 3: Tầng Lưu trữ & Lập Chỉ mục (Storage & Vector Indexing)
- [x] **Task 3.1**: Adapter cho Embedding Model (`src/infrastructure/ml_models/embedding_adapter.py`, `tests/unit/test_embedding_adapter.py`)
- [x] **Task 3.2**: Adapter cho Qdrant Vector Database (`src/infrastructure/persistence/qdrant_repo.py`, `tests/unit/test_qdrant_repo.py`)
- [x] **Task 3.3**: Adapter cho Sparse Index BM25 (`src/infrastructure/persistence/bm25_repo.py`, `tests/unit/test_bm25_repo.py`)
- [x] **Task 3.4**: Use Case Ingest Document Hoàn chỉnh (`src/application/use_cases/ingest_document.py`, `tests/integration/test_ingest_use_case.py`)

## Giai đoạn 4: Hybrid Retrieval, Reranking & Numeric Guardrails
- [ ] **Task 4.1**: Thuật toán Hợp nhất Điểm số Reciprocal Rank Fusion (`src/application/services/fusion_service.py`)
- [ ] **Task 4.2**: Adapter cho Cross-Encoder Reranker (`src/infrastructure/ml_models/reranker_adapter.py`)
- [ ] **Task 4.3**: Bộ Kiểm tra Số học Chống Ảo giác (`src/infrastructure/validators/numeric_guardrail.py`)
- [ ] **Task 4.4**: Use Case Hybrid Search & Answering (`src/application/use_cases/hybrid_search.py`, `src/application/use_cases/answer_with_verify.py`)

## Giai đoạn 5: Tầng API, Packaging & Đánh giá Định lượng (Evaluation Suite)
- [ ] **Task 5.1**: Xây dựng REST API bằng FastAPI (`src/presentation/`)
- [ ] **Task 5.2**: Xây dựng Bộ Đánh giá Định lượng (`tests/evaluation/`)
- [ ] **Task 5.3**: Hoàn thiện Tài liệu Kỹ thuật (`README.md`)
