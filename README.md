# FinDoc-Engine

**Multi-Modal Financial Parsing, Deterministic Hybrid Retrieval & Numeric Fact-Checking System**

Hệ thống trích xuất, lập chỉ mục và truy vấn ngữ nghĩa chuyên sâu cho Báo cáo Tài chính (BCTC) định dạng PDF của các doanh nghiệp niêm yết Việt Nam.

## Trạng thái Triển khai (WBS Roadmap)
- [x] **Giai đoạn 0: Thiết lập Hạ tầng & Môi trường (Foundation & Scaffolding)**
  - [x] Task 0.1: Khởi tạo Project, cấu trúc Clean Architecture & `pyproject.toml`
  - [x] Task 0.2: Cấu hình `docker-compose.yml` (Qdrant) & `Dockerfile`
  - [x] Task 0.3: Xây dựng Cấu hình & Base Settings (`src/infrastructure/config.py`) + Unit Tests
- [x] **Giai đoạn 1: Tầng Domain & Dữ liệu Cốt lõi (Domain Modeling)**
  - [x] Task 1.1: Định nghĩa các Thực thể Nghiệp vụ (`src/domain/entities/`) & Custom Exceptions (`src/domain/exceptions.py`)
  - [x] Task 1.2: Định nghĩa các Giao diện Cốt lõi (`src/domain/interfaces/`) + Unit Tests (`tests/unit/test_domain_entities.py`)
- [ ] **Giai đoạn 2: Parser Đa phương thức & Table Extraction (Ingestion Pipeline)**
- [ ] **Giai đoạn 3: Tầng Lưu trữ & Lập Chỉ mục (Storage & Vector Indexing)**
- [ ] **Giai đoạn 4: Hybrid Retrieval, Reranking & Numeric Guardrails**
- [ ] **Giai đoạn 5: Tầng API, Packaging & Đánh giá Định lượng (Evaluation Suite)**

## Khởi chạy nhanh (Bước 0)

```powershell
# 1. Kích hoạt môi trường ảo
.\.venv\Scripts\Activate.ps1

# 2. Kiểm tra Linter, Type Checker và Unit Tests
ruff check src tests
mypy src tests
pytest
```
