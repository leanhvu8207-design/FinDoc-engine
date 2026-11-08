"""Unit tests for Domain Entities, Exceptions, and Interface Contracts (Phase 1)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.domain.entities import (
    BlockType,
    BoundingBox,
    Chunk,
    ChunkType,
    Citation,
    DocumentBlock,
    DocumentMetadata,
    FinancialDocument,
    FinancialTable,
    PageBlock,
    QueryFilter,
    ReportType,
    SearchQuery,
    SearchResult,
    TableBlock,
    TableCell,
    ValidationReport,
    ValidationStatus,
)
from src.domain.exceptions import (
    FinDocError,
    HallucinationError,
    ParsingError,
    TableExtractionError,
)
from src.domain.interfaces import (
    IDocumentParser,
    IEmbeddingModel,
    INumericValidator,
    IReranker,
    ISparseRepository,
    IVectorRepository,
)


def test_bounding_box_geometry_and_validation() -> None:
    """Verify BoundingBox computes geometry accurately and rejects inverted coordinates."""
    bbox = BoundingBox(x0=10.0, y0=20.0, x1=110.0, y1=70.0, page_number=1)
    assert bbox.width == 100.0
    assert bbox.height == 50.0
    assert bbox.area == 5000.0

    with pytest.raises(ValidationError):
        BoundingBox(x0=100.0, y0=20.0, x1=50.0, y1=70.0, page_number=1)

    with pytest.raises(ValidationError):
        BoundingBox(x0=10.0, y0=80.0, x1=100.0, y1=30.0, page_number=1)


def test_page_block_instantiation() -> None:
    """Verify PageBlock and DocumentBlock alias instantiate with valid attributes."""
    bbox = BoundingBox(x0=0.0, y0=0.0, x1=500.0, y1=40.0, page_number=2)
    block = DocumentBlock(
        page_number=2,
        block_type=BlockType.HEADING,
        content="I. TÀI SẢN NGẮN HẠN",
        section_title="Bảng cân đối kế toán",
        bounding_box=bbox,
    )
    assert isinstance(block, PageBlock)
    assert block.page_number == 2
    assert block.block_type == BlockType.HEADING
    assert block.bounding_box == bbox


def test_table_block_matrix_and_numeric_extraction() -> None:
    """Verify TableBlock validates row lengths and extracts numeric cell values."""
    cells = [
        TableCell(row_index=0, col_index=0, raw_text="Tiền mặt", numeric_value=None),
        TableCell(row_index=0, col_index=1, raw_text="1.250.000", numeric_value=1250000.0),
        TableCell(row_index=1, col_index=1, raw_text="(45.000)", numeric_value=-45000.0),
    ]
    table = FinancialTable(
        page_number=3,
        end_page_number=4,
        title="Bảng cân đối kế toán hợp nhất",
        currency_unit="Triệu VND",
        headers=["Chỉ tiêu", "Số cuối kỳ"],
        rows=[
            ["Tiền mặt", "1.250.000"],
            ["Dự phòng giảm giá", "-45.000"],
        ],
        cells=cells,
        raw_markdown="| Chỉ tiêu | Số cuối kỳ |\n|---|---|\n| Tiền mặt | 1.250.000 |",
    )
    assert isinstance(table, TableBlock)
    assert table.row_count == 2
    assert table.col_count == 2
    assert table.extract_all_numbers() == [1250000.0, -45000.0]

    # Row column mismatch should fail validation
    with pytest.raises(ValidationError):
        TableBlock(
            page_number=1,
            headers=["Cột 1", "Cột 2"],
            rows=[["Chỉ có 1 cột"]],
            raw_markdown="| Cột 1 | Cột 2 |",
        )

    # Invalid end_page_number < page_number should fail validation
    with pytest.raises(ValidationError):
        TableBlock(
            page_number=5,
            end_page_number=3,
            raw_markdown="| A | B |",
        )


def test_document_metadata_and_chunk_consistency() -> None:
    """Verify DocumentMetadata normalizes ticker and Chunk enforces table consistency."""
    meta = DocumentMetadata(
        ticker=" fpt ",
        year=2025,
        quarter=4,
        report_type=ReportType.CONSOLIDATED,
        company_name="CTCP FPT",
    )
    assert meta.ticker == "FPT"
    header = meta.format_context_header(section_title="Thuyết minh số 5")
    expected_header = (
        "[Ticker: FPT | Period: Q4/2025 | Type: CONSOLIDATED | Section: Thuyết minh số 5]"
    )
    assert header == expected_header

    with pytest.raises(ValidationError):
        DocumentMetadata(ticker="FPT@#$", year=2025)

    table = TableBlock(
        page_number=5,
        headers=["Chỉ tiêu", "Giá trị"],
        rows=[["Doanh thu thuần", "62.849"]],
        raw_markdown="| Chỉ tiêu | Giá trị |\n|---|---|\n| Doanh thu thuần | 62.849 |",
    )

    table_chunk = Chunk(
        document_id="doc-001",
        content=f"{header}\n{table.raw_markdown}",
        chunk_type=ChunkType.TABLE,
        metadata=meta,
        page_number=5,
        section_title="Thuyết minh số 5",
        table_block=table,
    )
    assert table_chunk.chunk_type == ChunkType.TABLE
    assert table_chunk.table_block is table

    # Attaching a table_block to a TEXT chunk must raise ValidationError
    with pytest.raises(ValidationError):
        Chunk(
            document_id="doc-001",
            content="Đoạn văn bản thông thường",
            chunk_type=ChunkType.TEXT,
            metadata=meta,
            page_number=5,
            table_block=table,
        )


def test_financial_document_aggregate() -> None:
    """Verify FinancialDocument aggregate root initializes cleanly."""
    meta = DocumentMetadata(ticker="VNM", year=2025, report_type=ReportType.ANNUAL)
    doc = FinancialDocument(
        metadata=meta,
        file_path=Path("data/raw_pdfs/VNM_2025.pdf"),
        total_pages=68,
    )
    assert doc.metadata.ticker == "VNM"
    assert doc.total_pages == 68
    assert doc.blocks == []
    assert doc.tables == []
    assert doc.chunks == []


def test_search_query_and_filter_validation() -> None:
    """Verify QueryFilter and SearchQuery enforce domain constraints."""
    empty_filter = QueryFilter()
    assert empty_filter.is_empty is True

    active_filter = QueryFilter(ticker=" hpg ", year=2025, quarter=3)
    assert active_filter.is_empty is False
    assert active_filter.ticker == "HPG"

    query = SearchQuery(
        query_text="  Lợi nhuận sau thuế của HPG quý 3/2025 là bao nhiêu?  ",
        top_k=5,
        query_filter=active_filter,
    )
    assert query.query_text == "Lợi nhuận sau thuế của HPG quý 3/2025 là bao nhiêu?"
    assert query.top_k == 5

    with pytest.raises(ValidationError):
        SearchQuery(query_text="   ")


def test_citation_search_result_and_validation_report() -> None:
    """Verify Citation creation from Chunk, SearchResult, and ValidationReport logic."""
    meta = DocumentMetadata(ticker="FPT", year=2025, quarter=2)
    chunk = Chunk(
        chunk_id="chunk-123",
        document_id="doc-456",
        content="Doanh thu mảng Công nghệ đạt 18.450 tỷ đồng trong 6 tháng đầu năm.",
        chunk_type=ChunkType.TEXT,
        metadata=meta,
        page_number=12,
        section_title="Báo cáo bộ phận",
    )
    citation = Citation.from_chunk(chunk, max_excerpt_length=30)
    assert citation.chunk_id == "chunk-123"
    assert citation.ticker == "FPT"
    assert citation.page_number == 12
    assert len(citation.excerpt) <= 30

    result = SearchResult(
        chunk=chunk,
        score=0.92,
        rank=1,
        dense_score=0.88,
        sparse_score=14.2,
        rrf_score=0.032,
        rerank_score=0.92,
        citation=citation,
    )
    assert result.rank == 1
    assert result.citation == citation

    passed_report = ValidationReport(
        status=ValidationStatus.PASSED,
        extracted_numbers=[18450.0],
        verified_numbers=[18450.0],
        unverified_numbers=[],
        message="All numbers verified against source tables.",
    )
    assert passed_report.is_valid is True

    failed_report = ValidationReport(
        status=ValidationStatus.FAILED,
        extracted_numbers=[18450.0, 99999.0],
        verified_numbers=[18450.0],
        unverified_numbers=[99999.0],
        message="Detected unverified number: 99999.0",
    )
    assert failed_report.is_valid is False


def test_domain_exceptions_hierarchy() -> None:
    """Verify custom domain exceptions preserve details and inheritance."""
    err = TableExtractionError("Failed to detect wireframe", details={"page": 7})
    assert isinstance(err, ParsingError)
    assert isinstance(err, FinDocError)
    assert "page" in str(err)

    hallucination_err = HallucinationError(
        "Numeric mismatch detected",
        unverified_numbers=[42.5, 108.9],
    )
    assert isinstance(hallucination_err, FinDocError)
    assert hallucination_err.unverified_numbers == [42.5, 108.9]
    assert hallucination_err.details["unverified_numbers"] == [42.5, 108.9]


def test_interfaces_cannot_be_instantiated_directly() -> None:
    """Verify all 6 domain interfaces enforce ABC contract."""
    abstract_interfaces = [
        IDocumentParser,
        IEmbeddingModel,
        IVectorRepository,
        ISparseRepository,
        IReranker,
        INumericValidator,
    ]
    for interface_cls in abstract_interfaces:
        with pytest.raises(TypeError):
            interface_cls()
