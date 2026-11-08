"""Unit tests for TableExtractor and PdfPlumberAdapter (Phase 2 - Task 2.1 & 2.2)."""

from pathlib import Path

import pytest

from scripts.download_sample_bctc import ensure_sample_bctc_pdfs
from src.domain.entities.document import DocumentMetadata, ReportType
from src.domain.entities.page import BlockType
from src.domain.exceptions import ParsingError, TableExtractionError
from src.infrastructure.parsers import PdfPlumberAdapter, TableExtractor


def test_accounting_negative_parentheses_normalization() -> None:
    """Verify accounting negative numbers in parentheses convert to standard negative strings."""
    assert TableExtractor.normalize_cell_text("(123.456)") == "-123456"
    assert TableExtractor.normalize_cell_text("(123,456)") == "-123456"
    assert TableExtractor.normalize_cell_text(" ( 38.410.000 ) ") == "-38410000"
    assert TableExtractor.normalize_cell_text("(1,250,500)") == "-1250500"
    assert TableExtractor.normalize_cell_text("(12,5%)") == "-12.5%"
    assert TableExtractor.normalize_cell_text("Doanh thu\n thuần") == "Doanh thu thuần"
    assert TableExtractor.normalize_cell_text(None) == ""


def test_parse_financial_number_variants() -> None:
    """Verify numeric parsing handles VN/US separators, negatives, and non-numeric labels."""
    assert TableExtractor.parse_financial_number("(123.456)") == -123456.0
    assert TableExtractor.parse_financial_number("(123,456)") == -123456.0
    assert TableExtractor.parse_financial_number("-123456") == -123456.0
    assert TableExtractor.parse_financial_number("38.450.200") == 38450200.0
    assert TableExtractor.parse_financial_number("1,234,567.89") == 1234567.89
    assert TableExtractor.parse_financial_number("1.234,56") == 1234.56
    assert TableExtractor.parse_financial_number("15,75%") == 15.75
    assert TableExtractor.parse_financial_number("-") is None
    assert TableExtractor.parse_financial_number("V.01") is None
    assert TableExtractor.parse_financial_number("I. Tiền và tương đương tiền") is None


def test_multi_level_header_merging() -> None:
    """Verify 2-level hierarchical headers are forward-filled and joined accurately."""
    raw_matrix: list[list[str | None]] = [
        ["Chỉ tiêu", "Mã số", "Số cuối năm", "", "Số đầu năm"],
        ["", "", "Tập đoàn", "Công ty mẹ", "Tập đoàn"],
        ["Tiền mặt", "110", "9.320.150", "4.100.000", "8.140.000"],
        ["Dự phòng", "137", "(312.400)", "(120.000)", "(285.100)"],
    ]
    table_block = TableExtractor.build_table_block(
        raw_matrix=raw_matrix,
        page_number=2,
        title="Bảng cân đối kế toán",
        currency_unit="Triệu VND",
    )

    assert table_block.headers == [
        "Chỉ tiêu",
        "Mã số",
        "Số cuối năm - Tập đoàn",
        "Số cuối năm - Công ty mẹ",
        "Số đầu năm - Tập đoàn",
    ]
    assert table_block.row_count == 2
    assert table_block.col_count == 5
    assert table_block.rows[1] == ["Dự phòng", "137", "-312400", "-120000", "-285100"]

    extracted_nums = table_block.extract_all_numbers()
    assert extracted_nums == [
        110.0,
        9320150.0,
        4100000.0,
        8140000.0,
        137.0,
        -312400.0,
        -120000.0,
        -285100.0,
    ]
    assert "| Dự phòng | 137 | -312400 | -120000 | -285100 |" in table_block.raw_markdown


def test_empty_table_raises_extraction_error() -> None:
    """Verify empty or all-blank matrices raise TableExtractionError."""
    with pytest.raises(TableExtractionError):
        TableExtractor.build_table_block(raw_matrix=[], page_number=1)

    with pytest.raises(TableExtractionError):
        TableExtractor.build_table_block(raw_matrix=[[None, "  "]], page_number=1)


def test_pdfplumber_adapter_end_to_end_on_sample_bctc(tmp_path: Path) -> None:
    """Verify PdfPlumberAdapter extracts wireframe tables, negative numbers, and blocks from PDF."""
    saved_pdfs = ensure_sample_bctc_pdfs(output_dir=tmp_path)
    assert len(saved_pdfs) >= 3

    fpt_pdf = tmp_path / "FPT_2025_Q4_Consolidated.pdf"
    assert fpt_pdf.exists()

    adapter = PdfPlumberAdapter()
    tables = adapter.extract_tables(fpt_pdf)
    assert len(tables) == 2

    balance_sheet = tables[0]
    assert balance_sheet.page_number == 1
    assert balance_sheet.col_count == 5
    assert balance_sheet.row_count == 9
    assert balance_sheet.headers == [
        "Chi tieu",
        "Ma so",
        "Thuyet minh",
        "So cuoi ky",
        "So dau nam",
    ]

    bs_numbers = balance_sheet.extract_all_numbers()
    # Verify accounting negative numbers from parentheses (312.400) and (45.200) were extracted
    assert -312400.0 in bs_numbers
    assert -285100.0 in bs_numbers
    assert -45200.0 in bs_numbers
    assert 65341000.0 in bs_numbers
    assert "| Du phong phai thu ngan han kho doi | 137 | V.04 | -312400 | -285100 |" in (
        balance_sheet.raw_markdown
    )

    # Verify full document parsing
    meta = DocumentMetadata(
        ticker="FPT",
        year=2025,
        quarter=4,
        report_type=ReportType.CONSOLIDATED,
    )
    doc = adapter.parse_document(file_path=fpt_pdf, metadata=meta)
    assert doc.total_pages == 2
    assert len(doc.tables) == 2
    assert any(b.block_type == BlockType.TABLE for b in doc.blocks)
    assert any(b.block_type == BlockType.HEADING for b in doc.blocks)


def test_pdfplumber_adapter_missing_file_raises_parsing_error(tmp_path: Path) -> None:
    """Verify missing PDF path raises domain ParsingError."""
    adapter = PdfPlumberAdapter()
    missing_file = tmp_path / "non_existent.pdf"
    with pytest.raises(ParsingError):
        adapter.extract_tables(missing_file)
