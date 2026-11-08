"""Unit tests for deterministic NumericGuardrail (Phase 4 - Task 4.3)."""

from __future__ import annotations

import pytest

from src.domain.entities.response import ValidationStatus
from src.domain.entities.table import TableBlock, TableCell
from src.domain.exceptions import HallucinationError
from src.infrastructure.validators.numeric_guardrail import NumericGuardrail


def _make_source_table() -> TableBlock:
    """Create a ground-truth TableBlock with positive, negative, and percentage numbers."""
    cells = [
        TableCell(row_index=1, col_index=0, raw_text="Doanh thu thuần", numeric_value=None),
        TableCell(row_index=1, col_index=1, raw_text="62.724.500", numeric_value=62724500.0),
        TableCell(row_index=2, col_index=0, raw_text="Dự phòng phải thu", numeric_value=None),
        TableCell(row_index=2, col_index=1, raw_text="-312400", numeric_value=-312400.0),
        TableCell(row_index=3, col_index=0, raw_text="Lợi nhuận sau thuế", numeric_value=None),
        TableCell(row_index=3, col_index=1, raw_text="9.420.800", numeric_value=9420800.0),
        TableCell(row_index=4, col_index=0, raw_text="Biên lợi nhuận ròng", numeric_value=None),
        TableCell(row_index=4, col_index=1, raw_text="15,02%", numeric_value=15.02),
    ]
    return TableBlock(
        page_number=2,
        title="Báo cáo kết quả hoạt động kinh doanh hợp nhất",
        headers=["Chỉ tiêu", "Số cuối kỳ"],
        rows=[
            ["Doanh thu thuần", "62.724.500"],
            ["Dự phòng phải thu", "-312400"],
            ["Lợi nhuận sau thuế", "9.420.800"],
            ["Biên lợi nhuận ròng", "15,02%"],
        ],
        cells=cells,
        raw_markdown=(
            "| Chỉ tiêu | Số cuối kỳ |\n| --- | --- |\n"
            "| Doanh thu thuần | 62.724.500 |\n"
            "| Dự phòng phải thu | -312400 |\n"
            "| Lợi nhuận sau thuế | 9.420.800 |\n"
            "| Biên lợi nhuận ròng | 15,02% |"
        ),
    )


def test_numeric_guardrail_passes_when_all_numbers_match_source_table() -> None:
    """Verify answer containing exact numbers (negative and %) from source table passes."""
    guardrail = NumericGuardrail()
    source_table = _make_source_table()

    answer = (
        "Trong Quý 4/2025 (Trang 2, Thuyết minh V.01), Doanh thu thuần của FPT đạt "
        "62.724.500 triệu đồng, khoản dự phòng phải thu là (312.400) triệu đồng, "
        "và Lợi nhuận sau thuế đạt 9.420.800 triệu đồng (biên lợi nhuận ròng 15,02%)."
    )
    report = guardrail.validate(answer=answer, source_tables=[source_table])

    assert report.status == ValidationStatus.PASSED
    assert report.is_valid is True
    assert report.verified_numbers == [62724500.0, -312400.0, 9420800.0, 15.02]
    assert report.hallucinated_numbers == []
    assert report.unverified_numbers == []


def test_numeric_guardrail_fails_and_identifies_hallucinated_numbers() -> None:
    """Verify fabricated numbers absent from source tables trigger FAILED and record values."""
    guardrail = NumericGuardrail()
    source_table = _make_source_table()

    # 62.724.500 is real, but 99.888.777 and 42,5% are fabricated
    hallucinated_answer = (
        "Doanh thu thuần đạt 62.724.500 triệu đồng nhưng lợi nhuận sau thuế "
        "tăng vọt lên 99.888.777 triệu đồng với biên lợi nhuận 42,5%."
    )
    report = guardrail.validate(answer=hallucinated_answer, source_tables=[source_table])

    assert report.status == ValidationStatus.FAILED
    assert report.is_valid is False
    assert report.verified_numbers == [62724500.0]
    assert report.hallucinated_numbers == [99888777.0, 42.5]
    assert report.unverified_numbers == [99888777.0, 42.5]

    with pytest.raises(HallucinationError) as exc_info:
        guardrail.validate_or_raise(answer=hallucinated_answer, source_tables=[source_table])
    assert exc_info.value.unverified_numbers == [99888777.0, 42.5]


def test_numeric_guardrail_passes_when_answer_contains_no_numbers() -> None:
    """Verify qualitative answers with no numeric claims pass validation cleanly."""
    guardrail = NumericGuardrail()
    source_table = _make_source_table()

    qualitative_answer = (
        "Báo cáo tài chính hợp nhất trình bày tình hình tài sản và kết quả kinh doanh của tập đoàn."
    )
    report = guardrail.validate(answer=qualitative_answer, source_tables=[source_table])

    assert report.status == ValidationStatus.PASSED
    assert report.is_valid is True
    assert report.extracted_numbers == []
    assert report.hallucinated_numbers == []
