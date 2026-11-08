"""Financial table wireframe extraction, accounting number normalization, and Markdown conversion
(Phase 2 - Task 2.2).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from src.domain.entities.page import BoundingBox
from src.domain.entities.table import TableBlock, TableCell
from src.domain.exceptions import TableExtractionError

if TYPE_CHECKING:
    from pdfplumber.page import Page

# Matches accounting negative numbers in parentheses, e.g. (123.456), (123,456), (12,5%)
_ACCOUNTING_NEG_PAREN_RE = re.compile(
    r"^\(\s*(\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)\s*(%?)\s*\)$"
)

# Matches standalone financial numbers (with optional '-', thousand separators, or decimals)
_FINANCIAL_NUMBER_RE = re.compile(
    r"^[+-]?\s*(?:\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)\s*%?$"
)


class TableExtractor:
    """Extracts and normalizes complex accounting tables into structured TableBlock entities."""

    @staticmethod
    def _parse_number_token(token: str) -> float:
        """Convert a raw numeric string token (Vietnamese or international format) to float."""
        cleaned = token.strip().replace(" ", "")
        # Case 1: Multiple dots (e.g., 38.450.200) -> dots are thousand separators
        if cleaned.count(".") > 1 and "," not in cleaned:
            return float(cleaned.replace(".", ""))
        # Case 2: Multiple commas (e.g., 38,450,200) -> commas are thousand separators
        if cleaned.count(",") > 1 and "." not in cleaned:
            return float(cleaned.replace(",", ""))
        # Case 3: Both dot and comma present (e.g., 1.234,56 or 1,234.56)
        if "." in cleaned and "," in cleaned:
            if cleaned.rfind(",") > cleaned.rfind("."):
                # 1.234,56 -> 1234.56
                return float(cleaned.replace(".", "").replace(",", "."))
            # 1,234.56 -> 1234.56
            return float(cleaned.replace(",", ""))
        # Case 4: Single dot or single comma followed by exactly 3 digits (e.g., 123.456 or 123,456)
        single_sep_match = re.match(r"^([+-]?\d{1,3})[.,](\d{3})$", cleaned)
        if single_sep_match:
            return float(f"{single_sep_match.group(1)}{single_sep_match.group(2)}")
        # Case 5: Single comma as decimal separator (e.g., 12,5 -> 12.5)
        if "," in cleaned:
            return float(cleaned.replace(",", "."))
        return float(cleaned)

    @classmethod
    def normalize_cell_text(cls, raw_text: str | None) -> str:
        """Clean cell whitespace and convert accounting negative `(123.456)` -> `-123456`.

        Args:
            raw_text: Raw cell string extracted by PDF parser.

        Returns:
            Normalized single-line cell string with accounting negative parentheses standardized.
        """
        if raw_text is None:
            return ""
        compact = " ".join(raw_text.replace("\r", "\n").split()).strip()
        if not compact:
            return ""

        neg_match = _ACCOUNTING_NEG_PAREN_RE.match(compact)
        if neg_match:
            number_body, percent_sign = neg_match.group(1), neg_match.group(2)
            numeric_val = -abs(cls._parse_number_token(number_body))
            if numeric_val.is_integer():
                formatted = str(int(numeric_val))
            else:
                formatted = f"{numeric_val:g}"
            return f"{formatted}{percent_sign}"

        return compact

    @classmethod
    def parse_financial_number(cls, text: str | None) -> float | None:
        """Parse a normalized or raw cell string into a float if it represents a numeric value.

        Args:
            text: Raw or normalized cell string.

        Returns:
            Parsed float (negative for accounting parentheses), or None if cell is non-numeric.
        """
        if text is None:
            return None
        compact = " ".join(text.split()).strip()
        if not compact or compact in {"-", "--", "—", "N/A", "n/a"}:
            return None

        neg_match = _ACCOUNTING_NEG_PAREN_RE.match(compact)
        if neg_match:
            return -abs(cls._parse_number_token(neg_match.group(1)))

        if _FINANCIAL_NUMBER_RE.match(compact):
            stripped_pct = compact.rstrip("%").strip()
            sign = -1.0 if stripped_pct.startswith("-") else 1.0
            unsigned = stripped_pct.lstrip("+-").strip()
            return sign * cls._parse_number_token(unsigned)

        return None

    @classmethod
    def _is_sub_header_row(cls, row: list[str]) -> bool:
        """Return True if a row is a secondary multi-level header row (no numeric data)."""
        non_empty = [cell for cell in row if cell]
        if not non_empty:
            return False
        return all(cls.parse_financial_number(cell) is None for cell in non_empty)

    @classmethod
    def merge_headers(
        cls,
        cleaned_rows: list[list[str]],
        header_row_count: int | None = None,
    ) -> tuple[list[str], list[list[str]]]:
        """Extract and flatten single or multi-level headers from cleaned table rows."""
        if not cleaned_rows:
            raise TableExtractionError("Cannot extract headers from an empty table matrix")

        num_cols = max(len(r) for r in cleaned_rows)
        padded_rows = [r + [""] * (num_cols - len(r)) for r in cleaned_rows]

        if header_row_count is None:
            # Automatically detect 2-row hierarchical header if row 0 has blank cells (colspan)
            # and row 1 contains only non-numeric labels.
            has_span_gaps = any(cell == "" for cell in padded_rows[0][1:])
            if (
                len(padded_rows) >= 3
                and has_span_gaps
                and cls._is_sub_header_row(padded_rows[1])
            ):
                header_row_count = 2
            else:
                header_row_count = 1

        if header_row_count <= 1 or len(padded_rows) < header_row_count:
            headers = [
                cell if cell else f"Col_{idx + 1}"
                for idx, cell in enumerate(padded_rows[0])
            ]
            return headers, padded_rows[1:]

        # Forward-fill top header row across horizontal spans
        top_row: list[str] = []
        last_seen = ""
        for idx, cell in enumerate(padded_rows[0]):
            if cell:
                last_seen = cell
                top_row.append(cell)
            elif idx > 0 and padded_rows[1][idx]:
                top_row.append(last_seen)
            else:
                top_row.append("")

        sub_row = padded_rows[1]
        merged_headers: list[str] = []
        for idx in range(num_cols):
            top_val = top_row[idx]
            sub_val = sub_row[idx]
            if top_val and sub_val and top_val != sub_val:
                merged_headers.append(f"{top_val} - {sub_val}")
            elif top_val or sub_val:
                merged_headers.append(top_val or sub_val)
            else:
                merged_headers.append(f"Col_{idx + 1}")

        return merged_headers, padded_rows[header_row_count:]

    @staticmethod
    def to_markdown(headers: list[str], rows: list[list[str]]) -> str:
        """Render table headers and rows into a standard GitHub-Flavored Markdown table."""
        escaped_headers = [h.replace("|", "\\|") for h in headers]
        header_line = "| " + " | ".join(escaped_headers) + " |"
        separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        body_lines = [
            "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"
            for row in rows
        ]
        return "\n".join([header_line, separator_line, *body_lines])

    @classmethod
    def build_table_block(
        cls,
        raw_matrix: list[list[str | None]],
        page_number: int,
        title: str | None = None,
        currency_unit: str | None = "VND",
        bounding_box: BoundingBox | None = None,
        header_row_count: int | None = None,
    ) -> TableBlock:
        """Convert a raw 2D string matrix into a validated domain TableBlock entity.

        Args:
            raw_matrix: 2D list of raw cell strings from PDF table detection.
            page_number: 1-based page number.
            title: Optional section or table title.
            currency_unit: Currency unit of the table.
            bounding_box: Optional BoundingBox coordinates on the page.
            header_row_count: Explicit number of header rows (None for auto-detection).

        Returns:
            Populated TableBlock entity with normalized cells and Markdown table.

        Raises:
            TableExtractionError: If the matrix is empty or contains no valid columns.
        """
        if not raw_matrix:
            raise TableExtractionError(
                "Raw table matrix is empty",
                details={"page_number": page_number},
            )

        cleaned_rows: list[list[str]] = []
        for raw_row in raw_matrix:
            normalized_row = [cls.normalize_cell_text(cell) for cell in raw_row]
            if any(cell != "" for cell in normalized_row):
                cleaned_rows.append(normalized_row)

        if not cleaned_rows:
            raise TableExtractionError(
                "Table contains only empty cells after normalization",
                details={"page_number": page_number},
            )

        headers, data_rows = cls.merge_headers(
            cleaned_rows,
            header_row_count=header_row_count,
        )

        cells: list[TableCell] = []
        for col_idx, header_text in enumerate(headers):
            cells.append(
                TableCell(
                    row_index=0,
                    col_index=col_idx,
                    raw_text=header_text,
                    numeric_value=None,
                    is_header=True,
                )
            )

        for row_offset, row_values in enumerate(data_rows, start=1):
            for col_idx, cell_text in enumerate(row_values):
                num_val = cls.parse_financial_number(cell_text)
                cells.append(
                    TableCell(
                        row_index=row_offset,
                        col_index=col_idx,
                        raw_text=cell_text,
                        numeric_value=num_val,
                        is_header=False,
                    )
                )

        markdown_str = cls.to_markdown(headers=headers, rows=data_rows)
        return TableBlock(
            page_number=page_number,
            title=title,
            currency_unit=currency_unit,
            headers=headers,
            rows=data_rows,
            cells=cells,
            raw_markdown=markdown_str,
            bounding_box=bounding_box,
        )

    @classmethod
    def extract_tables_from_page(
        cls,
        page: Page,
        page_number: int,
        default_title: str | None = None,
        default_currency: str | None = "VND",
    ) -> list[TableBlock]:
        """Extract all wireframe tables from a `pdfplumber.page.Page` object."""
        table_settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 4,
            "join_tolerance": 4,
            "intersection_tolerance": 5,
        }
        found_tables = page.find_tables(table_settings=table_settings)
        results: list[TableBlock] = []

        for tbl in found_tables:
            raw_matrix = tbl.extract()
            if not raw_matrix or len(raw_matrix) < 2:
                continue
            x0, top, x1, bottom = tbl.bbox
            bbox = BoundingBox(
                x0=max(0.0, float(x0)),
                y0=max(0.0, float(top)),
                x1=max(0.0, float(x1)),
                y1=max(0.0, float(bottom)),
                page_number=page_number,
            )
            block = cls.build_table_block(
                raw_matrix=raw_matrix,
                page_number=page_number,
                title=default_title,
                currency_unit=default_currency,
                bounding_box=bbox,
            )
            results.append(block)

        return results
