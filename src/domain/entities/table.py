"""TableCell and TableBlock entities for financial tables (Phase 1 - Task 1.1)."""

from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.domain.entities.page import BoundingBox


class TableCell(BaseModel):
    """Represents an individual cell within a financial statement table matrix."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    row_index: int = Field(..., ge=0, description="0-based row index in the table matrix")
    col_index: int = Field(..., ge=0, description="0-based column index in the table matrix")
    raw_text: str = Field(default="", description="Original cell string extracted from PDF")
    numeric_value: float | None = Field(
        default=None,
        description="Normalized numeric value if cell contains a number (e.g. (123) -> -123.0)",
    )
    is_header: bool = Field(
        default=False,
        description="Whether this cell belongs to a table header row",
    )
    row_span: int = Field(default=1, ge=1, description="Number of rows spanned by this cell")
    col_span: int = Field(default=1, ge=1, description="Number of columns spanned by this cell")
    bounding_box: BoundingBox | None = Field(
        default=None,
        description="Optional spatial coordinates of the cell",
    )


class TableBlock(BaseModel):
    """Structured representation of an extracted accounting or financial table."""

    model_config = ConfigDict(extra="forbid")

    table_id: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
        description="Unique identifier for the table block",
    )
    page_number: int = Field(..., ge=1, description="1-based starting page number of the table")
    end_page_number: int | None = Field(
        default=None,
        ge=1,
        description="1-based ending page number if table spans multiple pages",
    )
    title: str | None = Field(
        default=None,
        description="Table title or statement section (e.g., 'Bảng cân đối kế toán hợp nhất')",
    )
    currency_unit: str | None = Field(
        default="VND",
        description="Reporting currency/unit of the table (e.g., 'VND', 'Triệu VND')",
    )
    headers: list[str] = Field(
        default_factory=list,
        description="Flattened column headers (multi-level headers joined cleanly)",
    )
    rows: list[list[str]] = Field(
        default_factory=list,
        description="2D string matrix of data rows (each row has len == len(headers))",
    )
    cells: list[TableCell] = Field(
        default_factory=list,
        description="Detailed cell objects with parsed numeric values and coordinates",
    )
    raw_markdown: str = Field(
        ...,
        min_length=1,
        description="Complete Markdown representation of the table",
    )
    bounding_box: BoundingBox | None = Field(
        default=None,
        description="Spatial bounding box of the table on its starting page",
    )

    @model_validator(mode="after")
    def validate_table_consistency(self) -> "TableBlock":
        """Validate page range and row column counts against headers when headers are defined."""
        if self.end_page_number is not None and self.end_page_number < self.page_number:
            raise ValueError(
                f"end_page_number ({self.end_page_number}) cannot be less than "
                f"page_number ({self.page_number})"
            )
        if self.headers and self.rows:
            expected_cols = len(self.headers)
            for idx, row in enumerate(self.rows):
                if len(row) != expected_cols:
                    raise ValueError(
                        f"Row {idx} has {len(row)} columns, expected {expected_cols} "
                        f"to match headers {self.headers}"
                    )
        return self

    @property
    def row_count(self) -> int:
        """Return total number of data rows in the table."""
        return len(self.rows)

    @property
    def col_count(self) -> int:
        """Return total number of columns in the table."""
        if self.headers:
            return len(self.headers)
        return len(self.rows[0]) if self.rows else 0

    def extract_all_numbers(self) -> list[float]:
        """Return all non-null normalized numeric values stored in table cells."""
        return [cell.numeric_value for cell in self.cells if cell.numeric_value is not None]


# Alias matching specification terminology
FinancialTable = TableBlock
