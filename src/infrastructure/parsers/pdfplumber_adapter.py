"""PDF parser adapter using pdfplumber and TableExtractor (Phase 2 - Task 2.2)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pdfplumber

from src.domain.entities.document import DocumentMetadata, FinancialDocument
from src.domain.entities.page import BlockType, BoundingBox, DocumentBlock, PageBlock
from src.domain.entities.table import TableBlock
from src.domain.exceptions import ParsingError, TableExtractionError
from src.domain.interfaces.i_parser import IDocumentParser
from src.infrastructure.parsers.table_extractor import TableExtractor

_HEADING_PATTERNS = re.compile(
    r"^(?:BẢNG CÂN ĐỐI|BANG CAN DOI|BÁO CÁO KẾT QUẢ|BAO CAO KET QUA|"
    r"BÁO CÁO LƯU CHUYỂN|BAO CAO LUU CHUYEN|THUYẾT MINH|THUYET MINH|"
    r"[A-Z]\.\s+[A-Z]|(?:I|II|III|IV|V|VI|VII|VIII|IX|X)+\.\s+)",
    re.IGNORECASE,
)

_CURRENCY_UNIT_RE = re.compile(
    r"(?:Đơn vị tính|Don vi tinh)\s*:\s*([A-Za-zÀ-ỹ0-9\s]+)",
    re.IGNORECASE,
)


class PdfPlumberAdapter(IDocumentParser):
    """Concrete implementation of IDocumentParser using pdfplumber and TableExtractor."""

    def __init__(self, table_extractor: TableExtractor | None = None) -> None:
        self._table_extractor = table_extractor or TableExtractor()

    @staticmethod
    def _validate_pdf_path(file_path: Path) -> None:
        """Verify the target PDF file exists and is a regular file."""
        if not file_path.exists() or not file_path.is_file():
            raise ParsingError(
                f"PDF file not found: {file_path}",
                details={"file_path": str(file_path)},
            )

    @staticmethod
    def _is_inside_any_bbox(
        word_top: float,
        word_bottom: float,
        word_x0: float,
        word_x1: float,
        bboxes: list[BoundingBox],
        margin: float = 2.0,
    ) -> bool:
        """Check whether a word rectangle overlaps with any extracted table bounding box."""
        for bbox in bboxes:
            horiz_overlap = word_x1 >= (bbox.x0 - margin) and word_x0 <= (bbox.x1 + margin)
            vert_overlap = word_bottom >= (bbox.y0 - margin) and word_top <= (bbox.y1 + margin)
            if horiz_overlap and vert_overlap:
                return True
        return False

    @staticmethod
    def _group_words_into_lines(words: list[dict[str, Any]], y_tolerance: float = 3.0) -> list[str]:
        """Group pdfplumber word dicts on the same vertical line into text strings."""
        if not words:
            return []
        sorted_words = sorted(words, key=lambda w: (float(w["top"]), float(w["x0"])))
        lines: list[list[dict[str, Any]]] = []
        for word in sorted_words:
            if not lines:
                lines.append([word])
                continue
            prev_top = float(lines[-1][0]["top"])
            if abs(float(word["top"]) - prev_top) <= y_tolerance:
                lines[-1].append(word)
            else:
                lines.append([word])

        return [
            " ".join(
                str(w["text"]) for w in sorted(line, key=lambda item: float(item["x0"]))
            ).strip()
            for line in lines
        ]

    @staticmethod
    def _detect_page_metadata(lines: list[str]) -> tuple[str | None, str]:
        """Detect section heading and reporting currency unit from page header lines."""
        section_title: str | None = None
        currency_unit: str = "VND"
        for line in lines:
            if section_title is None and _HEADING_PATTERNS.search(line):
                section_title = line.strip()
            unit_match = _CURRENCY_UNIT_RE.search(line)
            if unit_match:
                currency_unit = unit_match.group(1).strip()
        return section_title, currency_unit

    def extract_tables(self, file_path: Path) -> list[TableBlock]:
        """Extract all structured financial tables across all pages of a PDF file."""
        self._validate_pdf_path(file_path)
        all_tables: list[TableBlock] = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_idx, page in enumerate(pdf.pages, start=1):
                    words = page.extract_words() or []
                    lines = self._group_words_into_lines(words)
                    section_title, currency_unit = self._detect_page_metadata(lines)
                    page_tables = self._table_extractor.extract_tables_from_page(
                        page=page,
                        page_number=page_idx,
                        default_title=section_title,
                        default_currency=currency_unit,
                    )
                    all_tables.extend(page_tables)
        except TableExtractionError:
            raise
        except Exception as exc:
            raise TableExtractionError(
                f"Failed to extract tables from PDF: {file_path}",
                details={"file_path": str(file_path), "error": str(exc)},
            ) from exc

        return all_tables

    def extract_blocks(self, file_path: Path) -> list[DocumentBlock]:
        """Extract ordered structural page blocks (headings, narrative text, and tables)."""
        self._validate_pdf_path(file_path)
        blocks: list[DocumentBlock] = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_idx, page in enumerate(pdf.pages, start=1):
                    all_words = page.extract_words() or []
                    all_lines = self._group_words_into_lines(all_words)
                    section_title, currency_unit = self._detect_page_metadata(all_lines)

                    page_tables = self._table_extractor.extract_tables_from_page(
                        page=page,
                        page_number=page_idx,
                        default_title=section_title,
                        default_currency=currency_unit,
                    )
                    table_bboxes = [
                        tbl.bounding_box for tbl in page_tables if tbl.bounding_box is not None
                    ]

                    non_table_words = [
                        w
                        for w in all_words
                        if not self._is_inside_any_bbox(
                            word_top=float(w["top"]),
                            word_bottom=float(w["bottom"]),
                            word_x0=float(w["x0"]),
                            word_x1=float(w["x1"]),
                            bboxes=table_bboxes,
                        )
                    ]
                    narrative_lines = self._group_words_into_lines(non_table_words)
                    active_section = section_title

                    for line in narrative_lines:
                        if not line:
                            continue
                        is_heading = bool(_HEADING_PATTERNS.search(line))
                        if is_heading:
                            active_section = line
                        blocks.append(
                            PageBlock(
                                page_number=page_idx,
                                block_type=BlockType.HEADING if is_heading else BlockType.PARAGRAPH,
                                content=line,
                                section_title=active_section,
                            )
                        )

                    for tbl in page_tables:
                        blocks.append(
                            PageBlock(
                                page_number=page_idx,
                                block_type=BlockType.TABLE,
                                content=tbl.raw_markdown,
                                section_title=tbl.title or active_section,
                                bounding_box=tbl.bounding_box,
                            )
                        )
        except FinDocParserExceptionTuple as exc:
            raise exc
        except Exception as exc:
            raise ParsingError(
                f"Failed to parse blocks from PDF: {file_path}",
                details={"file_path": str(file_path), "error": str(exc)},
            ) from exc

        return blocks

    def parse_document(
        self,
        file_path: Path,
        metadata: DocumentMetadata,
    ) -> FinancialDocument:
        """Parse an entire PDF document into a complete FinancialDocument aggregate."""
        self._validate_pdf_path(file_path)
        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = max(1, len(pdf.pages))
            tables = self.extract_tables(file_path)
            blocks = self.extract_blocks(file_path)
            return FinancialDocument(
                metadata=metadata,
                file_path=file_path,
                total_pages=total_pages,
                blocks=blocks,
                tables=tables,
            )
        except (ParsingError, TableExtractionError):
            raise
        except Exception as exc:
            raise ParsingError(
                f"Failed to parse FinancialDocument: {file_path}",
                details={"file_path": str(file_path), "error": str(exc)},
            ) from exc


FinDocParserExceptionTuple = (ParsingError, TableExtractionError)
