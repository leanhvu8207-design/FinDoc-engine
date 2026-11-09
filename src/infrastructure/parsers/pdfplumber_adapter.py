"""PDF parser adapter using pdfplumber, TableExtractor, and Vision OCR fallback (Task 2.2)."""

from __future__ import annotations

import base64
import io
import json
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pdfplumber
import pypdfium2

from src.domain.entities.document import DocumentMetadata, FinancialDocument
from src.domain.entities.page import BlockType, BoundingBox, DocumentBlock, PageBlock
from src.domain.entities.table import TableBlock
from src.domain.exceptions import ParsingError, TableExtractionError
from src.domain.interfaces.i_parser import IDocumentParser
from src.infrastructure.config import get_settings
from src.infrastructure.parsers.table_extractor import TableExtractor

_HEADING_PATTERNS = re.compile(
    r"^(?:BẢNG CÂN ĐỐI|BANG CAN DOI|BÁO CÁO TÌNH HÌNH|BAO CAO TINH HINH|"
    r"BÁO CÁO KẾT QUẢ|BAO CAO KET QUA|"
    r"BÁO CÁO LƯU CHUYỂN|BAO CAO LUU CHUYEN|THUYẾT MINH|THUYET MINH|"
    r"[A-Z]\.\s+[A-Z]|(?:I|II|III|IV|V|VI|VII|VIII|IX|X)+\.\s+)",
    re.IGNORECASE,
)

_CURRENCY_UNIT_RE = re.compile(
    r"(?:Đơn vị tính|Don vi tinh|Đơn vị|Don vi)\s*:\s*([A-Za-zÀ-ỹ0-9\s]+)",
    re.IGNORECASE,
)

_MD_SEP_CELL_RE = re.compile(r"^:?-{2,}:?$")


class PdfPlumberAdapter(IDocumentParser):
    """Concrete implementation of IDocumentParser with Vision OCR fallback for scanned PDFs."""

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
            clean_line = line.replace("*", "").replace("#", "").strip()
            if section_title is None and _HEADING_PATTERNS.search(clean_line):
                section_title = clean_line
            unit_match = _CURRENCY_UNIT_RE.search(clean_line)
            if unit_match:
                currency_unit = unit_match.group(1).strip()
        return section_title, currency_unit

    @staticmethod
    def _ocr_single_scanned_page(file_path: Path, page_index_zero: int) -> str:
        """Render a scanned PDF page to JPEG and extract Markdown via Gemini Vision."""
        settings = get_settings()
        api_key = settings.gemini_api_key.strip()
        if not api_key:
            return ""

        doc = pypdfium2.PdfDocument(str(file_path))
        try:
            page = doc[page_index_zero]
            pil_img = page.render(scale=1.5).to_pil()
        finally:
            doc.close()

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        b64_img = base64.b64encode(buf.getvalue()).decode("ascii")

        prompt = (
            "Trích xuất chính xác toàn bộ văn bản và bảng biểu báo cáo tài chính trên trang này. "
            "Yêu cầu bắt buộc: "
            "1. Toàn bộ bảng biểu phải viết dưới dạng bảng Markdown chuẩn (dùng dấu |). "
            "2. Giữ nguyên chính xác 100% các con số và chỉ tiêu, không dùng dấu ** trong ô số."
        )
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-3.5-flash-lite:generateContent?key={api_key}"
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": b64_img}},
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.0},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            candidates = body.get("candidates", [])
            if not candidates:
                return ""
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(str(p.get("text", "")) for p in parts).strip()
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError):
            return ""

    def _parse_markdown_from_ocr(
        self,
        markdown_text: str,
        page_number: int,
    ) -> tuple[list[PageBlock], list[TableBlock]]:
        """Parse OCR Markdown output into PageBlock and TableBlock entities."""
        raw_lines = [line.strip() for line in markdown_text.splitlines() if line.strip()]
        section_title, currency_unit = self._detect_page_metadata(raw_lines)

        page_blocks: list[PageBlock] = []
        table_blocks: list[TableBlock] = []
        current_md_table_rows: list[list[str | None]] = []

        def _flush_table() -> None:
            nonlocal current_md_table_rows
            if len(current_md_table_rows) >= 2:
                try:
                    tbl = self._table_extractor.build_table_block(
                        raw_matrix=current_md_table_rows,
                        page_number=page_number,
                        title=section_title,
                        currency_unit=currency_unit,
                    )
                    table_blocks.append(tbl)
                    page_blocks.append(
                        PageBlock(
                            page_number=page_number,
                            block_type=BlockType.TABLE,
                            content=tbl.raw_markdown,
                            section_title=tbl.title or section_title,
                        )
                    )
                except TableExtractionError:
                    pass
            current_md_table_rows = []

        active_section = section_title
        for line in raw_lines:
            if line.startswith("|") and line.endswith("|") and line.count("|") >= 2:
                cells = [
                    c.replace("**", "").replace("*", "").strip()
                    for c in line.strip("|").split("|")
                ]
                if all(not c or bool(_MD_SEP_CELL_RE.match(c)) for c in cells):
                    continue
                current_md_table_rows.append(list(cells))
            else:
                _flush_table()
                clean_line = line.lstrip("#").replace("**", "").strip()
                if not clean_line:
                    continue
                is_heading = bool(_HEADING_PATTERNS.search(clean_line)) or line.startswith("#")
                if is_heading:
                    active_section = clean_line
                page_blocks.append(
                    PageBlock(
                        page_number=page_number,
                        block_type=BlockType.HEADING if is_heading else BlockType.PARAGRAPH,
                        content=clean_line,
                        section_title=active_section,
                    )
                )

        _flush_table()
        return page_blocks, table_blocks

    def _extract_scanned_pdf(
        self,
        file_path: Path,
        total_pages: int,
    ) -> tuple[list[DocumentBlock], list[TableBlock]]:
        """Run parallel Vision OCR across all pages of a scanned PDF document."""
        all_blocks: list[DocumentBlock] = []
        all_tables: list[TableBlock] = []

        with ThreadPoolExecutor(max_workers=min(5, total_pages)) as pool:
            futures = [
                pool.submit(self._ocr_single_scanned_page, file_path, idx)
                for idx in range(total_pages)
            ]
            ocr_texts = [f.result() for f in futures]

        for page_idx, md_text in enumerate(ocr_texts, start=1):
            if not md_text:
                continue
            p_blocks, t_blocks = self._parse_markdown_from_ocr(md_text, page_number=page_idx)
            all_blocks.extend(p_blocks)
            all_tables.extend(t_blocks)

        return all_blocks, all_tables

    def extract_tables(self, file_path: Path) -> list[TableBlock]:
        """Extract all structured financial tables across all pages of a PDF file."""
        self._validate_pdf_path(file_path)
        all_tables: list[TableBlock] = []
        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = max(1, len(pdf.pages))
                has_any_words = False
                for page_idx, page in enumerate(pdf.pages, start=1):
                    words = page.extract_words() or []
                    if words:
                        has_any_words = True
                    lines = self._group_words_into_lines(words)
                    section_title, currency_unit = self._detect_page_metadata(lines)
                    page_tables = self._table_extractor.extract_tables_from_page(
                        page=page,
                        page_number=page_idx,
                        default_title=section_title,
                        default_currency=currency_unit,
                    )
                    all_tables.extend(page_tables)

            if not all_tables and not has_any_words:
                _, all_tables = self._extract_scanned_pdf(file_path, total_pages)
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
                total_pages = max(1, len(pdf.pages))
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

            if not blocks:
                blocks, _ = self._extract_scanned_pdf(file_path, total_pages)
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
                has_any_words = any(bool(p.extract_words()) for p in pdf.pages)

            if not has_any_words:
                blocks, tables = self._extract_scanned_pdf(file_path, total_pages)
            else:
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
