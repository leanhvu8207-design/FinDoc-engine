"""Context-aware financial document and table chunking service (Phase 2 - Task 2.3)."""

from __future__ import annotations

import re

from src.domain.entities.document import (
    Chunk,
    ChunkType,
    DocumentMetadata,
    FinancialDocument,
)
from src.domain.entities.page import BlockType, PageBlock
from src.domain.entities.table import TableBlock
from src.domain.exceptions import ChunkingError

# Splits text at sentence-ending punctuation followed by whitespace or newline boundaries
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


class FinancialChunker:
    """Splits financial statements into context-enriched TEXT and intact TABLE chunks.

    Invariants:
    1. Table blocks (`ChunkType.TABLE`) are NEVER split across character boundaries; each
       `TableBlock` becomes a single atomic `Chunk` containing its complete Markdown matrix.
    2. Narrative paragraphs (`ChunkType.TEXT`) are grouped by page and section, split on
       sentence boundaries up to `chunk_size` with `chunk_overlap` overlap.
    3. Every chunk is prefixed with a deterministic context header:
       `[MÃ CP: {ticker} | NĂM: {year} | KỲ: {quarter} | LOẠI: {report_type} | TRANG: {page}]`
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        if chunk_size <= 0:
            raise ChunkingError(
                f"chunk_size must be positive, got {chunk_size}",
                details={"chunk_size": chunk_size},
            )
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ChunkingError(
                f"chunk_overlap ({chunk_overlap}) must be in [0, chunk_size ({chunk_size}))",
                details={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    @property
    def chunk_size(self) -> int:
        """Return target maximum body character length for narrative text chunks."""
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        """Return character overlap target between consecutive narrative text chunks."""
        return self._chunk_overlap

    @staticmethod
    def build_context_header(metadata: DocumentMetadata, page_number: int) -> str:
        """Format the mandatory Vietnamese financial context prefix for a chunk."""
        quarter_label = f"Q{metadata.quarter}" if metadata.quarter is not None else "FY"
        return (
            f"[MÃ CP: {metadata.ticker} | "
            f"NĂM: {metadata.year} | "
            f"KỲ: {quarter_label} | "
            f"LOẠI: {metadata.report_type.value} | "
            f"TRANG: {page_number}]"
        )

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        """Split narrative text into non-empty sentences without breaking mid-sentence."""
        raw_parts = _SENTENCE_SPLIT_RE.split(text.strip())
        return [part.strip() for part in raw_parts if part and part.strip()]

    def _split_oversized_sentence(self, sentence: str) -> list[str]:
        """Split a single extraordinarily long sentence at word boundaries only."""
        words = sentence.split()
        segments: list[str] = []
        current_words: list[str] = []
        current_len = 0

        for word in words:
            added_len = len(word) if not current_words else len(word) + 1
            if current_words and (current_len + added_len) > self._chunk_size:
                segments.append(" ".join(current_words))
                current_words = [word]
                current_len = len(word)
            else:
                current_words.append(word)
                current_len += added_len

        if current_words:
            segments.append(" ".join(current_words))
        return segments

    def split_narrative_text(self, text: str) -> list[str]:
        """Split narrative text into sentence-preserving windows of at most `chunk_size` chars."""
        sentences: list[str] = []
        for s in self._split_into_sentences(text):
            if len(s) > self._chunk_size:
                sentences.extend(self._split_oversized_sentence(s))
            else:
                sentences.append(s)

        if not sentences:
            return []

        windows: list[str] = []
        current_window: list[str] = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence) if not current_window else len(sentence) + 1
            if current_window and (current_len + sentence_len) > self._chunk_size:
                windows.append(" ".join(current_window))
                # Build overlap window from trailing whole sentences
                overlap_window: list[str] = []
                overlap_len = 0
                for prev_sent in reversed(current_window):
                    add_len = len(prev_sent) if not overlap_window else len(prev_sent) + 1
                    if overlap_len + add_len <= self._chunk_overlap:
                        overlap_window.insert(0, prev_sent)
                        overlap_len += add_len
                    else:
                        break
                current_window = overlap_window + [sentence]
                current_len = len(" ".join(current_window))
            else:
                current_window.append(sentence)
                current_len += sentence_len

        if current_window:
            windows.append(" ".join(current_window))

        return windows

    def _create_table_chunk(
        self,
        document_id: str,
        metadata: DocumentMetadata,
        table: TableBlock,
    ) -> Chunk:
        """Wrap an entire TableBlock into a single atomic TABLE chunk with context header."""
        header = self.build_context_header(metadata=metadata, page_number=table.page_number)
        title_line = f"Bảng: {table.title}\n" if table.title else ""
        full_content = f"{header}\n{title_line}{table.raw_markdown}"
        return Chunk(
            document_id=document_id,
            content=full_content,
            chunk_type=ChunkType.TABLE,
            metadata=metadata,
            page_number=table.page_number,
            section_title=table.title,
            bounding_box=table.bounding_box,
            table_block=table,
        )

    def chunk_blocks_and_tables(
        self,
        document_id: str,
        metadata: DocumentMetadata,
        blocks: list[PageBlock],
        tables: list[TableBlock],
    ) -> list[Chunk]:
        """Convert parsed PageBlocks and TableBlocks into ordered, context-aware Chunks.

        Args:
            document_id: Unique ID of the parent FinancialDocument.
            metadata: Corporate metadata of the report.
            blocks: Extracted PageBlock list from PDF parser.
            tables: Extracted TableBlock list from PDF parser.

        Returns:
            Ordered list of TEXT and TABLE Chunk entities.
        """
        if not document_id.strip():
            raise ChunkingError("document_id cannot be empty")

        chunks: list[Chunk] = []

        # Group narrative (non-table) blocks by (page_number, section_title)
        narrative_groups: list[tuple[int, str | None, list[str]]] = []
        for block in blocks:
            if block.block_type == BlockType.TABLE:
                continue
            cleaned = block.content.strip()
            if not cleaned:
                continue
            if (
                narrative_groups
                and narrative_groups[-1][0] == block.page_number
                and narrative_groups[-1][1] == block.section_title
            ):
                narrative_groups[-1][2].append(cleaned)
            else:
                narrative_groups.append((block.page_number, block.section_title, [cleaned]))

        # Map tables by page number to preserve page-level ordering
        tables_by_page: dict[int, list[TableBlock]] = {}
        for tbl in tables:
            tables_by_page.setdefault(tbl.page_number, []).append(tbl)

        processed_table_pages: set[int] = set()

        for page_num, section_title, text_lines in narrative_groups:
            combined_text = "\n".join(text_lines)
            header = self.build_context_header(metadata=metadata, page_number=page_num)
            for window_text in self.split_narrative_text(combined_text):
                chunks.append(
                    Chunk(
                        document_id=document_id,
                        content=f"{header}\n{window_text}",
                        chunk_type=ChunkType.TEXT,
                        metadata=metadata,
                        page_number=page_num,
                        section_title=section_title,
                    )
                )

            if page_num in tables_by_page and page_num not in processed_table_pages:
                for tbl in tables_by_page[page_num]:
                    chunks.append(
                        self._create_table_chunk(
                            document_id=document_id,
                            metadata=metadata,
                            table=tbl,
                        )
                    )
                processed_table_pages.add(page_num)

        # Append any tables on pages that had no narrative blocks
        for page_num in sorted(tables_by_page.keys()):
            if page_num not in processed_table_pages:
                for tbl in tables_by_page[page_num]:
                    chunks.append(
                        self._create_table_chunk(
                            document_id=document_id,
                            metadata=metadata,
                            table=tbl,
                        )
                    )

        return chunks

    def chunk_document(self, document: FinancialDocument) -> list[Chunk]:
        """Generate context-aware chunks directly from a FinancialDocument aggregate."""
        return self.chunk_blocks_and_tables(
            document_id=document.document_id,
            metadata=document.metadata,
            blocks=document.blocks,
            tables=document.tables,
        )
