"""Unit tests for FinancialChunker service (Phase 2 - Task 2.3)."""

from pathlib import Path

import pytest

from scripts.download_sample_bctc import ensure_sample_bctc_pdfs
from src.application.services.financial_chunker import FinancialChunker
from src.domain.entities.document import ChunkType, DocumentMetadata, ReportType
from src.domain.entities.page import BlockType, PageBlock
from src.domain.entities.table import TableBlock
from src.domain.exceptions import ChunkingError
from src.infrastructure.parsers import PdfPlumberAdapter


def test_context_header_injection_format() -> None:
    """Verify exact Vietnamese context header format on generated chunks."""
    meta = DocumentMetadata(
        ticker="FPT",
        year=2025,
        quarter=4,
        report_type=ReportType.CONSOLIDATED,
    )
    header = FinancialChunker.build_context_header(metadata=meta, page_number=7)
    assert header == "[MÃ CP: FPT | NĂM: 2025 | KỲ: Q4 | LOẠI: CONSOLIDATED | TRANG: 7]"

    annual_meta = DocumentMetadata(
        ticker="VNM",
        year=2025,
        quarter=None,
        report_type=ReportType.ANNUAL,
    )
    annual_header = FinancialChunker.build_context_header(metadata=annual_meta, page_number=1)
    assert annual_header == "[MÃ CP: VNM | NĂM: 2025 | KỲ: FY | LOẠI: ANNUAL | TRANG: 1]"


def test_table_chunk_never_split_even_when_exceeding_chunk_size() -> None:
    """Verify TableBlock is preserved as a single intact Chunk regardless of chunk_size."""
    meta = DocumentMetadata(
        ticker="HPG",
        year=2025,
        quarter=4,
        report_type=ReportType.CONSOLIDATED,
    )
    headers = ["Chỉ tiêu", "Mã số", "Số cuối năm", "Số đầu năm"]
    rows = [
        [f"Khoản mục tài sản số {idx}", f"{100 + idx}", f"{idx * 100000}", f"{idx * 90000}"]
        for idx in range(1, 20)
    ]
    md_lines = [
        "| Chỉ tiêu | Mã số | Số cuối năm | Số đầu năm |",
        "| --- | --- | --- | --- |",
        *[f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in rows],
    ]
    large_markdown = "\n".join(md_lines)
    assert len(large_markdown) > 800

    table_block = TableBlock(
        page_number=4,
        title="Bảng cân đối kế toán hợp nhất",
        headers=headers,
        rows=rows,
        raw_markdown=large_markdown,
    )

    # Set a very small chunk_size (120 chars) to prove TABLE chunks are never split
    chunker = FinancialChunker(chunk_size=120, chunk_overlap=20)
    chunks = chunker.chunk_blocks_and_tables(
        document_id="doc-hpg-2025",
        metadata=meta,
        blocks=[],
        tables=[table_block],
    )

    assert len(chunks) == 1
    table_chunk = chunks[0]
    assert table_chunk.chunk_type == ChunkType.TABLE
    assert table_chunk.table_block == table_block
    assert table_chunk.content.startswith(
        "[MÃ CP: HPG | NĂM: 2025 | KỲ: Q4 | LOẠI: CONSOLIDATED | TRANG: 4]"
    )
    assert large_markdown in table_chunk.content


def test_narrative_text_chunking_preserves_sentences_and_context_header() -> None:
    """Verify narrative text splits on sentence boundaries and attaches context header."""
    meta = DocumentMetadata(
        ticker="FPT",
        year=2025,
        quarter=3,
        report_type=ReportType.CONSOLIDATED,
    )
    s1 = "Doanh thu thuần hợp nhất quý 3 năm 2025 đạt 16.500 tỷ đồng."
    s2 = "Lợi nhuận trước thuế tăng trưởng 19,8% so với cùng kỳ năm trước."
    s3 = "Mảng dịch vụ công nghệ thông tin nước ngoài đóng góp tỷ trọng lớn nhất."
    blocks = [
        PageBlock(
            page_number=2,
            block_type=BlockType.PARAGRAPH,
            content=f"{s1} {s2} {s3}",
            section_title="Thuyết minh doanh thu",
        )
    ]

    # Set chunk_size=130 so s1 + s2 fit in chunk 1, and s3 goes to chunk 2 with s2 overlap
    chunker = FinancialChunker(chunk_size=130, chunk_overlap=70)
    chunks = chunker.chunk_blocks_and_tables(
        document_id="doc-fpt-q3",
        metadata=meta,
        blocks=blocks,
        tables=[],
    )

    assert len(chunks) == 2
    expected_prefix = "[MÃ CP: FPT | NĂM: 2025 | KỲ: Q3 | LOẠI: CONSOLIDATED | TRANG: 2]"
    for c in chunks:
        assert c.chunk_type == ChunkType.TEXT
        assert c.table_block is None
        assert c.content.startswith(expected_prefix)

    assert s1 in chunks[0].content
    assert s2 in chunks[0].content
    # Overlap sentence s2 should also appear in chunk 2 along with s3
    assert s2 in chunks[1].content
    assert s3 in chunks[1].content


def test_mixed_text_and_table_chunks_end_to_end(tmp_path: Path) -> None:
    """Verify PdfPlumberAdapter + FinancialChunker produce both TEXT and TABLE chunks."""
    ensure_sample_bctc_pdfs(output_dir=tmp_path)
    pdf_path = tmp_path / "VNM_2025_Q4_Consolidated.pdf"
    meta = DocumentMetadata(
        ticker="VNM",
        year=2025,
        quarter=4,
        report_type=ReportType.CONSOLIDATED,
    )

    parser = PdfPlumberAdapter()
    doc = parser.parse_document(file_path=pdf_path, metadata=meta)

    chunker = FinancialChunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.chunk_document(doc)

    text_chunks = [c for c in chunks if c.chunk_type == ChunkType.TEXT]
    table_chunks = [c for c in chunks if c.chunk_type == ChunkType.TABLE]

    assert len(text_chunks) >= 2
    assert len(table_chunks) == 2
    for chunk in chunks:
        assert chunk.content.startswith(
            f"[MÃ CP: VNM | NĂM: 2025 | KỲ: Q4 | LOẠI: CONSOLIDATED | TRANG: {chunk.page_number}]"
        )


def test_invalid_chunker_configuration_raises_error() -> None:
    """Verify invalid chunk_size or chunk_overlap raises ChunkingError."""
    with pytest.raises(ChunkingError):
        FinancialChunker(chunk_size=0, chunk_overlap=0)

    with pytest.raises(ChunkingError):
        FinancialChunker(chunk_size=256, chunk_overlap=256)

    with pytest.raises(ChunkingError):
        FinancialChunker(chunk_size=256, chunk_overlap=-5)
