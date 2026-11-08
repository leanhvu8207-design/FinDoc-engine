"""Script to download or generate Vietnamese financial statement PDFs (Phase 2 - Task 2.1).

Attempts to download public corporate financial statements and ensures at least 3 realistic,
wireframe-structured BCTC PDF files (FPT, VNM, HPG) exist in `data/raw_pdfs/` for deterministic
offline parsing and integration testing.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SampleBCTCSpec:
    """Specification for a sample corporate financial statement PDF."""

    filename: str
    ticker: str
    company_name: str
    period_label: str
    currency_unit: str
    remote_url: str | None
    balance_sheet_rows: list[tuple[str, str, str, str, str]]
    income_statement_rows: list[tuple[str, str, str, str, str]]


SAMPLE_BCTC_SPECS: list[SampleBCTCSpec] = [
    SampleBCTCSpec(
        filename="FPT_2025_Q4_Consolidated.pdf",
        ticker="FPT",
        company_name="CONG TY CO PHAN FPT",
        period_label="Quy 4 Nam 2025",
        currency_unit="Trieu VND",
        remote_url=None,
        balance_sheet_rows=[
            ("A. TAI SAN NGAN HAN", "100", "", "38.450.200", "32.110.500"),
            ("I. Tien va cac khoan tuong duong tien", "110", "V.01", "9.320.150", "8.140.000"),
            ("II. Dau tu tai chinh ngan han", "120", "V.02", "14.200.000", "11.500.000"),
            ("III. Cac khoan phai thu ngan han", "130", "V.03", "9.850.450", "8.420.300"),
            ("Du phong phai thu ngan han kho doi", "137", "V.04", "(312.400)", "(285.100)"),
            ("IV. Hang ton kho", "140", "V.05", "1.980.000", "1.825.200"),
            ("Du phong giam gia hang ton kho", "149", "", "(45.200)", "(38.900)"),
            ("B. TAI SAN DAI HAN", "200", "", "26.890.800", "24.120.000"),
            ("TONG CONG TAI SAN", "270", "", "65.341.000", "56.230.500"),
        ],
        income_statement_rows=[
            (
                "1. Doanh thu ban hang va cung cap dich vu",
                "01",
                "VI.01",
                "62.849.000",
                "52.618.000",
            ),
            ("2. Cac khoan giam tru doanh thu", "02", "VI.02", "(124.500)", "(98.000)"),
            ("3. Doanh thu thuan ve ban hang", "10", "", "62.724.500", "52.520.000"),
            ("4. Gia von hang ban", "11", "VI.03", "(38.410.000)", "(32.150.000)"),
            ("5. Loi nhuan gop ve ban hang", "20", "", "24.314.500", "20.370.000"),
            ("6. Chi phi tai chinh", "22", "VI.04", "(645.300)", "(590.200)"),
            ("7. Loi nhuan sau thue thu nhap doanh nghiep", "60", "", "9.420.800", "7.788.500"),
        ],
    ),
    SampleBCTCSpec(
        filename="VNM_2025_Q4_Consolidated.pdf",
        ticker="VNM",
        company_name="CONG TY CO PHAN SUA VIET NAM (VINAMILK)",
        period_label="Quy 4 Nam 2025",
        currency_unit="Trieu VND",
        remote_url=None,
        balance_sheet_rows=[
            ("A. TAI SAN NGAN HAN", "100", "", "36.125.000", "34.890.000"),
            ("I. Tien va cac khoan tuong duong tien", "110", "V.01", "2.910.400", "2.650.100"),
            ("II. Dau tu tai chinh ngan han", "120", "V.02", "20.450.000", "19.800.000"),
            ("III. Cac khoan phai thu ngan han", "130", "V.03", "6.120.300", "5.940.200"),
            ("Du phong phai thu ngan han kho doi", "137", "", "(18.750)", "(16.400)"),
            ("IV. Hang ton kho", "140", "V.05", "6.150.800", "6.020.500"),
            ("Du phong giam gia hang ton kho", "149", "", "(29.500)", "(24.800)"),
            ("B. TAI SAN DAI HAN", "200", "", "18.920.000", "17.760.000"),
            ("TONG CONG TAI SAN", "270", "", "55.045.000", "52.650.000"),
        ],
        income_statement_rows=[
            (
                "1. Doanh thu ban hang va cung cap dich vu",
                "01",
                "VI.01",
                "61.824.000",
                "60.479.000",
            ),
            ("2. Cac khoan giam tru doanh thu", "02", "", "(89.400)", "(110.200)"),
            ("3. Doanh thu thuan ve ban hang", "10", "", "61.734.600", "60.368.800"),
            ("4. Gia von hang ban", "11", "VI.03", "(36.210.500)", "(35.820.100)"),
            ("5. Loi nhuan gop ve ban hang", "20", "", "25.524.100", "24.548.700"),
            ("6. Chi phi ban hang", "25", "VI.05", "(13.450.200)", "(13.012.400)"),
            ("7. Loi nhuan sau thue thu nhap doanh nghiep", "60", "", "9.650.300", "9.019.300"),
        ],
    ),
    SampleBCTCSpec(
        filename="HPG_2025_Q4_Consolidated.pdf",
        ticker="HPG",
        company_name="CONG TY CO PHAN TAP DOAN HOA PHAT",
        period_label="Quy 4 Nam 2025",
        currency_unit="Trieu VND",
        remote_url=None,
        balance_sheet_rows=[
            ("A. TAI SAN NGAN HAN", "100", "", "84.560.000", "82.710.000"),
            ("I. Tien va cac khoan tuong duong tien", "110", "V.01", "12.450.000", "12.252.000"),
            ("II. Dau tu tai chinh ngan han", "120", "V.02", "22.980.000", "22.170.000"),
            ("III. Hang ton kho", "140", "V.04", "39.640.000", "34.504.000"),
            ("Du phong giam gia hang ton kho", "149", "", "(415.600)", "(628.900)"),
            ("B. TAI SAN DAI HAN", "200", "", "121.340.000", "105.072.000"),
            ("TONG CONG TAI SAN", "270", "", "205.900.000", "187.782.000"),
        ],
        income_statement_rows=[
            (
                "1. Doanh thu ban hang va cung cap dich vu",
                "01",
                "VI.01",
                "140.560.000",
                "120.355.000",
            ),
            ("2. Cac khoan giam tru doanh thu", "02", "", "(1.120.000)", "(1.402.000)"),
            ("3. Doanh thu thuan ve ban hang", "10", "", "139.440.000", "118.953.000"),
            ("4. Gia von hang ban", "11", "VI.02", "(120.850.000)", "(106.015.000)"),
            ("5. Loi nhuan gop ve ban hang", "20", "", "18.590.000", "12.938.000"),
            ("6. Chi phi tai chinh", "22", "VI.03", "(3.980.400)", "(5.195.000)"),
            ("7. Loi nhuan sau thue thu nhap doanh nghiep", "60", "", "12.020.500", "6.800.400"),
        ],
    ),
]


def _escape_pdf_text(text: str) -> str:
    """Escape parentheses and backslashes for PDF literal strings."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_table_page_stream(
    company_name: str,
    ticker: str,
    period_label: str,
    currency_unit: str,
    section_title: str,
    headers: list[str],
    rows: list[tuple[str, str, str, str, str]],
) -> bytes:
    """Generate a PDF content stream containing headings, narrative text, and a wireframe table."""
    ops: list[str] = []

    # Header text blocks
    header_lines = [
        (50.0, 790.0, 12, f"{company_name} (Ma CK: {ticker})"),
        (50.0, 772.0, 11, f"BAO CAO TAI CHINH HOP NHAT - {period_label}"),
        (50.0, 750.0, 13, section_title),
        (50.0, 732.0, 10, f"Don vi tinh: {currency_unit}"),
    ]
    for x, y, font_size, line in header_lines:
        escaped = _escape_pdf_text(line)
        ops.append(f"BT /F1 {font_size} Tf {x:.1f} {y:.1f} Td ({escaped}) Tj ET")

    # Table grid geometry (5 columns)
    col_x = [50.0, 260.0, 310.0, 380.0, 465.0, 550.0]
    top_y = 710.0
    row_height = 24.0
    all_rows: list[list[str]] = [headers] + [list(r) for r in rows]
    total_rows = len(all_rows)
    bottom_y = top_y - total_rows * row_height

    # Draw horizontal grid lines
    ops.append("0.75 w")
    for r_idx in range(total_rows + 1):
        y = top_y - r_idx * row_height
        ops.append(f"{col_x[0]:.1f} {y:.1f} m {col_x[-1]:.1f} {y:.1f} l S")

    # Draw vertical grid lines
    for x in col_x:
        ops.append(f"{x:.1f} {top_y:.1f} m {x:.1f} {bottom_y:.1f} l S")

    # Draw cell text inside each cell box
    for r_idx, row_values in enumerate(all_rows):
        cell_top = top_y - r_idx * row_height
        text_y = cell_top - 16.0
        for c_idx, cell_text in enumerate(row_values):
            if not cell_text:
                continue
            text_x = col_x[c_idx] + 4.0
            escaped = _escape_pdf_text(cell_text)
            ops.append(f"BT /F1 8 Tf {text_x:.1f} {text_y:.1f} Td ({escaped}) Tj ET")

    # Footer note below table
    note_y = bottom_y - 30.0
    note_text = _escape_pdf_text(
        f"Thuyet minh dinh kem la bo phan hop thanh cua Bao cao tai chinh {ticker}."
    )
    ops.append(f"BT /F1 9 Tf 50.0 {note_y:.1f} Td ({note_text}) Tj ET")

    return "\n".join(ops).encode("latin-1")


def build_two_page_bctc_pdf(spec: SampleBCTCSpec) -> bytes:
    """Create a valid 2-page PDF document with wireframe accounting tables."""
    headers = ["Chi tieu", "Ma so", "Thuyet minh", "So cuoi ky", "So dau nam"]
    stream1 = _build_table_page_stream(
        company_name=spec.company_name,
        ticker=spec.ticker,
        period_label=spec.period_label,
        currency_unit=spec.currency_unit,
        section_title="BANG CAN DOI KE TOAN HOP NHAT",
        headers=headers,
        rows=spec.balance_sheet_rows,
    )
    stream2 = _build_table_page_stream(
        company_name=spec.company_name,
        ticker=spec.ticker,
        period_label=spec.period_label,
        currency_unit=spec.currency_unit,
        section_title="BAO CAO KET QUA HOAT DONG KINH DOANH HOP NHAT",
        headers=headers,
        rows=spec.income_statement_rows,
    )

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>"
        ),
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 7 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream1)} >>\nstream\n".encode("latin-1") + stream1 + b"\nendstream",
        f"<< /Length {len(stream2)} >>\nstream\n".encode("latin-1") + stream2 + b"\nendstream",
    ]

    output = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = [0]
    for idx, obj_body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{idx} 0 obj\n".encode("latin-1"))
        output.extend(obj_body)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    output.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        output.extend(f"{off:010d} 00000 n \n".encode("latin-1"))

    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("latin-1")
    )
    return bytes(output)


def ensure_sample_bctc_pdfs(output_dir: Path | None = None) -> list[Path]:
    """Download or generate sample corporate financial statement PDFs into `data/raw_pdfs/`."""
    target_dir = output_dir or Path("data/raw_pdfs")
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for spec in SAMPLE_BCTC_SPECS:
        file_path = target_dir / spec.filename
        if spec.remote_url:
            try:
                req = urllib.request.Request(
                    spec.remote_url,
                    headers={"User-Agent": "FinDoc-Engine/0.1"},
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    pdf_bytes = response.read()
                if pdf_bytes.startswith(b"%PDF"):
                    file_path.write_bytes(pdf_bytes)
                    saved_paths.append(file_path)
                    continue
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                logger.warning(
                    "Remote download failed for %s (%s), using fixture.",
                    spec.ticker,
                    exc,
                )

        pdf_bytes = build_two_page_bctc_pdf(spec)
        file_path.write_bytes(pdf_bytes)
        saved_paths.append(file_path)

    return saved_paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    paths = ensure_sample_bctc_pdfs()
    for p in paths:
        logger.info("Prepared sample BCTC PDF: %s (%d bytes)", p, p.stat().st_size)
