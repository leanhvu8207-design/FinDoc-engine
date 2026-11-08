"""Deterministic numeric fact-checking guardrail (Phase 4 - Task 4.3)."""

from __future__ import annotations

import math
import re

from src.domain.entities.response import ValidationReport, ValidationStatus
from src.domain.entities.table import TableBlock
from src.domain.exceptions import HallucinationError
from src.domain.interfaces.i_validator import INumericValidator
from src.infrastructure.config import Settings, get_settings
from src.infrastructure.parsers.table_extractor import TableExtractor

# Patterns for non-financial reference numbers (dates, quarters, pages, note codes) to strip first
_NON_FINANCIAL_REF_RE = re.compile(
    r"\[MÃ CP:[^\]]+\]|"
    r"(?:năm|nam|fy|q[1-4]|quý|quy|tháng|thang|trang|page|thuyết minh|thuyet minh|"
    r"mã số|ma so|mục|muc)\s*(?:số\s*|so\s*)?[vViIxX0-9./_-]+|"
    r"\b(?:q[1-4]|quý\s*[1-4])\s*/\s*\d{4}\b|"
    r"\b[vViIxX]+\.\d+\b",
    re.IGNORECASE | re.UNICODE,
)

# Captures accounting negative numbers in parentheses OR standard financial numbers/percentages
_ANSWER_NUMBER_TOKEN_RE = re.compile(
    r"\(\s*\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?\s*%?\s*\)|"
    r"\(\s*\d+(?:[.,]\d+)?\s*%?\s*\)|"
    r"(?<![a-zA-ZÀ-ỹ0-9_])[+-]?\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?%?|"
    r"(?<![a-zA-ZÀ-ỹ0-9_])[+-]?\d+(?:[.,]\d+)?%?",
    re.UNICODE,
)


class NumericGuardrail(INumericValidator):
    """Pure deterministic validator that cross-checks numeric claims against source TableBlocks."""

    def __init__(
        self,
        settings: Settings | None = None,
        tolerance: float | None = None,
    ) -> None:
        cfg = settings or get_settings()
        self._tolerance: float = tolerance if tolerance is not None else cfg.numeric_tolerance

    @property
    def tolerance(self) -> float:
        """Return relative numeric comparison tolerance."""
        return self._tolerance

    @classmethod
    def extract_numbers_from_text(cls, text: str) -> list[float]:
        """Extract all financial numeric claims from a generated answer using pure Regex."""
        if not text or not text.strip():
            return []

        cleaned_text = _NON_FINANCIAL_REF_RE.sub(" ", text)
        matches = _ANSWER_NUMBER_TOKEN_RE.findall(cleaned_text)

        extracted: list[float] = []
        for raw_token in matches:
            parsed = TableExtractor.parse_financial_number(raw_token)
            if parsed is not None:
                extracted.append(parsed)
        return extracted

    @staticmethod
    def collect_ground_truth_numbers(source_tables: list[TableBlock]) -> list[float]:
        """Gather all valid numeric values from source TableBlocks (`cells` and `rows`)."""
        ground_truth: list[float] = []
        for table in source_tables:
            cell_numbers = table.extract_all_numbers()
            ground_truth.extend(cell_numbers)

            # Also inspect string rows in case TableBlock was created without explicit TableCells
            for row in table.rows:
                for cell_str in row:
                    parsed = TableExtractor.parse_financial_number(cell_str)
                    if parsed is not None:
                        ground_truth.append(parsed)

        return ground_truth

    def _is_number_verified(self, claimed: float, ground_truth_pool: list[float]) -> bool:
        """Return True if `claimed` matches any number in `ground_truth_pool` within tolerance."""
        return any(
            math.isclose(claimed, truth, rel_tol=self._tolerance, abs_tol=1e-4)
            for truth in ground_truth_pool
        )

    def validate(
        self,
        answer: str,
        source_tables: list[TableBlock],
    ) -> ValidationReport:
        """Verify that every numeric claim in `answer` exists in `source_tables`."""
        extracted_numbers = self.extract_numbers_from_text(answer)
        if not extracted_numbers:
            return ValidationReport(
                status=ValidationStatus.PASSED,
                extracted_numbers=[],
                verified_numbers=[],
                unverified_numbers=[],
                hallucinated_numbers=[],
                message="Answer contains no numeric claims; validation passed.",
            )

        ground_truth_pool = self.collect_ground_truth_numbers(source_tables)
        verified: list[float] = []
        hallucinated: list[float] = []

        for num in extracted_numbers:
            if self._is_number_verified(num, ground_truth_pool):
                verified.append(num)
            else:
                hallucinated.append(num)

        if hallucinated:
            return ValidationReport(
                status=ValidationStatus.FAILED,
                extracted_numbers=extracted_numbers,
                verified_numbers=verified,
                unverified_numbers=hallucinated,
                hallucinated_numbers=hallucinated,
                message=(
                    f"Numeric guardrail FAILED: detected {len(hallucinated)} "
                    f"hallucinated number(s) not present in source tables: {hallucinated}"
                ),
            )

        return ValidationReport(
            status=ValidationStatus.PASSED,
            extracted_numbers=extracted_numbers,
            verified_numbers=verified,
            unverified_numbers=[],
            hallucinated_numbers=[],
            message=f"All {len(verified)} numeric claim(s) verified against source tables.",
        )

    def validate_or_raise(
        self,
        answer: str,
        source_tables: list[TableBlock],
    ) -> ValidationReport:
        """Validate `answer` and raise `HallucinationError` if any number is unverified."""
        report = self.validate(answer=answer, source_tables=source_tables)
        if report.status == ValidationStatus.FAILED:
            raise HallucinationError(
                report.message,
                unverified_numbers=report.hallucinated_numbers,
            )
        return report
