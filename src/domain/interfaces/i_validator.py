"""Interface contract for deterministic numeric fact-checking validators (Phase 1 - Task 1.2)."""

from abc import ABC, abstractmethod

from src.domain.entities.response import ValidationReport
from src.domain.entities.table import TableBlock


class INumericValidator(ABC):
    """Port interface for deterministic numeric guardrail verification against source tables."""

    @abstractmethod
    def validate(
        self,
        answer: str,
        source_tables: list[TableBlock],
    ) -> ValidationReport:
        """Verify that every numeric claim in `answer` exists in `source_tables`.

        Args:
            answer: Generated natural language answer containing financial numbers.
            source_tables: Ground-truth TableBlock entities retrieved as context.

        Returns:
            ValidationReport detailing verified and unverified numbers and overall status.
        """
