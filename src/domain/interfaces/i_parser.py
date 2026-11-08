"""Interface contract for PDF document parsers (Phase 1 - Task 1.2)."""

from abc import ABC, abstractmethod
from pathlib import Path

from src.domain.entities.document import DocumentMetadata, FinancialDocument
from src.domain.entities.page import DocumentBlock
from src.domain.entities.table import TableBlock


class IDocumentParser(ABC):
    """Port interface for multi-modal financial PDF parsing adapters."""

    @abstractmethod
    def extract_blocks(self, file_path: Path) -> list[DocumentBlock]:
        """Extract ordered structural page blocks (text, headings, tables) from a PDF file.

        Args:
            file_path: Filesystem path to the target PDF document.

        Returns:
            Ordered list of extracted DocumentBlock entities.

        Raises:
            ParsingError: If the PDF file cannot be opened or parsed.
        """

    @abstractmethod
    def extract_tables(self, file_path: Path) -> list[TableBlock]:
        """Extract structured financial tables with normalized numbers and Markdown.

        Args:
            file_path: Filesystem path to the target PDF document.

        Returns:
            List of extracted TableBlock entities.

        Raises:
            TableExtractionError: If table wireframe or matrix extraction fails.
        """

    @abstractmethod
    def parse_document(
        self,
        file_path: Path,
        metadata: DocumentMetadata,
    ) -> FinancialDocument:
        """Parse an entire PDF document into a complete FinancialDocument aggregate.

        Args:
            file_path: Filesystem path to the target PDF document.
            metadata: Corporate metadata associated with the financial statement.

        Returns:
            Populated FinancialDocument aggregate root.

        Raises:
            ParsingError: If parsing fails at any stage.
        """
