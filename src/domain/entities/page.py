"""PageBlock and BoundingBox entities (Phase 1 - Task 1.1)."""

from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BlockType(StrEnum):
    """Structural classification of a parsed PDF page block."""

    HEADING = "HEADING"
    PARAGRAPH = "PARAGRAPH"
    TABLE = "TABLE"
    FOOTNOTE = "FOOTNOTE"
    METADATA = "METADATA"


class BoundingBox(BaseModel):
    """2D coordinate bounding box on a PDF page (origin at top-left)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    x0: float = Field(..., ge=0.0, description="Left X coordinate")
    y0: float = Field(..., ge=0.0, description="Top Y coordinate")
    x1: float = Field(..., ge=0.0, description="Right X coordinate")
    y1: float = Field(..., ge=0.0, description="Bottom Y coordinate")
    page_number: int = Field(..., ge=1, description="1-based page number")

    @model_validator(mode="after")
    def validate_coordinates(self) -> "BoundingBox":
        """Ensure right >= left and bottom >= top."""
        if self.x1 < self.x0:
            raise ValueError(f"x1 ({self.x1}) must be greater than or equal to x0 ({self.x0})")
        if self.y1 < self.y0:
            raise ValueError(f"y1 ({self.y1}) must be greater than or equal to y0 ({self.y0})")
        return self

    @property
    def width(self) -> float:
        """Calculate bounding box width."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Calculate bounding box height."""
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        """Calculate bounding box surface area."""
        return self.width * self.height


class PageBlock(BaseModel):
    """A discrete structural element extracted from a single PDF page."""

    model_config = ConfigDict(extra="forbid")

    block_id: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
        description="Unique identifier for the page block",
    )
    page_number: int = Field(..., ge=1, description="1-based page number where block appears")
    block_type: BlockType = Field(
        default=BlockType.PARAGRAPH,
        description="Structural category of the block",
    )
    content: str = Field(..., min_length=1, description="Normalized textual or markdown content")
    section_title: str | None = Field(
        default=None,
        description="Active financial statement section or note heading",
    )
    bounding_box: BoundingBox | None = Field(
        default=None,
        description="Spatial coordinates of the block on the PDF page",
    )


# Alias matching specification terminology in i_parser.py
DocumentBlock = PageBlock
