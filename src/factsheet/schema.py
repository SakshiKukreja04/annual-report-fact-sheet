"""Fact-sheet schema. Extra fields are rejected."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.extract.schema import KINDS, STATUSES

Kind = Literal[
    "product_or_service",
    "brand",
    "business_line",
    "segment",
    "subsidiary",
    "channel",
    "asset",
    "research_platform",
    "philanthropy",
    "other",
]
Status = Literal["current", "planned", "aspirational"]


class FactSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pdf_page: int = Field(ge=1)
    printed_page: str | None = None
    quote: str

    @field_validator("pdf_page")
    @classmethod
    def page_is_a_real_number(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("pdf_page must be a page number.")
        return value

    @field_validator("quote")
    @classmethod
    def quote_is_present(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("A source quote is empty.")
        return value


class FactItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Kind
    descriptor: str
    entity: str
    status: Status
    revenue_share_pct: float | None = None
    sources: list[FactSource] = Field(min_length=1, max_length=3)

    @field_validator("name", "entity")
    @classmethod
    def text_is_present(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("A required text field is empty.")
        return value

    @field_validator("descriptor")
    @classmethod
    def descriptor_is_text(cls, value: str) -> str:
        """A list subsidiary with no printed activity may keep an empty descriptor."""
        return value or ""


class Segment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    revenue: float | None = None
    unit: str | None = None
    share_pct: float | None = None

    @field_validator("name")
    @classmethod
    def name_is_present(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("A segment name is empty.")
        return value


class FactSheet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: Literal["POWERGRID", "HINDALCO", "COALINDIA"]
    period: Literal["FY25"]
    one_liner: str
    summary: str
    segments: list[Segment]
    items: list[FactItem]


assert set(KINDS) == {
    "product_or_service",
    "brand",
    "business_line",
    "segment",
    "subsidiary",
    "channel",
    "asset",
    "research_platform",
    "philanthropy",
    "other",
}
assert set(STATUSES) == {"current", "planned", "aspirational"}
