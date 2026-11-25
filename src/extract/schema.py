"""Candidate shape. Validation checks the response structure, not the quote wording."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

KINDS = (
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
)
STATUSES = ("current", "planned", "aspirational")


class Source(BaseModel):
    pdf_page: int
    printed_page: str | None = None
    quote: str

    @field_validator("printed_page", mode="before")
    @classmethod
    def printed_page_is_text(cls, value):
        if value is None or value == "":
            return None
        return str(value)

    @field_validator("quote")
    @classmethod
    def quote_has_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("A source quote is empty.")
        return value


class SupportingFigure(BaseModel):
    """A number printed in the report. This is not a calculated share."""

    label: str = ""
    value: str = ""
    pdf_page: int
    quote: str


class Candidate(BaseModel):
    name: str
    kind: Literal[
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
    descriptor: str
    entity: str
    status: Literal["current", "planned", "aspirational"]
    revenue_share_pct: float | None = None
    sources: list[Source] = Field(min_length=1, max_length=3)
    supporting_figures: list[SupportingFigure] = Field(default_factory=list)

    @field_validator("revenue_share_pct", mode="before")
    @classmethod
    def blank_share_is_null(cls, value):
        if value is None or value == "":
            return None
        if isinstance(value, str):
            cleaned = value.strip().rstrip("%")
            if not cleaned or cleaned.lower() == "null":
                return None
            return float(cleaned)
        return value


class ExtractionResponse(BaseModel):
    candidates: list[Candidate] = Field(default_factory=list)
