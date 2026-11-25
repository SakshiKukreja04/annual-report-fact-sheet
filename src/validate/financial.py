"""Printed percentages and explicit calculations. Segment revenue is not turned into a share."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|per\s*cent\b|percent\b)", re.IGNORECASE)


def printed_percentages(quote: str) -> list[Decimal]:
    found = []
    for match in _PERCENT.finditer(quote or ""):
        found.append(Decimal(match.group(1)))
    return found


def check_printed_share(claimed, quotes: list[str]) -> dict:
    """Keep a share only when that percentage is printed in a quote."""
    if claimed is None or claimed == "":
        return _record(None, "unavailable", "No revenue share was claimed.")
    try:
        share = Decimal(str(claimed))
    except (InvalidOperation, ValueError):
        return _record(None, "ambiguous", "The claimed share is not a number.")
    if share < 0 or share > 100:
        return _record(None, "out_of_bounds", "The percentage is outside 0 to 100.", share)
    printed = [value for quote in quotes for value in printed_percentages(quote)]
    if any(value == share for value in printed):
        return _record(share, "printed", "The percentage is printed in a source quote.", share)
    return _record(None, "not_printed", "The share is not printed in the source quotes.", share)


def calculate_share(numerator, denominator, unit: str, source_pages: list[int]) -> dict:
    """Divide two printed figures. This is not called unless both figures are supplied."""
    try:
        num = Decimal(str(numerator))
        den = Decimal(str(denominator))
    except (InvalidOperation, ValueError):
        return _record(None, "ambiguous", "The printed figures are not numbers.")
    record = {
        "numerator": format(num, "f"),
        "denominator": format(den, "f"),
        "unit": unit,
        "source_pages": list(source_pages),
    }
    if den == 0 or not source_pages or not unit:
        return _record(None, "ambiguous", "The calculation is missing a unit, a page, or a non-zero denominator.", extra=record)
    share = (num / den * Decimal(100)).quantize(Decimal("0.01"))
    if share < 0 or share > 100:
        return _record(None, "out_of_bounds", "The calculated percentage is outside 0 to 100.", share, record)
    return _record(share, "calculated_from_printed_figures", "Both figures were supplied from the report.", share, record)


def _record(share: Decimal | None, status: str, reason: str, claimed: Decimal | None = None, extra: dict | None = None) -> dict:
    body = {
        "revenue_share_pct": None if share is None else format(share, "f"),
        "status": status,
        "reason": reason,
        "claimed": None if claimed is None else format(claimed, "f"),
        "needs_review": status not in {"printed", "calculated_from_printed_figures", "unavailable"},
    }
    if extra:
        body.update(extra)
    return body
