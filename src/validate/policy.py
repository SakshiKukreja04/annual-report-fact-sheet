"""Keep business activities and correct kind and status from the quote.

The saved quote is not rewritten. A rejection is a reason string.
"""

from __future__ import annotations

import re

_ADMIN = re.compile(
    r"\b("
    r"agm|annual general meeting|speaker registration|e-?\s*voting|instavote|instameet|"
    r"book closure|iepf|dividend|record date|"
    r"re-?\s*appointment|appointment of (?:mr|ms|mrs|shri)|board appointment|"
    r"cost audit|statutory audit|secretarial audit|\bauditor\b|"
    r"share register|registrar and transfer|\bfolio\b"
    r")\b",
    re.I,
)
_CAPITAL = re.compile(
    r"\b(financial|human|intellectual|manufactured|natural|social and relationship)\s+capital\b",
    re.I,
)
_RESEARCH_SITE = re.compile(
    r"\b(research and development|research centre|research center|research platform|"
    r"r&d|laboratory|innovation centre|innovation center|technology centre|technology center)\b",
    re.I,
)
_TRAINING = re.compile(r"\b(training|workshop|awareness programme|awareness program)\b", re.I)
_CHANNEL = re.compile(r"\b(distribution|sales channel|dealer|retail network|stockist|e-commerce)\b", re.I)
_FOOTPRINT = re.compile(r"\b(global operations|footprint|presence in|operates in \d+)\b", re.I)
_COMPANY_NAME = re.compile(r"\b(limited|ltd|inc|incorporated|llc|gmbh)\b", re.I)
_PRODUCT = re.compile(r"\b(foil|sheet|ingot|rod|cathode|extrusion|beverage can|rolled products?)\b", re.I)
_BUSINESS_LINE = re.compile(r"\b(business|division)\b", re.I)
_PLANNED = re.compile(
    r"\b(plans? to|planned|proposed|under construction|to be commissioned|progressing|"
    r"will commission|will be (?:set up|built|commissioned)|expected to commission)\b",
    re.I,
)
_SITE = re.compile(r"\b(plant|smelter|refinery|mine|facility)\b", re.I)
_ASPIRATIONAL = re.compile(r"\b(aspire|aim to|vision to|strive to|committed to achieve)\b", re.I)
_CURRENT = re.compile(r"\b(operates|operating|produced|produces|manufactures|commissioned|during the year)\b", re.I)


def apply_policy(candidate: dict) -> tuple[dict, str | None]:
    """Return the corrected candidate and a rejection reason, if it is out of scope."""
    revised = dict(candidate)
    blob = _blob(revised)
    if _CAPITAL.search(revised.get("name") or "") or _CAPITAL.search(revised.get("descriptor") or ""):
        return revised, "reporting_framework"
    if _ADMIN.search(blob):
        return revised, "administrative"
    revised["kind"] = correct_kind(revised)
    revised["status"] = correct_status(revised)
    return revised, None


def correct_kind(candidate: dict) -> str:
    """Apply the kind rules. A footprint is not a channel and training is not a laboratory."""
    kind = candidate.get("kind") or "other"
    name = candidate.get("name") or ""
    blob = _blob(candidate)
    if kind == "research_platform" and _TRAINING.search(blob) and not _RESEARCH_SITE.search(blob):
        return "other"
    if kind == "channel" and not _CHANNEL.search(blob):
        if _FOOTPRINT.search(blob) or _COMPANY_NAME.search(name):
            return "subsidiary"
        return "other"
    if kind == "other" and _SITE.search(name):
        return "asset"
    if kind == "business_line" and _PRODUCT.search(name):
        return "product_or_service"
    if kind == "product_or_service" and _BUSINESS_LINE.search(name) and not _PRODUCT.search(name):
        return "business_line"
    if re.search(r"\bsegment\b", name, re.I) and kind == "product_or_service":
        return "segment"
    return kind


def correct_status(candidate: dict) -> str:
    """Use the quote. A mention of a future project is not a current activity."""
    quote = " ".join(source.get("quote") or "" for source in candidate.get("sources") or [])
    status = candidate.get("status") or "current"
    if status not in {"current", "planned", "aspirational"}:
        status = "current"
    if _CURRENT.search(quote) and not _PLANNED.search(quote):
        return "current"
    if _PLANNED.search(quote):
        return "planned"
    if _ASPIRATIONAL.search(quote):
        return "aspirational"
    if re.search(r"\b(expansion|project)\b", candidate.get("name") or "", re.I):
        return "planned"
    return status


def _blob(candidate: dict) -> str:
    quotes = " ".join(source.get("quote") or "" for source in candidate.get("sources") or [])
    return " ".join((candidate.get("name") or "", candidate.get("descriptor") or "", quotes))
