"""Generic rejects for corporate names that are not list entries."""

from __future__ import annotations

import re

# Running headers and sentence scraps. Not a company list.
HEADER_NAME = re.compile(
    r"corporate overview|statutory reports|financial statements|about the report|integrated annual report",
    re.I,
)
SENTENCE_NAME = re.compile(
    r"\b(made by|is hereby|hereby given|given to|pursuant to|members of|payment for|notice is)\b",
    re.I,
)
_PARTICLES = {"of", "and", "the", "for", "de", "du", "&"}
_LEGAL = re.compile(
    r"\b([A-Z][A-Za-z0-9&.'’\-]*(?:[ \t]+(?:&|[A-Za-z][A-Za-z0-9&.'’\-]*)){0,14}"
    r"[ \t]+(?:Limited|Ltd\.?|Private Limited|Pvt\.?\s*Ltd\.?|Inc\.?|LLC|GmbH))\b"
)


def is_running_header(name: str) -> bool:
    return bool(HEADER_NAME.search(name or ""))


def is_sentence_fragment(name: str) -> bool:
    return bool(SENTENCE_NAME.search(name or ""))


def _companyish(name: str) -> bool:
    if is_sentence_fragment(name) or is_running_header(name):
        return False
    if re.match(r"^(?:the|investment|incorporation|new|for|and|management|certified|unaudited)\b", name, re.I):
        return False
    if re.search(r"\b(?:statement|certified|management|investment in|incorporation of)\b", name, re.I):
        return False
    return True


def is_wrapped_fragment(name: str) -> bool:
    """A name joined from the middle of a line keeps a lowercase token."""
    for token in (name or "").split():
        if token[:1].islower() and token.casefold() not in _PARTICLES:
            return True
    return False


def legal_names(text: str) -> list[str]:
    return [" ".join(match.group(1).split()) for match in _LEGAL.finditer(text or "")]


def expand_suffix(name: str, text: str) -> str | None:
    """Replace a name that is only the ending of a longer legal name on the page."""
    if not name or not text:
        return None
    fragment = [token.casefold() for token in name.split()]
    if len(fragment) < 2:
        return None
    best = None
    for full in legal_names(text):
        tokens = [token.casefold() for token in full.split()]
        if len(tokens) <= len(fragment):
            continue
        if tokens[-len(fragment):] == fragment and _companyish(full):
            if best is None or len(tokens) < len(best.split()):
                best = full
    return best


def rebuild_wrapped_name(name: str, text: str) -> str | None:
    """Use a longer legal name on the page when this string is only its ending."""
    if not name or not text:
        return None
    return expand_suffix(name, text)
