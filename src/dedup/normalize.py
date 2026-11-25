"""Normalize names before comparison. The original text is left unchanged."""

from __future__ import annotations

import re

_ABBREVIATIONS = {
    "ltd": "limited",
    "pvt": "private",
    "inc": "incorporated",
    "corp": "corporation",
    "co": "company",
    "jv": "joint venture",
    "mgmt": "management",
    "mfg": "manufacturing",
}
_LEGAL = {"limited", "private", "incorporated", "corporation", "company", "plc"}
_KEEP_ENDING = {"business", "gas", "glass", "analysis", "basis"}
_STOP = {"the", "of", "for", "a", "an"}


def normalize(text: str) -> str:
    """Lowercase, punctuation, and/and, legal suffixes, and a light plural fold."""
    raw = (text or "").casefold().replace("&", " and ")
    raw = re.sub(r"[^a-z0-9\s]", " ", raw)
    tokens = []
    for token in raw.split():
        token = _ABBREVIATIONS.get(token, token)
        token = _singular(token)
        if token and token not in _STOP:
            tokens.append(token)
    return " ".join(tokens)


def content_tokens(text: str) -> set[str]:
    """Tokens that distinguish activities. Legal suffixes do not."""
    return {token for token in normalize(text).split() if token not in _LEGAL}


def entity_key(entity: str, ticker: str = "") -> str:
    """Parent mentions use company. A subsidiary keeps its own normalized name."""
    key = normalize(entity)
    parent = {"company", "parent", "the company"}
    if ticker:
        parent.add(normalize(ticker))
    if key in parent:
        return "company"
    stripped = " ".join(token for token in key.split() if token not in _LEGAL)
    return stripped or key


def _singular(token: str) -> str:
    if token in _KEEP_ENDING or len(token) <= 4:
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token
