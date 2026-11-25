"""Limit the 'already found' list to names that actually appear in the packet."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+", re.I)


def name_content_tokens(name: str) -> list[str]:
    stop = {"the", "and", "of", "for", "in", "to", "a", "an", "ltd", "limited", "pvt", "private", "inc", "llc"}
    tokens = []
    for token in _TOKEN.findall(name or ""):
        folded = token.casefold()
        if len(folded) < 3 or folded in stop:
            continue
        tokens.append(folded)
    return tokens


def relevant_known_names(known: list[str], packet: str, limit: int = 40) -> list[str]:
    """Keep names whose content tokens all appear in the packet text, capped at limit."""
    folded = (packet or "").casefold()
    chosen: list[str] = []
    seen: set[str] = set()
    for name in known:
        key = (name or "").casefold().strip()
        if not key or key in seen:
            continue
        tokens = name_content_tokens(name)
        if not tokens:
            continue
        if all(token in folded for token in tokens):
            chosen.append(name)
            seen.add(key)
        if len(chosen) >= limit:
            break
    return chosen
