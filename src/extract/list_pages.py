"""Select subsidiary / related-party list pages from selected page_text (Mistral-aware)."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from rapidfuzz import fuzz

from src.config.paths import PAGE_DIR
from src.extract.select import page_text
from src.ocr_mistral import load_mistral_page

LOGGER = logging.getLogger("factsheet")

_LIST_HINT = re.compile(
    r"related party|form\s*aoc-?\s*1|list of (?:subsidiar|joint venture|associate)|"
    r"\bsubsidiar(?:y|ies)\b|\bjoint ventures?\b|\bassociates?\b|"
    r"name of (?:the )?(?:subsidiary|company|entity)|enterprises? over which",
    re.I,
)
_SUFFIXES = re.compile(
    r"\b(?:private\s+)?limited\b|\bpvt\.?\s*ltd\.?\b|\bltd\.?\b|\binc\.?\b|\bllc\b|\bgmbh\b",
    re.I,
)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_entity_name(name: str) -> str:
    text = (name or "").casefold()
    text = _SUFFIXES.sub(" ", text)
    text = _NON_ALNUM.sub(" ", text)
    return " ".join(text.split())


def select_list_pages(ticker: str, *, page_dir: Path | None = None) -> list[int]:
    """Return PDF pages that look like subsidiary / JV / related-party lists."""
    ticker = ticker.upper()
    folder = page_dir or (PAGE_DIR / ticker)
    hits: list[int] = []
    for path in sorted(folder.glob("p*.json")):
        page = json.loads(path.read_text(encoding="utf-8"))
        pdf_page = int(page["pdf_page"])
        text = page_text(page) or ""
        mist = load_mistral_page(ticker, pdf_page)
        if mist and mist.get("markdown"):
            text = text + "\n" + mist["markdown"]
        if _LIST_HINT.search(text):
            hits.append(pdf_page)
    LOGGER.info("%s list_pages=%s count=%s", ticker, hits, len(hits))
    return hits


def dedup_entities(rows: list[dict], *, fuzzy_threshold: int = 92) -> list[dict]:
    """Keep one row per normalized name; prefer the longest grounded quote."""
    kept: list[dict] = []
    keys: list[str] = []
    for row in rows:
        name = (row.get("name") or "").strip()
        if not name:
            continue
        key = normalize_entity_name(name)
        if not key:
            continue
        match_index = None
        for index, existing in enumerate(keys):
            if existing == key or fuzz.ratio(existing, key) >= fuzzy_threshold:
                match_index = index
                break
        if match_index is None:
            keys.append(key)
            kept.append(row)
            continue
        current = kept[match_index]
        current_quote = len(current.get("quote") or "")
        new_quote = len(row.get("quote") or "")
        if new_quote > current_quote:
            kept[match_index] = row
            keys[match_index] = key
    return kept
