"""Shorten page text before a model call. Quotes are never checked against this text."""

from __future__ import annotations

import hashlib
import re

from src.extract.ledger import estimate_tokens
from src.extract.select import _CSR_SCOPE, page_text
from src.validate.quotes import fold_whitespace

_DOT_LINE = re.compile(r"^(?=.*\.{4,}).+$")
_PAGE_NUMBER = re.compile(r"^\d{1,4}$")


def repeated_lines(pages: list[dict]) -> set[str]:
    """Lines that show up on many pages are headers and footers."""
    counts: dict[str, int] = {}
    for page in pages:
        seen = set()
        for line in page_text(page).splitlines():
            key = fold_whitespace(line).casefold()
            if key and key not in seen and len(key.split()) <= 14:
                seen.add(key)
                counts[key] = counts.get(key, 0) + 1
    if len(pages) < 4:
        return set()
    threshold = max(4, len(pages) // 5)
    return {line for line, count in counts.items() if count >= threshold}


def paragraphs_for_page(page: dict, banners: set[str], *, scope_only: bool = False) -> list[dict]:
    """One record per block that is still a substring of the original page after folding."""
    original = fold_whitespace(page_text(page))
    records = []
    for block in page.get("text_blocks") or []:
        raw = (block.get("text") or "").strip()
        if not raw:
            continue
        if scope_only and not _CSR_SCOPE.search(raw):
            continue
        cleaned = clean_block(raw, banners)
        if not cleaned:
            continue
        folded = fold_whitespace(cleaned)
        if folded.casefold() not in original.casefold():
            continue
        records.append(
            {
                "hash": hashlib.sha256(folded.casefold().encode("utf-8")).hexdigest(),
                "text": folded,
                "pdf_page": int(page["pdf_page"]),
            }
        )
    return records


def clean_block(text: str, banners: set[str]) -> str:
    kept = []
    for line in text.splitlines():
        folded = fold_whitespace(line)
        if not folded or _PAGE_NUMBER.match(folded):
            continue
        if _DOT_LINE.match(folded):
            continue
        if folded.casefold() in banners:
            continue
        kept.append(folded)
    return re.sub(r"[ \t]{2,}", " ", " ".join(kept)).strip()


def render_new(paragraphs: list[dict], seen: set[str]) -> tuple[str, list[str]]:
    """Send each normalized paragraph once. The page tag is a page that contains it."""
    lines = []
    fresh = []
    for paragraph in paragraphs:
        if paragraph["hash"] in seen or paragraph["hash"] in fresh:
            continue
        lines.append(f"[P {paragraph['pdf_page']}] {paragraph['text']}")
        fresh.append(paragraph["hash"])
    return "\n".join(lines), fresh


def pack_pages(
    pages: list[dict],
    by_page: dict[int, list[dict]],
    max_tokens: int,
    max_pages: int = 8,
    *,
    preserve_order: bool = True,
) -> list[dict]:
    """Group pages until the next new paragraph would pass the token budget.

    Rank order is kept when preserve_order is true (default). Set false to pack
    by ascending pdf_page instead.
    """
    ordered = list(pages) if preserve_order else sorted(pages, key=lambda page: int(page["pdf_page"]))
    batches = []
    current: list[dict] = []
    current_tokens = 0
    seen: set[str] = set()
    pending: list[str] = []

    def close() -> None:
        nonlocal current, current_tokens, pending
        if not current:
            return
        batches.append(
            {
                "pages": current,
                "target_pages": [int(page["pdf_page"]) for page in current],
                "context_pages": [],
                "char_count": current_tokens * 4,
            }
        )
        seen.update(pending)
        current = []
        current_tokens = 0
        pending = []

    for page in ordered:
        packet, fresh = render_new(by_page.get(int(page["pdf_page"]), []), seen.union(pending))
        cost = estimate_tokens(packet)
        if current and (len(current) >= max_pages or (cost and current_tokens + cost > max_tokens)):
            close()
            packet, fresh = render_new(by_page.get(int(page["pdf_page"]), []), seen)
            cost = estimate_tokens(packet)
        current.append(page)
        current_tokens += cost
        pending.extend(fresh)
    close()
    return batches
