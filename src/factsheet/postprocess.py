"""Clean a saved fact sheet without calling a model. The original file is not written."""

from __future__ import annotations

import re
from collections import Counter

from src.validate.quotes import locate_quote, name_tokens, repair_named_quote

_DROP_NAME = re.compile(
    r"\b(agm|e-?\s*voting|instavote|instameet|dividend|book closure|iepf|remuneration|auditor|re-?\s*appointment)\b"
    r"|material topic"
    r"|\b(financial|human|intellectual|manufactured|natural|social and relationship)\s+capital\b"
    r"|risk management|\bhedg\w*"
    r"|\bghg\b|water[- ]?(saving|positivity)|zero[- ]waste"
    r"|planned investments|commitment of",
    re.I,
)
_GROUP = re.compile(
    r"the group runs \d+ hospitals|core sectors like cement|cement, metals, fibre|group.?s apex body",
    re.I,
)
_GOVERNANCE = re.compile(
    r"explanatory statement|notice of the .{0,40}annual general meeting|remote e-?\s*voting|instavote|instameet|book closure",
    re.I,
)
_COMPANY = {"", "company", "the company", "hindalco", "hindalco industries", "hindalco industries limited"}


_OPERATING = re.compile(r"aluminium|aluminum|copper|alumina|smelter|refinery|novelis|\bplant\b|mining", re.I)


def governance_pages(pages: list[dict]) -> set[int]:
    """A page is governance only from its own opening text, not from an earlier section label."""
    numbers = set()
    for page in pages:
        text = _text(page)[:1500]
        if not _GOVERNANCE.search(text) or _OPERATING.search(text):
            continue
        numbers.add(int(page["pdf_page"]))
    return numbers


def drop_items(items: list[dict], blocked_pages: set[int]) -> tuple[list[dict], list[dict]]:
    kept = []
    dropped = []
    for item in items:
        reason = _drop_reason(item, blocked_pages)
        if reason:
            dropped.append({"name": item.get("name"), "reason": reason, "pdf_page": _first_page(item)})
            continue
        kept.append(item)
    return kept, dropped


def merge_items(items: list[dict]) -> tuple[list[dict], int]:
    """One item per normalized name. Keep three quotes and the longer descriptor."""
    groups: dict[str, dict] = {}
    order = []
    for item in items:
        key = _name_key(item.get("name") or "")
        if not key:
            continue
        if key not in groups:
            groups[key] = {**item, "sources": list(item.get("sources") or [])}
            order.append(key)
            continue
        current = groups[key]
        if len(item.get("descriptor") or "") > len(current.get("descriptor") or ""):
            current["descriptor"] = item["descriptor"]
        if _kind_rank(item.get("kind")) > _kind_rank(current.get("kind")):
            current["kind"] = item["kind"]
        for source in item.get("sources") or []:
            if source not in current["sources"] and len(current["sources"]) < 3:
                current["sources"].append(source)
    return [groups[key] for key in order], len(items) - len(order)


def normalize_entities(items: list[dict], subsidiaries: list[str]) -> list[dict]:
    exact = {_name_key(name): name for name in subsidiaries}
    cleaned = []
    for item in items:
        entity = (item.get("entity") or "").strip()
        key = _name_key(entity)
        name_key = _name_key(item.get("name") or "")
        if entity.casefold() in _COMPANY:
            chosen = "company"
        elif key in exact:
            chosen = exact[key]
        elif item.get("kind") == "subsidiary" and name_key in exact:
            chosen = exact[name_key]
        elif item.get("kind") == "subsidiary":
            chosen = item.get("name") or "company"
        else:
            chosen = "company"
        cleaned.append({**item, "entity": chosen})
    return cleaned


def assign_printed_pages(items: list[dict], pages: list[dict]) -> list[dict]:
    by_page = {int(page["pdf_page"]): page for page in pages}
    updated = []
    for item in items:
        sources = []
        for source in item.get("sources") or []:
            page = by_page.get(source.get("pdf_page"))
            printed = _printed_for_quote(page, source.get("quote") or "") if page else source.get("printed_page")
            sources.append({**source, "printed_page": printed})
        updated.append({**item, "sources": sources})
    return updated


def repair_items(items: list[dict], texts: dict[int, str], pdf_texts: dict[int, str]) -> dict:
    """Snap each quote. Drop a quote that is not on the stored page or the PDF layer."""
    kept = []
    dropped = []
    counts = Counter()
    for item in items:
        sources = []
        for source in item.get("sources") or []:
            number = source.get("pdf_page")
            stored = texts.get(number, "") if isinstance(number, int) else ""
            original = source.get("quote") or ""
            repaired = repair_named_quote(original, stored, item.get("name") or "")
            quote = repaired.get("quote")
            if repaired["outcome"] == "dropped" or not quote:
                counts["dropped"] += 1
                continue
            stored_kind = locate_quote(quote, stored)
            pdf_kind = locate_quote(quote, pdf_texts.get(number, "")) if pdf_texts else None
            if stored_kind is None or pdf_kind is None:
                counts["dropped"] += 1
                continue
            if quote == original and stored_kind == "exact":
                counts["strict"] += 1
            elif quote == original:
                counts["normalized"] += 1
            else:
                counts["repaired" if repaired["outcome"] == "repaired" else "reanchored"] += 1
            if not _name_in_quote_or_sentence(item.get("name") or "", quote, stored):
                counts["dropped"] += 1
                continue
            sources.append({**source, "quote": quote})
        if not sources:
            dropped.append({"name": item.get("name"), "reason": "no_valid_quote", "pdf_page": _first_page(item)})
            continue
        kept.append({**item, "sources": sources[:3]})
    return {"items": kept, "dropped": dropped, "quotes": dict(counts)}


def _name_in_quote_or_sentence(name: str, quote: str, page_text: str) -> bool:
    tokens = name_tokens(name)
    if not tokens:
        return True
    sentence = quote
    if quote and quote in page_text:
        start = page_text.find(quote)
        left = max(page_text.rfind(".", 0, start), page_text.rfind("\n", 0, start))
        right_candidates = [index for index in (page_text.find(".", start + len(quote)), page_text.find("\n", start + len(quote))) if index >= 0]
        right = min(right_candidates) if right_candidates else len(page_text)
        sentence = page_text[left + 1 : right + 1]
    blob = f"{quote} {sentence}".casefold()
    return all(token.casefold() in blob for token in tokens)


def _drop_reason(item: dict, blocked_pages: set[int]) -> str | None:
    pages = [source.get("pdf_page") for source in item.get("sources") or [] if isinstance(source.get("pdf_page"), int)]
    blob = _fold_dashes(" ".join([item.get("name") or "", item.get("descriptor") or ""]))
    if _DROP_NAME.search(blob):
        return "excluded_topic"
    quotes = " ".join(source.get("quote") or "" for source in item.get("sources") or [])
    if _GROUP.search(quotes) or _GROUP.search(blob):
        return "group_level"
    if pages and all(page in blocked_pages for page in pages):
        return "governance_page"
    return None


def _printed_for_quote(page: dict, quote: str) -> str | None:
    label = str(page.get("printed_page") or "")
    parts = [part.strip() for part in label.split("/") if part.strip()]
    blocks = page.get("text_blocks") or []
    half = None
    for block in blocks:
        if quote and quote in (block.get("text") or ""):
            half = block.get("half") or "FULL"
            break
    if half is None:
        left = "\n".join((block.get("text") or "") for block in blocks if block.get("half") == "LEFT")
        right = "\n".join((block.get("text") or "") for block in blocks if block.get("half") == "RIGHT")
        if quote and locate_quote(quote, left):
            half = "LEFT"
        elif quote and locate_quote(quote, right):
            half = "RIGHT"
        else:
            half = "FULL"
    if half == "LEFT" and parts:
        return parts[0]
    if half == "RIGHT" and len(parts) > 1:
        return parts[1]
    if len(parts) == 1:
        return parts[0]
    return label or None


def _fold_dashes(text: str) -> str:
    return re.sub(r"[\u2010\u2011\u2012\u2013\u2014]", "-", text)


def _name_key(name: str) -> str:
    text = (name or "").casefold().replace("&", " and ")
    text = re.sub(r"\b(limited|ltd|inc|incorporated|pvt|private)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    tokens = []
    for word in text.split():
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        tokens.append(word)
    return " ".join(sorted(tokens))


def _kind_rank(kind: str | None) -> int:
    order = {"other": 0, "philanthropy": 1, "channel": 2, "research_platform": 3, "business_line": 4, "product_or_service": 5, "brand": 6, "asset": 7, "segment": 8, "subsidiary": 9}
    return order.get(kind or "", 0)


def _first_page(item: dict):
    sources = item.get("sources") or []
    return sources[0].get("pdf_page") if sources else None


def _text(page: dict) -> str:
    return "\n".join((block.get("text") or "") for block in page.get("text_blocks") or [])
