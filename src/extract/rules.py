"""Read subsidiary lists and the segment note without calling a model."""

from __future__ import annotations

import re

from src.extract.select import entity_spans, page_text
from src.validate.quotes import fold_whitespace

_SEGMENT = re.compile(r"\((\d+)\)\s+([A-Za-z][^:\n]{2,60}?)\s+segment\s*:", re.I)
_WORDS = re.compile(r"\S+")
_CORP_END = re.compile(r"\b(Limited|Ltd\.?|Pvt\.?|Private Limited|Inc\.?|LLC|GmbH)\s*$", re.I)
_PLANT_END = re.compile(r"\b(Plant|Mine)\s*$", re.I)
_DIVISION_END = re.compile(r"\bDivision\s*$", re.I)
_UNIT_END = re.compile(r"\bUnit\s*$", re.I)
_DROP_NAMES = re.compile(r"performance stock unit", re.I)
_ERSTWHILE = re.compile(r"^\s*erstwhile\s+", re.I)
_LIST_PAGE = re.compile(
    r"form aoc-?\s*1|investment in subsidiaries|investment in associates|joint ventures|"
    r"list of material unlisted subsidiar|statement containing salient features",
    re.I,
)
_SUB_CONTEXT = re.compile(
    r"form aoc-?\s*1|investment in subsidiaries|investment in associates|joint ventures|"
    r"list of material unlisted subsidiar|statement containing salient features|"
    r"\bsubsidiar(?:y|ies)\b|\bassociate(?:s|d)?\b|\bwholly[- ]owned\b",
    re.I,
)
_NAME = re.compile(
    r"(?m)^[ \t]*([A-Z][A-Za-z0-9&.'’\- ]{2,80}?(?:Limited|Ltd|Inc|LLC|GmbH|Private Limited|Pvt\.? Limited))\b"
)
_NAME_ANY = re.compile(
    r"\b([A-Z][A-Za-z0-9&.'’\-]*(?:[ \t]+(?:&|[A-Z][A-Za-z0-9&.'’\-]*)){0,12}"
    r"[ \t]+(?:Limited|Ltd\.?|Inc\.?|LLC|GmbH|Private Limited|Pvt\.? Limited))\b"
)
_JUNK_CORP = re.compile(
    r"^(?:to the members of|for |and |the |of |in |with |including )\b",
    re.I,
)
_REVENUE_LINE = re.compile(
    r"(?i)revenue for hindalco.s (aluminium upstream|aluminium downstream|copper|novelis) segment was.{0,80}?(?:₹|`|Rs\.?)\s*(\d[\d,]*)"
)
_SEGMENT_REVENUE_GENERIC = re.compile(
    r"(?i)^\s*([A-Za-z][A-Za-z0-9&/ \-]{2,40}?)\s+(?:segment\s+)?(?:revenue|sales).{0,40}?(?:₹|`|Rs\.?|INR)?\s*([\d,]+(?:\.\d+)?)\b"
)
_COAL_SEGMENT_LINE = re.compile(
    r"(?i)^\s*(coal mining|other(?:s)?|power|trading|coking|non[- ]coking)(?:\s+segment)?\s+"
    r".{0,60}?([\d,]+(?:\.\d+)?)\s*$"
)


def _fold_wrapped_lines(text: str) -> str:
    """Join soft wraps (next line continues lowercase); keep new entity lines separate."""
    lines = [line.strip() for line in (text or "").splitlines()]
    out: list[str] = []
    buf = ""
    for line in lines:
        if not line:
            if buf:
                out.append(buf)
                buf = ""
            continue
        if not buf:
            buf = line
            continue
        # A wrap continues mid-phrase; a new capitalised line is a new entry.
        if line[0].islower() and not (_CORP_END.search(buf) or buf.endswith((".", ":", ";"))):
            buf = fold_whitespace(f"{buf} {line}")
        else:
            out.append(buf)
            buf = line
    if buf:
        out.append(buf)
    return "\n".join(out)


def fold_block_text(page: dict) -> str:
    """Join line wraps inside each block, then join blocks. Used for name matching."""
    blocks = [
        block
        for block in page.get("text_blocks") or []
        if block.get("selected", True) and (block.get("text") or "").strip()
    ]
    blocks.sort(key=lambda block: block.get("reading_order", 0))
    parts = []
    for block in blocks:
        joined = _fold_wrapped_lines(block.get("text") or "")
        if joined:
            parts.append(joined)
    return "\n".join(parts)


def extract_rules(pages: list[dict]) -> list[dict]:
    """Read subsidiary, plant, and segment names from every page, including pages the filter drops."""
    drafts = []
    for page in pages:
        folded = fold_block_text(page)
        if not folded:
            continue
        number = int(page["pdf_page"])
        printed = page.get("printed_page")
        list_page = bool(_LIST_PAGE.search(folded))
        sub_context = False
        if list_page:
            for match in list(_NAME.finditer(folded)) + list(_NAME_ANY.finditer(folded)):
                name = _normalize_name(_strip_leading_label(" ".join(match.group(1).split())))
                name = _accept_list_name(name, folded)
                if not name or not _usable_corporate_name(name):
                    continue
                quote = match.group(1).strip()
                if quote not in folded:
                    quote = name if name in folded else quote
                drafts.append((name, "subsidiary", name, quote, number, printed, folded))
        for name, quote in entity_spans(folded):
            name = _normalize_name(_strip_leading_label(name))
            kind = _kind_for_name(name, list_page=list_page, sub_context=sub_context)
            if kind is None:
                continue
            if kind == "subsidiary" and list_page:
                # Already collected via corporate-name patterns above.
                continue
            if quote not in folded and fold_whitespace(quote) not in folded:
                continue
            drafts.append((name, kind, name, quote, number, printed, folded))
        for match in _SEGMENT.finditer(folded):
            label = " ".join(match.group(2).split())
            quote = _short_span(folded, match.start(), label)
            if quote:
                drafts.append((label, "segment", label, quote, number, printed, folded))
    page_count = max(1, len(pages))
    banner_at = max(8, page_count // 5)
    counts: dict[str, int] = {}
    for name, kind, *_rest in drafts:
        if kind == "subsidiary":
            counts[name.casefold()] = counts.get(name.casefold(), 0) + 1
    found = []
    seen = set()
    for name, kind, descriptor, quote, number, printed, text in drafts:
        if kind == "subsidiary" and counts.get(name.casefold(), 0) >= banner_at:
            continue
        _add(found, seen, name, kind, descriptor, quote, number, printed, text)
    return _merge_erstwhile(found)


def extract_listed(pages: list[dict]) -> dict:
    """Subsidiaries from the AOC-1 list pages, and segments from the segment note."""
    subsidiaries = []
    segments = []
    seen = set()
    for page in pages:
        folded = fold_block_text(page)
        if not folded:
            continue
        number = int(page["pdf_page"])
        printed = page.get("printed_page")
        if _LIST_PAGE.search(folded):
            for match in _NAME.finditer(folded):
                name = _normalize_name(" ".join(match.group(1).split()))
                if len(name.split()) < 2 or name.casefold().startswith("hindalco industries"):
                    continue
                if name.casefold().startswith(("associates of", "investment in", "list of")):
                    continue
                quote = match.group(1).strip()
                if quote not in folded:
                    quote = name if name in folded else quote
                _add(subsidiaries, seen, name, "subsidiary", name, quote, number, printed, folded)
        for match in _SEGMENT.finditer(folded):
            label = " ".join(match.group(2).split())
            quote = _short_span(folded, match.start(), label)
            if quote:
                _add(segments, seen, label, "segment", label, quote, number, printed, folded)
    revenue = _printed_revenue(pages)
    _apply_share_pct(revenue)
    return {"subsidiaries": subsidiaries, "segments": segments, "revenue": revenue}


def _kind_for_name(name: str, *, list_page: bool, sub_context: bool = False) -> str | None:
    if _DROP_NAMES.search(name):
        return None
    if _UNIT_END.search(name) and not _CORP_END.search(name):
        return None
    if _CORP_END.search(name):
        if not _usable_corporate_name(name):
            return None
        # Restore real corporate names on AOC-1/list pages and subsidiary-context pages.
        return "subsidiary" if (list_page or sub_context) else None
    if _PLANT_END.search(name):
        return "asset"
    if _DIVISION_END.search(name):
        return "business_line"
    return None


def _accept_list_name(name: str, text: str) -> str:
    from src.extract.name_quality import (
        is_running_header,
        is_sentence_fragment,
        is_wrapped_fragment,
        rebuild_wrapped_name,
    )

    if not name or is_running_header(name) or is_sentence_fragment(name):
        return ""
    if is_wrapped_fragment(name):
        return rebuild_wrapped_name(name, text) or ""
    return name


def _usable_corporate_name(name: str) -> bool:
    words = (name or "").split()
    if len(words) < 2:
        return False
    folded = name.casefold()
    if folded.startswith("hindalco industries"):
        return False
    if folded.startswith(("associates of", "investment in", "list of", "subsidiaries ", "to the members")):
        return False
    if _JUNK_CORP.search(name):
        return False
    # Drop truncated junk such as "Private Ltd" / "Company Private Limited".
    stem = re.sub(
        r"\b(Limited|Ltd\.?|Pvt\.?|Private Limited|Inc\.?|LLC|GmbH)\s*$",
        "",
        name,
        flags=re.I,
    ).strip().casefold()
    if stem in {"", "private", "company", "companies", "the company", "and company"}:
        return False
    return bool(_CORP_END.search(name))


def _normalize_name(name: str) -> str:
    cleaned = _ERSTWHILE.sub("", name or "").strip()
    return " ".join(cleaned.split())


def _strip_leading_label(name: str) -> str:
    return re.sub(
        r"^(?:Subsidiaries|Subsidiary|Associates|Associate|Investments?|Joint Ventures?|"
        r"To the Members of|For )\s+",
        "",
        name or "",
        flags=re.I,
    ).strip()


def _merge_erstwhile(items: list[dict]) -> list[dict]:
    """Keep one record per normalized name; prefer the non-erstwhile spelling."""
    by_key: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    for item in items:
        key = (item["name"].casefold(), item["kind"])
        current = by_key.get(key)
        if current is None:
            by_key[key] = item
            order.append(key)
            continue
        # Merge sources (cap 3).
        sources = list(current.get("sources") or [])
        for source in item.get("sources") or []:
            if source not in sources:
                sources.append(source)
        current["sources"] = sources[:3]
    return [by_key[key] for key in order]


def _printed_revenue(pages: list[dict]) -> dict[str, dict]:
    """A figure is kept only when the segment name and the number share a line."""
    found = {}
    for page in pages:
        folded = fold_block_text(page)
        segment_page = bool(re.search(r"reportable segment|segment information|operating segments", folded, re.I))
        for line in folded.splitlines():
            match = _REVENUE_LINE.search(line)
            if match and match.group(1).casefold() not in found:
                number = float(match.group(2).replace(",", ""))
                found[match.group(1).casefold()] = {
                    "name": match.group(1),
                    "share_pct": None,
                    "revenue": number,
                    "unit": "INR crore",
                    "quote": line.strip(),
                }
                continue
            if not segment_page:
                continue
            match = _COAL_SEGMENT_LINE.search(line)
            if match and match.group(1).casefold() not in found:
                label = " ".join(match.group(1).split())
                number = float(match.group(2).replace(",", ""))
                found[label.casefold()] = {
                    "name": label.title() if label.casefold() != "others" else "Others",
                    "share_pct": None,
                    "revenue": number,
                    "unit": "INR crore",
                    "quote": line.strip(),
                }
    return found


def _apply_share_pct(revenue: dict[str, dict]) -> None:
    total = sum(float(row["revenue"]) for row in revenue.values() if row.get("revenue") is not None)
    if total <= 0:
        return
    for row in revenue.values():
        row["share_pct"] = round(100.0 * float(row["revenue"]) / total, 2)


def short_names(candidates: list[dict]) -> list[str]:
    names = []
    for candidate in candidates:
        name = (candidate.get("name") or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def _add(found: list[dict], seen: set, name: str, kind: str, descriptor: str, quote: str, pdf_page: int, printed, text: str) -> None:
    key = (name.casefold(), kind, pdf_page)
    folded_quote = fold_whitespace(quote)
    if key in seen:
        return
    if quote not in text and folded_quote not in text:
        return
    seen.add(key)
    found.append(
        {
            "name": name,
            "kind": kind,
            "descriptor": " ".join(descriptor.split()[:12]),
            "entity": name if kind == "subsidiary" else "company",
            "status": "current",
            "revenue_share_pct": None,
            "sources": [{"pdf_page": pdf_page, "printed_page": None if printed is None else str(printed), "quote": quote}],
            "supporting_figures": [],
        }
    )


def _short_span(text: str, start: int, label: str) -> str | None:
    window = text[start : start + 220]
    words = _WORDS.findall(window)
    for count in range(min(15, len(words)), 1, -1):
        span = " ".join(words[:count])
        if span in text and label.casefold() in span.casefold():
            return span
    needle = f"{label} segment"
    if needle in text:
        return needle
    return label if label in text else None
