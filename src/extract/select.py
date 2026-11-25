"""Choose pages before any model call. Dropped pages are listed, not discarded quietly."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.config import select_keywords as keywords

HIGH_YIELD = {"business_overview", "mda", "directors_report", "segment_note", "subsidiaries"}
LOW_YIELD = {
    "auditor_report",
    "csr",
    "governance",
    "accounting_policy",
    "related_party",
    "remuneration",
    "tax",
    "financial_instruments",
    "notes_other",
    "toc",
    "front",
}

_HEADINGS = (
    (re.compile(r"directors['’] report|board['’]s report", re.I), "directors_report"),
    (re.compile(r"management discussion|\bmd\s*&\s*a\b", re.I), "mda"),
    (re.compile(r"business overview|our business|company overview", re.I), "business_overview"),
    (re.compile(r"reportable segment|segment information|operating segments", re.I), "segment_note"),
    (re.compile(r"form aoc-?\s*1|investment in subsidiaries|list of .*subsidiar", re.I), "subsidiaries"),
    (re.compile(r"independent auditor|key audit matters|auditor['’]s report", re.I), "auditor_report"),
    (re.compile(r"corporate social responsibility|annual report on csr", re.I), "csr"),
    (re.compile(r"corporate governance", re.I), "governance"),
    (re.compile(r"accounting polic", re.I), "accounting_policy"),
    (re.compile(r"related party", re.I), "related_party"),
    (re.compile(r"\bremuneration\b", re.I), "remuneration"),
    (re.compile(r"\bincome tax\b|tax expense", re.I), "tax"),
    (re.compile(r"financial instrument", re.I), "financial_instruments"),
    (re.compile(r"notes forming part|notes to the (?:standalone|consolidated)", re.I), "notes_other"),
)
_MARKERS = (
    (re.compile(r"form aoc-?\s*1|investment in subsidiaries|list of material unlisted subsidiar|joint ventures and associates", re.I), "subsidiaries"),
    (re.compile(r"reportable segment|segment information|operating segments", re.I), "segment_note"),
)
_STRONG_HEADING = re.compile(
    r"directors['’] report|board['’]s report|management discussion|\bmd\s*&\s*a\b|"
    r"business overview|reportable segment|form aoc-?\s*1|independent auditor|"
    r"corporate social responsibility|corporate governance|notes forming part|"
    r"notes to the (?:standalone|consolidated)",
    re.I,
)
_POSITIVE = keywords.compile_positive()
_NEGATIVE = keywords.compile_negative()
_ESG_HR = keywords.compile_esg_hr()
_CAPACITY = re.compile(keywords.CAPACITY_PATTERN, re.I)
_PRODUCT_PORTFOLIO = re.compile(keywords.PRODUCT_PORTFOLIO_PATTERN, re.I)
_REVENUE = re.compile(keywords.REVENUE_PATTERN, re.I)
_CSR_SCOPE = re.compile(keywords.CSR_SCOPE_PATTERN, re.I)
_CSR_SPEND = re.compile(keywords.CSR_SPEND_PATTERN, re.I)
_RD = re.compile(keywords.RD_PATTERN, re.I)
_PRIORITY = keywords.compile_priority()
_OCR_COMMON = {
    "the", "and", "of", "to", "in", "a", "for", "is", "on", "with", "as", "by", "or",
    "an", "be", "at", "from", "that", "this", "are", "was", "were", "company", "limited",
}
_DOTS = re.compile(r"\.{4,}")
_NOTE = re.compile(r"^\d+[A-Z]?\.\s+\S")
_NOTES_BEGIN = re.compile(r"notes forming part|notes to the (?:standalone|consolidated)|additional notes to the", re.I)
_GOVERNANCE_NOTICE = re.compile(keywords.GOVERNANCE_NOTICE_PATTERN, re.I)
PROTECTED_ENTITY_HITS = 3
_CORP_SUFFIX = r"(?:Limited|Ltd\.?|Pvt\.?|Inc\.?|LLC|GmbH|Private Limited)"
_ENTITY_NAME = re.compile(
    rf"\b([A-Z][A-Za-z0-9.'’\-]*(?:[ \t]+(?:&|[A-Z][A-Za-z0-9.'’\-]*)){{0,12}}[ \t]+(?:{_CORP_SUFFIX}|Division|Plant|Mine|Unit))\b"
)
_CORP_NAME = re.compile(
    rf"\b([A-Z][A-Za-z0-9.'’\-]*(?:[ \t]+(?:&|[A-Z][A-Za-z0-9.'’\-]*)){{0,12}}[ \t]+{_CORP_SUFFIX})\b"
)
_ENTITY_KEYWORD = re.compile(
    r"\b(?:Plants?|Divisions?|Mines?|Units?|Subsidiaries|Subsidiary|Joint Ventures?)\b",
    re.I,
)
_AOC_OR_SEGMENT = re.compile(
    r"form aoc-?\s*1|reportable segment|segment information|operating segments|"
    r"investment in subsidiaries|list of material unlisted subsidiar",
    re.I,
)


def page_text(page: dict) -> str:
    """Original block text, in reading order. Cleanup must not use this string as a quote source."""
    blocks = [block for block in page.get("text_blocks") or [] if block.get("selected", True) and (block.get("text") or "").strip()]
    blocks.sort(key=lambda block: block.get("reading_order", 0))
    return "\n".join(block["text"] for block in blocks)


_LEGAL_END = re.compile(r"^(.*\b(?:Limited|Ltd\.?|Inc\.?|LLC|GmbH))\b")
_PVT_END = re.compile(r"^(.*?\bPvt\.?)\b")
_SITE_END = re.compile(r"^(.*?\b(?:Division|Plant|Mine|Unit))\b")


def entity_spans(text: str) -> list[tuple[str, str]]:
    """Display name and the exact substring. An ampersand stays inside the name."""
    found = []
    seen = set()
    for match in _ENTITY_NAME.finditer(text or ""):
        raw = _clip_name(match.group(1))
        name = " ".join(raw.split())
        if not name or name.casefold() in seen or raw not in (text or ""):
            continue
        seen.add(name.casefold())
        found.append((name, raw))
    return found


def _clip_name(raw: str) -> str:
    """Stop at Limited when it is present, otherwise at the first plant or division."""
    if re.search(r"\b(?:Limited|Ltd\.?|Inc\.?|LLC|GmbH)\b", raw):
        matched = _LEGAL_END.match(raw)
        return matched.group(1) if matched else raw
    if re.search(r"\bPvt\.?\b", raw):
        matched = _PVT_END.match(raw)
        return matched.group(1) if matched else raw
    matched = _SITE_END.match(raw)
    return matched.group(1) if matched else raw


def corporate_name_count(text: str) -> int:
    """Count Limited/Ltd/Pvt/Inc/LLC/GmbH entity spans only."""
    return len(list(_CORP_NAME.finditer(text or "")))


def entity_hit_count(text: str) -> dict:
    """Corporate spans, generic keyword hits (lower weight), and AOC/segment markers."""
    raw = text or ""
    corp = corporate_name_count(raw)
    covered = [(match.start(), match.end()) for match in _CORP_NAME.finditer(raw)]
    keywords = 0
    for match in _ENTITY_KEYWORD.finditer(raw):
        if any(start <= match.start() < end for start, end in covered):
            continue
        keywords += 1
    markers = 1 if _AOC_OR_SEGMENT.search(raw) else 0
    # Protected only when enough corporate names OR an AOC/segment marker.
    protected = corp >= PROTECTED_ENTITY_HITS or markers > 0
    # Weighted score: corporate names dominate; keywords are weaker and do not protect alone.
    weighted = corp + markers * 3 + min(keywords, 4) * 0.25
    return {
        "corporate": corp,
        "keywords": keywords,
        "markers": markers,
        "protected": protected,
        "weighted": weighted,
        "total": corp + keywords + markers,
    }


def select_pages(pages: list[dict], *, protect: bool = True) -> dict:
    ordered = sorted(pages, key=lambda page: int(page["pdf_page"]))
    by_number = {int(page["pdf_page"]): page for page in ordered}
    rows = []
    notes_started = False
    for page in ordered:
        text = page_text(page)
        letters = sum(character.isalpha() for character in text)
        if _toc(text):
            rows.append(_row(page, "gap", "toc", "toc"))
            continue
        if _NOTES_BEGIN.search(text):
            notes_started = True
        section = _page_section(page, text, notes_started=notes_started)
        rows.append(_decide(page, text, letters, section, protect=protect))
    _mark_low_quality_ocr(rows, by_number)
    _pad(rows, by_number)
    kept = [row["pdf_page"] for row in rows if row["decision"] == "keep"]
    gaps = [row for row in rows if row["decision"] == "gap"]
    return {"pages": rows, "kept": kept, "gaps": gaps}


def write_selection(path: Path, ticker: str, selection: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ticker": ticker, "kept_count": len(selection["kept"]), "gap_count": len(selection["gaps"]), **selection}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def governance_notice_match(text: str) -> str | None:
    """The notice/proxy/e-voting/AGM phrase that marks a page as governance, if any."""
    match = _GOVERNANCE_NOTICE.search(text or "")
    if not match:
        return None
    return " ".join(match.group(0).split())


def _page_section(page: dict, text: str, *, notes_started: bool) -> str:
    """Classify from this page only. Strong headings win; notes reset carry-forward."""
    heading = _heading_section(page)
    marker = _marker_section(text)
    if heading:
        return heading
    if marker:
        return marker
    if governance_notice_match(text):
        return "governance"
    if notes_started or _NOTES_BEGIN.search(text):
        return "notes_other"
    if _note_reset(page):
        return "notes_other"
    return "front"


def _recovered_chars(text: str) -> int:
    return len((text or "").strip())


def _page_labels(page: dict) -> set[str]:
    labels = set(page.get("page_type") or [])
    labels.update(page.get("classifications") or [])
    return {str(label).upper() for label in labels}


def _is_nav_cover_index_toc(page: dict, text: str) -> bool:
    """True for TOC/cover/index-style pages that should stay gapped as front matter."""
    if _toc(text):
        return True
    labels = _page_labels(page)
    if labels & keywords.NAV_LABELS:
        return True
    return False


def _maybe_keep_recovered_front(page: dict, text: str, section: str, hits: dict, score: int) -> dict | None:
    """Do not gap recovered front pages that have real body text and are not nav/cover/TOC."""
    if section != "front":
        return None
    if _recovered_chars(text) < keywords.FRONT_RECOVERED_CHARS:
        return None
    if _is_nav_cover_index_toc(page, text):
        return None
    if score < keywords.FRONT_KEEP_SCORE and _is_nav_cover_index_toc(page, text):
        return None
    # Keep when score clears the floor, or when the page is not cover/index/TOC (already checked).
    if score >= keywords.FRONT_KEEP_SCORE or not _is_nav_cover_index_toc(page, text):
        return _row(
            page,
            "keep",
            "front_recovered",
            section,
            max(score, _rank(text, "front_recovered")),
            hits["total"],
            hit_detail=hits,
        )
    return None


def _has_selected_vision(page: dict) -> bool:
    """Legacy vision_ocr boost only. source=vision is context-only and does not force keep."""
    return any(
        block.get("selected") and block.get("source") == "vision_ocr" and (block.get("text") or "").strip()
        for block in page.get("text_blocks") or []
    )


def _decide(page: dict, text: str, letters: int, section: str, *, protect: bool = True) -> dict:
    hits = entity_hit_count(text)
    # Prefer OCR/digital ranking. Vision (source=vision) is context-only and must not
    # force a keep; legacy vision_ocr still gets a mild boost until pages are retiered.
    if _has_selected_vision(page):
        return _row(
            page,
            "keep",
            "vision_ocr",
            section,
            _rank(text, "vision_ocr") + 50,
            hits["total"],
            hit_detail=hits,
        )
    notice = governance_notice_match(text)
    # Notice, proxy, e-voting, and AGM language is skipped even when a low-yield
    # heading (CSR, remuneration) also matches. A high-yield section such as the
    # directors' report can mention an AGM without being the notice itself.
    if notice and section not in HIGH_YIELD:
        return _row(
            page,
            "gap",
            "governance_notice",
            "governance",
            _rank(text, "governance_notice"),
            hits["total"],
            hit_detail=hits,
        )
    if letters < 15 and _recovered_chars(text) < keywords.FRONT_RECOVERED_CHARS:
        return _row(page, "gap", "empty_or_short", section, entity_hits=hits["total"], hit_detail=hits)
    # Protected entities require corporate-name density or AOC/segment markers.
    # They must not override LOW_YIELD sections via generic plant/unit keywords alone.
    if protect and hits["protected"] and section not in LOW_YIELD:
        return _row(
            page,
            "keep",
            "protected_entities",
            section,
            _rank(text, "protected_entities") + 80,
            hits["total"],
            hit_detail=hits,
        )
    if protect and hits["protected"] and section in LOW_YIELD and hits["markers"]:
        # AOC-1 / segment note markers still keep even inside notes.
        return _row(
            page,
            "keep",
            "protected_entities",
            section,
            _rank(text, "protected_entities") + 80,
            hits["total"],
            hit_detail=hits,
        )
    if section == "csr":
        if _CSR_SCOPE.search(text):
            return _row(page, "keep", "csr_in_scope", section, _rank(text, "csr_in_scope"), hits["total"], hit_detail=hits)
        # Gap only when spend/policy boilerplate dominates and R&D keywords are absent.
        if _RD.search(text) or not _csr_spend_dominates(text):
            reason = "csr_with_research" if _RD.search(text) else "csr_not_boilerplate"
            return _row(page, "keep", reason, section, _rank(text, reason), hits["total"], hit_detail=hits)
        return _row(page, "gap", "csr_boilerplate", section, _rank(text, "csr_boilerplate"), hits["total"], hit_detail=hits)
    if section == "remuneration":
        if _CSR_SCOPE.search(text):
            return _row(page, "keep", "remuneration_in_scope", section, _rank(text, "remuneration_in_scope"), hits["total"], hit_detail=hits)
        return _row(page, "gap", "remuneration_boilerplate", section, _rank(text, "remuneration_boilerplate"), hits["total"], hit_detail=hits)
    if section == "auditor_report":
        return _row(page, "gap", "section:auditor_report", section, _rank(text, "section:auditor_report"), hits["total"], hit_detail=hits)
    if section in HIGH_YIELD:
        return _row(page, "keep", f"section:{section}", section, _rank(text, f"section:{section}"), hits["total"], hit_detail=hits)
    if _digit_ratio(text) > 0.5 and not _REVENUE.search(text):
        gap = _row(page, "gap", "numeric_page", section, _rank(text, "numeric_page"), hits["total"], hit_detail=hits)
        kept = _maybe_keep_recovered_front(page, text, section, hits, int(gap["score"] or 0))
        return kept or gap
    signal = (
        len(set(match.group(0).lower() for match in _POSITIVE.finditer(text)))
        - 2 * len(set(match.group(0).lower() for match in _NEGATIVE.finditer(text)))
        - len(set(match.group(0).lower() for match in _ESG_HR.finditer(text)))
        + int(hits["weighted"] // 2)
    )
    if section in LOW_YIELD and signal <= 0:
        gap = _row(page, "gap", f"section:{section}", section, _rank(text, f"section:{section}"), hits["total"], hit_detail=hits)
        kept = _maybe_keep_recovered_front(page, text, section, hits, int(gap["score"] or 0))
        return kept or gap
    if signal > 0:
        return _row(page, "keep", "keyword", section, _rank(text, "keyword"), hits["total"], hit_detail=hits)
    gap = _row(page, "gap", "no_business_signal", section, _rank(text, "no_business_signal"), hits["total"], hit_detail=hits)
    kept = _maybe_keep_recovered_front(page, text, section, hits, int(gap["score"] or 0))
    return kept or gap


def _pad(rows: list[dict], by_number: dict) -> None:
    kept = [row["pdf_page"] for row in rows if row["decision"] == "keep"]
    index = {row["pdf_page"]: row for row in rows}
    blocked = {
        "empty_or_short",
        "toc",
        "section:auditor_report",
        "csr_boilerplate",
        "remuneration_boilerplate",
        "low_quality_ocr",
        "governance_notice",
    }
    for number in kept:
        for neighbor in (number - 1, number + 1):
            row = index.get(neighbor)
            if row is None or neighbor not in by_number:
                continue
            if row["decision"] == "keep":
                continue
            if row["reason"] in blocked or row.get("section") == "auditor_report":
                continue
            row["decision"] = "keep"
            row["reason"] = "neighbor_pad"


def _row(
    page: dict,
    decision: str,
    reason: str,
    section: str,
    score: int | None = None,
    entity_hits: int | None = None,
    hit_detail: dict | None = None,
) -> dict:
    text = page_text(page)
    detail = hit_detail or entity_hit_count(text)
    return {
        "pdf_page": int(page["pdf_page"]),
        "decision": decision,
        "reason": reason,
        "section": section,
        "score": _rank(text, reason) if score is None else score,
        "entity_hits": detail["total"] if entity_hits is None else entity_hits,
        "corporate_hits": detail["corporate"],
        "keyword_hits": detail["keywords"],
    }


def _rank(text: str, reason: str) -> int:
    score = 10 * len(set(match.group(0).lower() for match in _POSITIVE.finditer(text)))
    if _PRIORITY.search(text):
        score += 100
    if _PRODUCT_PORTFOLIO.search(text):
        score += 60
    capacity_hits = len(_CAPACITY.findall(text or ""))
    score += min(capacity_hits, 8) * 20
    score -= 8 * len(set(match.group(0).lower() for match in _ESG_HR.finditer(text)))
    if _CSR_SCOPE.search(text):
        score += 15
    if reason == "neighbor_pad":
        score += 40
    return score


def _heading_section(page: dict) -> str | None:
    """Classify from headings on this page only (no carry-forward)."""
    for block in page.get("text_blocks") or []:
        text = (block.get("text") or "").strip()
        if not text:
            continue
        lines = [text] if len(text.split()) <= 12 else [line.strip() for line in text.splitlines()]
        for line in lines:
            if not line or len(line.split()) > 14:
                continue
            for pattern, name in _HEADINGS:
                if pattern.search(line):
                    return name
    return None


def _note_reset(page: dict) -> bool:
    for block in page.get("text_blocks") or []:
        text = (block.get("text") or "").strip()
        if not text or len(text.split()) > 12 or not _NOTE.match(text):
            continue
        if any(pattern.search(text) for pattern, _name in _HEADINGS[:5]):
            return False
        return True
    return False


def _marker_section(text: str) -> str | None:
    for pattern, name in _MARKERS:
        if pattern.search(text):
            return name
    return None


def _csr_spend_dominates(text: str) -> bool:
    """True when CSR spend/policy phrases outnumber research mentions."""
    spend = len(_CSR_SPEND.findall(text or ""))
    research = len(_RD.findall(text or ""))
    return spend > 0 and spend > research


def _ocr_selected(page: dict) -> bool:
    source = str((page.get("quality_metadata") or {}).get("selected_text_source") or "").lower()
    if source == "ocr":
        return True
    blocks = [
        block for block in page.get("text_blocks") or []
        if block.get("selected", True) and (block.get("text") or "").strip()
    ]
    if not blocks:
        return False
    sources = {str(block.get("source") or "").lower() for block in blocks}
    return bool(sources) and sources <= {"tesseract", "paddleocr", "ocr", "ocr_mistral", "ocr_vision"}


def ocr_quality(text: str) -> dict:
    """Latin-letter ratio and a small dictionary-token ratio. Generic, not company-specific."""
    raw = text or ""
    non_space = [char for char in raw if not char.isspace()]
    latin = sum(1 for char in non_space if ("a" <= char.lower() <= "z"))
    latin_ratio = (latin / len(non_space)) if non_space else 0.0
    tokens = re.findall(r"[A-Za-z]{2,}", raw)
    if tokens:
        hits = sum(1 for token in tokens if token.casefold() in _OCR_COMMON)
        dict_ratio = hits / len(tokens)
    else:
        dict_ratio = 0.0
    low = latin_ratio < keywords.OCR_LATIN_MIN or (len(raw.strip()) >= 80 and dict_ratio < keywords.OCR_DICT_MIN)
    return {"latin_ratio": round(latin_ratio, 4), "dict_ratio": round(dict_ratio, 4), "low": low}


def _mark_low_quality_ocr(rows: list[dict], by_number: dict) -> None:
    """Do not send OCR pages below the latin/dictionary floor to the LLM."""
    for row in rows:
        page = by_number.get(int(row["pdf_page"]))
        if page is None or not _ocr_selected(page):
            continue
        quality = ocr_quality(page_text(page))
        row["ocr_quality"] = quality
        if quality["low"] and row["decision"] == "keep":
            row["decision"] = "gap"
            row["reason"] = "low_quality_ocr"


def _toc(text: str) -> bool:
    return len(_DOTS.findall(text)) >= 8


def _digit_ratio(text: str) -> float:
    digits = sum(character.isdigit() for character in text)
    letters = sum(character.isalpha() for character in text)
    if digits + letters == 0:
        return 0.0
    return digits / (digits + letters)
