"""Keep subsidiary kind for list and table rows. Drop the rest."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from src.config.factsheet_keywords import LISTING_PAGE
from src.extract.name_quality import (
    expand_suffix,
    is_running_header,
    is_sentence_fragment,
    is_wrapped_fragment,
    rebuild_wrapped_name,
)
from src.extract.select import page_text
from src.factsheet.names import assignment_key

_ACTIVITY = re.compile(
    r"\b(engaged in|principal activity|nature of business|business of|operates|produces|mining|manufactur\w*)\b",
    re.I,
)


def clean_subsidiary_records(records: list[dict], pages: list[dict]) -> tuple[list[dict], dict]:
    """Return records for the sheet and a count of each rule that fired."""
    by_page = {int(page["pdf_page"]): page for page in pages}
    texts = {number: page_text(page) or "" for number, page in by_page.items()}
    listing = {number for number, text in texts.items() if LISTING_PAGE.search(text)}
    issuer = _issuer_name(texts)
    kept = []
    events = []
    allowed_keys = []
    for record in records:
        candidate = record.get("candidate") if isinstance(record.get("candidate"), dict) else record
        if (candidate or {}).get("kind") != "subsidiary":
            kept.append(record)
            continue
        updated, rule = _decide(record, candidate, texts, listing, issuer)
        events.append({"name": (updated.get("candidate") or updated).get("name") if updated else candidate.get("name") or "", "rule": rule, "original": candidate.get("name") or ""})
        if updated is None:
            continue
        if rule in {"list_no_activity", "descriptor_from_page", "rebuilt_wrapped", "list_keep"}:
            key = assignment_key((updated.get("candidate") or updated).get("name") or "")
            if key:
                allowed_keys.append(key)
        kept.append(updated)
    counts: dict[str, int] = {}
    for event in events:
        counts[event["rule"]] = counts.get(event["rule"], 0) + 1
    return kept, {
        "counts": counts,
        "events": events,
        "list_subsidiary_keys": sorted(set(allowed_keys)),
        "listing_pages": sorted(listing),
    }


def write_clean_report(ticker: str, report: dict, folder: Path) -> Path:
    path = folder / ticker / "subsidiary_clean.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "counts": report["counts"],
        "listing_pages": report["listing_pages"],
        "list_subsidiary_keys": report["list_subsidiary_keys"],
        "events": report["events"],
    }, indent=2), encoding="utf-8")
    keys_path = folder / ticker / "list_subsidiaries.json"
    keys_path.write_text(json.dumps({"keys": report["list_subsidiary_keys"]}, indent=2), encoding="utf-8")
    return path


def sample_subsidiaries(records: list[dict], limit: int = 20, seed: int = 25) -> list[dict]:
    rows = []
    for record in records:
        candidate = record.get("candidate") if isinstance(record.get("candidate"), dict) else record
        if (candidate or {}).get("kind") != "subsidiary":
            continue
        source = (candidate.get("sources") or [{}])[0]
        rows.append({
            "name": candidate.get("name"),
            "descriptor": candidate.get("descriptor"),
            "pdf_page": source.get("pdf_page"),
            "rule": candidate.get("clean_rule") or "",
        })
    picker = random.Random(seed)
    if len(rows) <= limit:
        return rows
    return picker.sample(rows, limit)


_CIN = re.compile(r"\b[A-Z]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}\b")
_ROW_PREFIX = re.compile(
    r"^(?:(?:\(?[ivx\d]+\)?[\).\s]+)|(?:operations|continued)\s+)+",
    re.I,
)
_LEGAL_LINE = re.compile(
    r"^([A-Z][A-Za-z0-9&.'’\-]*(?:[ \t]+(?:&|[A-Z][A-Za-z0-9&.'’\-]*)){0,14}"
    r"[ \t]+(?:Limited|Ltd\.?|Private Limited|Pvt\.?\s*Ltd\.?|Inc\.?|LLC|GmbH))\b"
)


def _decide(record: dict, candidate: dict, texts: dict[int, str], listing: set[int], issuer: str = "") -> tuple[dict | None, str]:
    from src.extract.name_quality import legal_names

    raw_name = str(candidate.get("name") or "")
    name = _clean_name(raw_name)
    cited = _cited_pages(candidate)
    page_blob = "\n".join(texts.get(page, "") for page in cited)
    if is_running_header(name) or not name:
        return None, "drop_header"
    if is_sentence_fragment(name):
        return None, "drop_sentence"
    rule = "list_keep"
    anchor = _list_row_page(name, texts, cited, listing)
    if anchor is None and (is_wrapped_fragment(raw_name) or expand_suffix(name, page_blob)):
        rebuilt = rebuild_wrapped_name(name, page_blob)
        if rebuilt:
            name = _clean_name(rebuilt)
            rule = "rebuilt_wrapped"
        elif is_wrapped_fragment(raw_name):
            return None, "drop_wrapped"
    joined = _join_previous_line(name, texts, cited)
    if joined and joined.casefold() != name.casefold():
        name = joined
        rule = "rebuilt_wrapped"
    elif anchor is not None and _bare_row(name, texts, cited) and expand_suffix(name, page_blob):
        name = _clean_name(expand_suffix(name, page_blob) or name)
        rule = "rebuilt_wrapped"
    elif anchor is not None and _bare_row(name, texts, cited):
        return None, "drop_wrapped"
    if is_running_header(name) or _header_only(name, texts, cited) or _same_issuer(name, issuer):
        return None, "drop_header"
    row_page = _list_row_page(name, texts, cited, listing) or anchor
    if rule == "rebuilt_wrapped" and anchor is not None:
        row_page = anchor
    if row_page is None and not _from_table(record, candidate):
        if rule == "rebuilt_wrapped" or is_wrapped_fragment(raw_name) or not legal_names(name):
            return None, "drop_wrapped"
        return None, "drop_not_list"
    printed = _printed_row_name(name, texts.get(row_page, "") if row_page is not None else "")
    if printed:
        name = printed
    phrase = _activity_phrase(name, texts.get(row_page, "") if row_page is not None else "")
    descriptor = phrase if phrase else ""
    if phrase:
        rule = "descriptor_from_page" if rule == "list_keep" else rule
    elif rule == "list_keep":
        rule = "list_no_activity"
    cloned = json.loads(json.dumps(record))
    target = cloned.get("candidate") if isinstance(cloned.get("candidate"), dict) else cloned
    target["name"] = name
    target["entity"] = name
    target["descriptor"] = descriptor
    target["list_subsidiary"] = True
    target["clean_rule"] = rule
    return cloned, rule


def _clean_name(name: str) -> str:
    cleaned = _CIN.sub(" ", name or "")
    cleaned = _ROW_PREFIX.sub("", " ".join(cleaned.split()))
    parts = cleaned.split()
    while parts and "." in parts[0] and parts[0].casefold() not in {"ltd.", "inc.", "pvt."}:
        parts = parts[1:]
    return " ".join(parts)


def _issuer_name(texts: dict[int, str]) -> str:
    pattern = re.compile(
        r"([A-Z][A-Za-z0-9&.'’\- ]{2,80}? Limited)\s+Integrated Annual Report",
    )
    for text in texts.values():
        match = pattern.search(text or "")
        if match:
            return " ".join(match.group(1).split())
    return ""


def _same_issuer(name: str, issuer: str) -> bool:
    from src.factsheet.names import assignment_key

    return bool(issuer) and assignment_key(name) == assignment_key(issuer)


def _header_only(name: str, texts: dict[int, str], cited: list[int]) -> bool:
    hits = 0
    header_hits = 0
    folded = name.casefold()
    for page in cited:
        for line in (texts.get(page) or "").splitlines():
            if folded not in " ".join(line.split()).casefold():
                continue
            hits += 1
            if is_running_header(line):
                header_hits += 1
    return hits > 0 and hits == header_hits


def _list_row_page(name: str, texts: dict[int, str], cited: list[int], listing: set[int]) -> int | None:
    from src.factsheet.names import assignment_key

    wanted = assignment_key(name)
    if not wanted:
        return None
    for page in cited:
        if page not in listing:
            continue
        for line in (texts.get(page) or "").splitlines():
            if is_running_header(line):
                continue
            row_name = _row_name(line)
            if row_name and assignment_key(row_name) == wanted:
                return page
    return None


def _row_name(line: str) -> str:
    compact = " ".join((line or "").split())
    compact = _ROW_PREFIX.sub("", compact)
    compact = " ".join(_CIN.sub(" ", compact).split())
    match = _LEGAL_LINE.match(compact)
    if not match:
        return ""
    tail = compact[match.end():]
    if re.search(r"\b(is|was|are|were|being|shall|has been|have been|incorporated)\b", tail, re.I):
        return ""
    if len(re.findall(r"[A-Za-z]{3,}", tail)) > 8:
        return ""
    return " ".join(match.group(1).split())


def _printed_row_name(name: str, text: str) -> str:
    from src.factsheet.names import assignment_key

    wanted = assignment_key(name)
    for line in (text or "").splitlines():
        if is_running_header(line):
            continue
        row_name = _row_name(line)
        if row_name and assignment_key(row_name) == wanted:
            return row_name
    return name


def _join_previous_line(name: str, texts: dict[int, str], cited: list[int]) -> str | None:
    """Join a wrapped row onto a previous line that is a name prefix."""
    from src.extract.name_quality import legal_names
    from src.factsheet.names import assignment_key

    for page in cited:
        lines = (texts.get(page) or "").splitlines()
        for index, line in enumerate(lines):
            row_name = _row_name(line)
            if not row_name or assignment_key(row_name) != assignment_key(name) or index == 0:
                continue
            if not _bare_line(line):
                continue
            previous = " ".join(lines[index - 1].split())
            if not _good_prefix(previous):
                continue
            found = legal_names(f"{previous} {row_name}")
            if not found:
                continue
            longest = max(found, key=len)
            if len(longest.split()) > len(name.split()):
                return longest
    return None


def _good_prefix(text: str) -> bool:
    from src.extract.name_quality import legal_names

    if not text or re.search(r"\d", text) or legal_names(text) or is_running_header(text):
        return False
    if re.search(r"\b(?:the|statement|certified|management|investment|incorporation|unaudited)\b", text, re.I):
        return False
    words = re.findall(r"[A-Za-z]{2,}", text)
    return 2 <= len(words) <= 8 and len(text) <= 80


def _bare_line(line: str) -> bool:
    """A wrapped fragment line has the name and no figures after it."""
    row_name = _row_name(line)
    if not row_name:
        return False
    compact = " ".join(line.split())
    found = compact.casefold().find(row_name.casefold())
    tail = compact[found + len(row_name):] if found >= 0 else ""
    return not re.search(r"\d", tail)


def _bare_row(name: str, texts: dict[int, str], cited: list[int]) -> bool:
    from src.factsheet.names import assignment_key

    wanted = assignment_key(name)
    saw_bare = False
    for page in cited:
        for line in (texts.get(page) or "").splitlines():
            row_name = _row_name(line)
            if not row_name or assignment_key(row_name) != wanted:
                continue
            if not _bare_line(line):
                return False
            saw_bare = True
    return saw_bare


def _cited_pages(candidate: dict) -> list[int]:
    pages = []
    for source in candidate.get("sources") or []:
        page = source.get("pdf_page")
        if isinstance(page, int) and not isinstance(page, bool):
            pages.append(page)
    return pages


def _from_table(record: dict, candidate: dict) -> bool:
    blob = " ".join(
        str(candidate.get(field) or "")
        for field in ("candidate_id", "batch_id")
    )
    blob += " " + str(record.get("candidate_id") or "")
    return "_tables_" in blob or str(record.get("source") or "") == "tables"


def _activity_phrase(name: str, text: str) -> str:
    if not text or name.casefold() not in text.casefold():
        return ""
    folded = text.casefold()
    start_at = 0
    while True:
        index = folded.find(name.casefold(), start_at)
        if index < 0:
            return ""
        left = max(text.rfind(".", 0, index), text.rfind("\n", 0, index))
        right = text.find(".", index + len(name))
        sentence = " ".join(text[left + 1: right if right >= 0 else index + len(name) + 160].split())
        start_at = index + len(name)
        if sentence.casefold() == name.casefold():
            continue
        if _ACTIVITY.search(sentence) and name.casefold() in sentence.casefold():
            return sentence[:180]
    return ""
