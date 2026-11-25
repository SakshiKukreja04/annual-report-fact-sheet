"""Generic precision pass applied after validation, before the fact sheet is written."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.config.factsheet_keywords import (
    ACCOUNTING_PAGE,
    AWARD_ITEM,
    BIO_START,
    CORE_PRODUCT,
    COUNTERPARTY,
    FACILITY_TAIL,
    FUTURE_CUE,
    GENERIC_HEAD,
    GENERIC_MODIFIER,
    GOVERNANCE_ITEM,
    LISTING_PAGE,
    PAST_CUE,
    POLICY_TERM,
    SALES_CHANNEL,
    TARGET_LINE,
    TOPIC_GENERIC,
)
from src.extract.select import page_text

_STOP = {"limited", "ltd", "pvt", "private", "the", "of", "and", "a", "an"}
_AMOUNT_END = re.compile(r"(?:rs\.?|inr|₹)?\s*\d[\d,]*(?:\.\d+)?\s*(?:crore|lakh|million|%)?\s*$", re.I)
_ACTIVITY = re.compile(
    r"\b(produces?|operates?|mines?|mining|engaged in|manufactur\w*|generates?|washes|sells?)\b",
    re.I,
)
_JV = re.compile(r"joint venture|\bjv\b", re.I)
_LEGAL = re.compile(
    r"\b([A-Z][A-Za-z0-9&.'’\-]*(?:\s+(?:&|[A-Z][A-Za-z0-9&.'’\-]*)){0,12}"
    r"\s+(?:Limited|Ltd\.?|Private Limited|Pvt\.?\s*Ltd\.?))\b"
)


def apply_precision(records: list[dict], pages: list[dict], ticker: str = "") -> tuple[list[dict], dict]:
    """Return kept records and a per-rule log with up to 10 before/after examples."""
    texts = {int(page["pdf_page"]): page_text(page) or "" for page in pages}
    by_page = {int(page["pdf_page"]): page for page in pages}
    log: dict[str, list] = {}
    kept: list[dict] = []
    flipped = []
    for record in records:
        candidate = record.get("candidate") if isinstance(record.get("candidate"), dict) else record
        if not isinstance(candidate, dict) or record.get("outcome") not in (None, "validated"):
            continue
        working = json.loads(json.dumps(record))
        target = working.get("candidate") if isinstance(working.get("candidate"), dict) else working
        renamed = _fix_name(target, ticker, texts.get(_page_of(target), ""))
        if renamed in {"company_name_drop", "fragment_name_drop"}:
            _note(log, renamed, candidate.get("name"), "", "")
            continue
        if renamed:
            _note(log, renamed, candidate.get("name"), target.get("name"), "")
        rule = _repair_quote(target, texts)
        if rule == "drop_quote":
            _note(log, "quote_drop", target.get("name"), target.get("name"), "")
            continue
        fitted = _constrain_quotes(target, texts)
        if fitted == "drop_quote":
            _note(log, "quote_drop", target.get("name"), "", "")
            continue
        if rule or fitted:
            _note(log, fitted or rule, candidate.get("name"), target.get("name"), (target.get("sources") or [{}])[0].get("quote"))
        page = texts.get(_page_of(target), "")
        junk = _junk_rule(target, page)
        if not junk:
            junk = _source_junk(target, texts)
        if junk == "fragment_rebuilt":
            _note(log, junk, candidate.get("name"), target.get("name"), "")
        elif junk:
            _note(log, junk, target.get("name"), "", (target.get("sources") or [{}])[0].get("quote"))
            continue
        _status(target, flipped, log)
        _folio(target, by_page, log)
        _polish_descriptor(target)
        _trim_amount(target)
        if _still_name_only(target):
            _note(log, "name_only_drop", target.get("name"), "", (target.get("sources") or [{}])[0].get("quote"))
            continue
        _sync_quotes(working, target)
        kept.append(working)
    before_jv = len(kept)
    kept = _collapse_joint_ventures(kept, texts, log)
    _note(log, "joint_venture_items", str(before_jv), str(len(kept)), "")
    kept.extend(_sales_and_products(pages, texts, log))
    for record in kept:
        target = record.get("candidate") if isinstance(record.get("candidate"), dict) else record
        if isinstance(target, dict):
            _polish_descriptor(target)
            _sync_quotes(record, target)
    return kept, {"counts": {name: len(rows) for name, rows in log.items()}, "examples": {name: rows[:10] for name, rows in log.items()}, "flipped": flipped}


def write_precision_report(ticker: str, report: dict, folder: Path) -> Path:
    path = folder / ticker / "precision_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def quote_supports_name(name: str, quote: str) -> bool:
    tokens = _name_tokens(name)
    if not tokens:
        return True
    folded = (quote or "").casefold()
    hits = sum(1 for token in tokens if re.search(rf"\b{re.escape(token)}\b", folded))
    return hits / len(tokens) >= 0.8


def generic_noun_only(name: str) -> bool:
    tokens = [token for token in re.sub(r"[^A-Za-z0-9\s]", " ", name or "").split() if token]
    if not 1 <= len(tokens) <= 3 or not GENERIC_HEAD.match(tokens[-1]):
        return False
    return all(token.casefold() in GENERIC_MODIFIER or GENERIC_HEAD.match(token) for token in tokens[:-1])


def _strip_controls(text: str) -> str:
    return "".join(ch for ch in (text or "") if ch >= " " or ch == "\n")


def _head_noun(name: str) -> str:
    tokens = _name_tokens(name)
    return tokens[-1] if tokens else ""


def _mentions_item(name: str, text: str) -> bool:
    if quote_supports_name(name, text):
        return True
    head = _head_noun(name)
    return bool(head) and bool(re.search(rf"\b{re.escape(head)}\b", text or "", re.I))


def _clip_around(name: str, sentence: str, limit: int) -> str:
    flat = " ".join(_strip_controls(sentence).split())
    if len(flat) <= limit:
        return flat
    folded = flat.casefold()
    index = folded.find((name or "").casefold())
    if index < 0:
        head = _head_noun(name)
        found = re.search(rf"\b{re.escape(head)}\b", flat, re.I) if head else None
        index = found.start() if found else 0
    start = max(0, index - 40)
    if start:
        space = flat.find(" ", start)
        start = space + 1 if space >= 0 else start
    end = min(len(flat), start + limit)
    if end < len(flat):
        space = flat.rfind(" ", start, end)
        if space > start:
            end = space
    return flat[start:end].strip(" ,;:-")


def _sentences_of(text: str) -> list[str]:
    compact = " ".join(_strip_controls(text).split())
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", compact) if part.strip()]


def _shortest_sentence(name: str, text: str) -> str:
    options = []
    for part in _sentences_of(text):
        if not _mentions_item(name, part):
            continue
        if BIO_START.match(part) or "integrated annual report" in part.casefold():
            continue
        options.append(part)
    if not options:
        return ""
    return min(options, key=len)


def descriptor_problem(name: str, descriptor: str, quote: str = "") -> str:
    """Empty string when the descriptor is one clean phrase of at most 200 characters."""
    text = " ".join(_strip_controls(descriptor).split())
    if not text:
        return "empty"
    if text.casefold() == " ".join((name or "").split()).casefold():
        return "name only"
    if len(text) > 200:
        return "longer than 200"
    if BIO_START.match(text):
        return "starts with He/She"
    if "integrated annual report" in text.casefold():
        return "annual-report boilerplate"
    if not re.search(r"[\w.!?%)]$", text):
        return "unclean ending"
    if not _mentions_item(name, text):
        return "does not mention the item"
    flat_quote = " ".join((quote or "").split())
    index = flat_quote.casefold().find(text.casefold()) if flat_quote else -1
    if index >= 0:
        end = index + len(text)
        if end < len(flat_quote) and flat_quote[end].isalnum():
            return "truncated"
    return ""


def _polish_descriptor(candidate: dict) -> None:
    name = str(candidate.get("name") or "").strip()
    quote = " ".join(source.get("quote") or "" for source in candidate.get("sources") or [])
    current = " ".join(_strip_controls(str(candidate.get("descriptor") or "")).split())
    if descriptor_problem(name, current, quote) == "":
        candidate["descriptor"] = current
        return
    sentence = _shortest_sentence(name, quote)
    candidate["descriptor"] = _clip_around(name, sentence, 200) if sentence else ""


def _constrain_quotes(candidate: dict, texts: dict[int, str]) -> str:
    kept = []
    changed = False
    name = str(candidate.get("name") or "")
    for source in candidate.get("sources") or []:
        page = texts.get(int(source.get("pdf_page") or 0), "")
        fitted = _sentence_quote(name, str(source.get("quote") or ""), page)
        if not fitted:
            changed = True
            continue
        if fitted != source.get("quote"):
            changed = True
        source["quote"] = fitted
        kept.append(source)
    candidate["sources"] = kept[:3]
    if not kept:
        return "drop_quote"
    return "quote_fitted" if changed else ""


def _sentence_quote(name: str, quote: str, page: str) -> str:
    """The sentence that holds this quote and the item name, clipped to 300 characters."""
    flat_page = " ".join(_strip_controls(page).split())
    flat_quote = " ".join(_strip_controls(quote).split())
    anchor = flat_quote[:80].casefold()
    options = []
    for part in _sentences_of(flat_page):
        if not _mentions_item(name, part):
            continue
        folded = part.casefold()
        if anchor and (anchor[:40] in folded or folded in flat_quote.casefold()):
            options.append(part)
    if not options:
        for part in _sentences_of(flat_quote):
            if _mentions_item(name, part) and (not flat_page or part.casefold() in flat_page.casefold()):
                options.append(part)
    if not options and flat_quote and _mentions_item(name, flat_quote):
        clipped = _clip_around(name, flat_quote, 300)
        if clipped and (not flat_page or clipped.casefold() in flat_page.casefold()):
            options.append(clipped)
    fitted = []
    for part in options:
        if len(part) > 300:
            part = _clip_around(name, part, 300)
        if part and len(part) <= 300 and _mentions_item(name, part):
            if not flat_page or part.casefold() in flat_page.casefold():
                fitted.append(part)
    if not fitted:
        return ""
    return min(fitted, key=len)


def _reporting_company(name: str, ticker: str) -> bool:
    compact = re.sub(r"[^a-z]", "", (name or "").casefold())
    tick = re.sub(r"[^a-z]", "", (ticker or "").casefold())
    if not compact or not tick:
        return False
    if compact in {tick, tick + "limited", tick + "ltd"}:
        return True
    return compact.startswith(tick) and compact.endswith("limited") and len(compact) <= len(tick) + 7


def _fix_name(candidate: dict, ticker: str, page: str) -> str:
    name = " ".join(str(candidate.get("name") or "").split())
    quote = " ".join(source.get("quote") or "" for source in candidate.get("sources") or [])
    cleaned = name
    if re.match(r"financial statements\b", cleaned, re.I):
        cleaned = re.sub(r"^financial statements\.?\s*(?:the\s+)?", "", cleaned, flags=re.I).strip()
    if re.fullmatch(r"[A-Za-z]+\s+Limited", cleaned):
        blob = quote or page or ""
        longer = []
        for match in _LEGAL.finditer(blob):
            full = " ".join(match.group(1).split())
            if full.casefold().count(cleaned.casefold()) != 1:
                continue
            if full.casefold().endswith(cleaned.casefold()) and len(full) > len(cleaned) + 1:
                longer.append(full)
        if longer:
            cleaned = min(longer, key=len)
        elif cleaned.split()[0].casefold() in TOPIC_GENERIC or cleaned.split()[0].casefold() in {"chemicals", "services", "industries", "projects", "systems"}:
            return "fragment_name_drop"
    if _reporting_company(cleaned, ticker):
        return "company_name_drop"
    if cleaned and cleaned != name:
        candidate["name"] = cleaned
        entity = str(candidate.get("entity") or "")
        if entity.casefold() in {name.casefold(), "company", "the company"}:
            candidate["entity"] = cleaned if entity.casefold() == name.casefold() else entity
        return "name_repaired"
    return ""


def _source_junk(candidate: dict, texts: dict[int, str]) -> str:
    name = str(candidate.get("name") or "").strip()
    quote = " ".join(source.get("quote") or "" for source in candidate.get("sources") or [])
    if BIO_START.match(quote.strip()) or BIO_START.match(name):
        return "biography_drop"
    if AWARD_ITEM.search(name) or (AWARD_ITEM.search(quote[:80]) and not _ACTIVITY.search(quote)):
        return "award_drop"
    if _facility_without_proper(name):
        return "generic_facility_drop"
    if candidate.get("kind") == "subsidiary":
        return ""
    pages = []
    for source in candidate.get("sources") or []:
        text = texts.get(int(source.get("pdf_page") or 0), "")
        if text:
            pages.append(text)
    if pages and all(_junk_page(text) for text in pages):
        # A notes page can still carry an operating sentence. Keep that item.
        if re.search(r"\b(joint venture|subsidiar\w*|operates|produces|sale of|customers)\b", quote, re.I):
            return ""
        return "schedule_page_drop"
    return ""


def _junk_page(text: str) -> bool:
    """True when the page heading is notes, CWIP, PPE, litigation, or a related-party disclosure."""
    head = "\n".join((text or "").splitlines()[:6])[:500]
    if LISTING_PAGE.search(head):
        return False
    return bool(re.search(
        r"(?m)^\s*(?:notes to the (?:standalone |consolidated )?financial|capital work|property,\s+plant|"
        r"contingent liabilit|related party disclosure|\blitigation\b)",
        head,
        re.I,
    ))


def _facility_without_proper(name: str) -> bool:
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", name or "")
    if len(tokens) < 2 or not FACILITY_TAIL.search(tokens[-1]):
        return False
    for token in tokens[:-1]:
        folded = token.casefold()
        if folded in GENERIC_MODIFIER or folded in {
            "water", "treatment", "rapid", "loading", "monitoring", "rescue", "medical",
            "regional", "area", "testing", "research", "training", "workshop", "underground", "opencast",
        } or len(token) <= 2:
            continue
        if token.isupper() or token[:1].isupper():
            return False
    return True


def _repair_quote(candidate: dict, texts: dict[int, str]) -> str:
    sources = candidate.get("sources") or []
    if not sources:
        return "drop_quote"
    source = sources[0]
    name = str(candidate.get("name") or "")
    quote = str(source.get("quote") or "")
    page = texts.get(int(source.get("pdf_page") or 0), "")
    widened = _widen(name, quote, page)
    if widened is None:
        return "drop_quote"
    if widened != quote:
        source["quote"] = widened
        candidate["sources"] = sources
        return "quote_widened" if quote_supports_name(name, widened) else "drop_quote"
    if not quote_supports_name(name, quote) or _mid_word(quote, page):
        return "drop_quote"
    return ""


def _widen(name: str, quote: str, page: str) -> str | None:
    cleaned = _snap_words(quote, page) or quote
    if quote_supports_name(name, cleaned) and not _mid_word(cleaned, page):
        return " ".join(cleaned.split())
    if not page:
        return None
    window = _window_for_name(name, page)
    if window and quote_supports_name(name, window) and not _mid_word(window, page):
        return window
    return None


def _snap_words(quote: str, page: str) -> str:
    compact_quote = " ".join((quote or "").split())
    compact_page = " ".join((page or "").split())
    if not compact_quote or not compact_page:
        return compact_quote
    index = compact_page.casefold().find(compact_quote.casefold())
    if index < 0:
        return compact_quote
    start, end = index, index + len(compact_quote)
    while start > 0 and compact_page[start - 1].isalnum():
        start -= 1
    while end < len(compact_page) and compact_page[end].isalnum():
        end += 1
    return compact_page[start:end]


def _mid_word(quote: str, page: str) -> bool:
    compact_quote = " ".join((quote or "").split())
    compact_page = " ".join((page or "").split())
    index = compact_page.casefold().find(compact_quote.casefold()) if compact_quote and compact_page else -1
    if index < 0:
        return bool(compact_quote) and (compact_quote[:1].islower() or not compact_quote[:1].isalnum())
    before = compact_page[index - 1] if index else " "
    after_at = index + len(compact_quote)
    after = compact_page[after_at] if after_at < len(compact_page) else " "
    return before.isalnum() or after.isalnum()


def _window_for_name(name: str, page: str) -> str:
    tokens = _name_tokens(name)
    folded = page.casefold()
    if name.casefold() in folded:
        index = folded.find(name.casefold())
        start = max(page.rfind(".", 0, index), page.rfind("\n", 0, index))
        end = page.find(".", index + len(name))
        return " ".join(page[start + 1: end if end >= 0 else index + len(name)].split())
    if not tokens:
        return ""
    positions = []
    for token in tokens:
        found = re.search(rf"\b{re.escape(token)}\b", folded)
        if not found:
            return ""
        positions.append(found.start())
    start, end = min(positions), max(positions)
    start = max(page.rfind(" ", 0, start), page.rfind("\n", 0, start))
    end = page.find(" ", end + 1)
    return " ".join(page[start + 1: end if end >= 0 else start + 180].split())


def _name_tokens(name: str) -> list[str]:
    raw = (name or "").casefold().replace("&", " and ")
    raw = re.sub(r"[^a-z0-9\s]", " ", raw)
    return [token for token in raw.split() if token not in _STOP and len(token) > 1]


def _junk_rule(candidate: dict, page: str) -> str:
    name = str(candidate.get("name") or "").strip()
    quote = str((candidate.get("sources") or [{}])[0].get("quote") or "")
    kind = candidate.get("kind") or ""
    folded_quote = " ".join(quote.split())
    if generic_noun_only(name):
        return "generic_noun_drop"
    if ACCOUNTING_PAGE.search(page or "") and kind in {"asset", "product_or_service"}:
        if POLICY_TERM.search(name) or POLICY_TERM.search(quote) or _AMOUNT_END.search(folded_quote):
            if not _ACTIVITY.search(folded_quote):
                return "accounting_schedule_drop"
    if GOVERNANCE_ITEM.search(name):
        return "governance_review"
    if _is_counterparty(name, quote):
        return "counterparty_review"
    if TARGET_LINE.search(name) or TARGET_LINE.search(quote):
        if not re.search(r"\b(project|plant|limited|ltd)\b", name, re.I):
            return "target_drop"
    rebuilt = _rebuild_fragment(name, quote)
    if rebuilt == "":
        return "fragment_drop"
    if rebuilt and rebuilt != name:
        candidate["name"] = rebuilt
        if (candidate.get("entity") or "").casefold() in {name.casefold(), "company", "the company"}:
            candidate["entity"] = rebuilt
        return "fragment_rebuilt"
    return ""


def _is_counterparty(name: str, quote: str) -> bool:
    if not name or name.casefold() not in (quote or "").casefold():
        return False
    return bool(re.search(r"(?:m/s|awarded to|handed over to|contractor)\s+.{0,40}" + re.escape(name), quote, re.I))


def _quote_is_name(name: str, quote: str) -> bool:
    folded = quote.casefold().strip()
    if folded == name.casefold():
        return True
    return bool(re.fullmatch(re.escape(name) + r"\s*\d+", quote, re.I))


def _rebuild_fragment(name: str, quote: str) -> str | None:
    if name[:1].islower() or re.match(r"^(?:mw|gw|kt)\b", name, re.I):
        longer = _LEGAL.search(quote or "")
        if longer:
            return " ".join(longer.group(1).split())
        prefix = re.search(r"\b\d+(?:\.\d+)?\s+(?:MW|GW|KT)\b[^.]{0,40}", quote or "", re.I)
        if prefix and name.casefold() in prefix.group(0).casefold():
            return " ".join(prefix.group(0).split())
        return ""
    return None


def _status(candidate: dict, flipped: list, log: dict) -> None:
    quote = " ".join((source.get("quote") or "") for source in candidate.get("sources") or [])
    current = candidate.get("status") or "current"
    if PAST_CUE.search(quote):
        new = "current"
    elif FUTURE_CUE.search(quote):
        new = "planned"
    else:
        return
    if new != current:
        candidate["status"] = new
        flipped.append({"name": candidate.get("name"), "before": current, "after": new})
        _note(log, "status_flip", f"{candidate.get('name')} [{current}]", new, quote[:160])


def _folio(candidate: dict, pages: dict[int, dict], log: dict) -> None:
    for source in candidate.get("sources") or []:
        page = pages.get(int(source.get("pdf_page") or 0))
        if not page:
            continue
        printed, rule = folio_for_quote(page, source.get("quote") or "")
        before = source.get("printed_page")
        if printed and str(before) != printed:
            source["printed_page"] = printed
            _note(log, "folio_" + rule, f"pdf {source.get('pdf_page')} {before}", printed, "")


def folio_for_quote(page: dict, quote: str) -> tuple[str | None, str]:
    """Pick the left or right folio when the page record stores both."""
    printed = page.get("printed_page")
    if printed in (None, ""):
        return None, "missing_folio"
    text = str(printed)
    if "/" not in text:
        return text, "page_folio"
    left, right = [part.strip() for part in text.split("/", 1)]
    halves = set()
    compact = " ".join((quote or "").split()).casefold()
    for block in page.get("text_blocks") or []:
        block_text = " ".join((block.get("text") or "").split()).casefold()
        if compact and compact[:80] in block_text:
            halves.add(str(block.get("half") or ""))
    if halves == {"LEFT"}:
        return left, "spread_left"
    if halves == {"RIGHT"}:
        return right, "spread_right"
    return text, "spread_pair"


def _collapse_joint_ventures(records: list[dict], texts: dict[int, str], log: dict) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    rest = []
    for record in records:
        candidate = record.get("candidate") or record
        blob = " ".join((source.get("quote") or "") for source in candidate.get("sources") or [])
        blob = f"{blob} {candidate.get('name') or ''}"
        if not _JV.search(blob):
            rest.append(record)
            continue
        key = " ".join(sorted(_name_tokens(candidate.get("name") or [])))
        grouped.setdefault(key or str(id(record)), []).append(record)
    for members in grouped.values():
        if len(members) == 1:
            rest.append(members[0])
            continue
        base = members[0]
        candidate = base.get("candidate") or base
        names = [((member.get("candidate") or member).get("name") or "") for member in members]
        legal = max(names, key=len)
        candidate["name"] = legal
        candidate["entity"] = legal
        candidate["kind"] = "business_line" if all((member.get("candidate") or member).get("kind") != "subsidiary" for member in members) else "subsidiary"
        sentence = _activity_sentence(legal, texts.get(_page_of(candidate), ""))
        if sentence:
            candidate["descriptor"] = sentence[:180]
            sources = list(candidate.get("sources") or [])
            if sentence not in {source.get("quote") for source in sources}:
                sources.append({**sources[0], "quote": sentence[:240]} if sources else {"pdf_page": _page_of(candidate), "printed_page": None, "quote": sentence[:240]})
            candidate["sources"] = sources[:3]
        _note(log, "joint_venture_merge", " | ".join(names), legal, candidate.get("descriptor") or "")
        rest.append(base)
    return rest


def _activity_sentence(name: str, text: str) -> str:
    if not name or name.casefold() not in (text or "").casefold():
        return ""
    folded = text.casefold()
    start = 0
    while True:
        index = folded.find(name.casefold(), start)
        if index < 0:
            return ""
        left = max(text.rfind(".", 0, index), text.rfind("\n", 0, index))
        right = text.find(".", index + len(name))
        sentence = " ".join(text[left + 1: right if right >= 0 else index + 180].split())
        start = index + len(name)
        if _ACTIVITY.search(sentence) and sentence.casefold() != name.casefold():
            return sentence
    return ""


def _sales_and_products(pages: list[dict], texts: dict[int, str], log: dict) -> list[dict]:
    added = []
    channel_pages = sorted(
        ((len(SALES_CHANNEL.findall(text)), number) for number, text in texts.items() if len(SALES_CHANNEL.findall(text)) >= 2),
        reverse=True,
    )
    chosen = [number for _count, number in channel_pages[:4]]
    for number, text in texts.items():
        if re.search(r"(?:\d{1,3}(?:\.\d+)?\s*%\s*\w{0,12}\s*)?(?:e-?\s*auction|\bfsa\b|\bnrs\b|power sector|non[-\s]?power|\bimports?\b)", text, re.I):
            if number not in chosen and len(chosen) < 8 and re.search(r"\d\s*%\s*(?:e-?\s*auction|\bfsa\b|\bnrs\b)|(?:e-?\s*auction|\bfsa\b|\bnrs\b|power sector|non[-\s]?power)\s*\d", text, re.I):
                chosen.append(number)
    best: dict[str, tuple] = {}
    for number in chosen:
        text = texts[number]
        for line in text.splitlines():
            compact = " ".join(line.split())
            channel = SALES_CHANNEL.search(compact)
            if not channel or not re.search(r"\d", compact):
                continue
            label = channel.group(0)
            key = re.sub(r"[\s\-]+", "", label.casefold())
            share, how, denominator = _share_from_line(compact, channel.start())
            if len(SALES_CHANNEL.findall(compact)) >= 2:
                share, how = None, "multi_label_chart"
            percent = re.search(rf"(?<!\d){re.escape(f'{share:g}')}\s*%", compact) if share is not None else None
            if share is not None and percent is None:
                share, how = None, "percent_not_in_quote"
            end = percent.end() if percent else channel.end()
            window = compact[max(0, channel.start() - 40): min(len(compact), end + 80)]
            quote = _verbatim(window, text) or _verbatim(label, text)
            if not quote:
                continue
            if share is not None and not re.search(rf"(?<!\d){re.escape(f'{share:g}')}\s*%", quote[:240]):
                share, how = None, "percent_outside_quote"
            score = (1 if share is not None else 0, min(len(window), 180))
            current = best.get(key)
            if current and current[0] >= score:
                continue
            name = _verbatim(label, text) or label
            best[key] = (score, name, window[:180], number, quote[:240], share, how, denominator)
    for key, (_score, name, descriptor, number, quote, share, how, denominator) in best.items():
        added.append(_made(name, "channel", descriptor, number, pages, quote, share, "printed" if share is not None else how))
        _note(log, "sales_channel", "", name, f"pdf {number} share {share} via {how} denom {denominator}")
    seen = set(best)
    found: dict[str, tuple] = {}
    for number, text in texts.items():
        for line in text.splitlines():
            compact = " ".join(line.split())
            product = CORE_PRODUCT.search(compact)
            if not product:
                continue
            key = product.group(0).casefold()
            if key in seen:
                continue
            window = compact[max(0, product.start() - 40): product.end() + 140]
            if len(window) <= len(product.group(0)) + 8:
                continue
            if _digit_heavy(window) and not _ACTIVITY.search(window):
                continue
            quote = _verbatim(window, text) or _verbatim(product.group(0), text)
            if not quote or not quote_supports_name(product.group(0), quote):
                continue
            score = (1 if _ACTIVITY.search(window) else 0, min(len(window), 180))
            current = found.get(key)
            if current and current[0] >= score:
                continue
            found[key] = (score, product.group(0), window[:180], number, quote[:240])
    for key, (_score, name, descriptor, number, quote) in found.items():
        seen.add(key)
        added.append(_made(name, "product_or_service", descriptor, number, pages, quote, None, ""))
        _note(log, "core_product", "", name, f"pdf {number}")
    return [row for row in added if row]


def _verbatim(needle: str, page: str) -> str:
    compact_page = " ".join((page or "").split())
    compact_needle = " ".join((needle or "").split())
    if not compact_needle:
        return ""
    index = compact_page.casefold().find(compact_needle.casefold())
    if index < 0:
        return ""
    return compact_page[index: index + len(compact_needle)]


def _share_from_line(line: str, anchor: int) -> tuple[float | None, str, str]:
    """A percent is a revenue share only when it sits on the channel label or is named as a share of sales."""
    figures = [float(token.replace(",", "")) for token in re.findall(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+\.\d+\b", line)]
    figure_note = ",".join(str(figure) for figure in figures[:4])
    for match in re.finditer(r"(\d{1,3}(?:\.\d+)?)\s*%", line):
        nearby = line[max(0, match.start() - 50): match.end() + 30]
        if re.search(r"premium|conformity|growth|decline|increase|higher|lower", nearby, re.I):
            continue
        close = abs(match.start() - anchor) <= 24
        named = re.search(r"\b(share|of sales|of revenue|of offtake|of dispatch|accounted for)\b", nearby, re.I)
        if close or named:
            return float(match.group(1)), "printed_percent", ""
    return None, "figures_seen_not_used", figure_note


def _made(name: str, kind: str, descriptor: str, pdf_page: int, pages: list[dict], quote: str, share, how: str) -> dict:
    page = next((item for item in pages if int(item["pdf_page"]) == pdf_page), {})
    printed, _rule = folio_for_quote(page, quote) if page else (None, "")
    record = {
        "outcome": "validated",
        "candidate": {
            "name": name,
            "kind": kind,
            "descriptor": descriptor or name,
            "entity": "company",
            "status": "current",
            "revenue_share_pct": share,
            "sources": [{"pdf_page": pdf_page, "printed_page": printed, "quote": quote[:240]}],
            "candidate_id": f"precision_{kind}_{pdf_page}",
        },
        "quotes": [{"pdf_page": pdf_page, "quote": quote[:240], "accepted": True}],
        "financial": {"status": how or "absent", "revenue_share_pct": share} if share is not None else {},
    }
    return record


def _describe_from_quote(candidate: dict) -> None:
    name = str(candidate.get("name") or "").strip()
    descriptor = str(candidate.get("descriptor") or "").strip()
    quote = " ".join(((candidate.get("sources") or [{}])[0].get("quote") or "").split())
    if not name or descriptor.casefold() != name.casefold():
        return
    if len(quote) <= len(name) + 8 or not quote_supports_name(name, quote):
        return
    candidate["descriptor"] = _around_name(name, quote)[:180]


def _describe_from_page(candidate: dict, texts: dict[int, str]) -> bool:
    """Use a longer page sentence that names the item when the descriptor is only the name."""
    name = str(candidate.get("name") or "").strip()
    descriptor = str(candidate.get("descriptor") or "").strip()
    if not name or (descriptor and descriptor.casefold() != name.casefold()):
        return False
    source_pages = []
    for source in candidate.get("sources") or []:
        page_no = int(source.get("pdf_page") or 0)
        if page_no and page_no not in source_pages:
            source_pages.append(page_no)
    order = source_pages + [page_no for page_no in texts if page_no not in source_pages]
    chosen: tuple[int, str] | None = None
    for page_no in order:
        sentence = _sentence_for_name(name, texts.get(page_no, ""))
        if not sentence:
            continue
        cleaned = _strip_trailing_amounts(sentence)
        if len(cleaned) <= len(name) + 8 or not quote_supports_name(name, cleaned) or _ends_bare(cleaned):
            continue
        chosen = (page_no, cleaned)
        if _ACTIVITY.search(cleaned):
            break
    if not chosen:
        return False
    page_no, sentence = chosen
    candidate["descriptor"] = sentence[:180]
    sources = list(candidate.get("sources") or [])
    snippet = sentence[:240]
    if len(sources) < 3 and snippet not in {str(item.get("quote") or "") for item in sources} and not _ends_bare(snippet):
        base = sources[0] if sources else {"pdf_page": page_no, "printed_page": None}
        sources.append({**base, "pdf_page": page_no, "quote": snippet})
        candidate["sources"] = sources
    return True


def _still_name_only(candidate: dict) -> bool:
    if candidate.get("kind") == "subsidiary":
        return False
    name = str(candidate.get("name") or "").strip()
    descriptor = str(candidate.get("descriptor") or "").strip()
    return not descriptor or descriptor.casefold() == name.casefold()


def _around_name(name: str, text: str) -> str:
    folded = text.casefold()
    index = folded.find(name.casefold())
    if index < 0:
        for token in sorted(_name_tokens(name), key=len, reverse=True):
            found = re.search(rf"\b{re.escape(token)}\b", folded)
            if found:
                index = found.start()
                break
    if index < 0:
        return text[:180]
    return text[max(0, index - 70): index + 110].strip()


def _ends_bare(text: str) -> bool:
    stripped = text.strip()
    if re.search(r"(?:\d{1,2}[./]\d{1,2}[./]\d{2,4}|(?:19|20)\d{2})\s*$", stripped):
        return False
    return bool(re.search(r"\d[\d,]*(?:\.\d+)?\s*$", stripped))


def _strip_trailing_amounts(text: str) -> str:
    cleaned = text.strip()
    previous = None
    while cleaned and cleaned != previous:
        previous = cleaned
        cleaned = re.sub(r"[\s\-–]*\d[\d,]*(?:\.\d+)?\s*$", "", cleaned).strip(" -–")
    return cleaned


def _digit_heavy(text: str) -> bool:
    digits = sum(character.isdigit() for character in text)
    return digits >= 8 and digits / max(len(text), 1) > 0.12


def _sentence_for_name(name: str, page: str) -> str:
    compact = " ".join((page or "").split())
    if not compact:
        return ""
    best = ""
    for part in re.split(r"(?<=[.!?])\s+", compact):
        if len(part) <= len(name) + 8 or not quote_supports_name(name, part) or _digit_heavy(part):
            continue
        if _ACTIVITY.search(part):
            return " ".join(part.split())
        if _ends_bare(part):
            continue
        if not best or len(part) < len(best):
            best = part
    return " ".join(best.split())


def _trim_amount(candidate: dict) -> None:
    if candidate.get("kind") not in {"asset", "product_or_service"}:
        return
    sources = candidate.get("sources") or []
    if not sources:
        return
    quote = sources[0].get("quote") or ""
    trimmed = re.sub(r"\s+\d[\d,]*(?:\.\d+)?\s*$", "", quote).strip()
    if trimmed and trimmed != quote and quote_supports_name(candidate.get("name") or "", trimmed):
        sources[0]["quote"] = trimmed


def _sync_quotes(record: dict, candidate: dict) -> None:
    record["quotes"] = [
        {"pdf_page": source.get("pdf_page"), "quote": source.get("quote"), "accepted": True, "printed_page": source.get("printed_page")}
        for source in candidate.get("sources") or []
        if source.get("quote")
    ]


def _page_of(candidate: dict) -> int:
    sources = candidate.get("sources") or []
    page = sources[0].get("pdf_page") if sources else 0
    return int(page) if isinstance(page, int) else 0


def _note(log: dict, rule: str, before: str, after: str, detail: str) -> None:
    log.setdefault(rule, []).append({"before": before or "", "after": after or "", "detail": detail or ""})
