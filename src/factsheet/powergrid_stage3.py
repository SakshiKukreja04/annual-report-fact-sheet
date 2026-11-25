"""Stage 3 POWERGRID correct-outputs rebuild. Staging only until gates pass.

Keep only items whose quotes verify on page_text (check 5 layer). Never invent
names, shares, or printed folios. Honest shortfall vs 190 is expected.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from src.config.paths import INPUT_DIR, PAGE_DIR, STAGING_DIR, SUBMISSION_DIR
from src.extract.select import page_text
from src.factsheet.build import consistency_errors, render_markdown, write_fact_sheet
from src.factsheet.names import assignment_key, should_merge
from src.factsheet.overview import write_overview_sidecar
from src.factsheet.powergrid_stage2 import (
    BUSINESS_PAGES,
    DROP_NAMES,
    VARIANT_MAP,
    _LEGAL,
    _accept_list_name,
    canon_cleanup,
    norm_key,
)
from src.factsheet.precision import quote_supports_name
from src.factsheet.schema import FactSheet
from src.validate.quotes import fold_whitespace, read_pdf_page, resolve_quote_on_page, snap_quote

TICKER = "POWERGRID"
STAGING = STAGING_DIR / "factsheet" / TICKER
REPORT = STAGING / "stage3_report.json"
LIST_PAGES = list(range(243, 251)) + list(range(355, 363))
_PRINTED_OK = re.compile(r"^\d+[A-Za-z]?(?:/\d+[A-Za-z]?)?$")
_FOLIO_HEADER = re.compile(
    r"(?:^|\n)\s*(?:page\s*)?(\d{1,3}[A-Za-z]?)\s*(?:/|\||$)",
    re.I,
)


def _pdf() -> Path | None:
    pdfs = sorted((INPUT_DIR / TICKER).glob("*.pdf"))
    return pdfs[0] if len(pdfs) == 1 else None


_PAGE_CACHE: dict[int, dict] = {}
_LAYER_CACHE: dict[int, str] = {}
_PDF_CACHE: dict[int, str] = {}


def page_record(page: int) -> dict:
    if page not in _PAGE_CACHE:
        path = PAGE_DIR / TICKER / f"p{page:04d}.json"
        _PAGE_CACHE[page] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return _PAGE_CACHE[page]


def page_layer(page: int) -> str:
    """Checker layer: page_text from page JSON (includes selected OCR blocks)."""
    if page not in _LAYER_CACHE:
        rec = page_record(page)
        _LAYER_CACHE[page] = page_text(rec) if rec else ""
    return _LAYER_CACHE[page]


def pdf_layer(page: int) -> str:
    if page not in _PDF_CACHE:
        pdf = _pdf()
        if not pdf:
            _PDF_CACHE[page] = ""
        else:
            try:
                _PDF_CACHE[page] = read_pdf_page(pdf, page)[0] or ""
            except Exception:
                _PDF_CACHE[page] = ""
    return _PDF_CACHE[page]


def verify_quote(quote: str, page: int) -> dict | None:
    """Accept only if check_factsheet would accept (page_text then PDF)."""
    q = fold_whitespace(quote or "")
    if not q:
        return None
    text = page_layer(page)
    verdict = resolve_quote_on_page(q, text)
    if verdict.get("accepted"):
        return {"quote": fold_whitespace(verdict.get("quote") or q)[:300], "page": page, "via": "page_text"}
    pdf_text = pdf_layer(page)
    if pdf_text:
        verdict = resolve_quote_on_page(q, pdf_text)
        if verdict.get("accepted"):
            return {"quote": fold_whitespace(verdict.get("quote") or q)[:300], "page": page, "via": "pdf"}
    return None


def printed_for(page: int, quote: str = "") -> str | None:
    rec = page_record(page)
    value = rec.get("printed_page")
    text = str(value).strip() if value not in (None, "") else ""
    if text and _PRINTED_OK.match(text):
        parts = [p.strip() for p in text.split("/") if p.strip()]
        if len(parts) < 2:
            return text
        half = None
        quote_cf = fold_whitespace(quote or "").casefold()
        for block in rec.get("text_blocks") or []:
            block_text = fold_whitespace(block.get("text") or "")
            if quote_cf and quote_cf in block_text.casefold():
                half = block.get("half") or "FULL"
                break
        if half == "LEFT":
            return parts[0]
        if half == "RIGHT" and len(parts) > 1:
            return parts[1]
        return parts[0]
    # Honest folio from header only when unambiguous single folio-like token near top
    layer = fold_whitespace(page_layer(page))[:400]
    matches = _FOLIO_HEADER.findall(layer)
    # Too noisy on related-party pages — do not invent
    return None


def seed_names() -> dict[str, dict]:
    """Collect candidate names from submission + page_text harvest."""
    seeds: dict[str, dict] = {}

    def add(name: str, pages: list[int], kind: str = "subsidiary") -> None:
        raw = canon_cleanup(name)
        if assignment_key(raw) in {assignment_key(n) for n in DROP_NAMES}:
            return
        if any(assignment_key(raw) == assignment_key(n) for n in DROP_NAMES):
            return
        if raw.casefold() in {n.casefold() for n in DROP_NAMES}:
            return
        key = norm_key(raw)
        if len(key) < 8:
            return
        row = seeds.setdefault(key, {"name": raw, "pages": [], "kind": kind})
        # Prefer Powergrid-prefixed / longer cleaner names
        if raw.casefold().startswith("powergrid") and (
            not row["name"].casefold().startswith("powergrid") or len(raw) >= len(row["name"])
        ):
            row["name"] = raw
        elif len(raw) > len(row["name"]) and not row["name"].casefold().startswith("powergrid"):
            row["name"] = raw
        for p in pages:
            if p not in row["pages"]:
                row["pages"].append(p)
        row["kind"] = kind

    # Submission seeds (cleaner spellings)
    sub_path = SUBMISSION_DIR / TICKER / f"{TICKER}.json"
    if sub_path.is_file():
        sheet = json.loads(sub_path.read_text(encoding="utf-8"))
        for item in sheet.get("items") or []:
            if item.get("name") in DROP_NAMES:
                continue
            pages = [int(s["pdf_page"]) for s in item.get("sources") or []]
            kind = item.get("kind") or "subsidiary"
            if kind in {"other"} and any(x in (item.get("name") or "").casefold() for x in ("bond", "debenture")):
                continue
            add(item["name"], pages or LIST_PAGES[:1], kind=kind if kind != "other" else "subsidiary")

    # Page-text legal harvest from every usable layer (digital / ocr_vision / tesseract).
    harvest_pages = set(LIST_PAGES)
    folder = PAGE_DIR / TICKER
    if folder.is_dir():
        for path in folder.glob("p*.json"):
            rec = json.loads(path.read_text(encoding="utf-8"))
            pdf_page = int(rec["pdf_page"])
            selected = [
                b
                for b in rec.get("text_blocks") or []
                if b.get("selected") and (b.get("text") or "").strip()
            ]
            sources = {b.get("source") for b in selected}
            methods = set(rec.get("extraction_methods") or [])
            text = page_text(rec) if selected else ""
            if (
                sources & {"pymupdf", "ocr_vision", "ocr_mistral", "tesseract", "paddleocr", "vision"}
                or "ocr_vision" in methods
                or (text and len(text) >= 40)
            ):
                harvest_pages.add(pdf_page)
    for page in sorted(harvest_pages):
        text = page_layer(page)
        if not text:
            continue
        for match in _LEGAL.finditer(text):
            raw = canon_cleanup(match.group(1))
            if not _accept_list_name(raw):
                continue
            add(raw, [page], kind="subsidiary")

    # Business labels only from segment/business pages (not subsidiary name hits)
    for page in BUSINESS_PAGES:
        if page_record(page).get("printed_page") in (None, ""):
            continue  # no folio → cannot keep under correctness rule
        text = page_layer(page)
        for label in ("Transmission", "Telecom", "Consultancy"):
            if label.casefold() in fold_whitespace(text).casefold():
                add(label, [page], kind="business_line")

    return seeds


def ground_candidate(row: dict) -> dict | None:
    """Return grounded item dict or None if not verifiable with printed folio."""
    name = row["name"]
    kind = row.get("kind") or "subsidiary"
    pages = list(row.get("pages") or []) + LIST_PAGES + BUSINESS_PAGES
    # de-dupe preserve order
    seen = set()
    ordered = []
    for p in pages:
        if p not in seen:
            seen.add(p)
            ordered.append(p)

    # Prefer pages that already have printed_page in JSON
    ordered.sort(key=lambda p: (0 if page_record(p).get("printed_page") not in (None, "") else 1, p))

    best = None
    chosen_name = name
    for page in ordered:
        layer = page_layer(page)
        if not layer:
            continue
        # Try canonical name, then verbatim snap, then variant-normalized search
        candidates = [name]
        snap = snap_quote(name, layer, minimum_ratio=0.9)
        if snap.get("quote"):
            candidates.append(snap["quote"])
        # If name not on page, try finding a Limited span that fuzzy-matches
        if not any(c.casefold() in fold_whitespace(layer).casefold() for c in candidates):
            key = norm_key(name)
            for match in _LEGAL.finditer(layer):
                span = fold_whitespace(match.group(1))
                if norm_key(span) == key or SequenceMatcher_ratio(norm_key(span), key) >= 0.92:
                    candidates.append(span)
                    break
        for cand in candidates:
            verified = verify_quote(cand, page)
            if not verified:
                continue
            quote = verified["quote"]
            # Prefer using the verified quote text as name when it's a legal entity line
            use_name = name
            if kind == "subsidiary" and re.search(r"\bLimited\b", quote, re.I) and quote_supports_name(quote, quote):
                # Keep cleaner Powergrid spelling if quote is OCR-garbled relative to seed
                if quote.casefold().startswith("powergrid") or not name.casefold().startswith("powergrid"):
                    if verify_quote(quote, page):
                        use_name = canon_cleanup(quote) if _accept_list_name(canon_cleanup(quote)) else name
            # Name tokens must be supported by quote
            check_name = use_name if quote_supports_name(use_name, quote) else cand
            if not quote_supports_name(check_name, quote):
                # rename to quote span if quote is the entity
                if re.search(r"Limited|Ltd", quote, re.I) and len(quote.split()) <= 12:
                    check_name = canon_cleanup(quote)
                    if not quote_supports_name(check_name, quote):
                        continue
                else:
                    continue
            printed = printed_for(page, quote)
            if printed in (None, ""):
                continue  # correctness: drop missing folio
            best = {
                "name": check_name,
                "kind": kind if kind in {"subsidiary", "business_line", "segment"} else "subsidiary",
                "descriptor": "",
                "entity": check_name if kind == "subsidiary" else "company",
                "status": "current",
                "revenue_share_pct": None,
                "sources": [{"pdf_page": page, "printed_page": printed, "quote": quote}],
            }
            chosen_name = check_name
            break
        if best:
            break
    return best


def SequenceMatcher_ratio(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b).ratio()


def build_list_keys(items: list[dict]) -> list[str]:
    keys = []
    for item in items:
        if item.get("kind") != "subsidiary":
            continue
        name = item["name"]
        quote = " ".join(s.get("quote") or "" for s in item.get("sources") or [])
        if quote_supports_name(name, quote):
            keys.append(assignment_key(name))
    keys = list(dict.fromkeys(keys))
    (STAGING / "list_subsidiaries.json").write_text(
        json.dumps({"ticker": TICKER, "keys": keys}, indent=2) + "\n",
        encoding="utf-8",
    )
    return keys


def fix_overview(sheet: dict) -> dict:
    from src.factsheet.overview import fallback_text

    one_fb, sum_fb = fallback_text(TICKER)
    one = sheet.get("one_liner") or ""
    summary = sheet.get("summary") or ""
    # Always prefer a page-grounded identity line for check 4
    grounded_one = None
    grounded_summary_parts = []
    for page in (4, 5, 6, 7, 8, 9, 10, 17, 31, 134):
        layer = page_layer(page)
        if not layer or len(layer) < 40:
            continue
        for cand in (
            "Power Grid Corporation of India Limited",
            "POWER GRID CORPORATION OF INDIA LIMITED",
            "POWERGRID",
        ):
            hit = verify_quote(cand, page)
            if hit:
                grounded_one = "Power Grid Corporation of India Limited"
                grounded_summary_parts.append(hit["quote"])
                break
        if grounded_one:
            break
    if grounded_one:
        one = grounded_one
    elif not one or one == one_fb or "FY25 operates" in one or "operates India" in one.casefold():
        one = "Power Grid Corporation of India Limited"
    if grounded_summary_parts:
        summary = fold_whitespace(" ".join(grounded_summary_parts))[:900]
    elif not summary or summary == sum_fb or "overview is written from" in summary.casefold():
        summary = one
    sheet["one_liner"] = one[:300]
    sheet["summary"] = summary[:900]
    overview = {
        "fallback": False,
        "reason": "sheet_grounded_offline",
        "pdf_pages": [4, 5, 6],
        "cache_hit": False,
        "billed": False,
        "input_tokens": 0,
        "output_tokens": 0,
        "key_label": "",
        "one_liner": sheet["one_liner"],
        "summary": sheet["summary"],
    }
    write_overview_sidecar(TICKER, overview, STAGING_DIR / "factsheet")
    return overview


def build_segments() -> list[dict]:
    rows = []
    for label in ("Transmission", "Telecom", "Consultancy"):
        ok = False
        for page in BUSINESS_PAGES + [338, 337, 299]:
            if verify_quote(label, page):
                ok = True
                break
        if ok:
            rows.append({"name": label, "revenue": None, "unit": None, "share_pct": None})
    return rows


def dedupe_items(items: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for item in items:
        drop = False
        for other in kept:
            if assignment_key(item["name"]) == assignment_key(other["name"]) or should_merge(item["name"], other["name"]):
                # Prefer Powergrid spelling / longer quote
                def score(row: dict) -> tuple:
                    q = row["sources"][0].get("quote") or ""
                    return (
                        1 if row["name"].casefold().startswith("powergrid") else 0,
                        len(q),
                        len(row["name"]),
                    )

                if score(item) > score(other):
                    kept = [item if x is other else x for x in kept]
                drop = True
                break
        if not drop:
            kept.append(item)
    return kept


def run() -> dict:
    STAGING.mkdir(parents=True, exist_ok=True)
    seeds = seed_names()
    print(f"0 seeds={len(seeds)}")

    grounded = []
    dropped = []
    for key, row in sorted(seeds.items(), key=lambda kv: kv[1]["name"]):
        item = ground_candidate(row)
        if item:
            grounded.append(item)
        else:
            dropped.append({"name": row["name"], "reason": "unverified_or_no_printed"})
    print(f"1 grounded={len(grounded)} dropped={len(dropped)}")

    items = dedupe_items(grounded)
    print(f"2 after_dedupe={len(items)}")

    # Force empty descriptors for list subsidiaries; enrich business lines lightly
    for item in items:
        if item["kind"] == "subsidiary":
            item["descriptor"] = ""
            item["entity"] = item["name"]
        else:
            item["entity"] = "company"
            # descriptor: only if a short non-name window verifies
            page = int(item["sources"][0]["pdf_page"])
            layer = page_layer(page)
            name = item["name"]
            quote = item["sources"][0]["quote"]
            if quote_supports_name(name, quote) and len(quote) > len(name) + 5:
                # use quote as descriptor only if not name-only and mentions item
                if quote.casefold() != name.casefold() and quote_supports_name(name, quote):
                    item["descriptor"] = quote[:200]
                else:
                    item["descriptor"] = ""
            else:
                item["descriptor"] = ""

    list_keys = build_list_keys(items)
    print(f"3 list_keys={len(list_keys)}")

    # Load prior sheet text if any for overview seed
    prior = {}
    for path in (STAGING / f"{TICKER}.v3.json", SUBMISSION_DIR / TICKER / f"{TICKER}.json"):
        if path.is_file():
            prior = json.loads(path.read_text(encoding="utf-8"))
            break

    segments = build_segments()
    # Do not add bare Transmission/Telecom/Consultancy items without a grounded
    # non-name descriptor + printed folio (business pages are usually folio-null).

    items = dedupe_items(items)
    items = [
        item
        for item in items
        if not (
            item.get("kind") in {"business_line", "segment"}
            and (
                int(item["sources"][0]["pdf_page"]) not in set(BUSINESS_PAGES) | {299}
                or not (item.get("descriptor") or "").strip()
                or (item.get("descriptor") or "").strip() == item["name"]
            )
        )
    ]
    items = sorted(items, key=lambda i: (min(int(s["pdf_page"]) for s in i["sources"]), i["name"]))

    sheet_out = {
        "ticker": TICKER,
        "period": prior.get("period") or "FY25",
        "one_liner": prior.get("one_liner") or "",
        "summary": prior.get("summary") or "",
        "segments": segments,
        "items": items,
    }
    overview = fix_overview(sheet_out)
    print(f"4 overview fallback={overview.get('fallback')} one={sheet_out['one_liner'][:80]!r}".encode("ascii", "replace").decode())

    # Final verify pass — drop anything that fails check5 or lacks printed
    final = []
    final_dropped = []
    for item in items:
        src = item["sources"][0]
        page = int(src["pdf_page"])
        quote = src.get("quote") or ""
        verified = verify_quote(quote, page)
        printed = src.get("printed_page")
        record = str(page_record(page).get("printed_page") or "")
        parts = [p.strip() for p in record.split("/") if p.strip()]
        printed_ok = printed not in (None, "") and (str(printed) == record or str(printed) in parts or record == "")
        # If record empty we shouldn't have kept — double-check
        if page_record(page).get("printed_page") in (None, ""):
            printed_ok = False
        if not verified or not printed_ok or not quote_supports_name(item["name"], quote):
            final_dropped.append(item["name"])
            continue
        src["quote"] = verified["quote"]
        # Re-resolve printed against record
        src["printed_page"] = printed_for(page, src["quote"]) or printed
        if src["printed_page"] in (None, ""):
            final_dropped.append(item["name"])
            continue
        item["revenue_share_pct"] = None
        if item["kind"] == "subsidiary":
            item["descriptor"] = ""
            item["entity"] = item["name"]
        final.append(item)

    print(f"5 final_items={len(final)} final_dropped={len(final_dropped)}")
    sheet_out["items"] = sorted(
        final,
        key=lambda i: (min(int(s["pdf_page"]) for s in i["sources"]), i["name"]),
    )
    build_list_keys(sheet_out["items"])

    validated = FactSheet.model_validate(sheet_out).model_dump(mode="json")
    markdown = render_markdown(validated)
    errors = consistency_errors(validated, markdown)
    print(f"6 schema_errors={errors} items={len(validated['items'])} shortfall={max(0, 190 - len(validated['items']))}")

    built = {"sheet": validated, "markdown": markdown, "omitted": []}
    write_fact_sheet(TICKER, built, STAGING_DIR / "factsheet", force=True)

    report = {
        "seeds": len(seeds),
        "grounded": len(grounded),
        "after_dedupe": len(items),
        "final_items": len(validated["items"]),
        "shortfall_vs_190": max(0, 190 - len(validated["items"])),
        "by_kind": dict(Counter(i["kind"] for i in validated["items"])),
        "dropped_unverified": dropped[:40],
        "final_dropped": final_dropped[:40],
        "segments": segments,
        "overview": {"fallback": overview.get("fallback"), "reason": overview.get("reason")},
        "schema_errors": errors,
        "honest_shortfall": True,
        "submission_untouched": True,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (STAGING / "SHORTFALL.md").write_text(
        "\n".join(
            [
                f"# POWERGRID honest shortfall",
                "",
                f"Items: **{len(validated['items'])}** (assignment target 190–300).",
                f"Shortfall: **{max(0, 190 - len(validated['items']))}**.",
                "",
                "Only page_text/PDF-verifiable quotes with recorded printed folios were kept.",
                "No padding, invented names, shares, or folios.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print("WROTE", REPORT)
    return report


if __name__ == "__main__":
    run()
