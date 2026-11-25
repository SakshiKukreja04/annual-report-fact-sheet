"""Local measurements for the filter, quotes, schedule, OCR, and a fact-sheet slice."""

from __future__ import annotations

import json
from pathlib import Path

from src.config.paths import CANDIDATE_DIR, INPUT_DIR, OCR_DIR, PAGE_DIR, PROJECT_ROOT
from src.extract.compact import paragraphs_for_page, render_new, repeated_lines, pack_pages
from src.extract.evidence import pages_for_extraction
from src.extract.ledger import estimate_tokens
from src.extract.rules import extract_rules
from src.extract.run import _covered_pages, load_saved_company
from src.extract.schedule import allocate
from src.extract.select import entity_hit_count, page_text, select_pages
from src.factsheet.assemble import make_final
from src.validate.quotes import locate_quote, repair_named_quote
from src.vision.usage import tokens_used, usage_path

PROMPT_TOKENS = 1677.7
TOKEN_RATIO = 1.132
MUST_KEEP = {"COALINDIA": [284, 60], "HINDALCO": [118, 138, 184, 185, 186]}


def _company_pages(ticker: str) -> list[dict]:
    return pages_for_extraction(load_saved_company(ticker, PAGE_DIR))


def _fitted(pages: list[dict], selection: dict) -> dict:
    kept = {int(number) for number in selection["kept"]}
    chosen = [page for page in pages if int(page["pdf_page"]) in kept]
    banners = repeated_lines(pages)
    by_page = {}
    reasons = {int(row["pdf_page"]): row.get("reason") or "" for row in selection["pages"]}
    for page in chosen:
        by_page[int(page["pdf_page"])] = paragraphs_for_page(
            page, banners, scope_only=reasons.get(int(page["pdf_page"]), "").endswith("_in_scope")
        )
    batches = pack_pages(chosen, by_page, 24000, max_pages=8)
    seen: set[str] = set()
    total = 0
    for batch in batches:
        paragraphs = []
        for page in batch["pages"]:
            paragraphs.extend(by_page.get(int(page["pdf_page"]), []))
        packet, fresh = render_new(paragraphs, seen)
        seen.update(fresh)
        total += round(PROMPT_TOKENS + TOKEN_RATIO * estimate_tokens(packet))
    return {"kept_pages": len(chosen), "batches": len(batches), "fitted_input_tokens": total}


def _quotes(ticker: str, texts: dict[int, str]) -> dict:
    counts = {"repaired": 0, "reanchored": 0, "dropped": 0, "checked": 0}
    silent = []
    items = []
    folder = CANDIDATE_DIR / ticker / "parsed"
    for path in sorted(folder.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("source") == "rules":
            continue
        items.extend(record.get("candidates") or [])
    for item in items:
        name = item.get("name") or ""
        for source in item.get("sources") or []:
            quote = source.get("quote") or ""
            cited = source.get("pdf_page")
            counts["checked"] += 1
            page = texts.get(cited, "") if isinstance(cited, int) else ""
            repaired = repair_named_quote(quote, page, name)
            counts[repaired["outcome"]] += 1
            if repaired["outcome"] != "dropped" and repaired.get("mentions_name") is False:
                silent.append({"name": name, "pdf_page": cited, "quote": (repaired.get("quote") or "")[:160]})
    return {"counts": counts, "quote_without_name": silent}


def _pdf_layer(ticker: str, texts: dict[int, str]) -> dict:
    folder = INPUT_DIR / ticker
    pdfs = sorted(folder.glob("*.pdf")) if folder.is_dir() else []
    if not pdfs:
        return {"engine": None, "error": "no pdf"}
    quotes = []
    parsed = CANDIDATE_DIR / ticker / "parsed"
    for path in sorted(parsed.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("source") == "rules":
            continue
        for item in record.get("candidates") or []:
            for source in item.get("sources") or []:
                if isinstance(source.get("pdf_page"), int) and source.get("quote"):
                    quotes.append((source["pdf_page"], source["quote"]))
    try:
        import pymupdf

        document = pymupdf.open(pdfs[0])
        engine = "pymupdf"
        layer = {number: document[number - 1].get_text("text") or "" for number in sorted({page for page, _quote in quotes})}
        document.close()
        error = None
    except Exception as exc:
        engine = None
        layer = {}
        error = f"{type(exc).__name__}: {exc}"
    stored_hits = 0
    layer_hits = 0
    for number, quote in quotes:
        if locate_quote(quote, texts.get(number, "")):
            stored_hits += 1
        if layer and locate_quote(quote, layer.get(number, "")):
            layer_hits += 1
    checked = len(quotes)
    return {
        "engine": engine,
        "error": error,
        "checked": checked,
        "stored_page_text_matches": stored_hits,
        "stored_match_rate": round(stored_hits / checked, 3) if checked else 0,
        "pdf_layer_matches": layer_hits,
        "pdf_layer_match_rate": round(layer_hits / checked, 3) if checked and engine else None,
    }


def _rules(ticker: str, pages: list[dict]) -> dict:
    found = extract_rules(pages)
    texts = {int(page["pdf_page"]): page_text(page) for page in pages}
    rejected = 0
    for item in found:
        for source in item["sources"]:
            if not locate_quote(source["quote"], texts.get(source["pdf_page"], "")):
                rejected += 1
    return {"items": len(found), "quotes_rejected": rejected}


def _ocr() -> dict:
    from src.ocr.pool import pending

    folder = OCR_DIR / "POWERGRID"
    done = list(folder.glob("POWERGRID_p*.json")) if folder.is_dir() else []
    waiting = pending("POWERGRID")
    stamps = sorted(path.stat().st_mtime for path in done)
    seconds = None
    if len(stamps) >= 2:
        seconds = (stamps[-1] - stamps[0]) / (len(stamps) - 1)
    return {
        "halves_done": len(done),
        "halves_pending": len(waiting),
        "seconds_per_half": None if seconds is None else round(seconds, 1),
        "hours_for_pending": None if seconds is None else round(seconds * len(waiting) / 3600, 1),
    }


def _slice_sheet(pages: list[dict], selection: dict) -> dict:
    texts = {int(page["pdf_page"]): page_text(page) for page in pages}
    candidates = []
    folder = CANDIDATE_DIR / "HINDALCO" / "parsed"
    for path in sorted(folder.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("source") == "rules" or record.get("request_status") != "success":
            continue
        for item in record.get("candidates") or []:
            sources = [
                source for source in item.get("sources") or []
                if isinstance(source.get("pdf_page"), int) and locate_quote(source.get("quote") or "", texts.get(source["pdf_page"], ""))
            ]
            if not sources or not item.get("name") or item.get("kind") is None:
                continue
            candidates.append({**item, "sources": sources[:3]})
            if len(candidates) >= 12:
                break
        if len(candidates) >= 12:
            break
    cited = {source["pdf_page"] for item in candidates for source in item["sources"]}
    used_pages = [page for page in pages if int(page["pdf_page"]) in cited]
    gaps = [row for row in selection["gaps"] if row["pdf_page"] in cited]
    destination = PROJECT_ROOT / "runs" / "slice_final"
    if destination.exists():
        import shutil

        shutil.rmtree(destination)
    return make_final(
        "HINDALCO",
        candidates,
        used_pages,
        destination / "outputs",
        destination,
        cost={"note": "slice only; existing outputs were not rewritten"},
        gaps=gaps,
    )


def _fallback_compare(texts: dict[int, str]) -> dict:
    from src.extract.client import extract_batch
    from src.extract.compact import paragraphs_for_page, render_new
    from src.extract.ledger import Ledger
    from src.extract.provider import fallback_caller, fallback_provider

    provider = fallback_provider()
    if provider is None:
        return {"configured": False, "live_call": False, "reason": "LLM_FALLBACK_PROVIDER is unset, is Groq, or uses the Groq key"}
    pages = []
    groq_names = set()
    folder = CANDIDATE_DIR / "HINDALCO" / "parsed"
    for path in sorted(folder.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("request_status") != "success" or record.get("source") == "rules":
            continue
        numbers = [int(number) for number in record.get("target_pages") or []]
        if len(numbers) != 1 or numbers[0] not in texts:
            continue
        pages.append(numbers[0])
        for item in record.get("candidates") or []:
            groq_names.add((item.get("name") or "").casefold())
        if len(pages) == 3:
            break
    packet, _fresh = render_new(
        [item for number in pages for item in paragraphs_for_page({"pdf_page": number, "text_blocks": [{"text": texts[number], "selected": True, "reading_order": 0}]}, set())],
        set(),
    )
    ledger = Ledger(PROJECT_ROOT / "eval" / "fallback_compare.jsonl")
    result = extract_batch(
        packet,
        ticker="HINDALCO",
        target_pages=pages,
        allowed_pages=set(pages),
        settings={"model": provider["model"] or "fallback", "max_output_tokens": 600, "max_attempts": 1, "timeout_seconds": 90},
        create=fallback_caller(provider),
        sleep=lambda _delay: None,
    )
    usage = result.get("usage") or {}
    ledger.add(
        ticker="HINDALCO",
        step="llm",
        pdf_pages=pages,
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        billed=True,
        provider=provider["name"],
    )
    fallback_names = {(item.get("name") or "").casefold() for item in result.get("candidates") or []}
    valid = 0
    total = 0
    for item in result.get("candidates") or []:
        for source in item.get("sources") or []:
            total += 1
            if locate_quote(source.get("quote") or "", texts.get(source.get("pdf_page"), "")):
                valid += 1
    shared = groq_names & fallback_names
    return {
        "configured": True,
        "live_call": True,
        "provider": provider["name"],
        "pages": pages,
        "groq_names": len(groq_names),
        "fallback_names": len(fallback_names),
        "names_in_both": len(shared),
        "agreement": round(len(shared) / len(groq_names | fallback_names), 3) if (groq_names or fallback_names) else None,
        "fallback_quotes_valid": valid,
        "fallback_quotes": total,
        "request_status": result.get("request_status"),
    }


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    confirm = {}
    still_dropped = []
    tokens = {}
    rules = {}
    queue = []
    for ticker in ("HINDALCO", "COALINDIA", "POWERGRID"):
        pages = _company_pages(ticker)
        current = select_pages(pages, protect=True)
        previous = select_pages(pages, protect=False)
        by_page = {row["pdf_page"]: row for row in current["pages"]}
        confirm[ticker] = {
            number: {"decision": by_page[number]["decision"], "reason": by_page[number]["reason"], "entity_hits": by_page[number]["entity_hits"]}
            for number in MUST_KEEP.get(ticker, [])
            if number in by_page
        }
        for row in current["gaps"]:
            if row.get("entity_hits", 0) >= 3:
                still_dropped.append({"ticker": ticker, "pdf_page": row["pdf_page"], "entity_hits": row["entity_hits"], "reason": row["reason"]})
        tokens[ticker] = {"old": _fitted(pages, previous), "new": _fitted(pages, current)}
        rules[ticker] = _rules(ticker, pages)
        covered = _covered_pages(CANDIDATE_DIR / ticker)
        scores = {int(row["pdf_page"]): row for row in current["pages"]}
        for number in current["kept"]:
            row = scores[int(number)]
            queue.append({
                "ticker": ticker,
                "pdf_page": int(number),
                "score": row.get("score") or 0,
                "tokens": max(1, round((tokens[ticker]["new"]["fitted_input_tokens"] / max(tokens[ticker]["new"]["kept_pages"], 1)))),
                "done": int(number) in covered,
            })
    hindalco_pages = _company_pages("HINDALCO")
    texts = {int(page["pdf_page"]): page_text(page) for page in hindalco_pages}
    used_today = tokens_used(usage_path(CANDIDATE_DIR))
    from src.extract.provider import fallback_provider

    fallback = fallback_provider()
    providers = [
        {"name": "groq", "daily_tokens": 200000, "remaining": max(0, 200000 - used_today) + 200000 * 2},
        {"name": "fallback", "daily_tokens": 0 if fallback is None else fallback["daily_token_cap"], "remaining": 0 if fallback is None else fallback["daily_token_cap"] * 2},
    ]
    plan = allocate(queue, providers)
    for company in plan["companies"].values():
        company["uncovered_count"] = len(company["pages_uncovered"])
        company["pages_uncovered"] = company["pages_uncovered"][:25]
    sheet = _slice_sheet(hindalco_pages, select_pages(hindalco_pages))
    report = {
        "must_keep": confirm,
        "high_hit_pages_still_dropped": still_dropped,
        "tokens": tokens,
        "rules": rules,
        "quotes": _quotes("HINDALCO", texts),
        "pdf_layer": _pdf_layer("HINDALCO", texts),
        "fallback": _fallback_compare(texts),
        "schedule": {"groq_used_today": used_today, "providers": plan["providers"], "companies": plan["companies"]},
        "powergrid_ocr": _ocr(),
        "slice_sheet": sheet,
    }
    if report["powergrid_ocr"]["halves_pending"] == 0:
        report["powergrid_after_ocr"] = tokens["POWERGRID"]["new"]
    else:
        report["powergrid_after_ocr"] = {"status": "ocr_not_finished", "text_pages_only": tokens["POWERGRID"]["new"]}
    destination = PROJECT_ROOT / "eval" / "slice_next.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    printable = dict(report)
    printable["quotes"] = {"counts": report["quotes"]["counts"], "quote_without_name": report["quotes"]["quote_without_name"][:15]}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
