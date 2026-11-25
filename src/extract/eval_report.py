"""Build eval/report.md from local pages and saved API usage. Does not write fact sheets."""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.config.paths import CANDIDATE_DIR, OUTPUT_DIR, PAGE_DIR, PROJECT_ROOT
from src.extract.compact import paragraphs_for_page, repeated_lines
from src.extract.evidence import pages_for_extraction, render_page
from src.extract.ledger import estimate_tokens, legacy_input_tokens, planned_input_tokens
from src.extract.rules import extract_rules
from src.extract.run import load_saved_company
from src.extract.select import select_pages
from src.validate.quotes import locate_quote
from src.validate.verify_quotes import verify_candidates

SEED = 23
SAMPLE_SIZE = 25
PREVIOUS_DROP_PAGES = [4, 6, 11, 15, 18, 19, 20, 21, 25, 26, 27, 28]


def _pages(ticker: str) -> list[dict]:
    return pages_for_extraction(load_saved_company(ticker, PAGE_DIR))


def _saved_candidates(ticker: str) -> list[dict]:
    folder = CANDIDATE_DIR / ticker / "parsed"
    items = []
    if not folder.is_dir():
        return items
    for path in sorted(folder.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("source") == "rules":
            continue
        for item in record.get("candidates") or []:
            items.append(item)
    return items


def _quote_rates(ticker: str, pages: list[dict]) -> dict:
    by_page = {int(page["pdf_page"]): page for page in pages}
    strict = normalized = rejected = 0
    for item in _saved_candidates(ticker):
        for source in item.get("sources") or []:
            page = by_page.get(source.get("pdf_page"))
            if page is None:
                rejected += 1
                continue
            text = "\n".join(block.get("text") or "" for block in page.get("text_blocks") or [])
            quote = source.get("quote") or ""
            if quote and quote in text:
                strict += 1
            elif locate_quote(quote, text) == "normalized":
                normalized += 1
            else:
                rejected += 1
    total = strict + normalized + rejected
    return {
        "checked": total,
        "strict": strict,
        "normalized": normalized,
        "rejected": rejected,
        "strict_rate": None if not total else round(strict / total, 3),
        "accepted_rate": None if not total else round((strict + normalized) / total, 3),
    }


def _actual_gap(ticker: str) -> dict:
    folder = CANDIDATE_DIR / ticker / "parsed"
    rows = []
    if not folder.is_dir():
        return {"calls": 0, "estimated_input": 0, "actual_input": 0, "actual_output": 0}
    pages = {int(page["pdf_page"]): page for page in _pages(ticker)}
    files = []
    for path in folder.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        usage = record.get("usage") or {}
        if record.get("request_status") != "success" or not usage.get("input_tokens"):
            continue
        files.append((path.stat().st_mtime, record))
    files.sort()
    for _mtime, record in files[:20]:
        estimate = 0
        for number in record.get("target_pages") or []:
            page = pages.get(int(number))
            if page is not None:
                estimate += estimate_tokens(render_page(page, "target"))
        rows.append(
            {
                "pages": record.get("target_pages"),
                "estimated_input": estimate,
                "actual_input": int(record["usage"]["input_tokens"]),
                "actual_output": int(record["usage"].get("output_tokens") or 0),
            }
        )
    return {
        "calls": len(rows),
        "estimated_input": sum(row["estimated_input"] for row in rows),
        "actual_input": sum(row["actual_input"] for row in rows),
        "actual_output": sum(row["actual_output"] for row in rows),
        "rows": rows,
    }


def _sheet_count(ticker: str) -> int:
    path = OUTPUT_DIR / ticker / f"{ticker}.json"
    if not path.is_file():
        return 0
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return len(payload.get("items") or [])


def main() -> None:
    root = PROJECT_ROOT / "eval"
    root.mkdir(parents=True, exist_ok=True)
    companies = []
    dropped_sample = {}
    recovered = []
    for ticker in ("HINDALCO", "COALINDIA", "POWERGRID"):
        pages = _pages(ticker)
        selection = select_pages(pages)
        kept_ids = set(selection["kept"])
        kept = [page for page in pages if int(page["pdf_page"]) in kept_ids]
        scope = {row["pdf_page"] for row in selection["pages"] if str(row["reason"]).endswith("_in_scope")}
        banners = repeated_lines(kept)
        by_page = {
            int(page["pdf_page"]): paragraphs_for_page(page, banners, scope_only=int(page["pdf_page"]) in scope)
            for page in kept
        }
        rules = extract_rules(kept)
        legacy = legacy_input_tokens(pages, ticker)["input_tokens"]
        planned = planned_input_tokens(kept, by_page, ticker, 2000, [item["name"] for item in rules[:40]])
        quotes = verify_candidates(rules, pages)
        candidates = _saved_candidates(ticker)
        lost = []
        for item in candidates:
            cited = {int(source["pdf_page"]) for source in item.get("sources") or [] if isinstance(source.get("pdf_page"), int)}
            if cited and cited.isdisjoint(kept_ids):
                lost.append(item.get("name"))
        gap_rows = [row for row in selection["gaps"]]
        draw = gap_rows
        if len(draw) > SAMPLE_SIZE:
            rng = random.Random(SEED)
            draw = rng.sample(draw, SAMPLE_SIZE)
        dropped_sample[ticker] = [{"pdf_page": row["pdf_page"], "reason": row["reason"], "section": row["section"]} for row in sorted(draw, key=lambda row: row["pdf_page"])]
        companies.append(
            {
                "ticker": ticker,
                "pages_total": len(pages),
                "pages_selected": len(kept),
                "pages_dropped": len(selection["gaps"]),
                "legacy_input_tokens": legacy,
                "planned_input_tokens": planned,
                "saved_pct": 0 if not legacy else round(100 * (legacy - planned) / legacy, 1),
                "existing_candidates": len(candidates),
                "existing_candidates_on_dropped_pages": len(lost),
                "rule_items": len(rules),
                "rule_quotes_accepted": quotes["accepted"],
                "rule_quotes_rejected": quotes["rejected"],
                "fact_sheet_items": _sheet_count(ticker),
                "quote_check": _quote_rates(ticker, pages),
                "actual_vs_estimate_first_20": _actual_gap(ticker),
                "dropped_sample_size": len(dropped_sample[ticker]),
            }
        )
        if ticker == "HINDALCO":
            for item in candidates:
                cited = sorted({int(source["pdf_page"]) for source in item.get("sources") or [] if isinstance(source.get("pdf_page"), int)})
                if cited and set(cited).issubset(set(PREVIOUS_DROP_PAGES)):
                    recovered.append({"name": item.get("name"), "pdf_page": cited[0], "kept": set(cited).issubset(kept_ids)})
    (root / "dropped_sample.json").write_text(json.dumps({"seed": SEED, "companies": dropped_sample}, indent=2), encoding="utf-8")
    used = 0
    usage_dir = CANDIDATE_DIR / "usage"
    if usage_dir.is_dir():
        for path in usage_dir.glob("2026-11-23*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            used += int(payload.get("input_tokens") or 0) + int(payload.get("output_tokens") or 0)
    report = {
        "companies": companies,
        "csr_pages_now_kept": recovered,
        "tokens_used_today_all_keys": used,
        "daily_cap": 200000,
        "note": "Dropped-page extraction was not sent. Today's Groq usage is already near the 200K cap, and the fact-sheet files were left unchanged.",
    }
    (root / "cost.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Token and recall report", "", "Fact-sheet JSON and Markdown files were not rewritten. No full pipeline run.", ""]
    for company in companies:
        gap = company["actual_vs_estimate_first_20"]
        lines.append(f"## {company['ticker']}")
        lines.append(f"- Pages with text: {company['pages_total']}. Selected: {company['pages_selected']}. Dropped: {company['pages_dropped']}.")
        lines.append(f"- Estimated input before filtering: {company['legacy_input_tokens']}. After filtering and cleanup: {company['planned_input_tokens']} ({company['saved_pct']}% lower).")
        lines.append(f"- Fact-sheet items: {company['fact_sheet_items']} (reference range 190-300). Saved candidates: {company['existing_candidates']}. Of those, {company['existing_candidates_on_dropped_pages']} sit only on pages the filter still drops.")
        lines.append(f"- Rule items: {company['rule_items']}. Quotes accepted on the original page text: {company['rule_quotes_accepted']}. Rejected: {company['rule_quotes_rejected']}.")
        check = company["quote_check"]
        lines.append(f"- Saved-candidate quotes: {check['checked']} checked, {check['strict']} strict, {check['normalized']} normalized, {check['rejected']} rejected.")
        if gap["calls"]:
            lines.append(f"- First {gap['calls']} billed calls: estimated input {gap['estimated_input']}, actual input {gap['actual_input']}, actual output {gap['actual_output']}.")
        lines.append(f"- Dropped-page sample size: {company['dropped_sample_size']} (seed {SEED}). Live extraction on that sample was not run.")
        lines.append("")
    lines.append("## CSR keep rule")
    lines.append("Pages in a CSR or remuneration section are kept when they mention a hospital, school, community, foundation, scholarship, health camp, skill, hostel, anganwadi, or village. Other paragraphs on those pages are not sent.")
    lines.append(f"- Previously dropped Hindalco candidates: {len(recovered)}. Now on a kept page: {sum(1 for row in recovered if row['kept'])}.")
    lines.append("")
    lines.append(f"Tokens recorded today across usage files: {used} of {200000}.")
    lines.append("PowerGrid OCR of the remaining scanned pages was not started in this slice. Extraction for PowerGrid stays blocked until those gaps are closed or listed in logs/accepted_gaps.json.")
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "companies": [{key: company[key] for key in ("ticker", "pages_total", "pages_selected", "pages_dropped", "planned_input_tokens", "saved_pct", "existing_candidates_on_dropped_pages", "fact_sheet_items")} for company in companies],
        "previously_dropped_items": len(recovered),
        "previously_dropped_now_kept": sum(1 for row in recovered if row["kept"]),
        "tokens_used_today": used,
    }, indent=2))


if __name__ == "__main__":
    main()
