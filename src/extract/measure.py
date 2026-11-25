"""Estimate token savings. This does not call the model or write fact sheets."""

from __future__ import annotations

import json
from pathlib import Path

from src.config.paths import CANDIDATE_DIR, PAGE_DIR, PROJECT_ROOT
from src.extract.compact import pack_pages, paragraphs_for_page, repeated_lines
from src.extract.evidence import pages_for_extraction
from src.extract.ledger import legacy_input_tokens, planned_input_tokens
from src.extract.rules import extract_rules
from src.extract.run import load_saved_company
from src.extract.select import select_pages
from src.validate.verify_quotes import verify_candidates

SAMPLE = {"HINDALCO": range(35, 60)}


def company_pages(ticker: str) -> list[dict]:
    loaded = load_saved_company(ticker, PAGE_DIR)
    return pages_for_extraction(loaded)


def compare(ticker: str, pages: list[dict]) -> dict:
    selection = select_pages(pages)
    kept = [page for page in pages if int(page["pdf_page"]) in set(selection["kept"])]
    banners = repeated_lines(kept)
    by_page = {int(page["pdf_page"]): paragraphs_for_page(page, banners) for page in kept}
    rules = extract_rules(kept)
    legacy = legacy_input_tokens(pages, ticker)["input_tokens"]
    planned = planned_input_tokens(kept, by_page, ticker, 2000, [item["name"] for item in rules])
    saved = 0 if legacy == 0 else round(100 * (legacy - planned) / legacy, 1)
    return {
        "ticker": ticker,
        "pages": len(pages),
        "kept": len(kept),
        "gaps": len(selection["gaps"]),
        "legacy_input_tokens": legacy,
        "planned_input_tokens": planned,
        "input_tokens_saved_pct": saved,
        "rules": rules,
        "selection": selection,
    }


def sample_recall(pages: list[dict], selection: dict, sample_pages: set[int]) -> dict:
    kept = set(selection["kept"])
    folder = CANDIDATE_DIR / "HINDALCO" / "parsed"
    items = []
    if folder.is_dir():
        for path in folder.glob("*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("source") == "rules":
                continue
            items.extend(record.get("candidates") or [])
    in_sample = []
    for item in items:
        pages_cited = {int(source["pdf_page"]) for source in item.get("sources") or [] if isinstance(source.get("pdf_page"), int)}
        if pages_cited & sample_pages:
            in_sample.append(pages_cited)
    kept_items = [pages_cited for pages_cited in in_sample if pages_cited & kept]
    return {
        "sample_pages": sorted(sample_pages),
        "existing_items": len(in_sample),
        "items_still_on_a_kept_page": len(kept_items),
        "filter_recall": None if not in_sample else round(len(kept_items) / len(in_sample), 3),
    }


def main() -> None:
    report = {"companies": [], "sample": {}, "rules": {}, "quotes": {}}
    for ticker in ("HINDALCO", "COALINDIA", "POWERGRID"):
        pages = company_pages(ticker)
        summary = compare(ticker, pages)
        rules = summary.pop("rules")
        selection = summary.pop("selection")
        report["companies"].append(summary)
        if ticker != "HINDALCO":
            continue
        sample_pages = set(SAMPLE["HINDALCO"])
        sample = [page for page in pages if int(page["pdf_page"]) in sample_pages]
        sample_summary = compare(ticker, sample)
        sample_summary.pop("rules")
        sample_summary.pop("selection")
        report["sample"] = {**sample_summary, **sample_recall(pages, selection, sample_pages)}
        missing = []
        for item in rules:
            for source in item["sources"]:
                if int(source["pdf_page"]) not in set(selection["kept"]):
                    missing.append({"name": item["name"], "pdf_page": source["pdf_page"], "kind": item["kind"]})
        kinds = {}
        for item in rules:
            kinds[item["kind"]] = kinds.get(item["kind"], 0) + 1
        report["rules"] = {"counts": kinds, "missing_from_selected_pages": missing[:30], "missing_count": len(missing)}
        report["quotes"] = {key: verify_candidates(rules, pages)[key] for key in ("checked", "accepted", "rejected")}
    destination = PROJECT_ROOT / "runs" / "token_slice" / "report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
