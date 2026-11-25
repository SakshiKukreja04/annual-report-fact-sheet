"""Build a fact sheet from the candidates already on disk. The real outputs directory is refused."""

from __future__ import annotations

import json
from pathlib import Path

from src.config.paths import OUTPUT_DIR
from src.dedup.resolve import resolve
from src.extract.select import page_text
from src.factsheet.build import build_fact_sheet, write_fact_sheet
from src.validate.quotes import locate_quote


def make_final(
    ticker: str,
    candidates: list[dict],
    pages: list[dict],
    output_dir: Path,
    report_dir: Path,
    *,
    cost: dict | None = None,
    gaps: list[dict] | None = None,
) -> dict:
    """Merge duplicates, keep quotes that are on the page, and write the sheet atomically."""
    if Path(output_dir).resolve() == OUTPUT_DIR.resolve():
        raise RuntimeError("Refusing to write a slice over the existing fact sheets.")
    texts = {int(page["pdf_page"]): page_text(page) for page in pages}
    checked = []
    rejected_quotes = 0
    for candidate in candidates:
        sources = []
        for source in candidate.get("sources") or []:
            quote = source.get("quote") or ""
            number = source.get("pdf_page")
            if not isinstance(number, int) or not locate_quote(quote, texts.get(number, "")):
                rejected_quotes += 1
                continue
            sources.append({"pdf_page": number, "printed_page": source.get("printed_page"), "quote": quote})
        if not sources:
            continue
        checked.append({**candidate, "sources": sources[:3]})
    resolution = resolve(checked, ticker)
    records = list(resolution["unchanged"]) + list(resolution["merged"])
    built = build_fact_sheet(ticker, records)
    written = write_fact_sheet(ticker, built, output_dir, force=False)
    coverage = _coverage(pages, records)
    report_dir.mkdir(parents=True, exist_ok=True)
    _write(report_dir / "cost.json", cost or {})
    _write(report_dir / "coverage.json", coverage)
    _write(report_dir / "known_gaps.json", {"gaps": gaps or []})
    readme = _readme(ticker, written, resolution, rejected_quotes, coverage, len(gaps or []))
    _write_text(report_dir / "README.md", readme)
    return {
        "ticker": ticker,
        "items": written["item_count"],
        "segments": written["segment_count"],
        "merged": resolution["merged_count"],
        "review": resolution["review_count"],
        "rejected_quotes": rejected_quotes,
        "json": written["json"],
        "markdown": written["markdown"],
        "schema_ok": True,
    }


def _coverage(pages: list[dict], records: list[dict]) -> dict:
    cited = set()
    for record in records:
        for source in record.get("sources") or []:
            if isinstance(source.get("pdf_page"), int):
                cited.add(source["pdf_page"])
    numbers = [int(page["pdf_page"]) for page in pages]
    return {
        "pages_in_slice": len(numbers),
        "pages_cited": len(cited),
        "pages_without_an_item": [number for number in numbers if number not in cited],
    }


def _readme(ticker: str, written: dict, resolution: dict, rejected_quotes: int, coverage: dict, gap_count: int) -> str:
    return "\n".join([
        f"# {ticker} slice",
        "",
        f"- Items: {written['item_count']}",
        f"- Segments: {written['segment_count']}",
        f"- Duplicates merged: {resolution['merged_count']}",
        f"- Duplicates left for review: {resolution['review_count']}",
        f"- Quotes rejected: {rejected_quotes}",
        f"- Pages in this slice: {coverage['pages_in_slice']}",
        f"- Pages cited by a kept item: {coverage['pages_cited']}",
        f"- Known gaps: {gap_count}",
        "",
    ])


def _write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
