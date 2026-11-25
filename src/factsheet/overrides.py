"""Apply image-verified manual overrides stored beside the staged fact sheet.

The file is data for one company. The rules that read it are generic.
"""

from __future__ import annotations

import json

from src.config.paths import STAGING_DIR
from src.extract.select import page_text
from src.factsheet.names import assignment_key, topic_match
from src.validate.quotes import resolve_quote_on_page


def apply_manual_overrides(ticker: str, items: list[dict], pages: dict[int, dict]) -> list[dict]:
    path = STAGING_DIR / "factsheet" / ticker / "manual_overrides.json"
    if not path.is_file():
        return items
    payload = json.loads(path.read_text(encoding="utf-8"))
    specs = [spec for spec in (payload.get("items") or []) if spec.get("name")]
    if not specs:
        return items
    removed = set()
    for spec in specs:
        names = [spec.get("name") or "", *list(spec.get("merge_names") or [])]
        keys = {assignment_key(name) for name in names if name}
        for item in items:
            item_name = item.get("name") or ""
            if assignment_key(item_name) in keys or any(topic_match(item_name, name) for name in names if name):
                removed.add(id(item))
    kept = [item for item in items if id(item) not in removed]
    for spec in specs:
        item = _item_from_spec(spec, pages)
        if item is not None:
            kept.append(item)
    return kept


def _item_from_spec(spec: dict, pages: dict[int, dict]) -> dict | None:
    pdf_page = int(spec.get("pdf_page") or 0)
    page = pages.get(pdf_page) or {}
    text = page_text(page) if page else ""
    sources = []
    for quote in spec.get("quotes") or []:
        verdict = resolve_quote_on_page(str(quote), text)
        if not verdict.get("accepted"):
            continue
        sources.append({
            "pdf_page": pdf_page,
            "printed_page": None,
            "quote": verdict.get("quote") or str(quote),
        })
        if len(sources) == 3:
            break
    if not sources:
        return None
    share = spec.get("revenue_share_pct")
    return {
        "name": str(spec["name"]),
        "kind": spec.get("kind") or "other",
        "descriptor": str(spec.get("descriptor") or ""),
        "entity": "company",
        "status": spec.get("status") or "current",
        "revenue_share_pct": None if share is None else float(share),
        "sources": sources,
    }
