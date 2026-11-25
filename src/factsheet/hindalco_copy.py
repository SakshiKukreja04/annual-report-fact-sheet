"""Post-process the Hindalco copy. HINDALCO.json is not opened for writing."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from src.config.paths import CANDIDATE_DIR, INPUT_DIR, OUTPUT_DIR, PAGE_DIR, PROJECT_ROOT
from src.dedup.resolve import resolve
from src.extract.evidence import pages_for_extraction
from src.extract.rules import extract_listed
from src.extract.run import load_saved_company
from src.extract.schema import KINDS, STATUSES
from src.extract.settings import text_settings
from src.factsheet.build import render_markdown
from src.factsheet.postprocess import (
    assign_printed_pages,
    drop_items,
    governance_pages,
    merge_items,
    normalize_entities,
    repair_items,
    _name_key,
    _text,
)
from src.factsheet.schema import FactSheet
from src.validate.quotes import locate_quote

def run() -> dict:
    source = OUTPUT_DIR / "HINDALCO" / "HINDALCO.v0.json"
    original = OUTPUT_DIR / "HINDALCO" / "HINDALCO.json"
    sheet = json.loads(source.read_text(encoding="utf-8"))
    pages = pages_for_extraction(load_saved_company("HINDALCO", PAGE_DIR))
    texts = {int(page["pdf_page"]): _text(page) for page in pages}
    steps = {"items_at_start": len(sheet["items"])}
    blocked = governance_pages(pages)
    kept, dropped = drop_items(sheet["items"], blocked)
    steps["after_drop"] = len(kept)
    steps["dropped"] = dropped
    merged, merged_away = merge_items(kept)
    steps["after_merge"] = len(merged)
    steps["duplicates_folded"] = merged_away
    listed = extract_listed(pages)
    steps["listed_subsidiaries"] = len(listed["subsidiaries"])
    steps["listed_segments"] = len(listed["segments"])
    names = [item["name"] for item in listed["subsidiaries"]]
    combined = _add_listed(merged, listed["subsidiaries"] + listed["segments"])
    steps["after_listed_merge"] = len(combined)
    combined = normalize_entities(combined, names)
    combined = assign_printed_pages(combined, pages)
    cited = {source["pdf_page"] for item in combined for source in item.get("sources") or [] if isinstance(source.get("pdf_page"), int)}
    pdf_layer = _pdf_layer(cited)
    repaired = repair_items(combined, texts, pdf_layer)
    steps["after_quote_repair"] = len(repaired["items"])
    steps["quote_counts"] = repaired["quotes"]
    steps["dropped"].extend(repaired["dropped"])
    resolved = resolve(repaired["items"], "HINDALCO")
    final_items = [cleaned for item in list(resolved["unchanged"]) + list(resolved["merged"]) if (cleaned := _clean(item))]
    steps["after_dedupe"] = len(final_items)
    steps["dedupe_merged_groups"] = resolved["merged_count"]
    steps["dedupe_review"] = resolved["review_count"]
    segments = _segments(listed)
    one_liner = f"HINDALCO FY25 has {len(final_items)} items kept from the annual report."
    summary = "Each item is kept only when its quote is on the cited PDF page and on the PDF text layer."
    built = {
        "ticker": "HINDALCO",
        "period": "FY25",
        "one_liner": one_liner,
        "summary": summary,
        "segments": segments,
        "items": final_items,
    }
    validated = FactSheet.model_validate(built).model_dump(mode="json")
    steps["schema_ok"] = True
    steps["kinds"] = dict(Counter(item["kind"] for item in validated["items"]))
    checked = _quote_report(validated["items"], texts, pdf_layer)
    steps["final_quote_check"] = checked
    cited_after = {source["pdf_page"] for item in validated["items"] for source in item["sources"]}
    steps["pages_cited"] = len(cited_after)
    steps["pages_total"] = len(pages)
    destination = OUTPUT_DIR / "HINDALCO" / "HINDALCO.post.json"
    markdown_path = OUTPUT_DIR / "HINDALCO" / "HINDALCO.post.md"
    _atomic(destination, json.dumps(validated, indent=2, ensure_ascii=False) + "\n")
    _atomic_text(markdown_path, render_markdown(validated))
    steps["wrote"] = str(destination)
    steps["original_unchanged"] = original.read_bytes() == source.read_bytes()
    steps["model"] = _model_followup(validated, pages, texts, destination, markdown_path)
    sample = random.Random(23).sample(validated["items"], min(20, len(validated["items"])))
    steps["sample"] = [
        {"name": item["name"], "kind": item["kind"], "entity": item["entity"], "pdf_page": item["sources"][0]["pdf_page"], "quote": item["sources"][0]["quote"][:140]}
        for item in sample
    ]
    report_path = PROJECT_ROOT / "eval" / "hindalco_post.json"
    _atomic(report_path, json.dumps({key: value for key, value in steps.items() if key != "sample"}, indent=2))
    return steps


def _model_followup(sheet: dict, pages: list[dict], texts: dict[int, str], destination: Path, markdown_path: Path) -> dict:
    """One short overview call, then pages 31-58 only while the daily cap still has room."""
    from src.extract.client import extract_batch
    from src.extract.compact import paragraphs_for_page, render_new, repeated_lines
    from src.extract.ledger import estimate_tokens
    from src.extract.prompt import PROMPT_VERSION, SYSTEM_PROMPT
    from src.extract.select import select_pages
    from src.factsheet.postprocess import drop_items
    from src.validate.quotes import fold_whitespace
    from src.vision.usage import record_attempt, tokens_used, usage_path

    cap = text_settings()["daily_token_cap"]
    usage = usage_path(CANDIDATE_DIR)
    report = {"tokens_before": tokens_used(usage), "cap": cap}
    selection = select_pages(pages)
    chosen = []
    for row in selection["pages"]:
        if row.get("section") in {"business_overview", "segment_note"} and row.get("decision") == "keep":
            chosen.append(texts.get(int(row["pdf_page"]), "")[:700])
        if sum(len(part) for part in chosen) > 2400:
            break
    packet = "\n\n".join(part for part in chosen if part)[:2800]
    overview_estimate = estimate_tokens(packet) + 500
    if len(packet) < 200:
        report["overview"] = "no_source_text"
    elif tokens_used(usage) + overview_estimate > cap:
        report["overview"] = "skipped_cap"
    else:
        from groq import Groq

        from src.vision.client import usage_counts
        from src.vision.settings import api_key

        grounded = False
        parsed = {}
        try:
            response = Groq(api_key=api_key(), timeout=90).chat.completions.create(
                model=text_settings()["model"],
                temperature=0,
                max_completion_tokens=400,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Return JSON with one_liner and summary. one_liner is one sentence. summary is one paragraph. Use only the user text. Do not add facts that are not in that text."},
                    {"role": "user", "content": packet},
                ],
            )
            raw = response.choices[0].message.content or ""
            usage_row = usage_counts(response.usage)
            record_attempt(usage, {"ticker": "HINDALCO", "pdf_page": 319, "request_status": "success", "usage": usage_row, "model": text_settings()["model"]})
            parsed = _overview_json(raw)
            grounded = _grounded(parsed.get("one_liner") or "", packet) and _grounded(parsed.get("summary") or "", packet)
            report["overview"] = {"status": "success", "grounded": grounded, "usage": usage_row}
        except Exception as exc:
            report["overview"] = {"status": "failed", "error": type(exc).__name__}
        if grounded:
            sheet["one_liner"] = parsed["one_liner"]
            sheet["summary"] = parsed["summary"]
            validated = FactSheet.model_validate(sheet).model_dump(mode="json")
            _atomic(destination, json.dumps(validated, indent=2, ensure_ascii=False) + "\n")
            _atomic_text(markdown_path, render_markdown(validated))
            sheet = validated
    banners = repeated_lines(pages)
    by_page = {int(page["pdf_page"]): page for page in pages}
    known = [item["name"] for item in sheet["items"]]
    sent = []
    unsent = []
    added = 0
    for number in range(31, 59):
        page = by_page.get(number)
        if page is None:
            unsent.append({"pdf_page": number, "reason": "missing_page"})
            continue
        body, _fresh = render_new(paragraphs_for_page(page, banners), set())
        estimate = int((estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(body)) * 1.8 + 700)
        if tokens_used(usage) + estimate > cap:
            unsent.append({"pdf_page": number, "reason": "daily_cap"})
            continue
        if not body.strip():
            unsent.append({"pdf_page": number, "reason": "no_text"})
            continue
        result = extract_batch(
            body,
            ticker="HINDALCO",
            target_pages=[number],
            allowed_pages={number},
            settings={**text_settings(), "max_attempts": 1},
            known_names=known,
        )
        record_attempt(usage, {**result, "pdf_page": number})
        spent = int((result.get("usage") or {}).get("input_tokens") or 0) + int((result.get("usage") or {}).get("output_tokens") or 0)
        if spent:
            import time
            time.sleep(min(spent / 8000 * 60, 75))
        parsed_path = CANDIDATE_DIR / "HINDALCO" / "parsed" / f"HINDALCO_p{number:04d}-p{number:04d}_reextract_{PROMPT_VERSION}.json"
        _atomic(parsed_path, json.dumps({**result, "source": "reextract"}, indent=2))
        sent.append(number)
        fresh, _rejected = drop_items(result.get("candidates") or [], set())
        for item in fresh:
            sources = []
            for source in item.get("sources") or []:
                quote = source.get("quote") or ""
                if locate_quote(quote, texts.get(number, "")):
                    sources.append({"pdf_page": number, "printed_page": source.get("printed_page"), "quote": quote})
            if not sources:
                continue
            candidate = {**item, "sources": sources, "revenue_share_pct": None, "entity": item.get("entity") or "company"}
            cleaned = _clean(candidate)
            if not cleaned:
                continue
            sheet["items"].append(cleaned)
            known.append(cleaned["name"])
            added += 1
    if added:
        if not (isinstance(report.get("overview"), dict) and report["overview"].get("grounded")):
            sheet["one_liner"] = f"HINDALCO FY25 has {len(sheet['items'])} items kept from the annual report."
        validated = FactSheet.model_validate(sheet).model_dump(mode="json")
        _atomic(destination, json.dumps(validated, indent=2, ensure_ascii=False) + "\n")
        _atomic_text(markdown_path, render_markdown(validated))
    report["reextract_sent"] = sent
    report["reextract_unsent"] = unsent
    report["items_added"] = added
    report["tokens_after"] = tokens_used(usage)
    report["prompt_version"] = PROMPT_VERSION
    return report


def _overview_json(raw: str) -> dict:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        payload = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _grounded(text: str, source: str) -> bool:
    from src.validate.quotes import fold_whitespace

    words = fold_whitespace(text).casefold().split()
    folded = fold_whitespace(source).casefold()
    if len(words) < 4:
        return False
    return any(" ".join(words[index : index + 4]) in folded for index in range(len(words) - 3))


def _add_listed(items: list[dict], listed: list[dict]) -> list[dict]:
    present = {_name_key(item.get("name") or "") for item in items}
    added = list(items)
    for item in listed:
        key = _name_key(item.get("name") or "")
        if not key or key in present:
            continue
        present.add(key)
        added.append(item)
    return added


def _segments(listed: dict) -> list[dict]:
    rows = []
    for item in listed["segments"]:
        label = item["name"]
        figures = listed["revenue"].get(label.casefold(), {})
        rows.append({
            "name": label,
            "revenue": figures.get("revenue"),
            "unit": figures.get("unit"),
            "share_pct": figures.get("share_pct"),
        })
    return rows


def _clean(item: dict) -> dict:
    kind = item.get("kind") if item.get("kind") in KINDS else "other"
    status = item.get("status") if item.get("status") in STATUSES else "current"
    sources = []
    for source in (item.get("sources") or [])[:3]:
        if not source.get("quote") or not isinstance(source.get("pdf_page"), int):
            continue
        sources.append({
            "pdf_page": source["pdf_page"],
            "printed_page": None if not source.get("printed_page") else str(source["printed_page"]),
            "quote": source["quote"],
        })
    if not sources or not str(item.get("name") or "").strip():
        return {}
    return {
        "name": item.get("name"),
        "kind": kind,
        "descriptor": item.get("descriptor") or item.get("name"),
        "entity": item.get("entity") or "company",
        "status": status,
        "revenue_share_pct": None,
        "sources": sources,
    }


def _quote_report(items: list[dict], texts: dict[int, str], pdf_texts: dict[int, str]) -> dict:
    counts = Counter()
    for item in items:
        for source in item["sources"]:
            stored = locate_quote(source["quote"], texts.get(source["pdf_page"], ""))
            layer = locate_quote(source["quote"], pdf_texts.get(source["pdf_page"], ""))
            counts[f"stored_{stored or 'missing'}"] += 1
            counts[f"pdf_{layer or 'missing'}"] += 1
    return dict(counts)


def _pdf_layer(numbers: set[int]) -> dict[int, str]:
    folder = INPUT_DIR / "HINDALCO"
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        return {}
    import pymupdf

    document = pymupdf.open(pdfs[0])
    layer = {}
    for number in numbers:
        if 1 <= number <= document.page_count:
            layer[number] = document[number - 1].get_text("text") or ""
    document.close()
    return layer


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    _atomic(path, text)


if __name__ == "__main__":
    result = run()
    printable = {key: value for key, value in result.items() if key not in {"dropped", "sample"}}
    print(json.dumps(printable, indent=2))
    print("DROPPED")
    for row in result["dropped"]:
        print(f"{row['pdf_page']}\t{row['reason']}\t{row['name']}".encode("ascii", "replace").decode("ascii"))
    print("SAMPLE")
    for row in result["sample"]:
        print(f"{row['pdf_page']}\t{row['kind']}\t{row['entity']}\t{row['name']}\t{row['quote']}".encode("ascii", "replace").decode("ascii"))
