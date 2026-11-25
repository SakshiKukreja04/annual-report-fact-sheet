"""Print PASS or FAIL for the assignment fact-sheet checks. Does not add items."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from src.config.paths import CANDIDATE_DIR, PAGE_DIR, STAGING_DIR
from src.extract.prompt import PROMPT_VERSION
from src.extract.select import page_text
from src.factsheet.build import consistency_errors
from src.factsheet.names import actor_entity, assignment_key, distinct_proper_conflict, should_merge, topic_match
from src.factsheet.precision import descriptor_problem, generic_noun_only, quote_supports_name
from src.factsheet.overview import fallback_text
from src.factsheet.schema import FactSheet
from src.validate.quotes import read_pdf_page, resolve_quote_on_page


def check_factsheet(ticker: str, path: str | Path | None = None) -> list[dict]:
    ticker = ticker.upper()
    folder = Path(path) if path else (STAGING_DIR / "factsheet" / ticker)
    if folder.is_file():
        folder = folder.parent
    sidecar = STAGING_DIR / "factsheet" / ticker
    json_path = folder / f"{ticker}.json"
    markdown_path = folder / f"{ticker}.md"
    results = []
    sheet = None
    markdown = ""
    if not json_path.is_file() or not markdown_path.is_file():
        results.append(_row(1, False, "JSON and Markdown must both exist", []))
        for number in range(2, 11):
            results.append(_row(number, False, "sheet missing", []))
        return results
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        sheet = FactSheet.model_validate(payload).model_dump(mode="json")
        markdown = markdown_path.read_text(encoding="utf-8")
        errors = consistency_errors(sheet, markdown)
        results.append(_row(1, not errors, "schema valid" if not errors else "; ".join(errors), errors))
    except Exception as exc:
        results.append(_row(1, False, f"{type(exc).__name__}: {exc}", []))
        for number in range(2, 11):
            results.append(_row(number, False, "schema invalid", []))
        return results

    count = len(sheet["items"])
    shortfall = max(0, 190 - count)
    results.append(_row(
        2,
        shortfall == 0,
        f"items={count}" if shortfall == 0 else f"items={count} shortfall={shortfall}",
        [] if shortfall == 0 else [f"shortfall {shortfall}"],
    ))

    segment_report = _read(folder / "segments_report.json") or _read(sidecar / "segments_report.json")
    table_found = bool(segment_report.get("table_found"))
    segment_ok = (not table_found) or bool(sheet["segments"])
    detail = "no segment table" if not table_found else f"segments={len(sheet['segments'])}"
    if not table_found:
        detail = f"no segment table ({segment_report.get('reason') or 'not found'})"
    results.append(_row(3, segment_ok, detail, [] if segment_ok else ["segment table found but segments is empty"]))

    one_fallback, summary_fallback = fallback_text(ticker)
    overview = _read(folder / "overview.json") or _read(sidecar / "overview.json")
    fallback = bool(overview.get("fallback")) or sheet["one_liner"] == one_fallback or sheet["summary"] == summary_fallback
    results.append(_row(4, not fallback, "grounded overview" if not fallback else f"fallback ({overview.get('reason') or 'generic text'})", []))

    quote_failures = _quote_failures(ticker, sheet)
    results.append(_row(5, not quote_failures, f"quote failures={len(quote_failures)}", quote_failures[:30]))

    duplicates = _duplicates(sheet)
    results.append(_row(6, not duplicates, f"duplicate groups={len(duplicates)}", duplicates[:30]))

    entity_failures = _entity_failures(sheet)
    results.append(_row(7, not entity_failures, f"company-entity conflicts={len(entity_failures)}", entity_failures[:30]))

    name_only_allowed = set(_name_only_names(ticker))
    list_keys = set(_list_subsidiary_keys(ticker))
    descriptor_failures = []
    for item in sheet["items"]:
        descriptor = (item.get("descriptor") or "").strip()
        same = descriptor == item["name"].strip()
        empty = not descriptor
        key = assignment_key(item["name"])
        list_row = item["kind"] == "subsidiary" and (key in list_keys or key in name_only_allowed)
        if list_row and (empty or same):
            continue
        if same or empty:
            descriptor_failures.append(item["name"])
    results.append(_row(
        8,
        not descriptor_failures,
        f"descriptor-equals-name or empty count={len(descriptor_failures)}",
        descriptor_failures[:30],
    ))

    share_failures = _share_failures(ticker, sheet)
    results.append(_row(9, not share_failures, f"unsupported shares={len(share_failures)}", share_failures[:30]))

    missing_printed = []
    for item in sheet["items"]:
        sources = item["sources"]
        if not 1 <= len(sources) <= 3:
            missing_printed.append(f"{item['name']}: {len(sources)} sources")
        for source in sources:
            if source.get("printed_page") in (None, ""):
                missing_printed.append(f"{item['name']} pdf {source.get('pdf_page')}: missing printed_page")
            if not source.get("pdf_page"):
                missing_printed.append(f"{item['name']}: missing pdf_page")
    results.append(_row(10, not missing_printed, f"source gaps={len(missing_printed)}", missing_printed[:40]))

    unsupported = []
    bare_amounts = []
    generic_names = []
    for item in sheet["items"]:
        quote = " ".join(source.get("quote") or "" for source in item["sources"])
        if not quote_supports_name(item["name"], quote):
            unsupported.append(item["name"])
        if item["kind"] in {"asset", "product_or_service"} and _bare_amount(quote):
            bare_amounts.append(item["name"])
        if generic_noun_only(item["name"]):
            generic_names.append(item["name"])
    results.append(_row(11, not unsupported, f"quote missing name tokens={len(unsupported)}", unsupported[:30]))
    results.append(_row(12, not bare_amounts, f"quote ending in a bare amount={len(bare_amounts)}", bare_amounts[:30]))
    results.append(_row(13, not generic_names, f"generic-noun-only names={len(generic_names)}", generic_names[:30]))
    duplicate_pairs = []
    items = sheet["items"]
    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            if should_merge(items[left]["name"], items[right]["name"]):
                duplicate_pairs.append(f"{items[left]['name']} == {items[right]['name']}")
    results.append(_row(14, not duplicate_pairs, f"extended duplicate pairs={len(duplicate_pairs)}", duplicate_pairs[:30]))
    proper_conflicts = _proper_noun_merge_conflicts(ticker)
    results.append(_row(15, not proper_conflicts, f"proper-noun merge conflicts={len(proper_conflicts)}", proper_conflicts[:30]))
    entity_quote = _entity_not_in_quote(sheet)
    results.append(_row(16, not entity_quote, f"entity missing from quote={len(entity_quote)}", entity_quote[:30]))
    folio_mismatches = _printed_page_mismatches(ticker, sheet)
    results.append(_row(17, not folio_mismatches, f"printed page mismatches={len(folio_mismatches)}", folio_mismatches[:30]))
    descriptor_shape = _descriptor_shape(sheet, ticker)
    results.append(_row(18, not descriptor_shape, f"descriptor shape={len(descriptor_shape)}", descriptor_shape[:30]))
    long_quotes = _long_quotes(sheet)
    results.append(_row(19, not long_quotes, f"quotes over 300 or with controls={len(long_quotes)}", long_quotes[:30]))
    actor_mismatch = _actor_mismatch(sheet)
    results.append(_row(20, not actor_mismatch, f"entity mismatches={len(actor_mismatch)}", actor_mismatch[:30]))
    topic_dupes = _topic_duplicates(sheet)
    results.append(_row(21, not topic_dupes, f"topic duplicates={len(topic_dupes)}", topic_dupes[:30]))
    share_labels = _share_label_failures(ticker, sheet)
    results.append(_row(22, not share_labels, f"share label mismatches={len(share_labels)}", share_labels[:30]))
    long_desc = []
    numeric_desc = []
    cin_names = []
    for item in sheet["items"]:
        descriptor = item.get("descriptor") or ""
        if len(descriptor) > 200:
            long_desc.append(item["name"])
        if len(re.findall(r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?%?", descriptor)) >= 6:
            numeric_desc.append(item["name"])
        if re.match(r"^U\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}\b", (item.get("name") or "").strip()):
            cin_names.append(item["name"])
    results.append(_row(23, not long_desc, f"descriptor>200 chars={len(long_desc)}", long_desc[:30]))
    results.append(_row(24, not numeric_desc, f"descriptor>=6 numbers={len(numeric_desc)}", numeric_desc[:30]))
    results.append(_row(25, not cin_names, f"CIN-prefixed names={len(cin_names)}", cin_names[:30]))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a staged fact sheet against the assignment.")
    parser.add_argument("--company", required=True)
    parser.add_argument(
        "--path",
        default="",
        help="Folder containing <TICKER>.json and <TICKER>.md (default: staging/factsheet/<TICKER>).",
    )
    args = parser.parse_args(argv)
    results = check_factsheet(args.company, path=args.path or None)
    failed = False
    for row in results:
        status = "PASS" if row["passed"] else "FAIL"
        print(f"{status} {row['check']}. {row['detail']}")
        if not row["passed"]:
            failed = True
            for item in row["examples"]:
                print(f"  - {item}")
    return 1 if failed else 0


def _quote_failures(ticker: str, sheet: dict) -> list[str]:
    pages = {}
    folder = PAGE_DIR / ticker
    native_cache: dict[int, str] = {}
    from src.config.paths import INPUT_DIR

    pdfs = sorted((INPUT_DIR / ticker).glob("*.pdf"))
    pdf_path = pdfs[0] if len(pdfs) == 1 else None
    failures = []
    for item in sheet["items"]:
        for source in item["sources"]:
            page = int(source["pdf_page"])
            if page not in pages:
                path = folder / f"p{page:04d}.json"
                pages[page] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            text = page_text(pages[page]) if pages[page] else ""
            verdict = resolve_quote_on_page(source.get("quote") or "", text)
            if not verdict["accepted"] and pdf_path is not None:
                if page not in native_cache:
                    native_cache[page] = read_pdf_page(pdf_path, page)[0]
                verdict = resolve_quote_on_page(source.get("quote") or "", native_cache[page])
            if not verdict["accepted"]:
                failures.append(f"{item['name']} pdf {page}: {verdict.get('reason')}")
    return failures


def _duplicates(sheet: dict) -> list[str]:
    seen: dict[tuple, str] = {}
    failures = []
    for item in sheet["items"]:
        key = (item["kind"], assignment_key(item["name"]))
        if not key[1]:
            continue
        if key in seen:
            failures.append(f"{seen[key]} == {item['name']}")
        else:
            seen[key] = item["name"]
    return failures


def _entity_failures(sheet: dict) -> list[str]:
    names = [item["name"] for item in sheet["items"] if item["kind"] == "subsidiary"]
    failures = []
    for item in sheet["items"]:
        if item["entity"].casefold() not in {"company", "the company"}:
            continue
        for source in item["sources"]:
            actor = actor_entity(source.get("quote") or "", item["entity"], names)
            if actor.casefold() not in {"company", "the company", "parent"}:
                failures.append(f"{item['name']} quote names {actor}")
                break
    return failures


def _bare_amount(quote: str) -> bool:
    text = quote.strip()
    if re.search(r"(?:\d{1,2}[./]\d{1,2}[./]\d{2,4}|(?:19|20)\d{2})\s*$", text):
        return False
    return bool(re.search(r"\d[\d,]*(?:\.\d+)?\s*$", text))


def _percent_in_quote(share: float, quote: str) -> bool:
    rendered = f"{share:g}"
    return bool(re.search(rf"(?<!\d){re.escape(rendered)}\s*%", quote or ""))


def _proper_noun_merge_conflicts(ticker: str) -> list[str]:
    path = STAGING_DIR / "factsheet" / ticker / "precision_report.json"
    if not path.is_file():
        return []
    report = json.loads(path.read_text(encoding="utf-8"))
    conflicts = []
    for group in report.get("merge_groups") or []:
        names = [name for name in group if name]
        for left in range(len(names)):
            for right in range(left + 1, len(names)):
                if distinct_proper_conflict(names[left], names[right]):
                    conflicts.append(f"{names[left]} == {names[right]}")
    return conflicts


def _entity_not_in_quote(sheet: dict) -> list[str]:
    coalfields = {
        "ecl": "eastern coalfields limited",
        "bccl": "bharat coking coal limited",
        "ccl": "central coalfields limited",
        "ncl": "northern coalfields limited",
        "wcl": "western coalfields limited",
        "secl": "south eastern coalfields limited",
        "mcl": "mahanadi coalfields limited",
        "cmpdi": "central mine planning & design institute limited",
        "cmpdil": "central mine planning & design institute limited",
    }
    failures = []
    for item in sheet["items"]:
        entity = str(item.get("entity") or "")
        if entity.casefold() in {"company", "the company", "parent"}:
            continue
        quote = " ".join(source.get("quote") or "" for source in item.get("sources") or [])
        folded = quote.casefold().replace("&", " and ")
        target = entity.casefold().replace("&", " and ")
        if target in folded:
            continue
        # Accept printed abbreviations that expand to the entity (or vice versa).
        ok = False
        for abbr, full in coalfields.items():
            if target in {abbr, full} and (abbr in folded or full in folded):
                ok = True
                break
        if not ok:
            failures.append(f"{item['name']} entity {entity}")
    return failures


def _printed_page_mismatches(ticker: str, sheet: dict) -> list[str]:
    failures = []
    for item in sheet["items"]:
        for source in item.get("sources") or []:
            path = PAGE_DIR / ticker / f"p{int(source['pdf_page']):04d}.json"
            if not path.is_file():
                failures.append(f"{item['name']} pdf {source.get('pdf_page')}: no page record")
                continue
            record = json.loads(path.read_text(encoding="utf-8")).get("printed_page")
            got = source.get("printed_page")
            if _printed_matches(got, record):
                continue
            failures.append(f"{item['name']} pdf {source.get('pdf_page')}: sheet {got} record {record}")
    return failures


def _printed_matches(got, record) -> bool:
    if got in (None, "") and record in (None, ""):
        return True
    if str(got) == str(record):
        return True
    parts = [part.strip() for part in str(record or "").split("/") if part.strip()]
    return len(parts) == 2 and str(got) in parts


def _share_failures(ticker: str, sheet: dict) -> list[str]:
    allowed = {}
    path = CANDIDATE_DIR / ticker / "validation" / "validated.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("validated") if isinstance(payload, dict) else payload
        for row in rows or []:
            candidate = row.get("candidate") if isinstance(row.get("candidate"), dict) else row
            status = (row.get("financial") or {}).get("status")
            if status in {"printed", "calculated_from_printed_figures"}:
                allowed[assignment_key(candidate.get("name") or "")] = status
    failures = []
    for item in sheet["items"]:
        if item.get("revenue_share_pct") is None:
            continue
        quote = " ".join(source.get("quote") or "" for source in item.get("sources") or [])
        if _percent_in_quote(item["revenue_share_pct"], quote):
            continue
        if assignment_key(item["name"]) not in allowed:
            failures.append(f"{item['name']} share {item['revenue_share_pct']}")
    return failures


def _list_subsidiary_keys(ticker: str) -> list[str]:
    path = STAGING_DIR / "factsheet" / ticker / "list_subsidiaries.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("keys") or [])


def _name_only_names(ticker: str) -> list[str]:
    path = CANDIDATE_DIR / ticker / "parsed" / f"{ticker}_tables_{PROMPT_VERSION}.json"
    if not path.is_file():
        return []
    record = json.loads(path.read_text(encoding="utf-8"))
    keys = []
    for name in record.get("name_only_names") or []:
        key = assignment_key(name)
        if key:
            keys.append(key)
    return keys


def _read(path: Path) -> dict:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _descriptor_shape(sheet: dict, ticker: str = "") -> list[str]:
    list_keys = set(_list_subsidiary_keys(ticker or sheet.get("ticker") or ""))
    failures = []
    for item in sheet["items"]:
        descriptor = item.get("descriptor") or ""
        if not descriptor.strip():
            continue
        key = assignment_key(item["name"])
        if item.get("kind") == "subsidiary" and (key in list_keys or re.search(r"\([A-Z]{2,12}\)", descriptor)):
            # Activity phrase from the consolidated subsidiary table is allowed.
            continue
        quote = " ".join(source.get("quote") or "" for source in item.get("sources") or [])
        problem = descriptor_problem(item["name"], descriptor, quote)
        if problem:
            failures.append(f"{item['name']}: {problem}")
    return failures


def _long_quotes(sheet: dict) -> list[str]:
    failures = []
    for item in sheet["items"]:
        for source in item.get("sources") or []:
            quote = source.get("quote") or ""
            if len(quote) > 300 or re.search(r"[\x00-\x1f\x7f]", quote):
                failures.append(f"{item['name']} pdf {source.get('pdf_page')}: {len(quote)} chars")
    return failures


def _actor_mismatch(sheet: dict) -> list[str]:
    names = [item["name"] for item in sheet["items"] if item["kind"] == "subsidiary"]
    abbr = {
        "ecl": "eastern coalfields",
        "bccl": "bharat coking",
        "ccl": "central coalfields",
        "ncl": "northern coalfields",
        "wcl": "western coalfields",
        "secl": "south eastern",
        "mcl": "mahanadi",
        "cmpdi": "central mine planning",
        "cmpdil": "central mine planning",
    }
    failures = []
    for item in sheet["items"]:
        entity = str(item.get("entity") or "")
        for source in item.get("sources") or []:
            actor = actor_entity(source.get("quote") or "", entity, names)
            folded = actor.casefold()
            ent = entity.casefold()
            if folded in {"company", "the company", "parent", ent}:
                continue
            if folded in ent or ent in folded:
                continue
            # CMPDI / ECL style abbreviations vs full entity
            ok = False
            for key, stem in abbr.items():
                if (folded == key or stem in folded) and (ent == key or stem in ent or key in ent):
                    ok = True
                    break
            if ok:
                continue
            failures.append(f"{item['name']} entity {entity} quote actor {actor}")
            break
    return failures


def _topic_duplicates(sheet: dict) -> list[str]:
    failures = []
    items = sheet["items"]
    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            if topic_match(items[left]["name"], items[right]["name"]):
                failures.append(f"{items[left]['name']} == {items[right]['name']}")
    return failures


def _share_label_failures(ticker: str, sheet: dict) -> list[str]:
    verified = {}
    path = STAGING_DIR / "factsheet" / ticker / "manual_overrides.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for spec in payload.get("items") or []:
            if spec.get("image_verified") and spec.get("revenue_share_pct") is not None:
                verified[assignment_key(spec.get("name") or "")] = float(spec["revenue_share_pct"])
    failures = []
    for item in sheet["items"]:
        share = item.get("revenue_share_pct")
        if share is None:
            continue
        quote = " ".join(source.get("quote") or "" for source in item.get("sources") or [])
        if not _percent_in_quote(share, quote):
            failures.append(f"{item['name']} share {share} missing from quote")
            continue
        if verified.get(assignment_key(item["name"])) == float(share):
            continue
        if not _label_near_share(item["name"], float(share), quote):
            failures.append(f"{item['name']} share {share} is not next to its label")
    return failures


def _label_near_share(name: str, share: float, quote: str) -> bool:
    rendered = f"{share:g}"
    folded = quote or ""
    for match in re.finditer(rf"(?<!\d){re.escape(rendered)}\s*%", folded):
        window = folded[max(0, match.start() - 24): match.end() + 24]
        if name.casefold() in window.casefold():
            return True
        head = name.split()[-1]
        if len(head) > 2 and re.search(rf"\b{re.escape(head)}\b", window, re.I):
            return True
    return False


def _row(number: int, passed: bool, detail: str, examples: list[str]) -> dict:
    return {"check": number, "passed": passed, "detail": detail, "examples": examples}


if __name__ == "__main__":
    sys.exit(main())
