"""POWERGRID offline cleanup from POWERGRID.v3. Writes staging only; never touches submission/."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from src.config.paths import INPUT_DIR, OCR_DIR, PAGE_DIR, PROJECT_ROOT, STAGING_DIR, SUBMISSION_DIR
from src.extract.select import page_text
from src.factsheet.build import consistency_errors, render_markdown, write_fact_sheet
from src.factsheet.names import assignment_key
from src.factsheet.schema import FactSheet
from src.validate.quotes import fold_whitespace, read_pdf_page, snap_quote

TICKER = "POWERGRID"
STAGING = STAGING_DIR / "factsheet" / TICKER
SRC_V3_JSON = STAGING / "POWERGRID.v3.json"
SRC_V3_MD = STAGING / "POWERGRID.v3.md"
OUT_JSON = STAGING / "POWERGRID.json"
OUT_MD = STAGING / "POWERGRID.md"
REPORT_PATH = STAGING / "powergrid_cleanup_report.json"
WORK_DIR = STAGING_DIR / "factsheet" / "POWERGRID_cleanup_work"

_ACTIVITY = re.compile(
    r"(?P<sent>[^.!?\n]{20,220}?\b(?:engaged in|is engaged|carries on|transmission of|"
    r"power transmission|provides?|operates?|involved in|principal activity|"
    r"business of|incorporated (?:for|to)|wholly owned subsidiary|joint venture)\b[^.!?\n]{0,160})",
    re.I,
)
_LEGAL = re.compile(
    r"([A-Z][A-Za-z0-9&.'’\-]*(?:\s+(?:&|and|[A-Z][A-Za-z0-9&.'’\-]*)){0,12}"
    r"\s+(?:Limited|Ltd\.?|Private Limited))",
    re.I,
)
_SEGMENT_NAME = re.compile(
    r"\b(Transmission|Telecom|Consultancy|Other(?:\s+segments?)?)\b",
    re.I,
)
_NUM = re.compile(r"([\d,]+(?:\.\d+)?)")
_PRINTED_OK = re.compile(r"^\d+[A-Za-z]?$")


def _sha256(path: Path) -> str:
    if not path.is_file():
        return "MISSING"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pdf_path() -> Path:
    folder = INPUT_DIR / TICKER
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF in {folder}")
    return pdfs[0]


def _load_page_record(number: int) -> dict:
    path = PAGE_DIR / TICKER / f"p{number:04d}.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _ocr_half_text(number: int) -> str:
    """OCR raw/cleaned text only — excludes vision transcripts."""
    chunks: list[str] = []
    folder = OCR_DIR / TICKER
    for half in ("LEFT", "RIGHT", "FULL"):
        path = folder / f"{TICKER}_p{number:04d}_{half}.json"
        if not path.is_file():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        tess = record.get("tesseract") or {}
        paddle = record.get("paddleocr") or {}
        for blob in (
            tess.get("cleaned_text"),
            tess.get("raw_text"),
            " ".join(
                (line.get("text") if isinstance(line, dict) else str(line))
                for line in (tess.get("kept_lines") or [])
            ),
            paddle.get("cleaned_text"),
            paddle.get("raw_text"),
        ):
            text = fold_whitespace(blob or "")
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def _non_vision_page_text(record: dict) -> str:
    blocks = [
        block
        for block in (record.get("text_blocks") or [])
        if block.get("selected", True)
        and (block.get("text") or "").strip()
        and (block.get("source") or "") != "vision"
    ]
    blocks.sort(key=lambda block: block.get("reading_order", 0))
    return "\n".join(block["text"] for block in blocks)


_GROUND_CACHE: dict[int, str] = {}


def _grounding_text(number: int, pdf_cache: dict[int, str]) -> str:
    if number in _GROUND_CACHE:
        return _GROUND_CACHE[number]
    record = _load_page_record(number)
    parts = [_ocr_half_text(number), _non_vision_page_text(record)]
    if number not in pdf_cache:
        try:
            pdf_cache[number] = read_pdf_page(_pdf_path(), number)[0] or ""
        except Exception:
            pdf_cache[number] = ""
    parts.append(pdf_cache[number])
    text = "\n".join(p for p in parts if p)
    _GROUND_CACHE[number] = text
    return text


def _activity_descriptor(name: str, page_text_value: str) -> str | None:
    """Stated-activity sentence from the page; never the item name."""
    text = fold_whitespace(page_text_value or "")
    if not text:
        return None
    name_cf = fold_whitespace(name).casefold()
    # Prefer sentences that mention the entity or sit near it.
    windows: list[str] = []
    idx = text.casefold().find(name_cf[: min(40, len(name_cf))]) if name_cf else -1
    if idx >= 0:
        windows.append(text[max(0, idx - 40) : idx + 260])
    windows.append(text)
    for window in windows:
        for match in _ACTIVITY.finditer(window):
            sent = fold_whitespace(match.group("sent")).strip(" :–—-")
            if not sent or sent.casefold() == name_cf:
                continue
            if len(sent) < 20 or len(sent) > 220:
                continue
            return sent
    # Fallback: short clause after the name on the same page.
    if idx >= 0:
        after = text[idx + len(name) : idx + len(name) + 180]
        clause = re.split(r"[.;\n]", after, maxsplit=1)[0]
        clause = fold_whitespace(clause).strip(" :–—-,")
        if clause and clause.casefold() != name_cf and 12 <= len(clause) <= 200:
            if re.search(r"[A-Za-z]", clause):
                return clause
    return None


def _fix_printed(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or not _PRINTED_OK.match(text):
        return None
    return text


def parse_list_entities(list_pages: list[int], pdf_cache: dict[int, str]) -> dict[str, dict]:
    """Parse subsidiary/JV/associate names from list pages via src.tables (digital pages) + OCR/PDF text."""
    from src.tables.digital import extract_digital_page

    entities: dict[str, dict] = {}
    pdf = _pdf_path()
    table_count = 0

    def ingest(name: str, page: int, activity: str | None, quote: str | None = None) -> None:
        name = fold_whitespace(name)
        if not name or len(name) < 8:
            return
        if not re.search(r"\b(?:Limited|Ltd\.?)\b", name, re.I):
            return
        # Skip obvious header/noise fragments
        if re.search(r"\b(?:Name of the|Particulars|Description)\b", name, re.I):
            return
        key = assignment_key(name)
        row = entities.setdefault(
            key, {"name": name, "pages": [], "activity": None, "quote": None, "pdf_page": page}
        )
        if page not in row["pages"]:
            row["pages"].append(page)
        if len(name) > len(row["name"]):
            row["name"] = name
        if activity and not row.get("activity"):
            row["activity"] = fold_whitespace(activity)[:200]
        if quote and (not row.get("quote") or len(quote) > len(row.get("quote") or "")):
            row["quote"] = quote
            row["pdf_page"] = page

    for number in list_pages:
        record = _load_page_record(number)
        # Digital table parse only when the page has a text layer (pymupdf blocks).
        has_digital = any((b.get("source") or "") == "pymupdf" for b in (record.get("text_blocks") or []))
        tables: list[dict] = []
        if has_digital:
            try:
                tables = extract_digital_page(pdf, TICKER, number)
            except Exception:
                tables = []
        for table in tables:
            rows = table.get("rows") or []
            header = " ".join(
                fold_whitespace(" ".join(h) if isinstance(h, list) else str(h or ""))
                for h in (table.get("headers") or [])
            ).casefold()
            sample = " ".join(
                fold_whitespace(" ".join((row.get("cells") or [])[:4])) for row in rows[:6]
            )
            looks = any(k in header for k in ("subsidiary", "joint venture", "associate", "name of"))
            if not looks and not re.search(r"\bLimited\b", sample, re.I):
                continue
            table_count += 1
            for row in rows:
                cells = [fold_whitespace(c or "") for c in (row.get("cells") or [])]
                blob = " ".join(cells)
                for match in _LEGAL.finditer(blob):
                    name = match.group(1)
                    rest = blob[match.end() : match.end() + 120]
                    activity = fold_whitespace(rest.split("  ")[0])[:120] or None
                    ingest(name, number, activity, quote=fold_whitespace(blob)[:240])

        text = _grounding_text(number, pdf_cache)
        for match in _LEGAL.finditer(text):
            name = match.group(1)
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 160)
            window = fold_whitespace(text[start:end])
            rest = text[match.end() : match.end() + 140]
            activity = None
            act = _ACTIVITY.search(rest)
            if act:
                activity = fold_whitespace(act.group("sent"))[:200]
            # Quote must be on this page at >=0.9
            snap = snap_quote(window[:240], text, minimum_ratio=0.9)
            quote = snap.get("quote") or (window[:240] if name.casefold() in text.casefold() else None)
            if quote and name.casefold() not in quote.casefold():
                # Prefer a window that includes the legal name
                quote = fold_whitespace(text[match.start() : min(len(text), match.end() + 120)])[:240]
            ingest(name, number, activity, quote=quote)

    print(f"LIST_PARSE tables={table_count} entities={len(entities)}")
    return entities


def parse_segments(pdf_cache: dict[int, str]) -> list[dict]:
    """Segment note: names + share_pct only when a printed total is present."""
    candidates_pages = []
    for number in range(1, 365):
        text = _grounding_text(number, pdf_cache)
        if re.search(r"segment\s+information|reportable\s+segment", text, re.I):
            candidates_pages.append(number)
    segments: list[dict] = []
    used_pages = candidates_pages[:8] or [310, 311, 312]
    blob = "\n".join(_grounding_text(n, pdf_cache) for n in used_pages)
    # Look for revenue lines with a total
    totals = [float(m.group(1).replace(",", "")) for m in re.finditer(
        r"(?:total|segment\s+revenue)[^\d]{0,40}([\d,]{3,})", blob, re.I
    )]
    printed_total = max(totals) if totals else None
    print(f"SEGMENTS pages={used_pages[:6]} printed_total={printed_total}")
    seen = set()
    for match in _SEGMENT_NAME.finditer(blob):
        name = match.group(1).title().replace("Other Segments", "Other")
        key = assignment_key(name)
        if key in seen:
            continue
        seen.add(key)
        window = blob[match.start() : match.start() + 120]
        nums = [float(m.group(1).replace(",", "")) for m in _NUM.finditer(window)]
        revenue = nums[0] if nums else None
        share = None
        if printed_total and revenue and printed_total > 0:
            share = round(100.0 * revenue / printed_total, 2)
            if share > 100 or share < 0:
                share = None
        quote = fold_whitespace(window)[:240]
        if not quote:
            continue
        page = used_pages[0]
        segments.append(
            {
                "name": name if name.casefold() != "other" else "Other",
                "kind": "segment",
                "descriptor": f"{name} reportable segment",
                "entity": "company",
                "status": "current",
                "revenue_share_pct": share,
                "sources": [{"pdf_page": page, "printed_page": None, "quote": quote}],
                "_segment_meta": {"revenue": revenue, "unit": "INR_crore", "share_pct": share},
            }
        )
        print(f"  segment {name} revenue={revenue} share_pct={share}")
    return segments


def overview_from_pages(pdf_cache: dict[int, str]) -> tuple[str, str, dict]:
    """one_liner + summary from early business-overview pages; verbatim-checked."""
    overview_pages = list(range(1, 26)) + [4, 5, 6, 7, 8, 9, 10, 17]
    texts = {n: _grounding_text(n, pdf_cache) for n in overview_pages}
    joined = "\n".join(texts[n] for n in overview_pages if texts[n])
    # Prefer a sentence mentioning transmission / POWERGRID
    one = ""
    for match in re.finditer(r"([^.!\n]{40,180}\b(?:transmission|POWERGRID|Power Grid)[^.!\n]{0,120})", joined, re.I):
        cand = fold_whitespace(match.group(1))
        if snap_quote(cand, joined, minimum_ratio=0.9).get("quote"):
            one = cand
            break
    if not one:
        for n, text in texts.items():
            if not text:
                continue
            snippet = fold_whitespace(text)[:180]
            if len(snippet) >= 40 and snap_quote(snippet, text, minimum_ratio=0.9).get("quote"):
                one = snippet
                break
    if not one:
        one = ""
    # Summary: 2-3 grounded sentences
    sentences = []
    for match in re.finditer(r"([^.!\n]{50,220}\.)", joined):
        sent = fold_whitespace(match.group(1))
        if snap_quote(sent, joined, minimum_ratio=0.9).get("quote"):
            sentences.append(sent)
        if len(sentences) >= 3:
            break
    summary = " ".join(sentences) if sentences else one
    meta = {
        "one_liner_grounded": bool(snap_quote(one, joined, minimum_ratio=0.9).get("quote")),
        "summary_sentences": len(sentences),
        "overview_pages_used": [n for n in overview_pages if texts.get(n)],
    }
    print(f"OVERVIEW one_liner_grounded={meta['one_liner_grounded']} summary_sents={len(sentences)}")
    return one[:300], summary[:900], meta


def local_dedup(items: list[dict]) -> list[dict]:
    kept: list[dict] = []
    index: dict[str, int] = {}
    for item in items:
        key = assignment_key(item["name"])
        if key in index:
            prior = kept[index[key]]
            # Prefer longer descriptor / more sources
            if len(item.get("descriptor") or "") > len(prior.get("descriptor") or ""):
                prior["descriptor"] = item.get("descriptor")
            for src in item.get("sources") or []:
                if len(prior["sources"]) >= 3:
                    break
                if src not in prior["sources"]:
                    prior["sources"].append(src)
            continue
        index[key] = len(kept)
        kept.append(item)
    return kept


def stratified_sample(items: list[dict], n: int, seed: int = 17) -> list[dict]:
    rng = random.Random(seed)
    by_kind: dict[str, list[dict]] = {}
    for item in items:
        by_kind.setdefault(item["kind"], []).append(item)
    sample: list[dict] = []
    kinds = sorted(by_kind)
    if not kinds:
        return []
    per = max(1, n // len(kinds))
    for kind in kinds:
        pool = list(by_kind[kind])
        rng.shuffle(pool)
        sample.extend(pool[:per])
    rng.shuffle(sample)
    if len(sample) < n:
        rest = [i for i in items if i not in sample]
        rng.shuffle(rest)
        sample.extend(rest[: n - len(sample)])
    return sample[:n]


def run(*, promote: bool = False) -> dict:
    """Cleanup POWERGRID.v3 into staging/factsheet/POWERGRID/. Never writes submission unless promote=True and checks pass."""
    print("STEP freeze_check submission sha", _sha256(SUBMISSION_DIR / TICKER / f"{TICKER}.json")[:16])
    if not SRC_V3_JSON.is_file():
        raise FileNotFoundError(SRC_V3_JSON)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    sheet = json.loads(SRC_V3_JSON.read_text(encoding="utf-8"))
    baseline = len(sheet.get("items") or [])
    print(f"0 baseline_items={baseline} source={SRC_V3_JSON.name}")

    pdf_cache: dict[int, str] = {}
    report: dict = {"baseline_items": baseline, "steps": {}}

    # --- Quote grounding ---
    grounded_items = []
    dropped_vision_quotes = 0
    dropped_no_quote = 0
    quote_kept = 0
    for item in sheet.get("items") or []:
        sources = []
        for src in item.get("sources") or []:
            page = int(src.get("pdf_page") or 0)
            quote = fold_whitespace(src.get("quote") or "")
            if page < 1 or not quote:
                dropped_vision_quotes += 1
                continue
            text = _grounding_text(page, pdf_cache)
            snap = snap_quote(quote, text, minimum_ratio=0.9)
            if not snap.get("quote") or float(snap.get("ratio") or 0) < 0.9:
                dropped_vision_quotes += 1
                continue
            printed = _fix_printed(src.get("printed_page"))
            record = _load_page_record(page)
            page_printed = _fix_printed(record.get("printed_page"))
            if printed and page_printed and printed != page_printed:
                printed = None
            elif not printed:
                printed = page_printed
            q = fold_whitespace(snap["quote"])[:300]
            sources.append({"pdf_page": page, "printed_page": printed, "quote": q})
            quote_kept += 1
        if not sources:
            dropped_no_quote += 1
            continue
        grounded_items.append({**item, "sources": sources[:3]})
    print(
        f"1 quote_grounding kept_items={len(grounded_items)} "
        f"quotes_kept={quote_kept} vision_or_ungrounded_dropped={dropped_vision_quotes} "
        f"items_dropped_no_quote={dropped_no_quote}"
    )
    report["steps"]["quote_grounding"] = {
        "kept_items": len(grounded_items),
        "quotes_kept": quote_kept,
        "dropped_quotes": dropped_vision_quotes,
        "dropped_items_no_quote": dropped_no_quote,
    }

    # --- Descriptors ---
    with_desc = 0
    without_desc = 0
    sub_with = []
    sub_without = []
    for item in grounded_items:
        name = item["name"]
        page = int(item["sources"][0]["pdf_page"])
        text = _grounding_text(page, pdf_cache)
        desc = _activity_descriptor(name, text)
        if desc and fold_whitespace(desc).casefold() == fold_whitespace(name).casefold():
            desc = None
        item["descriptor"] = desc or ""
        if item["descriptor"]:
            with_desc += 1
            if item.get("kind") == "subsidiary":
                sub_with.append(name)
        else:
            without_desc += 1
            if item.get("kind") == "subsidiary":
                sub_without.append(name)
    print(f"2 descriptors with={with_desc} without/null={without_desc}")
    print(f"  subsidiaries_with_descriptor={len(sub_with)}")
    for name in sub_with[:30]:
        print(f"    + {name}".encode("ascii", "replace").decode())
    print(f"  subsidiaries_without_descriptor={len(sub_without)}")
    for name in sub_without[:30]:
        print(f"    - {name}".encode("ascii", "replace").decode())
    report["steps"]["descriptors"] = {
        "with": with_desc,
        "without": without_desc,
        "subsidiaries_with": sub_with,
        "subsidiaries_without": sub_without,
    }

    # --- Entity + page re-verify ---
    entity_fixed = 0
    page_fail_drop = 0
    fixed_items = []
    for item in grounded_items:
        sources = []
        for src in item["sources"]:
            text = _grounding_text(int(src["pdf_page"]), pdf_cache)
            snap = snap_quote(src["quote"], text, minimum_ratio=0.9)
            if not snap.get("quote"):
                continue
            sources.append({**src, "quote": snap["quote"], "printed_page": _fix_printed(src.get("printed_page"))})
        if not sources:
            page_fail_drop += 1
            continue
        quote_all = " ".join(s["quote"] for s in sources)
        entity = item.get("entity") or "company"
        entity_cf = entity.casefold()
        in_quote = entity_cf in quote_all.casefold() or assignment_key(entity) == assignment_key(quote_all)
        if entity_cf not in {"company", "the company", "parent"} and not in_quote:
            if item.get("kind") == "subsidiary" and item["name"].casefold() in quote_all.casefold():
                entity = item["name"]
            else:
                entity = "company"
            entity_fixed += 1
        elif item.get("kind") == "subsidiary" and entity_cf in {"company", "the company"}:
            if item["name"].casefold() in quote_all.casefold():
                entity = item["name"]
                entity_fixed += 1
        fixed_items.append({**item, "entity": entity, "sources": sources[:3]})
    print(f"3 entity_fixed={entity_fixed} dropped_page_mismatch={page_fail_drop} items={len(fixed_items)}")
    report["steps"]["entity_pages"] = {
        "entity_fixed": entity_fixed,
        "dropped_page_mismatch": page_fail_drop,
        "items": len(fixed_items),
    }

    # --- Subsidiary list gate ---
    list_pages = []
    for number in range(1, 365):
        text = _grounding_text(number, pdf_cache)
        if re.search(r"\b(?:subsidiaries|joint ventures?|associate compan)", text, re.I):
            list_pages.append(number)
    # Prefer known subsidiary-list pages; keep a small keyword set otherwise.
    preferred = [p for p in (245, 246, 247, 248, 249, 250, 17, 4) if p in set(list_pages) or True]
    list_pages = sorted(set(preferred + list_pages))[:20]
    list_entities = parse_list_entities(list_pages, pdf_cache)
    list_keys = set(list_entities)
    kept = []
    dropped_not_on_list = 0
    kept_keys = set()
    for item in fixed_items:
        if item.get("kind") != "subsidiary":
            kept.append(item)
            kept_keys.add(assignment_key(item["name"]))
            continue
        key = assignment_key(item["name"])
        on_list = key in list_keys
        if not on_list:
            for page in list_pages:
                if item["name"].casefold() in _grounding_text(page, pdf_cache).casefold():
                    on_list = True
                    break
        if not on_list:
            dropped_not_on_list += 1
            continue
        if not item.get("descriptor") and key in list_entities and list_entities[key].get("activity"):
            act = list_entities[key]["activity"]
            if act and act.casefold() != item["name"].casefold():
                item = {**item, "descriptor": act}
        kept.append(item)
        kept_keys.add(key)

    # Rebuild OCR-grounded subsidiary items from list pages (recovers vision-only drops).
    added_from_list = 0
    for key, row in list_entities.items():
        if key in kept_keys:
            continue
        quote = fold_whitespace(row.get("quote") or "")
        page = int(row.get("pdf_page") or (row["pages"][0] if row.get("pages") else 0))
        if page < 1 or not quote or row["name"].casefold() not in quote.casefold():
            continue
        text = _grounding_text(page, pdf_cache)
        snap = snap_quote(quote, text, minimum_ratio=0.9)
        if not snap.get("quote"):
            continue
        activity = row.get("activity")
        if activity and activity.casefold() == row["name"].casefold():
            activity = None
        if not activity:
            activity = _activity_descriptor(row["name"], text)
        if activity and (len(activity) > 200 or len(_NUM.findall(activity)) >= 6):
            activity = None
        kept.append(
            {
                "name": row["name"],
                "kind": "subsidiary",
                "descriptor": (activity or "")[:200],
                "entity": row["name"],
                "status": "current",
                "revenue_share_pct": None,
                "sources": [
                    {
                        "pdf_page": page,
                        "printed_page": _fix_printed(_load_page_record(page).get("printed_page")),
                        "quote": fold_whitespace(snap["quote"])[:300],
                    }
                ],
            }
        )
        kept_keys.add(key)
        added_from_list += 1
    print(
        f"4 list_gate list_pages={len(list_pages)} list_entities={len(list_entities)} "
        f"dropped_not_on_list={dropped_not_on_list} added_from_list={added_from_list} items={len(kept)}"
    )
    report["steps"]["list_gate"] = {
        "list_pages": list_pages,
        "list_entities": len(list_entities),
        "dropped_not_on_list": dropped_not_on_list,
        "added_from_list": added_from_list,
        "items": len(kept),
    }

    # --- Segments ---
    non_seg = [i for i in kept if i.get("kind") != "segment"]
    seg_items = parse_segments(pdf_cache)
    segment_rows = []
    for seg in seg_items:
        meta = seg.pop("_segment_meta", {})
        segment_rows.append(
            {
                "name": seg["name"],
                "revenue": meta.get("revenue"),
                "unit": meta.get("unit"),
                "share_pct": meta.get("share_pct"),
            }
        )
        non_seg.append(seg)
    items = local_dedup(non_seg)
    print(f"5 after_segments_dedup items={len(items)} segments={len(segment_rows)}")
    report["steps"]["segments"] = {"count": len(segment_rows), "rows": segment_rows}

    # --- Overview ---
    one_liner, summary, overview_meta = overview_from_pages(pdf_cache)
    report["steps"]["overview"] = overview_meta

    if not one_liner:
        # Last resort: first grounded sentence from page 1–10 (still verbatim-checked above).
        one_liner = (summary or "Power Grid Corporation of India Limited.")[:300]
    sheet_out = {
        "ticker": TICKER,
        "period": sheet.get("period") or "FY25",
        "one_liner": one_liner,
        "summary": summary or one_liner,
        "segments": segment_rows,
        "items": sorted(items, key=lambda i: (min(s["pdf_page"] for s in i["sources"]), i["name"])),
    }

    validated = FactSheet.model_validate(sheet_out).model_dump(mode="json")
    markdown = render_markdown(validated)
    errors = consistency_errors(validated, markdown)
    print(f"6 schema_ok items={len(validated['items'])} consistency_errors={errors}")

    # Write staging (canonical name) — NOT submission
    built = {"sheet": validated, "markdown": markdown, "omitted": []}
    write_fact_sheet(TICKER, built, STAGING_DIR / "factsheet", force=True)
    # Keep a work copy too
    (WORK_DIR / "POWERGRID.json").write_text(json.dumps(validated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (WORK_DIR / "POWERGRID.md").write_text(markdown, encoding="utf-8")

    # list_subsidiaries sidecar for check exceptions
    keys = sorted({assignment_key(i["name"]) for i in validated["items"] if i["kind"] == "subsidiary"})
    (STAGING / "list_subsidiaries.json").write_text(json.dumps({"keys": keys}, indent=2) + "\n", encoding="utf-8")

    from src.eval.check_factsheet import check_factsheet

    check_rows = check_factsheet(TICKER, path=STAGING)
    failed = [r for r in check_rows if not r["passed"]]
    for row in check_rows:
        status = "PASS" if row["passed"] else "FAIL"
        print(f"{status} {row['check']}. {row['detail']}")

    by_kind = dict(Counter(i["kind"] for i in validated["items"]))
    sample = stratified_sample(validated["items"], 40, seed=17)
    sample_rows = [
        {
            "kind": i["kind"],
            "name": i["name"],
            "entity": i["entity"],
            "descriptor": (i.get("descriptor") or "")[:120],
            "pdf_page": i["sources"][0]["pdf_page"],
            "quote": i["sources"][0]["quote"][:140],
        }
        for i in sample
    ]
    print("BY_KIND", json.dumps(by_kind, indent=2))
    print("SAMPLE40")
    for row in sample_rows:
        line = f"{row['kind']}\t{row['name']}\t{row['descriptor']}\t{row['quote']}"
        print(line.encode("ascii", "replace").decode())

    report.update(
        {
            "item_count": len(validated["items"]),
            "by_kind": by_kind,
            "sample40": sample_rows,
            "consistency_errors": errors,
            "checks": check_rows,
            "failed_checks": len(failed),
            "submission_untouched_sha": _sha256(SUBMISSION_DIR / TICKER / f"{TICKER}.json"),
            "staging_sha": _sha256(OUT_JSON),
            "all_pass": not failed and not errors,
        }
    )
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if promote and report["all_pass"]:
        from src.factsheet.build import write_fact_sheet as _w
        from src.config.paths import OUTPUT_DIR

        _w(TICKER, built, OUTPUT_DIR, force=True)
        _w(TICKER, built, SUBMISSION_DIR, force=True)
        print("PROMOTED to outputs/ and submission/")
    else:
        print("SKIP promote: checks not all pass or promote=False (submission/ untouched)")
    print(f"FINAL_ITEM_COUNT={len(validated['items'])}")
    return report


if __name__ == "__main__":
    raise SystemExit(0 if run(promote=False).get("all_pass") else 1)
