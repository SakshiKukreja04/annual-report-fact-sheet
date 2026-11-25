"""Stage 2 POWERGRID offline rebuild. Staging only. No Groq."""

from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from src.config.paths import INPUT_DIR, OCR_DIR, PAGE_DIR, STAGING_DIR, SUBMISSION_DIR
from src.extract.select import page_text
from src.factsheet.build import consistency_errors, render_markdown, write_fact_sheet
from src.factsheet.names import assignment_key
from src.factsheet.schema import FactSheet
from src.validate.quotes import fold_whitespace, snap_quote

TICKER = "POWERGRID"
STAGING = STAGING_DIR / "factsheet" / TICKER
REPORT = STAGING / "stage2_report.json"
WORK = STAGING_DIR / "factsheet" / "POWERGRID_stage2_work"
V3 = STAGING / "POWERGRID.v3.json"

LIST_PAGES = list(range(245, 251)) + [355, 356, 357, 358, 359, 360, 361, 362]
BUSINESS_PAGES = [337, 338]
DROP_NAMES = {
    "Debentures / Bonds",
    "Debentures/Bonds",
    "Green Bond",
    "Zero Coupon Bond",
    "Cross Border",
    "Operating Segments",
    "Other Operating Revenue - Transmission",
}
_SKIP_NAME = re.compile(
    r"\b("
    r"Particulars|Disclosure|Statement|Annexure|Dividend|Interest|Investment|"
    r"For the year|Independent|Consolidated Financial|And \d|Services provided|"
    r"Joint Ventures|Other Related|Operating Segment|Related Party|"
    r"Corporation of India Limited|Companies Act|auditors?|report of"
    r")\b",
    re.I,
)
_NAME_START = re.compile(
    r"^(?:Power\s*g(?:rid|nd|ed|rio|rit|rn?d)?|Powargrid|Powegrid|Powengrist|Fowergrid|"
    r"National|Cross\s*Bord(?:er|ey)|Bihar\s+Grid|Energy\s+Efficiency|RINL|"
    r"Sikkim|Warora|Bidar|Bulwal|Fatehgarh|Central\s+Transmission|"
    r"Khavda|Chitr?a?durga|Bahoskuntio|Banas\s*kantha|Kumoal|Rojosthan|"
    r"Kunool|Notianal|POWERGRID)",
    re.I,
)

# OCR variant stem -> preferred canonical fragment
VARIANT_MAP = {
    "medipur jeral": "medinipur jeerat",
    "madinpur jeerak": "medinipur jeerat",
    "medipur jeral transmission": "medinipur jeerat transmission",
    "nithanchal": "nithanal",
    "midhanchal": "mithilanchal",
    "kurwae": "kurawar",
    "otpad": "olpad",
    "prayabad": "firozabad",
    "aymer phagi": "ajmer phagi",
    "bihul": "bikaner",
    "garmt yamuna": "garhmukteshwar",
    "ghorar": "gurgaon",
    "erwf": "erwr",
    "jeerkat": "jeerat",
    "jeerat": "jeerat",
    "koppil gadug": "koppal gadag",
    "khayda": "khavda",
    "banusk": "banaskantha",
    "unchohar": "unchahar",
    "noida": "noida",
}

_LEGAL = re.compile(
    r"("
    r"(?:Power\s*g(?:rid|nd|ed|rio|rit|rn?d)?|Powargrid|Powegrid|Powengrist|Fowergrid|POWERGRID|"
    r"National|Cross\s*Bord(?:er|ey)|Bihar\s+Grid|Energy\s+Efficiency|RINL|Sikkim|"
    r"Warora|Bidar|Bulwal|Fatehgarh|Central\s+Transmission|Khavda|"
    r"Chitr?a?durga|Bahoskuntio|Banas\s*kantha|Kumoal|Rojosthan|Kunool|Notianal)"
    r"[A-Za-z0-9&./'\-]*(?:\s+[A-Za-z0-9&./'\-]+){0,10}"
    r"\s+(?:Limited|Ltd\.?|Private\s+Limited|Limitest|Lumited|Limiiod|Limtied|Limled|Limiied|Umited)"
    r")",
    re.I,
)


def ocr_raw(page: int, *, include_vision: bool = False) -> str:
    chunks: list[str] = []
    for half in ("LEFT", "RIGHT", "FULL"):
        path = OCR_DIR / TICKER / f"{TICKER}_p{page:04d}_{half}.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        tess = data.get("tesseract") or {}
        paddle = data.get("paddleocr") or {}
        for blob in (
            tess.get("raw_text"),
            tess.get("cleaned_text"),
            " ".join((ln.get("text") if isinstance(ln, dict) else str(ln)) for ln in (tess.get("kept_lines") or [])),
            paddle.get("raw_text"),
            paddle.get("cleaned_text"),
        ):
            if blob:
                chunks.append(blob)
        for variant in tess.get("raw_variants") or []:
            if isinstance(variant, dict) and variant.get("text"):
                chunks.append(variant["text"])
            elif isinstance(variant, str):
                chunks.append(variant)
        if include_vision:
            vis = (data.get("vision_ocr") or {}).get("transcript") or ""
            if vis:
                chunks.append(vis)
    return "\n".join(chunks)


def page_non_vision(page: int) -> str:
    path = PAGE_DIR / TICKER / f"p{page:04d}.json"
    if not path.is_file():
        return ""
    data = json.loads(path.read_text(encoding="utf-8"))
    blocks = [
        b.get("text") or ""
        for b in (data.get("text_blocks") or [])
        if (b.get("source") or "") != "vision" and (b.get("text") or "").strip()
    ]
    return "\n".join(blocks)


def page_all(page: int) -> str:
    path = PAGE_DIR / TICKER / f"p{page:04d}.json"
    text = ""
    if path.is_file():
        text = page_text(json.loads(path.read_text(encoding="utf-8"))) or ""
    return "\n".join(x for x in (ocr_raw(page, include_vision=True), page_non_vision(page), text) if x)


def canon_cleanup(name: str) -> str:
    name = fold_whitespace(name)
    name = re.sub(r"^\d+\s*", "", name)
    name = re.sub(r"\b(Limitest|Lumited|Limiiod|Limtied)\b", "Limited", name, flags=re.I)
    name = re.sub(r"\bPower\s*gid\b", "Powergrid", name, flags=re.I)
    name = re.sub(r"\bPower\s*grid\b", "Powergrid", name, flags=re.I)
    name = re.sub(r"\bPoworgrid\b", "Powergrid", name, flags=re.I)
    name = re.sub(r"\bPowergrnd\b", "Powergrid", name, flags=re.I)
    name = re.sub(r"\bPowerged\b", "Powergrid", name, flags=re.I)
    name = re.sub(r"\bfransmasion\b", "Transmission", name, flags=re.I)
    name = re.sub(r"\bTranamiesion\b", "Transmission", name, flags=re.I)
    name = re.sub(r"\bTronamission\b", "Transmission", name, flags=re.I)
    name = re.sub(r"\bAugmentotion\b", "Augmentation", name, flags=re.I)
    name = fold_whitespace(name)
    return name


def norm_key(name: str) -> str:
    text = canon_cleanup(name).casefold()
    for bad, good in VARIANT_MAP.items():
        text = text.replace(bad, good)
    text = text.replace("&", " and ")
    text = re.sub(r"\bprivate\b", "pvt", text)
    text = re.sub(r"\blimited\b", "ltd", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _accept_list_name(raw: str) -> bool:
    if len(raw) < 12 or len(raw) > 80:
        return False
    words = raw.split()
    if not (2 <= len(words) <= 10):
        return False
    if _SKIP_NAME.search(raw):
        return False
    if not _NAME_START.search(raw):
        return False
    # reject glued multi-entity / table-number OCR
    lim_pat = r"\b(?:Limited|Ltd\.?|Limitest|Lumited|Limiiod|Limtied|Limled|Limiied|Umited|Lamited|Lemated|tmitest|timited|Simited)\b"
    if len(re.findall(lim_pat, raw, re.I)) != 1:
        return False
    if re.search(r"\d+\.\d+", raw):  # 0.74 etc.
        return False
    if re.search(r"\b\d{2,}\b", raw) and not re.search(r"\b(?:II|III|IV|I+|\d)\s+(?:Transmission|Power|Transco)", raw, re.I):
        # allow Roman/ordinal project markers; reject table cells like L03 / 1022
        if re.search(r"\bL?\d{2,}\b", raw):
            return False
    if re.search(
        r"\b(?:Power\s*g\w*|Powargrid|Powegrid|Fowergrid|Powengrist|POWERGRID)\b"
        r".+\b(?:Power\s*g\w*|Powargrid|Powegrid|Fowergrid|Powengrist|POWERGRID)\b",
        raw,
        re.I,
    ):
        return False
    # must end with Limited/Ltd variant
    if not re.search(lim_pat + r"\s*$", raw, re.I):
        return False
    return True


def merge_near_duplicates(entities: dict[str, dict], *, threshold: float = 0.92) -> dict[str, dict]:
    """Collapse OCR variants of the same subsidiary into one canonical row."""
    keys = sorted(entities.keys(), key=lambda k: (-len(entities[k]["name"]), k))
    kept: dict[str, dict] = {}
    for key in keys:
        row = entities[key]
        matched = None
        for kk in list(kept.keys()):
            if SequenceMatcher(None, key, kk).ratio() >= threshold:
                matched = kk
                break
        if matched is None:
            kept[key] = row
            continue
        host = kept[matched]
        for p in row["pages"]:
            if p not in host["pages"]:
                host["pages"].append(p)
        host["quotes"].update(row.get("quotes") or {})
        # Prefer cleaner Powergrid-spelled longer names without OCR digit junk
        cand = row["name"]
        cur = host["name"]
        prefer = (
            (cand.casefold().startswith("powergrid") and not cur.casefold().startswith("powergrid"))
            or (cand.casefold().startswith("powergrid") and len(cand) > len(cur) and "  " not in cand)
        )
        if prefer:
            host["name"] = cand
    return kept


def parse_list_entities() -> dict[str, dict]:
    entities: dict[str, dict] = {}
    page_hits: dict[int, int] = {}
    for page in LIST_PAGES:
        page_hits[page] = 0
        ocr = ocr_raw(page, include_vision=False) or page_non_vision(page)
        # Line-first: related-party tables are one entity per line
        lines = (ocr or page_all(page)).splitlines()
        candidates: list[str] = []
        for line in lines:
            line = fold_whitespace(line)
            if len(line) < 12:
                continue
            for match in _LEGAL.finditer(line):
                candidates.append(match.group(1))
        # Also scan folded OCR for single-line misses (JV block on 362)
        for match in _LEGAL.finditer(fold_whitespace(ocr or "")):
            candidates.append(match.group(1))
        for cand in candidates:
            raw = canon_cleanup(cand)
            if not _accept_list_name(raw):
                continue
            key = norm_key(raw)
            if len(key) < 8:
                continue
            row = entities.setdefault(key, {"name": raw, "pages": [], "quotes": {}})
            if page not in row["pages"]:
                row["pages"].append(page)
            if raw.casefold().startswith("powergrid") and len(raw) >= len(row["name"]):
                row["name"] = raw
            elif not row["name"].casefold().startswith("powergrid") and len(raw) > len(row["name"]):
                row["name"] = raw
            span = fold_whitespace(raw)
            layer = ocr or page_non_vision(page)
            if not layer:
                continue
            folded_layer = fold_whitespace(layer)
            if span.casefold() in folded_layer.casefold():
                q = span
            else:
                snap = snap_quote(span, layer, minimum_ratio=0.9)
                q = snap.get("quote")
            if q and len(q.split()) <= 15:
                row["quotes"][page] = q
                page_hits[page] += 1
    print(f"LIST_PARSE entities={len(entities)} page_hits={page_hits}")
    merged = merge_near_duplicates(entities)
    print(f"LIST_MERGE {len(entities)} -> {len(merged)}")
    return merged


def best_fuzzy(name: str, entities: dict[str, dict], *, threshold: float = 0.78) -> tuple[str, dict] | None:
    key = norm_key(name)
    if key in entities:
        return key, entities[key]
    best = None
    best_ratio = 0.0
    for ek, row in entities.items():
        ratio = SequenceMatcher(None, key, ek).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best = (ek, row)
    if best and best_ratio >= threshold:
        return best
    # variant substitutions then retry
    altered = key
    for bad, good in VARIANT_MAP.items():
        if bad in altered:
            altered = altered.replace(bad, good)
    if altered != key and altered in entities:
        return altered, entities[altered]
    for ek, row in entities.items():
        ratio = SequenceMatcher(None, altered, ek).ratio()
        if ratio >= threshold and ratio > best_ratio:
            best_ratio = ratio
            best = (ek, row)
    return best if best and best_ratio >= threshold else None


def activity_sentence(name: str, text: str) -> str | None:
    blob = fold_whitespace(text or "")
    if not blob:
        return None
    name_cf = fold_whitespace(name).casefold()
    for match in re.finditer(r"[^.!?]{25,220}", blob):
        sent = fold_whitespace(match.group(0)).strip(" :–—-")
        if name_cf in sent.casefold() and sent.casefold() != name_cf and 25 <= len(sent) <= 200:
            return sent
    if re.search(r"principal business|transmission of electricity|consultancy|telecom", blob, re.I):
        m = re.search(
            r"([^.!?]{0,40}(?:principal business|transmission of electricity|consultancy|telecom)[^.!?]{0,120})",
            blob,
            re.I,
        )
        if m:
            return fold_whitespace(m.group(1))[:200]
    return None


def header_from_overview() -> tuple[str, str, list[int]]:
    pages = [4, 5, 6, 7, 8, 9, 10, 17, 31, 134]
    one = ""
    one_page = pages[0]
    # Prefer a short grounded company identity quote
    for p in pages:
        layer = ocr_raw(p, include_vision=False) or page_non_vision(p) or page_all(p)
        if not layer:
            continue
        for cand in (
            "Power Grid Corporation of India Limited",
            "POWERGRID",
            "transmission of electricity",
        ):
            snap = snap_quote(cand, layer, minimum_ratio=0.9)
            if snap.get("quote"):
                one = snap["quote"]
                one_page = p
                break
            if cand.casefold() in fold_whitespace(layer).casefold():
                one = cand
                one_page = p
                break
        if one:
            break
    sents = []
    for p in pages:
        layer = ocr_raw(p, include_vision=False) or page_non_vision(p) or page_all(p)
        if not layer:
            continue
        for match in re.finditer(
            r"([^.!?\n]{40,180}\b(?:transmission|interstate|POWERGRID|Power Grid)[^.!?\n]{0,80}\.)",
            layer,
            re.I,
        ):
            sent = fold_whitespace(match.group(1))
            if re.search(r"\b(Cost Auditor|appointed by the Board|Agenda|Notice)\b", sent, re.I):
                continue
            snap = snap_quote(sent, layer, minimum_ratio=0.9)
            if snap.get("quote"):
                sents.append(snap["quote"])
            elif sent.casefold() in fold_whitespace(layer).casefold():
                sents.append(sent)
            if len(sents) >= 3:
                break
        if len(sents) >= 3:
            break
    summary = " ".join(sents) if sents else one
    used = sorted({one_page, *pages[:3]})
    return one or "Power Grid Corporation of India Limited", summary or one, used


def parse_segments() -> list[dict]:
    """Segment labels from OCR segment note; no invented share_pct / revenue."""
    pages = [338, 337, 299]
    rows = []
    for label in ("Transmission", "Telecom", "Consultancy"):
        found_page = None
        quote = None
        for p in pages:
            layer = ocr_raw(p, include_vision=False) or page_non_vision(p) or page_all(p)
            if not layer:
                continue
            snap = snap_quote(label, layer, minimum_ratio=0.9)
            if snap.get("quote"):
                found_page = p
                quote = snap["quote"]
                break
            if label.casefold() in fold_whitespace(layer).casefold():
                found_page = p
                quote = label
                break
        if not found_page or not quote:
            continue
        rows.append(
            {
                "name": label,
                "revenue": None,
                "unit": None,
                "share_pct": None,  # no unambiguous printed total without inventing
            }
        )
    print(f"SEGMENTS found={len(rows)} pages_used=labels_on_338 (share_pct/revenue null)")
    return rows


def quote_validity_ocr(items: list[dict]) -> dict:
    total = ok = 0
    failing = []
    for item in items:
        for src in item.get("sources") or []:
            total += 1
            page = int(src["pdf_page"])
            quote = src.get("quote") or ""
            # measure vs OCR raw (and non-vision page text), not vision-only
            layer = ocr_raw(page, include_vision=False) or page_non_vision(page)
            if not quote:
                failing.append({"name": item["name"], "page": page, "reason": "empty"})
                continue
            snap = snap_quote(quote, layer, minimum_ratio=0.9)
            if snap.get("quote") and float(snap.get("ratio") or 0) >= 0.9:
                ok += 1
            elif fold_whitespace(quote).casefold() in fold_whitespace(layer).casefold():
                ok += 1
            else:
                failing.append({"name": item["name"], "page": page, "quote": quote[:120], "reason": "not_on_ocr"})
    pct = round(100.0 * ok / total, 2) if total else 0.0
    return {"total": total, "ok": ok, "pct": pct, "failing": failing}


def run() -> dict:
    WORK.mkdir(parents=True, exist_ok=True)
    src = V3 if V3.is_file() else (SUBMISSION_DIR / TICKER / f"{TICKER}.json")
    sheet = json.loads(src.read_text(encoding="utf-8"))
    print(f"0 baseline_items={len(sheet['items'])} source={src.name}")

    entities = parse_list_entities()
    for key, row in sorted(entities.items(), key=lambda kv: kv[1]["name"])[:40]:
        print(f"  LIST {row['name']} pages={row['pages'][:4]}".encode("ascii", "replace").decode())

    # Rebuild subsidiaries from list with OCR-verified name quotes
    sub_items = []
    skipped_no_quote = []
    for key, row in entities.items():
        name = row["name"]
        # Prefer a page with an OCR-verified quote
        quote = None
        page = None
        if row.get("quotes"):
            page, quote = next(iter(row["quotes"].items()))
        else:
            # try each page
            for p in row["pages"]:
                layer = ocr_raw(p, include_vision=False) or page_non_vision(p)
                span = fold_whitespace(name)
                snap = snap_quote(span, layer, minimum_ratio=0.9)
                if snap.get("quote") and len(snap["quote"].split()) <= 15:
                    quote = snap["quote"]
                    page = p
                    break
                if span.casefold() in fold_whitespace(layer).casefold() and len(span.split()) <= 15:
                    quote = span
                    page = p
                    break
        if not quote or not page:
            skipped_no_quote.append(name)
            continue
        sub_items.append(
            {
                "name": name,
                "kind": "subsidiary",
                "descriptor": "",
                "entity": name,
                "status": "current",
                "revenue_share_pct": None,
                "sources": [{"pdf_page": int(page), "printed_page": None, "quote": fold_whitespace(quote)[:300]}],
            }
        )
    print(f"1 subsidiaries_built={len(sub_items)} skipped_no_ocr_quote={len(skipped_no_quote)}")

    # Business lines from segment/COOM pages (337 description + 338 labels)
    biz_items = []
    for label, nice in (
        ("Transmission", "Transmission"),
        ("Telecom", "Telecom"),
        ("Consultancy", "Consultancy"),
    ):
        page_hit = None
        quote = None
        desc = ""
        for p in BUSINESS_PAGES:
            layer = ocr_raw(p, include_vision=False) or page_non_vision(p) or page_all(p)
            if not layer:
                continue
            snap = snap_quote(label, layer, minimum_ratio=0.9)
            if snap.get("quote"):
                page_hit = p
                quote = snap["quote"]
                desc = activity_sentence(label, layer) or ""
                break
            if label.casefold() in fold_whitespace(layer).casefold():
                page_hit = p
                quote = label
                desc = activity_sentence(label, layer) or ""
                break
        if not page_hit or not quote:
            continue
        # Prefer empty descriptor over OCR-garbled activity fragments
        if desc:
            if (
                desc.casefold() == label.casefold()
                or len(desc) < 40
                or re.search(r"\d{2,}", desc)
                or "Depreci" in desc
                or "information:" in desc.casefold()
            ):
                desc = ""
            else:
                layer = ocr_raw(page_hit, include_vision=False) or page_non_vision(page_hit) or page_all(page_hit)
                if not snap_quote(desc, layer, minimum_ratio=0.9).get("quote"):
                    if desc.casefold() not in fold_whitespace(layer).casefold():
                        desc = ""
        biz_items.append(
            {
                "name": nice,
                "kind": "business_line",
                "descriptor": desc[:200],
                "entity": "company",
                "status": "current",
                "revenue_share_pct": None,
                "sources": [{"pdf_page": page_hit, "printed_page": None, "quote": fold_whitespace(quote)[:300]}],
            }
        )
    # dedup business names
    biz_dedup = {}
    for item in biz_items:
        biz_dedup[assignment_key(item["name"])] = item
    biz_items = list(biz_dedup.values())
    print(f"2 business_lines_from_337={len(biz_items)}")
    for item in biz_items:
        print(f"  BIZ {item['name']} desc={item['descriptor'][:80]!r}".encode("ascii", "replace").decode())

    # Drop AGM 1-24 items and bond names from any residual carryover (we rebuild clean)
    dropped_agm_bonds = sorted(DROP_NAMES)
    print(f"3 drop_bonds_agm_names={dropped_agm_bonds}")

    one, summary, header_pages = header_from_overview()
    print(f"4 header pages={header_pages} one={one[:120]!r}".encode("ascii", "replace").decode())
    segments = parse_segments()

    items = sub_items + biz_items
    # Ensure every subsidiary matches list (by construction) and no empty quotes
    items = [i for i in items if i.get("sources") and (i["sources"][0].get("quote") or "").strip()]
    # null descriptors except business lines from 337
    for item in items:
        if item["kind"] == "subsidiary":
            item["descriptor"] = ""
        elif not item.get("descriptor"):
            item["descriptor"] = ""

    # Dedup
    deduped = []
    seen = {}
    for item in items:
        key = assignment_key(item["name"])
        if key in seen:
            continue
        seen[key] = len(deduped)
        deduped.append(item)

    sheet_out = {
        "ticker": TICKER,
        "period": sheet.get("period") or "FY25",
        "one_liner": one[:300],
        "summary": (summary or one)[:900],
        "segments": segments,
        "items": sorted(deduped, key=lambda i: (min(s["pdf_page"] for s in i["sources"]), i["name"])),
    }
    validated = FactSheet.model_validate(sheet_out).model_dump(mode="json")
    markdown = render_markdown(validated)
    errors = consistency_errors(validated, markdown)

    built = {"sheet": validated, "markdown": markdown, "omitted": []}
    write_fact_sheet(TICKER, built, STAGING_DIR / "factsheet", force=True)
    (WORK / f"{TICKER}.json").write_text(json.dumps(validated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (WORK / f"{TICKER}.md").write_text(markdown, encoding="utf-8")

    qv = quote_validity_ocr(validated["items"])
    subs = [i for i in validated["items"] if i["kind"] == "subsidiary"]
    all_subs_on_list = True
    off = []
    for item in subs:
        if not best_fuzzy(item["name"], entities, threshold=0.78):
            all_subs_on_list = False
            off.append(item["name"])
    empty_quotes = sum(1 for i in validated["items"] for s in i["sources"] if not (s.get("quote") or "").strip())

    gates = {
        "every_subsidiary_on_list": all_subs_on_list and not off,
        "quote_validity_ocr_pct": qv["pct"],
        "empty_quotes": empty_quotes,
        "schema_ok": not errors,
        "item_count": len(validated["items"]),
        "shortfall_vs_190": max(0, 190 - len(validated["items"])),
        "all_pass": all_subs_on_list and not off and empty_quotes == 0 and not errors and qv["pct"] >= 90.0,
    }
    # Stage2 gates: every subsidiary matches list, quote validity vs OCR, 0 empty quotes, item count reported
    print(f"5 items={len(validated['items'])} subs={len(subs)} qv_ocr={qv['pct']}% empty={empty_quotes} schema_err={errors}")

    report = {
        "source": str(src),
        "list_entities": len(entities),
        "skipped_no_ocr_quote": skipped_no_quote[:40],
        "dropped_bonds_agm": dropped_agm_bonds,
        "business_lines": [i["name"] for i in biz_items],
        "header_pages": header_pages,
        "segments": segments,
        "by_kind": dict(Counter(i["kind"] for i in validated["items"])),
        "quote_validity_ocr": qv,
        "subs_off_list": off,
        "gates": gates,
        "submission_untouched": True,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n=== STAGE 2 GATE METRICS (stop; wait for go) ===")
    for k, v in gates.items():
        print(f"  {k}: {v}")
    print(f"BY_KIND {report['by_kind']}")
    print("WROTE", REPORT)
    return report


if __name__ == "__main__":
    raise SystemExit(0 if run().get("gates", {}).get("all_pass") else 1)
