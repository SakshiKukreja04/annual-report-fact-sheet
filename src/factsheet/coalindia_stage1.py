"""Stage 1 COALINDIA offline cleanup. Writes staging/factsheet/COALINDIA only. No Groq."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from src.config.paths import INPUT_DIR, PAGE_DIR, STAGING_DIR, SUBMISSION_DIR
from src.extract.select import page_text
from src.factsheet.build import consistency_errors, render_markdown, write_fact_sheet
from src.factsheet.names import assignment_key
from src.factsheet.schema import FactSheet
from src.validate.quotes import fold_whitespace, normalize_with_map, read_pdf_page, snap_quote

TICKER = "COALINDIA"
STAGING = STAGING_DIR / "factsheet" / TICKER
REPORT = STAGING / "stage1_report.json"
WORK = STAGING_DIR / "factsheet" / "COALINDIA_stage1_work"

DROP_NAMES = {
    "Corporate Social Responsibility Expenses",
    "Evacuation facilitating Charges",
    "Loading and additional transportation charges",
    "Revenue from Other Services",
    "Subsidy for Sand Stowing & Protective Works",
    "Subsidy for Sand Stowing",
    "Exploration and Evaluation Assets",
    "Stripping activity asset",
    "Import and other sale",
    "Safety Committees",
    "Coal Mines Pension Scheme (CMPS)",
    "Defined Contribution Superannuation Pension Scheme (DCSPS)",
    "Vigilance Division",
    "Information Security Audit",
    "Outbound Training program for Women Executives",
    "Ropan Abhiyaan",
}
DROP_NAME_KEYS = {assignment_key(n) for n in DROP_NAMES}
DROP_NAME_KEYS |= {assignment_key("Subsidy for Sand Stowing & Protective Works")}

PLACEHOLDER_RX = re.compile(
    r"(?:—|--|-)\s*(?:reported activity|mining project|support scheme|plant project)\s*$",
    re.I,
)
# Accept plain folios (58), lettered (58A), and spread labels (58/59).
_PRINTED_OK = re.compile(r"^\d+[A-Za-z]?(?:/\d+[A-Za-z]?)?$")
_WORD = re.compile(r"\S+")


def _pdf() -> Path:
    return sorted((INPUT_DIR / TICKER).glob("*.pdf"))[0]


_PDF_CACHE: dict[int, str] = {}


def pdf_layer(page: int) -> str:
    if page not in _PDF_CACHE:
        try:
            _PDF_CACHE[page] = read_pdf_page(_pdf(), page)[0] or ""
        except Exception:
            _PDF_CACHE[page] = ""
    return _PDF_CACHE[page]


def printed_for(page: int, quote: str = "") -> str | None:
    """Resolve printed folio from page JSON; prefer LEFT/RIGHT half when spread."""
    path = PAGE_DIR / TICKER / f"p{page:04d}.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    value = record.get("printed_page")
    text = str(value).strip() if value not in (None, "") else ""
    if not text or not _PRINTED_OK.match(text):
        return None
    parts = [part.strip() for part in text.split("/") if part.strip()]
    if len(parts) < 2:
        return text
    half = None
    quote_cf = fold_whitespace(quote or "").casefold()
    for block in record.get("text_blocks") or []:
        block_text = fold_whitespace(block.get("text") or "")
        if quote_cf and quote_cf in block_text.casefold():
            half = block.get("half") or "FULL"
            break
    if half == "LEFT":
        return parts[0]
    if half == "RIGHT":
        return parts[1]
    # Default to left half for spreads (matches prior sheet style) when quote half unknown
    return parts[0]


def shortest_name_span(name: str, page_text_value: str, *, max_words: int = 15) -> str | None:
    """Shortest verbatim span (<=max_words, word-boundary) containing name, in one contiguous region."""
    raw = page_text_value or ""
    if not raw or not name:
        return None
    # Prefer exact casefold find on folded whitespace lines/blocks
    blocks = re.split(r"\n{2,}|\r\n{2,}", raw)
    if len(blocks) == 1:
        blocks = raw.splitlines() or [raw]
    name_cf = fold_whitespace(name).casefold()
    candidates: list[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        folded = fold_whitespace(block)
        if name_cf not in folded.casefold():
            # try snap for hyphenation
            snap = snap_quote(name, block, minimum_ratio=0.9)
            if not snap.get("quote"):
                continue
            # expand around snapped name inside this block
            block_for_search = block
            anchor = snap["quote"]
        else:
            block_for_search = folded
            anchor = None
            # locate raw slice via normalize map
            snap = snap_quote(name, block, minimum_ratio=0.9)
            if snap.get("quote"):
                anchor = snap["quote"]
            else:
                continue
        # Word tokens of the block (whitespace split keeps alignment loose but page-verbatim enough)
        words = list(_WORD.finditer(fold_whitespace(block)))
        if not words:
            continue
        # Find a window of <=15 words containing the name tokens
        name_tokens = fold_whitespace(anchor or name).casefold().split()
        if not name_tokens:
            continue
        token_texts = [fold_whitespace(m.group(0)).casefold() for m in words]
        # find contiguous match of name_tokens inside token_texts (fuzzy start)
        starts = []
        first = name_tokens[0]
        for i, tok in enumerate(token_texts):
            if first in tok or tok in first:
                starts.append(i)
        for i in starts or range(len(token_texts)):
            for length in range(len(name_tokens), max_words + 1):
                j = i + length
                if j > len(words):
                    break
                window = fold_whitespace(" ".join(m.group(0) for m in words[i:j]))
                if name_cf not in window.casefold() and fold_whitespace(anchor or "").casefold() not in window.casefold():
                    continue
                # Must appear verbatim (normalized) on the page
                if snap_quote(window, page_text_value, minimum_ratio=0.9).get("quote"):
                    exact = snap_quote(window, page_text_value, minimum_ratio=0.9)["quote"]
                    if exact and len(exact.split()) <= max_words:
                        candidates.append(exact)
                break
    if not candidates:
        # fallback: snap name alone if present
        snap = snap_quote(name, page_text_value, minimum_ratio=0.9)
        if snap.get("quote") and len(snap["quote"].split()) <= max_words:
            return snap["quote"]
        return None
    return min(candidates, key=lambda s: (len(s.split()), len(s)))


def activity_sentence(name: str, page_text_value: str) -> str | None:
    text = fold_whitespace(page_text_value or "")
    if not text:
        return None
    name_cf = fold_whitespace(name).casefold()
    for match in re.finditer(r"[^.!?]{20,220}[.!?]?", text):
        sent = fold_whitespace(match.group(0)).strip(" :–—-")
        if not sent or sent.casefold() == name_cf:
            continue
        if name_cf in sent.casefold() or snap_quote(name, sent, minimum_ratio=0.9).get("quote"):
            if PLACEHOLDER_RX.search(sent):
                continue
            if 20 <= len(sent) <= 200:
                return sent
    return None


def is_placeholder(desc: str) -> bool:
    d = (desc or "").strip()
    if not d:
        return False
    if PLACEHOLDER_RX.search(d):
        return True
    if re.search(r"\b(reported activity|support scheme|plant project|mining project)\b", d, re.I) and (
        d.casefold().startswith((d.split("—")[0] if "—" in d else d).casefold()[:20])
    ):
        # name — role pattern
        if "—" in d or " - " in d or d.endswith("project") or d.endswith("activity") or d.endswith("scheme"):
            left = re.split(r"\s+[—\-]\s+", d, maxsplit=1)[0].strip()
            if left and assignment_key(left) == assignment_key(d.split("—")[0].split(" - ")[0]):
                return bool(PLACEHOLDER_RX.search(d) or re.search(r"—\s*(reported activity|mining project|support scheme|plant project)", d, re.I))
    return bool(re.search(r"—\s*(reported activity|mining project|support scheme|plant project)\s*$", d, re.I))


def quote_validity(items: list[dict]) -> dict:
    total = ok = 0
    failing = []
    for item in items:
        for src in item.get("sources") or []:
            total += 1
            page = int(src["pdf_page"])
            quote = src.get("quote") or ""
            layer = pdf_layer(page)
            snap = snap_quote(quote, layer, minimum_ratio=0.9)
            if snap.get("quote") and float(snap.get("ratio") or 0) >= 0.9:
                ok += 1
            else:
                # also accept exact folded containment
                if fold_whitespace(quote) and fold_whitespace(quote) in fold_whitespace(layer):
                    ok += 1
                else:
                    try:
                        nq, _ = normalize_with_map(quote)
                        nl, _ = normalize_with_map(layer)
                        if nq and nq in nl:
                            ok += 1
                            continue
                    except Exception:
                        pass
                    failing.append({"name": item["name"], "page": page, "quote": quote[:160]})
    pct = round(100.0 * ok / total, 2) if total else 0.0
    return {"total": total, "ok": ok, "pct": pct, "failing": failing}


def run() -> dict:
    WORK.mkdir(parents=True, exist_ok=True)
    src = SUBMISSION_DIR / TICKER / f"{TICKER}.json"
    sheet = json.loads(src.read_text(encoding="utf-8"))
    before_n = len(sheet["items"])
    print(f"0 baseline_items={before_n} from={src}")

    # List pdf-41-only biography items first
    bio_only = []
    for item in sheet["items"]:
        pages = {int(s["pdf_page"]) for s in item.get("sources") or []}
        if pages == {41}:
            bio_only.append(
                {
                    "name": item["name"],
                    "kind": item["kind"],
                    "descriptor": item.get("descriptor"),
                    "quote": (item["sources"][0].get("quote") or "")[:160],
                }
            )
    print(f"BIO_ONLY_PDF41 count={len(bio_only)}")
    for row in bio_only:
        print(f"  BIO {row['name']} | {row['quote']}".encode("ascii", "replace").decode())

    # Drop named junk + bio-only p41
    drop_keys = set(DROP_NAME_KEYS) | {assignment_key(r["name"]) for r in bio_only}
    kept = []
    dropped = []
    for item in sheet["items"]:
        key = assignment_key(item["name"])
        name = item["name"]
        if key in drop_keys or name in DROP_NAMES:
            dropped.append(name)
            continue
        # fuzzy drop for sand stowing / import sale variants
        if re.search(
            r"^(Corporate Social Responsibility Expenses|Evacuation facilitating|Loading and additional|"
            r"Revenue from Other Services|Subsidy for Sand Stowing|Exploration and Evaluation Assets|"
            r"Stripping activity asset|Import and other sale|Safety Committees|Vigilance Division|"
            r"Information Security Audit|Outbound Training|Ropan Abhiyaan|Coal Mines Pension|"
            r"Defined Contribution Superannuation)",
            name,
            re.I,
        ):
            dropped.append(name)
            continue
        kept.append(item)
    print(f"1 dropped_junk_or_bio={len(dropped)} remaining={len(kept)}")
    for name in dropped:
        print(f"  DROP {name}".encode("ascii", "replace").decode())

    # Quote snapping
    snapped_items = []
    quotes_before = sum(len(i.get("sources") or []) for i in kept)
    quotes_kept = quotes_dropped = items_dropped_no_quote = 0
    for item in kept:
        sources = []
        for src in item.get("sources") or []:
            page = int(src["pdf_page"])
            layer = pdf_layer(page)
            span = shortest_name_span(item["name"], layer, max_words=15)
            if not span:
                quotes_dropped += 1
                continue
            # verify on PDF layer
            if not snap_quote(span, layer, minimum_ratio=0.9).get("quote") and fold_whitespace(span) not in fold_whitespace(layer):
                quotes_dropped += 1
                continue
            printed = src.get("printed_page")
            page_printed = printed_for(page, span)
            printed_text = str(printed).strip() if printed not in (None, "") else ""
            if not printed_text or not _PRINTED_OK.match(printed_text):
                printed = page_printed
            else:
                # Keep a half that matches the page record; only blank true mismatches
                record_path = PAGE_DIR / TICKER / f"p{page:04d}.json"
                record_label = ""
                if record_path.is_file():
                    record_label = str(json.loads(record_path.read_text(encoding="utf-8")).get("printed_page") or "")
                parts = [p.strip() for p in record_label.split("/") if p.strip()]
                if record_label and printed_text != record_label and printed_text not in parts:
                    printed = page_printed
                else:
                    printed = printed_text
            sources.append({"pdf_page": page, "printed_page": printed, "quote": fold_whitespace(span)[:300]})
            quotes_kept += 1
        if not sources:
            items_dropped_no_quote += 1
            continue
        snapped_items.append({**item, "sources": sources[:3]})
    print(
        f"2 quote_snap before_quotes={quotes_before} kept_quotes={quotes_kept} "
        f"dropped_quotes={quotes_dropped} items_dropped_no_quote={items_dropped_no_quote} items={len(snapped_items)}"
    )

    # Descriptor placeholders → page sentence or null
    placeholders_fixed = 0
    for item in snapped_items:
        desc = (item.get("descriptor") or "").strip()
        if is_placeholder(desc) or PLACEHOLDER_RX.search(desc) or re.search(
            r"—\s*(reported activity|mining project|support scheme|plant project)\s*$", desc, re.I
        ):
            page = int(item["sources"][0]["pdf_page"])
            sent = activity_sentence(item["name"], pdf_layer(page))
            item["descriptor"] = sent or ""
            placeholders_fixed += 1
        elif desc and desc.casefold() == item["name"].casefold():
            # leave name-only as empty rather than inventing
            page = int(item["sources"][0]["pdf_page"])
            sent = activity_sentence(item["name"], pdf_layer(page))
            item["descriptor"] = sent or ""
    print(f"3 placeholders_rewritten_or_nulled={placeholders_fixed}")

    # Shares on p209: keep FSA 79, E-Auction 16; null ambiguous washed/other
    print("4 page209 share policy: keep FSA=79 E-Auction=16; null Other By Product / Washed Coal (Coking|Non Coking) unless unambiguous")
    print("  (pie chart is image-only on p209; text does not print an unambiguous 1%/2%/2% mapping — nulling those three)")
    for item in snapped_items:
        name = item["name"]
        if name == "FSA":
            item["revenue_share_pct"] = 79.0
        elif name == "E-Auction":
            item["revenue_share_pct"] = 16.0
        elif name in {"Other By Product", "Washed Coal (Coking)", "Washed Coal (Non Coking)"}:
            item["revenue_share_pct"] = None

    # Segments: document states no separate reportable segments
    segments = []
    segment_note = {
        "pages": [249, 292],
        "finding": "Segment Reporting notes state there are no separate reportable segments for the company/group.",
        "segments_added": 0,
    }
    print(f"5 segments: none added ({segment_note['finding']})")

    # Dedup by normalized name
    deduped = []
    seen = {}
    for item in snapped_items:
        key = assignment_key(item["name"])
        if key in seen:
            prior = deduped[seen[key]]
            if len(item.get("descriptor") or "") > len(prior.get("descriptor") or ""):
                prior["descriptor"] = item.get("descriptor")
            continue
        seen[key] = len(deduped)
        deduped.append(item)

    # Trim 'other' to <=5: drop admin/HR/ESG/policy-shaped others first
    other_drop_rx = re.compile(
        r"\b(pension|training|vigilance|committee|audit|policy|swachh|award|scholarship|"
        r"MoU with IIT|Howrah Bridge|sports promotion|engagement|CSR|safai|"
        r"first mile connectivity|defined contribution|coal india sports)\b",
        re.I,
    )
    others = [i for i in deduped if i.get("kind") == "other"]
    non_others = [i for i in deduped if i.get("kind") != "other"]
    kept_others = []
    dropped_others = []
    for item in others:
        blob = f"{item.get('name')} {item.get('descriptor')}"
        if other_drop_rx.search(blob or ""):
            dropped_others.append(item["name"])
            continue
        kept_others.append(item)
    # If still >5, keep the 5 with longest grounded quotes
    if len(kept_others) > 5:
        kept_others = sorted(kept_others, key=lambda i: len(i["sources"][0]["quote"]), reverse=True)
        dropped_others.extend(i["name"] for i in kept_others[5:])
        kept_others = kept_others[:5]
    deduped = non_others + kept_others
    print(f"3b other_trimmed kept={len(kept_others)} dropped={len(dropped_others)}")

    # Count remaining placeholders / other
    placeholder_left = [
        i["name"]
        for i in deduped
        if is_placeholder(i.get("descriptor") or "")
        or re.search(r"—\s*(reported activity|mining project|support scheme|plant project)\s*$", i.get("descriptor") or "", re.I)
    ]
    other_count = sum(1 for i in deduped if i.get("kind") == "other")

    sheet_out = {
        "ticker": TICKER,
        "period": sheet.get("period") or "FY25",
        "one_liner": sheet.get("one_liner") or "",
        "summary": sheet.get("summary") or "",
        "segments": segments,
        "items": sorted(deduped, key=lambda i: (min(s["pdf_page"] for s in i["sources"]), i["name"])),
    }
    validated = FactSheet.model_validate(sheet_out).model_dump(mode="json")
    markdown = render_markdown(validated)
    errors = consistency_errors(validated, markdown)
    print(f"6 schema_errors={errors} items={len(validated['items'])} other={other_count} placeholders_left={len(placeholder_left)}")

    # Write staging only
    built = {"sheet": validated, "markdown": markdown, "omitted": []}
    write_fact_sheet(TICKER, built, STAGING_DIR / "factsheet", force=True)
    (WORK / f"{TICKER}.json").write_text(json.dumps(validated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (WORK / f"{TICKER}.md").write_text(markdown, encoding="utf-8")

    qv = quote_validity(validated["items"])
    print(f"7 quote_validity_normalized_vs_pdf={qv['pct']}% ({qv['ok']}/{qv['total']}) failing={len(qv['failing'])}")
    for row in qv["failing"][:20]:
        print(f"  FAILQ {row['name']} p{row['page']}: {row['quote']}".encode("ascii", "replace").decode())

    by_kind = dict(Counter(i["kind"] for i in validated["items"]))
    gates = {
        "quote_validity_pct": qv["pct"],
        "quote_validity_pass": qv["pct"] >= 99.0,
        "placeholder_descriptors": len(placeholder_left),
        "placeholder_pass": len(placeholder_left) == 0,
        "other_count": other_count,
        "other_pass": other_count <= 5,
        "schema_ok": not errors,
        "dup_groups": 0,
    }
    # dup check
    keys = [assignment_key(i["name"]) for i in validated["items"]]
    gates["dup_groups"] = len(keys) - len(set(keys))
    gates["dup_pass"] = gates["dup_groups"] == 0
    gates["all_pass"] = all(
        [
            gates["quote_validity_pass"],
            gates["placeholder_pass"],
            gates["other_pass"],
            gates["schema_ok"],
            gates["dup_pass"],
        ]
    )

    report = {
        "baseline_items": before_n,
        "bio_only_pdf41": bio_only,
        "dropped": dropped,
        "quote_snap": {
            "before_quotes": quotes_before,
            "kept_quotes": quotes_kept,
            "dropped_quotes": quotes_dropped,
            "items_dropped_no_quote": items_dropped_no_quote,
        },
        "placeholders_fixed": placeholders_fixed,
        "placeholders_left": placeholder_left,
        "segment_note": segment_note,
        "shares": {"FSA": 79.0, "E-Auction": 16.0, "Other By Product": None, "Washed Coal (Coking)": None, "Washed Coal (Non Coking)": None},
        "final_items": len(validated["items"]),
        "by_kind": by_kind,
        "quote_validity": qv,
        "gates": gates,
        "consistency_errors": errors,
        "submission_untouched": True,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n=== STAGE 1 GATE METRICS (stop; wait for go) ===")
    for k, v in gates.items():
        print(f"  {k}: {v}")
    print(f"BY_KIND {by_kind}")
    print(f"FINAL_ITEMS {len(validated['items'])}")
    print("WROTE", REPORT)
    return report


if __name__ == "__main__":
    raise SystemExit(0 if run().get("gates", {}).get("all_pass") else 1)
