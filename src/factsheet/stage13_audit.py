"""Stage 13: audit HINDALCO.continued.json against cached page text and the PDF.

continued.json is never overwritten. Outputs are written beside it.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

from src.config.paths import INPUT_DIR, OUTPUT_DIR, PAGE_DIR
from src.dedup.normalize import content_tokens, entity_key, normalize
from src.dedup.resolve import compare, resolve
from src.extract.rules import extract_listed
from src.extract.evidence import pages_for_extraction
from src.extract.run import load_saved_company
from src.extract.select import page_text, select_pages
from src.factsheet.build import consistency_errors, render_markdown
from src.factsheet.schema import FactSheet
from src.ingestion.audit import locate_pdfs
from src.validate.policy import apply_policy, correct_kind, correct_status
from src.validate.quotes import locate_quote, name_tokens

_FALSE_SUBS = {
    "bse limited",
    "nse limited",
    "national stock exchange",
    "mufg intime india private limited",
    "name mufg intime india private limited",
    "ceat limited",
    "nestle india limited",
    "securities and exchange board of india",
}
_FALSE_SUB_PAT = re.compile(
    r"\b(stock exchange|registrar and (?:share )?transfer|depository|"
    r"\bbank\b|reserve bank|income tax|ministry of|government of|"
    r"chartered accountant|cost accountant)\b",
    re.I,
)
_FRAGMENT = re.compile(
    r"^(quantity unit|cash generating unit|face value per unit|property plant|"
    r"property plant and equipment|unit|share capital|equity share)$",
    re.I,
)
_RELATION = re.compile(
    r"\b(wholly[- ]owned subsidiary|subsidiary(?:\s+company)?|joint venture|"
    r"associate(?:\s+company)?|consolidated subsidiary|step[- ]down subsidiary|"
    r"held subsidiary|our subsidiary)\b",
    re.I,
)
_UNRELATED = re.compile(r"\bunrelated(?:\s+part(?:y|ies))?\b", re.I)
_CUSTOMER_SUPPLIER = re.compile(r"\b(customer|supplier|vendor|lender|banker|auditor|registrar)\b", re.I)
_TRAINING = re.compile(r"\b(training|workshop|awareness programme|awareness program)\b", re.I)
_ENV = re.compile(r"\b(ghg|emissions?|water[- ]?positiv\w*|zero[- ]waste|recycling target|carbon intensit\w*)\b", re.I)
_PROCESS = re.compile(r"\b(process|procedure|methodology|framework|policy)\b", re.I)
_MW_ONLY = re.compile(r"^\d+(\.\d+)?\s*mw$", re.I)
_INDUSTRY = re.compile(r"\b(automotive|packaging|construction|electrical|aerospace)\b", re.I)
_STRATEGY = re.compile(r"\b(strategy|roadmap|vision|mission|priority|focus area)\b", re.I)
_ADMIN_EXTRA = re.compile(
    r"\b(agm|e-?\s*voting|instavote|speaker registration|book closure|dividend|"
    r"cost audit|re-?\s*appointment|shareholder|folio|registrar)\b",
    re.I,
)


def load_page_texts(ticker: str = "HINDALCO") -> dict[int, str]:
    folder = PAGE_DIR / ticker
    texts = {}
    for path in folder.glob("p*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        texts[int(record["pdf_page"])] = page_text(record)
    return texts


def listed_subsidiary_keys(pages: list[dict] | None = None) -> set[str]:
    pages = pages or pages_for_extraction(load_saved_company("HINDALCO", PAGE_DIR))
    listed = extract_listed(pages)
    keys = set()
    for item in listed.get("subsidiaries") or []:
        keys.add(entity_key(item["name"], "HINDALCO"))
        keys.add(normalize(item["name"]))
    return keys


def audit_item(item: dict, page_texts: dict[int, str], listed: set[str], index: int) -> dict:
    """Return one audit row. The original item fields are always preserved."""
    original = {
        "candidate_id": f"c{index}",
        "original_name": item.get("name"),
        "original_kind": item.get("kind"),
        "original_entity": item.get("entity"),
        "original_status": item.get("status"),
        "original_descriptor": item.get("descriptor"),
        "source_pdf_page": (item.get("sources") or [{}])[0].get("pdf_page"),
        "original_quote": (item.get("sources") or [{}])[0].get("quote"),
    }
    sources = list(item.get("sources") or [])
    valid_sources = []
    invalid_quotes = 0
    missing_pages = 0
    for source in sources:
        page = source.get("pdf_page")
        quote = source.get("quote") or ""
        if page not in page_texts:
            missing_pages += 1
            continue
        text = page_texts[page]
        if locate_quote(quote, text) is None:
            invalid_quotes += 1
            continue
        tokens = name_tokens(item.get("name") or "")
        folded = f"{quote}\n{text}".casefold()
        if tokens and not all(token.casefold() in folded for token in tokens):
            invalid_quotes += 1
            continue
        valid_sources.append(dict(source))
    if missing_pages and not valid_sources:
        return _row(original, "REJECT", reason="missing_pdf_page", supporting_page=None, supporting_quote=None)
    if not valid_sources:
        return _row(original, "REJECT", reason="invalid_or_unsupported_quote", supporting_page=original["source_pdf_page"], supporting_quote=original["original_quote"])

    revised = {
        **item,
        "sources": valid_sources[:3],
    }
    revised, policy_reason = apply_policy(revised)
    if policy_reason:
        return _row(
            original,
            "REJECT",
            reason=policy_reason,
            supporting_page=valid_sources[0]["pdf_page"],
            supporting_quote=valid_sources[0]["quote"],
        )
    blob = _blob(revised)
    if _ADMIN_EXTRA.search(blob):
        return _row(original, "REJECT", reason="administrative", supporting_page=valid_sources[0]["pdf_page"], supporting_quote=valid_sources[0]["quote"])
    if _FRAGMENT.match((revised.get("name") or "").strip()):
        return _row(original, "REJECT", reason="non_company_fragment", supporting_page=valid_sources[0]["pdf_page"], supporting_quote=valid_sources[0]["quote"])
    if _MW_ONLY.match((revised.get("name") or "").strip()):
        return _row(original, "REJECT", reason="capacity_not_asset_name", supporting_page=valid_sources[0]["pdf_page"], supporting_quote=valid_sources[0]["quote"])

    kind = revised.get("kind") or "other"
    name = revised.get("name") or ""
    if kind == "subsidiary":
        sub = _subsidiary_decision(revised, page_texts, listed)
        if sub["decision"] != "ok":
            return _row(
                original,
                sub["decision"],
                reason=sub["reason"],
                corrected_kind=sub.get("kind"),
                corrected_name=sub.get("name"),
                corrected_entity=sub.get("entity"),
                supporting_page=valid_sources[0]["pdf_page"],
                supporting_quote=valid_sources[0]["quote"],
                item=revised if sub["decision"] == "CORRECT" else None,
            )
    kind = _reclassify(revised)
    revised["kind"] = kind
    revised["status"] = correct_status(revised)
    revised["entity"] = _entity_for(revised)

    changed = (
        revised["name"] != item.get("name")
        or revised["kind"] != item.get("kind")
        or revised["entity"] != item.get("entity")
        or revised["status"] != item.get("status")
        or revised.get("descriptor") != item.get("descriptor")
        or revised["sources"] != sources
    )
    decision = "CORRECT" if changed else "KEEP"
    return _row(
        original,
        decision,
        reason="corrected_fields" if changed else "verified",
        corrected_name=revised["name"],
        corrected_kind=revised["kind"],
        corrected_entity=revised["entity"],
        corrected_status=revised["status"],
        supporting_page=revised["sources"][0]["pdf_page"],
        supporting_quote=revised["sources"][0]["quote"],
        item=revised,
    )


def _subsidiary_decision(item: dict, page_texts: dict[int, str], listed: set[str]) -> dict:
    name = item.get("name") or ""
    key = entity_key(name, "HINDALCO")
    if normalize(name) in _FALSE_SUBS or key in {entity_key(n, "HINDALCO") for n in _FALSE_SUBS}:
        return {"decision": "REJECT", "reason": "false_subsidiary"}
    if _FALSE_SUB_PAT.search(name) or _FALSE_SUB_PAT.search(item.get("descriptor") or ""):
        return {"decision": "REJECT", "reason": "false_subsidiary"}
    if key in listed or normalize(name) in listed:
        return {"decision": "ok"}
    pages = [source.get("pdf_page") for source in item.get("sources") or []]
    context = []
    for page in pages:
        for offset in (-1, 0, 1):
            text = page_texts.get(int(page) + offset, "")
            if text:
                context.append(text)
    joined = "\n".join(context)
    if name and _name_tied_to_relation(joined, name):
        return {"decision": "ok"}
    quote = " ".join(source.get("quote") or "" for source in item.get("sources") or [])
    if _name_tied_to_relation(quote, name):
        return {"decision": "ok"}
    # Geographic footprint labeled subsidiary without legal proof.
    if re.search(r"global operations|footprint", name, re.I):
        return {
            "decision": "CORRECT",
            "reason": "footprint_not_subsidiary",
            "kind": "other",
            "name": name,
            "entity": "company",
        }
    return {"decision": "REVIEW", "reason": "subsidiary_relationship_unproven"}


def _name_tied_to_relation(text: str, name: str, radius: int = 90) -> bool:
    """Relation words must attach to this name, not a different company nearby."""
    if not text or not name:
        return False
    folded = text.casefold()
    needle = name.casefold()
    start = 0
    while True:
        index = folded.find(needle, start)
        if index < 0:
            break
        window = text[max(0, index - radius) : index + len(name) + radius]
        if _UNRELATED.search(window) or _CUSTOMER_SUPPLIER.search(window):
            start = index + len(needle)
            continue
        # Prefer patterns where the relation word is immediately around the name.
        left = text[max(0, index - radius) : index]
        right = text[index + len(name) : index + len(name) + radius]
        if _RELATION.search(left) or _RELATION.search(right):
            # Reject windows that clearly attach subsidiary language to another Limited/Ltd first.
            other = re.search(
                r"([A-Z][A-Za-z0-9&.,' -]{2,80}\b(?:Limited|Ltd\.?|LLC|Inc\.?)),?\s+"
                r"(?:a |an |the )?(?:wholly[- ]owned )?subsidiary",
                window,
            )
            if other and name.casefold() not in other.group(1).casefold():
                start = index + len(needle)
                continue
            return True
        start = index + len(needle)
    return False


def _reclassify(item: dict) -> str:
    kind = correct_kind(item)
    name = item.get("name") or ""
    blob = _blob(item)
    if kind == "business_line" and _TRAINING.search(blob):
        return "other"
    if kind == "asset" and _ENV.search(blob) and not re.search(r"\b(plant|smelter|refinery|mine|facility)\b", name, re.I):
        return "other"
    if kind == "research_platform" and _PROCESS.search(blob) and not re.search(r"\b(lab|centre|center|r&d|research)\b", blob, re.I):
        return "other"
    if kind == "segment" and _INDUSTRY.search(name) and not re.search(r"\bsegment\b", name, re.I):
        return "other"
    if kind == "business_line" and _STRATEGY.search(name) and not re.search(r"\b(aluminium|copper|novelis|alumina)\b", name, re.I):
        return "other"
    return kind


def _entity_for(item: dict) -> str:
    if item.get("kind") == "subsidiary":
        return item.get("name") or item.get("entity") or "company"
    entity = item.get("entity") or "company"
    if entity_key(entity, "HINDALCO") == "company":
        return "company"
    return entity


def _blob(item: dict) -> str:
    quotes = " ".join(source.get("quote") or "" for source in item.get("sources") or [])
    return " ".join((item.get("name") or "", item.get("descriptor") or "", quotes))


def _row(
    original: dict,
    decision: str,
    *,
    reason: str,
    corrected_name=None,
    corrected_kind=None,
    corrected_entity=None,
    corrected_status=None,
    supporting_page=None,
    supporting_quote=None,
    merged_into=None,
    item=None,
) -> dict:
    return {
        **original,
        "decision": decision,
        "corrected_name": corrected_name,
        "corrected_kind": corrected_kind,
        "corrected_entity": corrected_entity,
        "corrected_status": corrected_status,
        "reason": reason,
        "supporting_pdf_page": supporting_page,
        "supporting_quote": supporting_quote,
        "merged_into": merged_into,
        "item": item,
    }


def merge_survivors(rows: list[dict]) -> list[dict]:
    """Merge KEEP/CORRECT items. MERGE rows replace the duplicates."""
    survivors = [row for row in rows if row["decision"] in {"KEEP", "CORRECT"} and row.get("item")]
    items = []
    for row in survivors:
        item = dict(row["item"])
        item["candidate_id"] = row["candidate_id"]
        items.append(item)
    result = resolve(items, "HINDALCO")
    by_id = {row["candidate_id"]: row for row in rows}
    final_items = []
    for group in result["merged"]:
        member_ids = group.get("member_ids") or []
        clean = {
            key: value
            for key, value in group.items()
            if key not in {"member_ids", "confidence", "reason", "retained_sources", "candidate_id"}
        }
        clean["sources"] = (group.get("sources") or [])[:3]
        final_items.append(clean)
        for member in member_ids[1:]:
            row = by_id.get(member)
            if not row:
                continue
            row["decision"] = "MERGE"
            row["merged_into"] = group.get("name")
            row["reason"] = "duplicate_of_retained_item"
            row["item"] = None
    for item in result["unchanged"]:
        clean = {key: value for key, value in item.items() if key != "candidate_id"}
        final_items.append(clean)
    return final_items


def _fact_item(item: dict) -> dict:
    return {
        "name": item["name"],
        "kind": item["kind"],
        "descriptor": item["descriptor"],
        "entity": item["entity"],
        "status": item["status"],
        "revenue_share_pct": item.get("revenue_share_pct"),
        "sources": [
            {
                "pdf_page": int(source["pdf_page"]),
                "printed_page": source.get("printed_page"),
                "quote": source["quote"],
            }
            for source in (item.get("sources") or [])[:3]
        ],
    }


def overview_text(page_texts: dict[int, str], pages: list[dict]) -> str:
    selection = select_pages(pages)
    chosen = []
    for row in selection.get("pages") or []:
        if row.get("section") in {"business_overview", "segment_note", "mda"} and row.get("decision") == "keep":
            number = int(row["pdf_page"])
            text = page_texts.get(number, "")
            if _looks_like_toc(text):
                continue
            chosen.append(number)
    if not chosen:
        chosen = [number for number in range(31, 58) if not _looks_like_toc(page_texts.get(number, ""))]
    parts = []
    for number in chosen[:12]:
        text = page_texts.get(number, "")
        if text:
            parts.append(text[:900])
    return "\n".join(parts)


def company_overview(source_text: str) -> tuple[str, str]:
    """Prefer a Hindalco business sentence over a cover slogan."""
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.])\s+", source_text or "") if sentence.strip()]
    ranked = []
    for sentence in sentences:
        folded = sentence.casefold()
        words = sentence.split()
        if len(words) < 10 or len(words) > 80:
            continue
        if re.search(r"\b(contents|statutory reports|financial statements|page)\b", folded):
            continue
        if "aluminium" not in folded and "aluminum" not in folded and "copper" not in folded:
            continue
        if not re.search(r"[a-z]{4,}", folded):
            continue
        score = 0
        if folded.startswith("hindalco") or "hindalco has" in folded or "hindalco is" in folded:
            score += 5
        if "aluminium" in folded and "copper" in folded:
            score += 3
        if "novelis" in folded or "rolled" in folded or "downstream" in folded:
            score += 2
        if "force for sustainable" in folded or "shopfloor" in folded or "tagline" in folded:
            score -= 5
        if len(words) < 12:
            score -= 2
        cleaned = sentence if sentence.endswith(".") else sentence + "."
        ranked.append((score, cleaned))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    if not ranked or ranked[0][0] < 1:
        return (
            "Hindalco Industries Limited produces aluminium and copper products.",
            "Hindalco operates aluminium and copper businesses, including upstream metal, "
            "downstream products, and the Novelis rolled-products business, as described in its FY25 report.",
        )
    one_liner = ranked[0][1]
    summary = " ".join(sentence for _score, sentence in ranked[:3] if _score >= 1)
    return one_liner, summary


def _looks_like_toc(text: str) -> bool:
    head = (text or "")[:500].casefold()
    if "contents" in head and head.count("\n") >= 8:
        return True
    digits = sum(character.isdigit() for character in head)
    return digits >= 40 and "aluminium" not in head and "copper" not in head


def build_segments(items: list[dict], pages: list[dict]) -> list[dict]:
    listed = extract_listed(pages)
    revenue = listed.get("revenue") or {}
    segments = []
    seen = set()
    for item in items:
        if item.get("kind") != "segment":
            continue
        key = normalize(item["name"])
        if key in seen:
            continue
        seen.add(key)
        money = revenue.get(item["name"].casefold()) or {}
        share = item.get("revenue_share_pct")
        if share is None:
            share = money.get("share_pct")
        segments.append(
            {
                "name": item["name"],
                "revenue": money.get("revenue"),
                "unit": money.get("unit"),
                "share_pct": share,
            }
        )
    for item in listed.get("segments") or []:
        key = normalize(item["name"])
        if key in seen:
            continue
        seen.add(key)
        money = revenue.get(item["name"].casefold()) or {}
        segments.append(
            {
                "name": item["name"],
                "revenue": money.get("revenue"),
                "unit": money.get("unit"),
                "share_pct": money.get("share_pct"),
            }
        )
    # Also surface revenue-only segments (e.g. COALINDIA) that have printed figures.
    for key, money in revenue.items():
        name = money.get("name") or key
        norm = normalize(name)
        if norm in seen:
            continue
        seen.add(norm)
        segments.append(
            {
                "name": name,
                "revenue": money.get("revenue"),
                "unit": money.get("unit"),
                "share_pct": money.get("share_pct"),
            }
        )
    return segments


def run_audit(
    continued_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    continued_path = continued_path or (OUTPUT_DIR / "HINDALCO" / "HINDALCO.continued.json")
    output_dir = output_dir or (OUTPUT_DIR / "HINDALCO")
    if not continued_path.is_file():
        raise FileNotFoundError(f"Missing continued sheet: {continued_path}")
    pdf = locate_pdfs(INPUT_DIR).get("HINDALCO")
    if pdf is None or not Path(pdf).is_file():
        raise FileNotFoundError("HINDALCO FY25 PDF is unavailable.")

    sheet = json.loads(continued_path.read_text(encoding="utf-8-sig"))
    originals = list(sheet.get("items") or [])
    page_texts = load_page_texts("HINDALCO")
    pages = pages_for_extraction(load_saved_company("HINDALCO", PAGE_DIR))
    listed = listed_subsidiary_keys(pages)

    rows = [audit_item(item, page_texts, listed, index) for index, item in enumerate(originals)]
    final_items = merge_survivors(rows)
    final_items = [_fact_item(item) for item in final_items if item]
    overview = overview_text(page_texts, pages)
    one_liner, summary = company_overview(overview)
    segments = build_segments(final_items, pages)
    final_sheet = {
        "ticker": "HINDALCO",
        "period": "FY25",
        "one_liner": one_liner,
        "summary": summary,
        "segments": segments,
        "items": sorted(final_items, key=lambda item: (min(s["pdf_page"] for s in item["sources"]), item["name"])),
    }
    validated = FactSheet.model_validate(final_sheet).model_dump(mode="json")
    markdown = render_markdown(validated)
    errors = consistency_errors(validated, markdown)
    if errors:
        raise RuntimeError("Final Markdown and JSON differ: " + "; ".join(errors))

    audit_rows = []
    for row in rows:
        audit_rows.append({key: value for key, value in row.items() if key != "item"})

    decisions = Counter(row["decision"] for row in audit_rows)
    reasons = Counter(row["reason"] for row in audit_rows if row["decision"] == "REJECT")
    original_kinds = Counter(row["original_kind"] for row in audit_rows)
    final_kinds = Counter(item["kind"] for item in validated["items"])
    pages_cited = Counter()
    for row in audit_rows:
        page = row.get("source_pdf_page")
        if page is not None:
            pages_cited[str(page)] += 1
    verified_quotes = sum(1 for row in audit_rows if row["decision"] in {"KEEP", "CORRECT", "MERGE"} and row.get("supporting_quote"))
    invalid_quotes = sum(1 for row in audit_rows if row["reason"] in {"invalid_or_unsupported_quote", "missing_pdf_page"})
    missing_evidence = sum(1 for row in audit_rows if row["decision"] == "REVIEW")

    if sum(decisions.values()) != len(originals):
        raise RuntimeError("Audit decisions do not reconcile with the original candidate count.")

    summary_payload = {
        "original_candidate_count": len(originals),
        "kept_count": decisions.get("KEEP", 0),
        "corrected_count": decisions.get("CORRECT", 0),
        "merged_count": decisions.get("MERGE", 0),
        "rejected_count": decisions.get("REJECT", 0),
        "review_required_count": decisions.get("REVIEW", 0),
        "final_unique_item_count": len(validated["items"]),
        "verified_quote_count": verified_quotes,
        "invalid_quote_count": invalid_quotes,
        "missing_evidence_count": missing_evidence,
        "counts_by_original_kind": dict(original_kinds),
        "counts_by_final_kind": dict(final_kinds),
        "counts_by_decision": dict(decisions),
        "counts_by_source_page": dict(sorted(pages_cited.items(), key=lambda pair: int(pair[0]))),
        "rejected_reason_distribution": dict(reasons),
        "pdf": str(pdf),
        "source_sheet": str(continued_path),
        "review_names": [row["original_name"] for row in audit_rows if row["decision"] == "REVIEW"],
        "false_subsidiary_rejects": [
            row["original_name"]
            for row in audit_rows
            if row["reason"] in {"false_subsidiary", "non_company_fragment"}
        ],
        "administrative_rejects": [row["original_name"] for row in audit_rows if row["reason"] == "administrative"],
        "json_markdown_consistent": True,
        "every_final_quote_verified": all(
            locate_quote(source["quote"], page_texts.get(source["pdf_page"], "")) is not None
            for item in validated["items"]
            for source in item["sources"]
        ),
    }

    _write_outputs(output_dir, validated, markdown, audit_rows, summary_payload)
    return summary_payload


def _write_outputs(output_dir: Path, sheet: dict, markdown: str, audit_rows: list[dict], summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "HINDALCO_final.json",
        "md": output_dir / "HINDALCO_final.md",
        "audit_json": output_dir / "HINDALCO_item_audit.json",
        "audit_csv": output_dir / "HINDALCO_item_audit.csv",
        "summary": output_dir / "HINDALCO_audit_summary.json",
    }
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"{path.name} already exists. Refusing to overwrite.")
    _atomic(paths["json"], json.dumps(sheet, indent=2, ensure_ascii=False) + "\n")
    _atomic(paths["md"], markdown if markdown.endswith("\n") else markdown + "\n")
    _atomic(paths["audit_json"], json.dumps(audit_rows, indent=2, ensure_ascii=False) + "\n")
    _atomic(paths["summary"], json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    fieldnames = [
        "candidate_id",
        "original_name",
        "original_kind",
        "original_entity",
        "original_status",
        "source_pdf_page",
        "original_quote",
        "decision",
        "corrected_name",
        "corrected_kind",
        "corrected_entity",
        "corrected_status",
        "reason",
        "supporting_pdf_page",
        "supporting_quote",
        "merged_into",
    ]
    with paths["audit_csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in audit_rows:
            writer.writerow(row)


def _atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    summary = run_audit()
    print(json.dumps({key: summary[key] for key in (
        "original_candidate_count",
        "kept_count",
        "corrected_count",
        "merged_count",
        "rejected_count",
        "review_required_count",
        "final_unique_item_count",
        "verified_quote_count",
        "invalid_quote_count",
        "every_final_quote_verified",
    )}, indent=2))


if __name__ == "__main__":
    main()
