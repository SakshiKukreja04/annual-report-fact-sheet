"""Generic post-validate filters. Prompt text stays unchanged."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from src.validate.quotes import normalize_cosmetic, snap_quote

EVENT_PREFIXES = (
    "Incorporation of ",
    "Commissioning of ",
    "Inauguration of ",
    "Launch of ",
    "Signing of ",
    "Award of ",
)
_EVENT_RE = re.compile(
    r"^(Incorporation|Commissioning|Inauguration|Launch|Signing|Award)\s+of\s+",
    re.I,
)
_GOVERNANCE = re.compile(
    r"\b(vigilance|secretarial|audit committee|board of directors|grievance|"
    r"nomination|remuneration committee|company secretary|compliance officer|"
    r"internal audit|whistle\s*blower)\b",
    re.I,
)
_ACTIVITY = re.compile(
    r"\b(produces?|operates?|manufactures?|mines?|supplies?|sells?|owns?|"
    r"develops?|commissions?|constructs?|generates?|transmits?|engaged in|"
    r"subsidiary|joint venture|washery|plant|mine|capacity)\b",
    re.I,
)


def apply_postfilters(
    validated: list[dict],
    review: list[dict],
    rejected: list[dict],
    *,
    page_texts: dict[int, str] | None = None,
) -> dict:
    """Rewrite outcomes with generic cleanup rules. Returns new buckets + counts."""
    page_texts = page_texts or {}
    counts: Counter[str] = Counter()
    out_validated: list[dict] = []
    out_review: list[dict] = []
    out_rejected: list[dict] = list(rejected)

    for row in validated + review:
        item = _apply_one(row, page_texts, counts)
        outcome = item["outcome"]
        if outcome == "validated":
            out_validated.append(item)
        elif outcome == "review":
            out_review.append(item)
        else:
            out_rejected.append(item)

    return {
        "validated": out_validated,
        "review": out_review,
        "rejected": out_rejected,
        "validated_count": len(out_validated),
        "review_count": len(out_review),
        "rejected_count": len(out_rejected),
        "rule_counts": dict(counts),
    }


def _apply_one(row: dict, page_texts: dict[int, str], counts: Counter) -> dict:
    result = {
        "candidate": dict(row.get("candidate") or {}),
        "outcome": row.get("outcome") or "validated",
        "rejection_reasons": list(row.get("rejection_reasons") or []),
        "review_reasons": list(row.get("review_reasons") or []),
        "quotes": list(row.get("quotes") or []),
        "financial": row.get("financial"),
        "coverage": row.get("coverage"),
        "postfilter_rules": [],
    }
    candidate = result["candidate"]
    name = normalize_cosmetic((candidate.get("name") or "").strip())
    descriptor = normalize_cosmetic((candidate.get("descriptor") or "").strip())
    entity = normalize_cosmetic((candidate.get("entity") or "").strip())
    candidate["name"] = name
    candidate["descriptor"] = descriptor
    candidate["entity"] = entity
    sources = list(candidate.get("sources") or [])
    quote = ""
    pdf_page = None
    if sources:
        quote = sources[0].get("quote") or ""
        pdf_page = sources[0].get("pdf_page")
    elif result["quotes"]:
        quote = result["quotes"][0].get("quote") or ""
        pdf_page = result["quotes"][0].get("pdf_page")

    # 1) Event-prefix names
    match = _EVENT_RE.match(name)
    if match:
        counts["event_prefix"] += 1
        stripped = _EVENT_RE.sub("", name).strip()
        candidate["name"] = stripped
        if entity.casefold().startswith(match.group(0).casefold()):
            candidate["entity"] = _EVENT_RE.sub("", entity).strip() or stripped
        if descriptor.casefold() == name.casefold():
            candidate["descriptor"] = stripped
        name = stripped
        page = page_texts.get(int(pdf_page)) if pdf_page else ""
        if page and not _name_supported(stripped, quote, page):
            result["outcome"] = "rejected"
            result["rejection_reasons"] = list(result["rejection_reasons"]) + ["event_prefix_unsupported"]
            result["postfilter_rules"].append("event_prefix_drop")
            counts["event_prefix_drop"] += 1
            return result
        result["postfilter_rules"].append("event_prefix_strip")
        counts["event_prefix_strip"] += 1

    # 2) Governance / admin units → review
    blob = f"{name} {descriptor} {quote}"
    if _GOVERNANCE.search(blob):
        counts["governance_review"] += 1
        result["outcome"] = "review"
        if "governance_unit" not in result["review_reasons"]:
            result["review_reasons"] = list(result["review_reasons"]) + ["governance_unit"]
        result["postfilter_rules"].append("governance_review")
        return result

    # 3) Name-only items
    if name and descriptor and name.casefold() == descriptor.casefold() and quote.strip().casefold() == name.casefold():
        counts["name_only"] += 1
        if _from_subsidiary_table(candidate, quote):
            result["postfilter_rules"].append("name_only_subsidiary_keep")
            counts["name_only_subsidiary_keep"] += 1
            return result
        page = page_texts.get(int(pdf_page)) if pdf_page else ""
        enriched = _enrich_name_only(name, page) if page else None
        if enriched:
            candidate["descriptor"] = enriched["descriptor"]
            if sources:
                sources[0] = {**sources[0], "quote": enriched["quote"]}
                candidate["sources"] = sources
            if result["quotes"]:
                result["quotes"][0] = {**result["quotes"][0], "quote": enriched["quote"], "accepted": True}
            result["postfilter_rules"].append("name_only_enriched")
            counts["name_only_enriched"] += 1
            return result
        result["outcome"] = "review"
        if "name_only_no_activity" not in result["review_reasons"]:
            result["review_reasons"] = list(result["review_reasons"]) + ["name_only_no_activity"]
        result["postfilter_rules"].append("name_only_review")
        counts["name_only_review"] += 1
        return result

    return result


def _from_subsidiary_table(candidate: dict, quote: str) -> bool:
    kind = (candidate.get("kind") or "").casefold()
    if kind == "subsidiary":
        return True
    batch = str(candidate.get("batch_id") or candidate.get("candidate_id") or "")
    if "rules" in batch.casefold() and "limited" in (quote or "").casefold():
        return True
    return False


def _name_supported(name: str, quote: str, page: str) -> bool:
    folded_name = name.casefold()
    if folded_name and folded_name in (quote or "").casefold():
        return True
    snap = snap_quote(name, page, minimum_ratio=0.9)
    return bool(snap.get("quote"))


def _enrich_name_only(name: str, page: str) -> dict | None:
    """Find a verbatim sentence with the name plus an activity verb/noun."""
    raw = page or ""
    pattern = re.compile(re.escape(name), re.I)
    for match in pattern.finditer(raw):
        start = match.start()
        left = max(raw.rfind(".", 0, start), raw.rfind("\n", 0, start))
        right_dot = raw.find(".", start + len(name))
        right_break = raw.find("\n", start + len(name))
        ends = [index for index in (right_dot, right_break) if index >= 0]
        right = min(ends) if ends else min(len(raw), start + 220)
        sentence = raw[left + 1 : right + (1 if right < len(raw) and raw[right:right + 1] == "." else 0)].strip()
        if not sentence or name.casefold() not in sentence.casefold():
            continue
        if not _ACTIVITY.search(sentence):
            continue
        if sentence == name:
            continue
        # Descriptor: short noun phrase after stripping the name once.
        descriptor = re.sub(re.escape(name), "", sentence, count=1, flags=re.I).strip(" ,;-")
        descriptor = " ".join(descriptor.split()[:12]) or sentence[:80]
        return {"quote": sentence[:240], "descriptor": descriptor}
    return None


def write_postfilter_report(destination: Path, payload: dict) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "postfiltered.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (destination / "postfilter_summary.json").write_text(
        json.dumps(
            {
                "validated_count": payload.get("validated_count"),
                "review_count": payload.get("review_count"),
                "rejected_count": payload.get("rejected_count"),
                "rule_counts": payload.get("rule_counts"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
