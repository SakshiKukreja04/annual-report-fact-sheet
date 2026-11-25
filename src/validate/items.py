"""Accept, reject, or hold a candidate. A quote on a page does not prove the item."""

from __future__ import annotations

import re

from src.dedup.normalize import content_tokens, entity_key, normalize
from src.extract.schema import KINDS, STATUSES
from src.validate.financial import check_printed_share
from src.validate.policy import apply_policy
from src.validate.quotes import normalize_cosmetic, validate_quote

_COMPANY_PHRASE = re.compile(r"\bthe company\b|\bour company\b", re.I)


def validate_candidate(candidate: dict, context: dict) -> dict:
    candidate = dict(candidate)
    # Cosmetic hyphen/quote normalization on name/descriptor/entity (not prompt).
    for field in ("name", "descriptor", "entity"):
        if candidate.get(field):
            candidate[field] = normalize_cosmetic(str(candidate[field]))
    candidate, policy_reason = apply_policy(candidate)
    reasons = []
    if policy_reason:
        reasons.append(policy_reason)
    review = []
    name = (candidate.get("name") or "").strip()
    if not name:
        reasons.append("empty_name")
    if candidate.get("kind") not in KINDS:
        reasons.append("invalid_kind")
    if candidate.get("status") not in STATUSES:
        reasons.append("invalid_status")
    sources = list(candidate.get("sources") or [])
    if len(sources) > 3:
        reasons.append("too_many_quotes")
    if not sources:
        reasons.append("no_quote")
    quote_results = [validate_quote(source, context) for source in sources[:3] if "too_many_quotes" not in reasons]
    if "too_many_quotes" in reasons:
        quote_results = [validate_quote(source, context) for source in sources]
    accepted = [result for result in quote_results if result["accepted"]]
    if sources and not accepted and "too_many_quotes" not in reasons:
        reasons.append("no_valid_quote")
    kept_sources = []
    for source, result in zip(sources, quote_results):
        if result["accepted"]:
            row = dict(source)
            # Replace stored quote with the exact page substring after a normalized match.
            if result.get("quote"):
                row["quote"] = result["quote"]
            kept_sources.append(row)
    entity_ok = _entity_supported(candidate.get("entity") or "", accepted, context)
    descriptor_ok = _descriptor_supported(candidate.get("descriptor") or "", accepted)
    if accepted and not entity_ok:
        review.append("unsupported_entity")
    if accepted and not descriptor_ok:
        review.append("weak_semantic_evidence")
    financial = check_printed_share(
        candidate.get("revenue_share_pct"),
        [result["quote"] for result in accepted],
    )
    if financial["needs_review"]:
        review.append(financial["status"])
    if reasons:
        outcome = "rejected"
    elif review:
        outcome = "review"
    else:
        outcome = "validated"
    checked = {
        key: value
        for key, value in candidate.items()
        if key != "sources"
    }
    checked["sources"] = kept_sources
    checked["revenue_share_pct"] = financial["revenue_share_pct"] if outcome != "rejected" else candidate.get("revenue_share_pct")
    return {
        "candidate": checked,
        "outcome": outcome,
        "rejection_reasons": reasons,
        "review_reasons": review,
        "quotes": quote_results,
        "financial": financial,
        "coverage": {
            "name": candidate.get("name"),
            "candidate_id": candidate.get("candidate_id"),
            "source_count": len(sources),
            "valid_quotes": len(accepted),
            "rejected_quotes": len(quote_results) - len(accepted),
            "pages": [result["pdf_page"] for result in quote_results],
            "match_statuses": [result["match"] for result in quote_results],
            "verifications": [result["verification"] for result in quote_results],
            "outcome": outcome,
        },
    }


def _entity_supported(entity: str, accepted: list[dict], context: dict) -> bool:
    if not accepted:
        return False
    ticker = context.get("ticker") or ""
    if entity_key(entity, ticker) == "company":
        for result in accepted:
            text = result["quote"]
            page = (context.get("native_text") or {}).get(result["pdf_page"], "")
            ocr = (context.get("ocr_text") or {}).get(result["pdf_page"], "")
            blob = f"{text} {page} {ocr}"
            folded = normalize(blob)
            if "company" in folded.split() or (ticker and normalize(ticker) in folded):
                return True
            # Annual reports usually say "the Company" rather than the ticker code.
            if _COMPANY_PHRASE.search(blob):
                return True
            # Letterhead / footer often prints an issuer "* Limited" without the word company.
            head = (page or ocr or "")[:800]
            if re.search(r"\b[A-Za-z][A-Za-z0-9&.'’\- ]{1,60}\bLimited\b", head):
                return True
        return False
    wanted = normalize(entity)
    if not wanted:
        return False
    for result in accepted:
        page = (context.get("native_text") or {}).get(result["pdf_page"], "")
        ocr = (context.get("ocr_text") or {}).get(result["pdf_page"], "")
        if wanted in normalize(f"{result['quote']} {page} {ocr}"):
            return True
    return False


def _descriptor_supported(descriptor: str, accepted: list[dict]) -> bool:
    tokens = content_tokens(descriptor)
    if not tokens or not accepted:
        return False
    evidence = content_tokens(" ".join(result["quote"] for result in accepted))
    return len(tokens & evidence) / len(tokens) >= 0.5
