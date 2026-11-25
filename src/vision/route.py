"""Choose the few pages that may go to vision. Illegible pages never qualify."""

from __future__ import annotations

import math

COMPLEX_TABLE = "complex_table_text_unreliable"
DIAGRAM_LABELS = "readable_diagram_labels"
LAYOUT_LOST = "layout_relationships_lost"
OCR_AMBIGUOUS = "ocr_text_structurally_ambiguous"
OCR_WEAK = "ocr_text_weak_or_garbled"

ILLEGIBLE = "illegible_low_dpi"
NO_TEXT = "no_meaningful_text"
LOCALLY_ADEQUATE = "locally_adequate"
NOT_ASSESSED = "not_assessed"
CAP_REACHED = "vision_cap_reached"

# Selected OCR shorter than this (or garbled) may use Qwen as a hint only.
_WEAK_OCR_CHARS = 120


def vision_cap(page_count: int, max_percent: float) -> int:
    """Whole pages allowed by the percentage. A fraction below one page allows none."""
    if page_count <= 0 or max_percent <= 0:
        return 0
    return math.floor(page_count * max_percent / 100)


def route_page(page: dict, *, table_statuses: list[str] | None = None, diagram_labels: bool = False) -> dict:
    """Return one routing decision. Eligible decisions always include a reason."""
    warnings = set(page.get("warnings") or [])
    quality = page.get("quality_metadata") or {}
    recoverability = quality.get("ocr_recoverability") or {}
    status = page.get("content_status")
    statuses = set(table_statuses or [])
    if (
        status == "illegible"
        or "illegible_low_dpi" in warnings
        or any(value == "ILLEGIBLE" for value in recoverability.values())
    ):
        return _decision(False, ILLEGIBLE)
    if status == "pending_ocr":
        return _decision(False, NOT_ASSESSED)
    captions = [caption for caption in (page.get("captions") or []) if (caption.get("text") or "").strip()]
    selected = quality.get("selected_text_source")
    if status == "image_only_visual" and not captions and not diagram_labels:
        return _decision(False, NO_TEXT)
    if diagram_labels and captions:
        return _decision(True, DIAGRAM_LABELS)
    if statuses.intersection({"STRUCTURE_ONLY", "PARTIAL", "MANUAL_REVIEW"}) or "scanned_table_text_is_unverified" in warnings:
        return _decision(True, COMPLEX_TABLE)
    if quality.get("layout_relationships_lost"):
        return _decision(True, LAYOUT_LOST)
    if selected == "ocr" and quality.get("structurally_ambiguous"):
        return _decision(True, OCR_AMBIGUOUS)
    selected_blocks = [
        block for block in page.get("text_blocks") or []
        if block.get("selected") and (block.get("text") or "").strip()
    ]
    selected_chars = sum(len(block.get("text") or "") for block in selected_blocks)
    selected_sources = {str(block.get("source") or "") for block in selected_blocks}
    ocr_selected = selected == "ocr" or bool(selected_sources & {"tesseract", "paddleocr", "ocr", "ocr_mistral"})
    # Vision is a hint only: never quote-grounding. Use when OCR is short/garbled.
    if ocr_selected and (quality.get("garbled_text_layer") or selected_chars < _WEAK_OCR_CHARS):
        return _decision(True, OCR_WEAK)
    if quality.get("mistral_weak") and selected_chars < _WEAK_OCR_CHARS:
        return _decision(True, OCR_WEAK)
    return _decision(False, LOCALLY_ADEQUATE)


def select_pages(
    pages: list[dict],
    *,
    table_statuses: dict[int, list[str]] | None = None,
    diagram_pages: set[int] | None = None,
    max_percent: float = 2.0,
    limit: int | None = None,
) -> list[dict]:
    """Keep eligible pages inside the percentage cap and the pilot limit."""
    allowed = vision_cap(len(pages), max_percent)
    if limit is not None:
        allowed = min(allowed, limit)
    chosen = []
    decisions = []
    statuses = table_statuses or {}
    diagrams = diagram_pages or set()
    for page in pages:
        decision = route_page(
            page,
            table_statuses=statuses.get(page["pdf_page"]),
            diagram_labels=page["pdf_page"] in diagrams,
        )
        if decision["eligible"] and len(chosen) >= allowed:
            decision = _decision(False, CAP_REACHED)
        if decision["eligible"]:
            chosen.append(page)
        decisions.append({**decision, "pdf_page": page["pdf_page"], "ticker": page.get("ticker")})
    return decisions


def _decision(eligible: bool, reason: str) -> dict:
    if eligible and not reason:
        raise ValueError("An eligible vision page needs an explicit routing reason.")
    return {"eligible": eligible, "routing_reason": reason}
