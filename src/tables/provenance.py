"""Printed page numbers, units, and footnotes around a table."""

from __future__ import annotations

import re

_UNIT = re.compile(
    r"(₹|\brs\.?\b|\binr\b|\busd\b|\$|\bcrore\b|\bmillion\b|\blakh\b|\bpercent\b|%)",
    re.IGNORECASE,
)
_FOOTNOTE = re.compile(
    r"^(note\b|notes\b|source\b|\(?\d+\)|[\*\#†‡])",
    re.IGNORECASE,
)
_PRINTED = re.compile(r"^(?:page\s+)?(\d{1,3})(?:\s+of\s+\d{1,3})?$", re.IGNORECASE)
_FOLIO_BAND = 0.12
OFFSET_MIN_COUNT = 5
OFFSET_MIN_CONFIDENCE = 70.0


def _word_xy(word: dict) -> tuple[float, float]:
    """PDF words use x0/top. OCR words use bbox [x, y, w, h]."""
    if any(key in word for key in ("x0", "top", "y0")):
        return float(word.get("x0", word.get("x", 0)) or 0), float(word.get("top", word.get("y0", 0)) or 0)
    bbox = word.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return float(bbox[0]), float(bbox[1])
    return 0.0, 0.0


def _word_confidence(word: dict) -> float | None:
    if "confidence" not in word or word.get("confidence") is None:
        return None
    return float(word["confidence"])


def printed_page(
    words: list[dict],
    page_height: float,
    page_width: float | None = None,
    *,
    allow_pair: bool = False,
) -> str | None:
    """A margin folio. A pair is returned only for a real spread of consecutive numbers."""
    detail = printed_page_detail(words, page_height, page_width=page_width, allow_pair=allow_pair)
    if not detail:
        return None
    return detail["printed_page"]


def printed_page_detail(
    words: list[dict],
    page_height: float,
    page_width: float | None = None,
    *,
    allow_pair: bool = False,
) -> dict | None:
    """Best folio in the top or bottom 12% band.

    Outer corners outrank center-margin digits. A pair is the best-scoring
    consecutive pair, and only when allow_pair is set (a full spread, not one
    half and not a FULL page). Two numbers that are not consecutive stay a
    single folio — the higher-scoring one — and are never joined.
    """
    if page_height <= 0:
        return None
    band = page_height * _FOLIO_BAND
    tight_bottom = page_height * 0.045
    width = float(page_width) if page_width and page_width > 0 else None
    candidates = []
    for word in words:
        text = str(word.get("text") or "").strip()
        match = _PRINTED.match(text)
        if not match:
            continue
        raw = match.group(1)
        x0, top = _word_xy(word)
        in_top = top <= band
        in_bottom = top >= page_height - band
        if not (in_top or in_bottom):
            continue
        bottom_score = 0.0
        if in_bottom:
            bottom_score = min(1.0, (top - (page_height - band)) / max(band, 1.0))
            if top >= page_height - tight_bottom:
                bottom_score += 1.0
        edge_score = 0.0
        if width:
            if x0 <= width * 0.12 or x0 >= width * 0.88:
                edge_score = 1.0
            elif x0 <= width * 0.25 or x0 >= width * 0.75:
                edge_score = 0.4
        if len(raw) > 1 and raw.startswith("0") and edge_score < 1.0:
            continue
        confidence = _word_confidence(word)
        conf_score = 1.0 if confidence is None else max(0.0, confidence) / 100.0
        candidates.append(
            {
                "top": top,
                "x0": x0,
                "number": raw,
                "value": int(raw),
                "in_bottom": in_bottom,
                "confidence": confidence,
                "score": bottom_score + edge_score + (0.2 if in_bottom else 0.0) + conf_score,
            }
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item["score"], -item["top"] if item["in_bottom"] else item["top"], item["x0"]))
    pair = _best_consecutive_pair(candidates, width) if allow_pair else None
    if pair is not None:
        left, right = pair
        confidences = [item["confidence"] for item in (left, right) if item["confidence"] is not None]
        return {
            "printed_page": f"{left['value']}/{right['value']}",
            "confidence": min(confidences) if confidences else None,
            "score": left["score"] + right["score"],
            "value": left["value"],
            "pair": True,
        }
    best = candidates[0]
    return {
        "printed_page": best["number"],
        "confidence": best["confidence"],
        "score": best["score"],
        "value": best["value"],
        "pair": False,
    }


def _best_consecutive_pair(candidates: list[dict], width: float | None) -> tuple[dict, dict] | None:
    """Highest-scoring (n, n+1) whose smaller number sits to the left."""
    by_value: dict[int, dict] = {}
    for item in candidates:
        current = by_value.get(item["value"])
        if current is None or item["score"] > current["score"]:
            by_value[item["value"]] = item
    best: tuple[dict, dict] | None = None
    best_score = -1.0
    for value, left in by_value.items():
        right = by_value.get(value + 1)
        if right is None:
            continue
        if width is not None and left["x0"] > right["x0"]:
            continue
        score = left["score"] + right["score"]
        if score > best_score:
            best = (left, right)
            best_score = score
    return best


def combine_half_folios(choices: list[dict]) -> dict | None:
    """Join LEFT and RIGHT only when the numbers are consecutive.

    A single half, or a FULL page, never becomes a pair. Disagreeing numbers
    are left unset rather than joined.
    """
    by_half: dict[str, dict] = {}
    for item in choices:
        if not item or item.get("pair"):
            continue
        half = str(item.get("half") or "")
        if half not in {"LEFT", "RIGHT", "FULL"}:
            continue
        current = by_half.get(half)
        if current is None or float(item.get("score") or 0) > float(current.get("score") or 0):
            by_half[half] = item
    left = by_half.get("LEFT")
    right = by_half.get("RIGHT")
    if left and right:
        if int(right["value"]) != int(left["value"]) + 1:
            return None
        confidences = [item.get("confidence") for item in (left, right) if item.get("confidence") is not None]
        return {
            "printed_page": f"{int(left['value'])}/{int(right['value'])}",
            "confidence": min(confidences) if confidences else None,
            "score": float(left.get("score") or 0) + float(right.get("score") or 0),
            "value": int(left["value"]),
            "pair": True,
            "half": "PAIR",
        }
    if left or right:
        only = left or right
        return {**only, "pair": False, "half": only["half"]}
    full = by_half.get("FULL")
    if full:
        return {**full, "pair": False, "half": "FULL"}
    return None


def spread_folio_inference(pages: list[dict]) -> dict:
    """Fill blank spreads when five confident OCR folios share one offset.

    offset = left printed number - 2 * pdf_page. Pages that already have a
    folio are anchors only; they are not rewritten.
    """
    groups: dict[int, list[dict]] = {}
    for page in pages:
        anchor = _spread_anchor(page)
        if anchor is None:
            continue
        groups.setdefault(anchor["offset"], []).append(anchor)
    if not groups:
        return {"offset": None, "anchors": [], "filled": {}}
    offset, anchors = max(groups.items(), key=lambda item: (len(item[1]), -abs(item[0])))
    if len(anchors) < OFFSET_MIN_COUNT:
        return {"offset": None, "anchors": anchors, "filled": {}, "rejected_offset": offset, "rejected_count": len(anchors)}
    filled = {}
    for page in pages:
        if page.get("printed_page") not in (None, ""):
            continue
        if "TWO_PAGE_SPREAD" not in (page.get("page_type") or []):
            continue
        left = 2 * int(page["pdf_page"]) + offset
        if left <= 0:
            continue
        filled[int(page["pdf_page"])] = f"{left}/{left + 1}"
    return {"offset": offset, "anchors": anchors, "filled": filled}


def _spread_anchor(page: dict) -> dict | None:
    meta = page.get("quality_metadata") or {}
    if meta.get("folio_source") != "ocr":
        return None
    confidence = meta.get("folio_confidence")
    if confidence is None or float(confidence) < OFFSET_MIN_CONFIDENCE:
        return None
    if "TWO_PAGE_SPREAD" not in (page.get("page_type") or []):
        return None
    printed = str(page.get("printed_page") or "")
    parts = [part for part in printed.split("/") if part.strip().isdigit()]
    half = meta.get("folio_half")
    if len(parts) == 2:
        left, right = int(parts[0]), int(parts[1])
        if right != left + 1:
            return None
    elif len(parts) == 1 and half == "LEFT":
        left = int(parts[0])
    elif len(parts) == 1 and half == "RIGHT":
        left = int(parts[0]) - 1
        if left <= 0:
            return None
    else:
        return None
    return {
        "pdf_page": int(page["pdf_page"]),
        "printed_page": printed,
        "confidence": float(confidence),
        "offset": left - 2 * int(page["pdf_page"]),
        "left": left,
    }


def _lines_in_band(words: list[dict], y_min: float, y_max: float) -> list[str]:
    chosen = [word for word in words if y_min <= float(word.get("top", 0)) < y_max]
    chosen.sort(key=lambda word: (round(float(word.get("top", 0)), 0), float(word.get("x0", 0))))
    lines: list[list[str]] = []
    current_top = None
    for word in chosen:
        top = round(float(word.get("top", 0)), 0)
        text = str(word.get("text") or "").strip()
        if not text:
            continue
        if current_top is None or abs(top - current_top) > 3 or not lines:
            lines.append([text])
            current_top = top
        else:
            lines[-1].append(text)
    return [" ".join(parts) for parts in lines if parts]


def units_above(words: list[dict], bbox: tuple[float, float, float, float]) -> str | None:
    """Unit text sitting just above the table. Nothing is calculated from it."""
    _x0, top, _x1, _bottom = bbox
    lines = _lines_in_band(words, top - 48, top + 2)
    matches = [line for line in lines if _UNIT.search(line)]
    if not matches:
        return None
    return matches[-1][:240]


def footnotes_below(words: list[dict], bbox: tuple[float, float, float, float]) -> list[str]:
    _x0, _top, _x1, bottom = bbox
    lines = _lines_in_band(words, bottom - 1, bottom + 56)
    notes = [line for line in lines if _FOOTNOTE.search(line.strip())]
    return notes[:8]
