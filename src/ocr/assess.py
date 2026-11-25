"""Quality gate, recoverability, and character error rate.

Low-confidence words stay in the raw OCR record. They are not evidence.
"""

from __future__ import annotations

import re

from src.ocr.settings import MIN_CONTIGUOUS_WORDS, MIN_RUN_CONFIDENCE, WORD_CONFIDENCE_THRESHOLD

RECOVERABLE = "RECOVERABLE"
ILLEGIBLE = "ILLEGIBLE"
IMAGE_ONLY_VISUAL = "IMAGE_ONLY_VISUAL"
ILLEGIBLE_REASON = "illegible_low_dpi"


def character_error_rate(reference: str, hypothesis: str) -> float | None:
    """CER = (substitutions + deletions + insertions) / reference length, in percent."""
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    if not ref:
        return None
    return round(100.0 * _edit_distance(ref, hyp) / len(ref), 2)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _edit_distance(left: str, right: str) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def _token_is_usable(word: dict, threshold: float) -> bool:
    text = (word.get("text") or "").strip()
    letters = sum(character.isalpha() for character in text)
    return letters >= 2 and float(word.get("confidence", -1)) >= threshold


def kept_word_runs(
    words: list[dict],
    threshold: float = WORD_CONFIDENCE_THRESHOLD,
    minimum: int = MIN_CONTIGUOUS_WORDS,
    min_run_confidence: float = MIN_RUN_CONFIDENCE,
) -> list[dict]:
    """Keep runs of adjacent same-line words. A single confident word is not a quote."""
    runs: list[dict] = []
    current: list[dict] = []
    current_line = None
    for word in words:
        line_id = word.get("line_id")
        usable = _token_is_usable(word, threshold)
        if current and line_id != current_line:
            _store_run(runs, current, minimum, min_run_confidence)
            current = []
        if not usable:
            _store_run(runs, current, minimum, min_run_confidence)
            current = []
            current_line = line_id
            continue
        current.append(word)
        current_line = line_id
    _store_run(runs, current, minimum, min_run_confidence)
    return runs


def _store_run(runs: list[dict], words: list[dict], minimum: int, min_run_confidence: float) -> None:
    if len(words) < minimum:
        return
    confidences = [float(word["confidence"]) for word in words]
    if sum(confidences) / len(confidences) < min_run_confidence:
        return
    runs.append(
        {
            "text": " ".join(word["text"].strip() for word in words),
            "word_count": len(words),
            "line_confidence": round(sum(confidences) / len(confidences), 2),
            "line_id": words[0].get("line_id"),
            "bbox": _union_bbox(words),
        }
    )


def _union_bbox(words: list[dict]) -> list[int]:
    boxes = [word["bbox"] for word in words if word.get("bbox")]
    if not boxes:
        return [0, 0, 0, 0]
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[0] + box[2] for box in boxes)
    y1 = max(box[1] + box[3] for box in boxes)
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


def cleaned_text(runs: list[dict]) -> str:
    return "\n".join(run["text"] for run in runs)


def meaningful_captions(runs: list[dict]) -> list[dict]:
    """Keep a caption when it is a real word sequence, not binarization noise."""
    kept = []
    for run in runs:
        compact = run["text"].replace(" ", "")
        letters = sum(character.isalpha() for character in compact)
        if letters >= 12 and letters / max(len(compact), 1) >= 0.7:
            kept.append(run)
    return kept


def is_document_like(gray) -> bool:
    """A text page has a large near-white background. A photograph usually does not."""
    import cv2

    small = cv2.resize(gray, (160, 220), interpolation=cv2.INTER_AREA)
    return float((small > 220).mean()) >= 0.35


def recoverability(classifications: list[str], document_like: bool, runs: list[dict]) -> dict:
    """Choose RECOVERABLE, ILLEGIBLE, or IMAGE_ONLY_VISUAL. Every page gets a status."""
    labels = set(classifications)
    visual_page = "IMAGE_ONLY_VISUAL" in labels or not document_like
    if visual_page:
        captions = meaningful_captions(runs)
        return {
            "recoverability": IMAGE_ONLY_VISUAL,
            "failure_reason": None,
            "pass_to_extraction": False,
            "use_ocr_text": False,
            "caption_text": cleaned_text(captions),
        }
    if runs:
        return {
            "recoverability": RECOVERABLE,
            "failure_reason": None,
            "pass_to_extraction": True,
            "use_ocr_text": True,
            "caption_text": "",
        }
    has_text_layer = bool(labels & {"DIGITAL_TEXT", "MIXED_TEXT_IMAGE", "GARBLED_OR_LOW_QUALITY_TEXT"})
    if has_text_layer:
        # A readable digital page is not a low-DPI failure when OCR keeps no line.
        return {
            "recoverability": RECOVERABLE,
            "failure_reason": None,
            "pass_to_extraction": True,
            "use_ocr_text": False,
            "caption_text": "",
        }
    return {
        "recoverability": ILLEGIBLE,
        "failure_reason": ILLEGIBLE_REASON,
        "pass_to_extraction": False,
        "use_ocr_text": False,
        "caption_text": "",
    }
