"""Find a quote on the cited page. snap_quote may return a page substring. It does not edit the file."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path


def fold_whitespace(text: str) -> str:
    """Search copy only. Line breaks and repeated spaces become one space."""
    cleaned = (text or "").replace("\u00a0", " ")
    cleaned = re.sub(r"-\s*\n", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


_LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
}
_QUOTES = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
    "\u00ab": '"',
    "\u00bb": '"',
}
_DASHES = {
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u2212": "-",
}
_STRIP_CHARS = {
    "\u00ad",  # soft hyphen
    "\u200b",  # zero-width space
    "\u200c",
    "\u200d",
    "\ufeff",
    "\u2060",
}
_ELLIPSIS_SPLIT = re.compile(r"\.\.\.|…")


def normalize_cosmetic(text: str) -> str:
    """Normalize hyphen/quote variants for display fields (names, descriptors)."""
    if not text:
        return text
    raw = unicodedata.normalize("NFKC", text)
    out = []
    for character in raw:
        if character in _STRIP_CHARS:
            continue
        if character in _DASHES:
            out.append("-")
        elif character in _QUOTES:
            out.append(_QUOTES[character])
        elif character in _LIGATURES:
            out.append(_LIGATURES[character])
        else:
            out.append(character)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Search copy plus the original index of each kept character."""
    output = []
    mapping = []
    index = 0
    raw = unicodedata.normalize("NFKC", text or "")
    while index < len(raw):
        character = raw[index]
        if character in _STRIP_CHARS:
            index += 1
            continue
        if character in "-\u2010\u2011\u2012\u2013\u2014\u2015\u2212" and index + 1 < len(raw) and raw[index + 1] in "\n\r":
            index += 2
            if index < len(raw) and raw[index] in "\n\r":
                index += 1
            continue
        replacement = _LIGATURES.get(character) or _QUOTES.get(character) or _DASHES.get(character)
        if replacement is not None:
            for piece in replacement:
                output.append(piece.lower())
                mapping.append(index)
            index += 1
            continue
        if character.isspace() or character == "\u00a0":
            if output and output[-1] != " ":
                output.append(" ")
                mapping.append(index)
            index += 1
            continue
        output.append(character.lower())
        mapping.append(index)
        index += 1
    if output and output[-1] == " ":
        output.pop()
        mapping.pop()
    return "".join(output), mapping


def snap_quote(quote: str, page_text: str, *, minimum_ratio: float = 0.9) -> dict:
    """Return the exact page substring when the normalized window matches closely enough."""
    norm_quote, _quote_map = normalize_with_map(quote)
    norm_page, page_map = normalize_with_map(page_text)
    if not norm_quote or not norm_page:
        return {"quote": None, "ratio": 0.0, "reason": "empty"}
    found = norm_page.find(norm_quote)
    if found >= 0:
        return {
            "quote": _slice_from_raw(page_text, page_map, found, found + len(norm_quote)),
            "ratio": 1.0,
            "reason": "normalized_window",
        }
    window = len(norm_quote)
    best_ratio = 0.0
    best_start = 0
    step = max(1, window // 10)
    limit = max(1, len(norm_page) - max(window // 2, 1))
    anchors = sorted((word for word in norm_quote.split() if len(word) >= 5), key=len, reverse=True)[:2]
    starts = []
    for anchor in anchors:
        offset = norm_quote.find(anchor)
        position = 0
        while len(starts) < 30:
            found = norm_page.find(anchor, position)
            if found < 0:
                break
            starts.append(min(limit - 1, max(0, found - offset)))
            position = found + len(anchor)
    if not starts:
        starts = list(range(0, limit, max(step, 20)))
    for start in starts:
        end = min(len(norm_page), start + window)
        ratio = SequenceMatcher(None, norm_quote, norm_page[start:end], autojunk=False).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = start
    for start in range(max(0, best_start - step), min(limit, best_start + step + 1)):
        end = min(len(norm_page), start + window)
        ratio = SequenceMatcher(None, norm_quote, norm_page[start:end]).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = start
    if best_ratio < minimum_ratio:
        return {"quote": None, "ratio": round(best_ratio, 3), "reason": "below_0.9"}
    snapped = _slice_from_raw(page_text, page_map, best_start, min(len(page_map), best_start + window))
    if not snapped:
        return {"quote": None, "ratio": round(best_ratio, 3), "reason": "below_0.9"}
    return {"quote": snapped, "ratio": round(best_ratio, 3), "reason": "fuzzy_window"}


def _slice_from_raw(page_text: str, mapping: list[int], start: int, end: int) -> str:
    """Slice the NFKC page text that produced `mapping`."""
    raw = unicodedata.normalize("NFKC", page_text or "")
    if start >= len(mapping) or end <= start:
        return ""
    last = mapping[min(end, len(mapping)) - 1]
    return raw[mapping[start] : last + 1]


def _slice(page_text: str, mapping: list[int], start: int, end: int) -> str:
    return _slice_from_raw(page_text, mapping, start, end)


_NAME_STOP = {
    "limited", "ltd", "pvt", "private", "company", "the", "and", "for", "with", "from", "into", "over",
}


def name_tokens(name: str) -> list[str]:
    """Distinctive words from an item name. Suffixes such as Limited are not anchors."""
    words = [word for word in re.findall(r"[A-Za-z0-9]{3,}", name or "") if word.casefold() not in _NAME_STOP]
    long = [word for word in words if len(word) >= 5]
    return long or words[:3]


def repair_named_quote(quote: str, page_text: str, name: str, *, minimum_ratio: float = 0.9) -> dict:
    """Snap a quote, then keep it only when the item name is in that quote or its sentence."""
    tokens = name_tokens(name)
    snap = snap_quote(quote, page_text, minimum_ratio=minimum_ratio)
    snapped = snap.get("quote")
    if snapped and float(snap.get("ratio") or 0) >= minimum_ratio and _mentions(snapped, tokens, page_text):
        return {
            "quote": snapped,
            "ratio": snap["ratio"],
            "reason": snap["reason"],
            "outcome": "repaired",
            "mentions_name": _mentions_text(snapped, tokens),
        }
    anchored = _name_window(page_text, tokens)
    if anchored:
        return {
            "quote": anchored,
            "ratio": snap.get("ratio") or 0.0,
            "reason": "name_anchor",
            "outcome": "reanchored",
            "mentions_name": True,
        }
    return {
        "quote": None,
        "ratio": snap.get("ratio") or 0.0,
        "reason": "name_missing" if snapped else (snap.get("reason") or "below_0.9"),
        "outcome": "dropped",
        "mentions_name": False,
    }


def _mentions_text(text: str, tokens: list[str]) -> bool:
    if not tokens:
        return True
    folded = (text or "").casefold()
    return all(token.casefold() in folded for token in tokens)


def _mentions(quote: str, tokens: list[str], page_text: str) -> bool:
    if _mentions_text(quote, tokens):
        return True
    sentence = _sentence_containing(page_text, quote)
    return _mentions_text(sentence, tokens)


def _sentence_containing(page_text: str, quote: str) -> str:
    raw = unicodedata.normalize("NFKC", page_text or "")
    if not quote or quote not in raw:
        return ""
    start = raw.find(quote)
    left = max(raw.rfind(".", 0, start), raw.rfind("\n", 0, start))
    right_dot = raw.find(".", start + len(quote))
    right_break = raw.find("\n", start + len(quote))
    ends = [index for index in (right_dot, right_break) if index >= 0]
    right = min(ends) if ends else len(raw)
    return raw[left + 1 : right + 1]


def _name_window(page_text: str, tokens: list[str]) -> str | None:
    if not tokens:
        return None
    raw = unicodedata.normalize("NFKC", page_text or "")
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", raw):
        if not _mentions_text(sentence, tokens):
            continue
        words = list(re.finditer(r"\S+", sentence))
        if not words:
            continue
        folded = [match.group(0).casefold() for match in words]
        cover = _earliest_cover(folded, tokens)
        if cover is None:
            continue
        start, end = cover
        window = sentence[words[start].start() : words[end - 1].end()]
        if window in raw and _mentions_text(window, tokens):
            return window
        if len(words) <= 15 and sentence.strip() in raw:
            return sentence.strip()
    return None


def _earliest_cover(words: list[str], tokens: list[str]) -> tuple[int, int] | None:
    """Shortest run of at most 15 words that contains every name token."""
    needles = [token.casefold() for token in tokens]
    best = None
    for start in range(len(words)):
        needed = set(needles)
        limit = min(len(words), start + 15)
        for end in range(start, limit):
            for needle in list(needed):
                if needle in words[end]:
                    needed.discard(needle)
            if not needed:
                span = (start, end + 1)
                if best is None or span[1] - span[0] < best[1] - best[0]:
                    best = span
                break
    return best


def locate_quote(quote: str, page_text: str) -> str | None:
    if not quote or not page_text:
        return None
    if quote in page_text:
        return "exact"
    if fold_whitespace(quote) in fold_whitespace(page_text):
        return "normalized"
    snap = snap_quote(quote, page_text, minimum_ratio=1.0)
    if snap.get("quote") and float(snap.get("ratio") or 0) >= 1.0:
        return "normalized"
    return None


def _ellipsis_fragments(quote: str) -> list[str]:
    parts = [part.strip() for part in _ELLIPSIS_SPLIT.split(quote or "") if part and part.strip()]
    return parts


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text or ""))


def resolve_quote_on_page(quote: str, page_text: str) -> dict:
    """Match and snap a quote. Ellipsis quotes are never accepted whole."""
    raw_quote = quote or ""
    if _ELLIPSIS_SPLIT.search(raw_quote):
        fragments = _ellipsis_fragments(raw_quote)
        # Prefer the longest fragment with at least 8 words.
        ranked = sorted(fragments, key=lambda part: (_word_count(part), len(part)), reverse=True)
        for fragment in ranked:
            if _word_count(fragment) < 8:
                continue
            snap = snap_quote(fragment, page_text, minimum_ratio=0.9)
            if snap.get("quote") and float(snap.get("ratio") or 0) >= 0.9:
                return {
                    "accepted": True,
                    "match": "ellipsis_fragment",
                    "quote": snap["quote"],
                    "reason": None,
                    "fragment": fragment,
                }
        return {
            "accepted": False,
            "match": None,
            "quote": raw_quote,
            "reason": "ellipsis_unsupported",
            "fragment": None,
        }
    if raw_quote in (page_text or ""):
        return {"accepted": True, "match": "exact_match", "quote": raw_quote, "reason": None, "fragment": None}
    snap = snap_quote(raw_quote, page_text, minimum_ratio=0.9)
    if snap.get("quote") and float(snap.get("ratio") or 0) >= 0.9:
        match = "normalized_match" if float(snap["ratio"]) >= 1.0 or snap.get("reason") == "normalized_window" else "snapped_match"
        if snap.get("reason") == "normalized_window":
            match = "normalized_match"
        return {
            "accepted": True,
            "match": match,
            "quote": snap["quote"],
            "reason": None,
            "fragment": None,
        }
    folded_q = fold_whitespace(raw_quote)
    folded_p = fold_whitespace(page_text or "")
    if folded_q and folded_q in folded_p:
        # Fall back to snap at lower ratio to recover exact bytes.
        snap = snap_quote(raw_quote, page_text, minimum_ratio=0.8)
        if snap.get("quote"):
            return {
                "accepted": True,
                "match": "normalized_match",
                "quote": snap["quote"],
                "reason": None,
                "fragment": None,
            }
    return {
        "accepted": False,
        "match": None,
        "quote": raw_quote,
        "reason": "quote_not_on_page",
        "fragment": None,
    }


def validate_quote(source: dict, context: dict) -> dict:
    """Check one source. On a normalized hit, `quote` becomes the exact page substring."""
    quote = source.get("quote") or ""
    page = source.get("pdf_page")
    saved = {
        "quote": quote,
        "pdf_page": page,
        "printed_page": source.get("printed_page"),
        "accepted": False,
        "match": None,
        "verification": None,
        "reason": None,
        "page_image": None,
    }
    page_count = int(context.get("page_count") or 0)
    if not isinstance(page, int) or isinstance(page, bool) or page < 1 or (page_count and page > page_count):
        saved["reason"] = "invalid_page"
        return saved
    ocr_only = page in set(context.get("ocr_only_pages") or [])
    if ocr_only:
        text = (context.get("ocr_text") or {}).get(page, "")
        saved["verification"] = "ocr"
        saved["page_image"] = (context.get("page_images") or {}).get(page)
        resolved = resolve_quote_on_page(quote, text)
        if resolved["accepted"]:
            saved.update(accepted=True, match=f"ocr_verified_{resolved['match']}".replace("ocr_verified_exact_match", "ocr_verified_exact").replace("ocr_verified_normalized_match", "ocr_verified_normalized"), quote=resolved["quote"])
            if saved["match"] not in {"ocr_verified_exact", "ocr_verified_normalized"}:
                saved["match"] = "ocr_verified_normalized" if "normalized" in (resolved["match"] or "") or "snapped" in (resolved["match"] or "") or "ellipsis" in (resolved["match"] or "") else "ocr_verified_exact"
        else:
            saved["reason"] = resolved["reason"] or "quote_not_in_ocr"
            if resolved["reason"] == "ellipsis_unsupported":
                saved["reason"] = "ellipsis_unsupported"
        return saved
    text = (context.get("native_text") or {}).get(page, "")
    saved["verification"] = "native_pdf"
    resolved = resolve_quote_on_page(quote, text)
    if resolved["accepted"]:
        saved.update(accepted=True, match=resolved["match"], quote=resolved["quote"])
    else:
        saved["reason"] = resolved["reason"] or "quote_not_on_page"
    return saved


def read_pdf_page(pdf_path: Path, pdf_page: int) -> tuple[str, str]:
    """Read one page from the original PDF. PyMuPDF is preferred when it can load."""
    try:
        import pymupdf

        with pymupdf.open(pdf_path) as document:
            if pdf_page < 1 or pdf_page > document.page_count:
                raise ValueError(f"{pdf_path.name} has no PDF page {pdf_page}")
            return document[pdf_page - 1].get_text("text") or "", "pymupdf"
    except Exception as pymupdf_error:
        if "has no PDF page" in str(pymupdf_error):
            raise
        import pdfplumber

        with pdfplumber.open(pdf_path) as document:
            if pdf_page < 1 or pdf_page > len(document.pages):
                raise ValueError(f"{pdf_path.name} has no PDF page {pdf_page}") from pymupdf_error
            return document.pages[pdf_page - 1].extract_text() or "", "pdfplumber"


def pdf_page_count(pdf_path: Path) -> tuple[int, str]:
    try:
        import pymupdf

        with pymupdf.open(pdf_path) as document:
            return document.page_count, "pymupdf"
    except Exception:
        import pdfplumber

        with pdfplumber.open(pdf_path) as document:
            return len(document.pages), "pdfplumber"
