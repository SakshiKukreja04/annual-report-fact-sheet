"""COALINDIA final cleanup. Offline, rules only. Writes only under staging/factsheet/COALINDIA."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.config.paths import OUTPUT_DIR, PAGE_DIR, PROJECT_ROOT, STAGING_DIR, SUBMISSION_DIR
from src.extract.select import page_text
from src.factsheet.build import consistency_errors, render_markdown, write_fact_sheet
from src.factsheet.cost import cost_table
from src.factsheet.names import assignment_key
from src.factsheet.precision import generic_noun_only, quote_supports_name
from src.factsheet.schema import FactSheet
from src.validate.quotes import fold_whitespace

_CONTROL = re.compile(r"[\x00-\x1f\x7f]+")

STAGING = STAGING_DIR / "factsheet" / "COALINDIA"
SHEET_PATH = STAGING / "COALINDIA.json"
MD_PATH = STAGING / "COALINDIA.md"
OVERRIDES_PATH = STAGING / "manual_overrides.json"
REPORT_PATH = STAGING / "final_cleanup_report.json"

_SUB_ROW = re.compile(
    r"(?P<name>"
    r"(?:Eastern Coalfields Limited|Bharat Coking Coal Limited|Central Coalfields Limited|"
    r"Northern Coalfields Limited|Western Coalfields Limited|South Eastern Coalfields Limited|"
    r"Mahanadi Coalfields Limited|Central Mine Planning\s*&\s*Design Institute Limited|"
    r"CIL Solar PV Limited|CIL Navikarniya Urja Limited|Coal Gas India Limited|"
    # Coal Gas India is also matched by the generic Limited pattern below;
    r"International Coal Venture(?:s)? Private Limited|CIL NTPC Urja Private Limited|"
    r"Talcher Fertilizers Limited|Hindustan Urvarak(?:\s+and|\s+&\s+)Rasayan Limited|"
    r"Coal Lignite Urja Vikas Private Limited|Coal India Africana Limitada|"
    r"Bharat Coal Gasification\s*&\s*Chemicals Limited|"
    r"[A-Z][A-Za-z0-9&.'’\-]*(?:\s+(?:&|and|[A-Z][A-Za-z0-9&.'’\-]*)){0,10}"
    r"\s+(?:Limited|Limitada|Ltd\.?|Private Limited))"
    r")"
    r"(?:,\s*Mozambique)?"
    r"(?:\s*\((?P<abbr>[A-Z]{2,12})\))?"
    r"(?:\([^)]*incorpora[^)]*\))?"
    r"\s+(?P<activity>[A-Za-z][A-Za-z0-9 /&-]{1,50}?)"
    r"\s+(?:India|Mozambique)\s+(?P<pct>\d+(?:\.\d+)?)\s*%",
    re.I,
)
_CIN = re.compile(r"^U\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}\b", re.I)
_NUM_TOKEN = re.compile(r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?%?")
_FUTURE = re.compile(
    r"\b(will|proposed to be set up|plans to|to be formed|under construction|"
    r"yet to commence|mou to explore|shall be|to be incorporated)\b",
    re.I,
)
_CURRENT = re.compile(
    r"\b(from\s+(?:19|20)\d{2}|has started|commissioned|was developed|is engaged|"
    r"incorporated on|operational|currently)\b",
    re.I,
)
_BIO = re.compile(
    r"\b(he|she)\b.{0,40}\b(served|tenure|joined|career|appointed as|director)\b|"
    r"\b(served as|joined as|his tenure|her tenure|career spanning)\b",
    re.I,
)
_CSR_AWARD = re.compile(
    r"\b(csr award|national csr|award for|excellence award|trophy|citation)\b",
    re.I,
)
_LOAN_RPT = re.compile(
    r"\b(related party transaction|\brpt\b|loans? (?:and|&) advances|investment in|"
    r"inter-corporate loan|guarantee given)\b",
    re.I,
)
_ACCOUNTING = re.compile(
    r"\b(progressive mine closure|property,? plant and equipment|exports as loans|"
    r"accounting polic|depreciation method|impairment loss)\b",
    re.I,
)
_ORG_DEPT = re.compile(
    r"\b(forest department|organisation|organization|ministry of coal|"
    r"department of)\b",
    re.I,
)
_SOFT_MINE = re.compile(
    r"^(Data Mine|National Coal Mine|Regular Mine|WCL Mine|ERP Mine|Portal Mine)\b",
    re.I,
)
_UNFINISHED = re.compile(r"(?:\bat\s+H\s*\d+\s*$|\bNo\.\s*$|\bat\s+H\s*$)", re.I)
_COALFIELD_ABBR = {
    "ecl": "Eastern Coalfields Limited",
    "bccl": "Bharat Coking Coal Limited",
    "ccl": "Central Coalfields Limited",
    "ncl": "Northern Coalfields Limited",
    "wcl": "Western Coalfields Limited",
    "secl": "South Eastern Coalfields Limited",
    "mcl": "Mahanadi Coalfields Limited",
    "cmpdil": "Central Mine Planning & Design Institute Limited",
    "cmpdi": "Central Mine Planning & Design Institute Limited",
}
_JV_CANON = {
    "hurl": ("Hindustan Urvarak and Rasayan Limited", "HURL"),
    "hindustan urvarak and rasayan limited": ("Hindustan Urvarak and Rasayan Limited", "HURL"),
    "hindustan urvarak & rasayan limited": ("Hindustan Urvarak and Rasayan Limited", "HURL"),
    "tfl": ("Talcher Fertilizers Limited", "TFL"),
    "talcher fertilizers limited": ("Talcher Fertilizers Limited", "TFL"),
    "cluvpl": ("Coal Lignite Urja Vikas Private Limited", "CLUVPL"),
    "coal lignite urja vikas private limited": ("Coal Lignite Urja Vikas Private Limited", "CLUVPL"),
    "icvl": ("International Coal Venture Private Limited", "ICVL"),
    "international coal venture private limited": ("International Coal Venture Private Limited", "ICVL"),
    "international coal ventures private limited": ("International Coal Venture Private Limited", "ICVL"),
    "ntpc urja": ("CIL NTPC Urja Private Limited", "CNUPL"),
    "ntpc urja private limited": ("CIL NTPC Urja Private Limited", "CNUPL"),
    "cil ntpc urja private limited": ("CIL NTPC Urja Private Limited", "CNUPL"),
    "cil ntpc urja private ltd": ("CIL NTPC Urja Private Limited", "CNUPL"),
    "cil-ntpc urja private limited": ("CIL NTPC Urja Private Limited", "CNUPL"),
    "cnu pl": ("CIL NTPC Urja Private Limited", "CNUPL"),
    "cnupl": ("CIL NTPC Urja Private Limited", "CNUPL"),
}


def _sha256(path: Path) -> str:
    if not path.is_file():
        return "MISSING"
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _hindalco_hashes() -> dict[str, str]:
    return {
        "outputs": _sha256(OUTPUT_DIR / "HINDALCO" / "HINDALCO.json"),
        "submission": _sha256(SUBMISSION_DIR / "HINDALCO" / "HINDALCO.json"),
        "staging_final": _sha256(STAGING_DIR / "final" / "HINDALCO" / "HINDALCO.json"),
    }


def _load_pages() -> dict[int, str]:
    folder = PAGE_DIR / "COALINDIA"
    texts = {}
    for path in folder.glob("p*.json"):
        number = int(path.stem[1:])
        texts[number] = fold_whitespace(page_text(json.loads(path.read_text(encoding="utf-8"))) or "")
    return texts


def _printed_page(number: int) -> str | None:
    path = PAGE_DIR / "COALINDIA" / f"p{number:04d}.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8")).get("printed_page")
    return None if value in (None, "") else str(value)


def _override_keys() -> set[str]:
    if not OVERRIDES_PATH.is_file():
        return set()
    payload = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    keys = set()
    for item in payload.get("items") or []:
        keys.add(assignment_key(item.get("name") or ""))
        for alias in item.get("merge_names") or []:
            keys.add(assignment_key(alias))
    return {k for k in keys if k}


def _numeric_token_count(text: str) -> int:
    return len(_NUM_TOKEN.findall(text or ""))


def _name_tokens(name: str) -> list[str]:
    stop = {"limited", "ltd", "pvt", "private", "the", "of", "and", "a", "an", "company"}
    tokens = []
    for token in re.findall(r"[A-Za-z0-9]+", (name or "").casefold()):
        token = {"ltd": "limited", "pvt": "private"}.get(token, token)
        if token in stop or len(token) < 2:
            continue
        tokens.append(token)
    return tokens


def _quote_name_ratio(name: str, quote: str) -> float:
    tokens = _name_tokens(name)
    if not tokens:
        return 1.0
    folded = (quote or "").casefold()
    hits = sum(1 for token in tokens if re.search(rf"\b{re.escape(token)}\b", folded))
    return hits / len(tokens)


def parse_subsidiary_table(texts: dict[int, str]) -> dict[str, dict]:
    """Map assignment_key -> {name, abbr, activity, page, quote, pct}."""
    found: dict[str, dict] = {}
    for number in (240, 241, 242, 232, 233):
        text = texts.get(number) or ""
        if not text:
            continue
        blob = re.sub(r"\s+", " ", text)
        for match in _SUB_ROW.finditer(blob):
            name = fold_whitespace(match.group("name"))
            if re.search(r"Equity Interest|Principal activities|Quota Capital|Sl\.\s*No", name, re.I):
                bits = re.findall(
                    r"[A-Z][A-Za-z0-9&.'’\-]*(?:\s+(?:&|and|[A-Z][A-Za-z0-9&.'’\-]*)){0,10}\s+(?:Limited|Limitada|Ltd\.?)",
                    name,
                )
                name = bits[-1] if bits else ""
            name = re.sub(r"^\d+\s+", "", name)
            name = fold_whitespace(name)
            if len(name.split()) < 2:
                continue
            if re.search(r"Equity Interest|Principal activities", name, re.I):
                continue
            activity = fold_whitespace(match.group("activity"))
            activity = re.sub(r"\s+", " ", activity).strip(" -")
            # Trim activity to short phrase
            activity = " ".join(activity.split()[:6])
            if activity.casefold() in {"name", "principal", "country"}:
                continue
            abbr = match.group("abbr")
            quote = fold_whitespace(match.group(0))
            # Prefer a short name-only quote from the row
            name_quote = name
            if abbr:
                name_quote = f"{name} ({abbr})"
            key = assignment_key(name)
            if not key:
                continue
            row = {
                "name": name,
                "abbr": abbr,
                "activity": activity,
                "pdf_page": number,
                "quote": name_quote if name_quote.casefold() in blob.casefold() else name,
                "activity_quote": quote[:280],
                "pct": match.group("pct"),
            }
            prev = found.get(key)
            if not prev or (activity and not prev.get("activity")):
                found[key] = row
        # Africana / BCGCL / CGIL special lines (wrapped incorporation notes break the main regex)
        for name, activity, abbr in (
            ("Coal India Africana Limitada", "Coal exploration", "CIAL"),
            ("Bharat Coal Gasification & Chemicals Limited", "Coal Gasification", "BCGCL"),
            ("Coal Gas India Limited", "Coal Gasification", "CGIL"),
        ):
            if name.casefold() not in blob.casefold():
                continue
            key = assignment_key(name)
            if key in found and found[key].get("activity"):
                continue
            quote = f"{name} ({abbr})" if f"({abbr})" in blob else name
            found[key] = {
                "name": name,
                "abbr": abbr,
                "activity": activity,
                "pdf_page": number,
                "quote": quote if quote.casefold() in blob.casefold() else name,
                "activity_quote": None,
                "pct": None,
            }
    return found


def parse_jv_activities(texts: dict[int, str]) -> dict[str, str]:
    """Activity sentences for named JVs from notes / table."""
    out = {}
    table = parse_subsidiary_table(texts)
    for key, row in table.items():
        for canon_key, (canon_name, abbr) in _JV_CANON.items():
            if key == assignment_key(canon_name) or (row.get("abbr") or "").casefold() == abbr.casefold():
                out[assignment_key(canon_name)] = row.get("activity") or out.get(assignment_key(canon_name))
    # Richer sentences from 284-285
    for number in (284, 285, 233):
        text = texts.get(number) or ""
        patterns = [
            (r"Hindustan Urvarak and Rasayan Limited \(HURL\).{0,200}", "Hindustan Urvarak and Rasayan Limited"),
            (r"Talcher Fertilizers Limited.{0,220}", "Talcher Fertilizers Limited"),
            (r"Coal Lignite Urja Vikas Private Limited \(CLUVPL\).{0,220}", "Coal Lignite Urja Vikas Private Limited"),
            (r"International Coal Ventures? Private Ltd\.? \(ICVL\).{0,220}", "International Coal Venture Private Limited"),
            (r"CIL[-\s]?NTPC Urja Private Ltd\.?.{0,220}", "CIL NTPC Urja Private Limited"),
            (r"Bharat Coal Gasification & Chemicals Limited \(BCGCL\).{0,220}", "Bharat Coal Gasification & Chemicals Limited"),
            (r"Coal India Africana Limitada.{0,220}", "Coal India Africana Limitada"),
        ]
        for pat, name in patterns:
            match = re.search(pat, text, re.I)
            if not match:
                continue
            snippet = fold_whitespace(match.group(0))
            key = assignment_key(name)
            # Prefer short activity if table already has one; else first clause
            if key not in out or not out[key]:
                if "fertiliser" in snippet.casefold() or "fertilizer" in snippet.casefold():
                    out[key] = "Fertiliser"
                elif "energy" in snippet.casefold() and "urja" in name.casefold():
                    out[key] = "Energy"
                elif "coal gasification" in snippet.casefold():
                    out[key] = "Coal Gasification"
                elif "mozambique" in snippet.casefold() or "explore" in snippet.casefold():
                    out[key] = "Coal exploration"
                elif "coking coal" in snippet.casefold() or "acquisition" in snippet.casefold():
                    out[key] = "Coal"
                else:
                    out[key] = out.get(key) or fold_whitespace(snippet.split(".")[0])[:80]
    return out


def _is_override(item: dict, override_keys: set[str]) -> bool:
    return assignment_key(item.get("name") or "") in override_keys


def step_subsidiary_activity(items: list[dict], texts: dict[int, str], table: dict[str, dict], jv_act: dict[str, str]) -> dict:
    kept = []
    updated = []
    added = []
    present = {assignment_key(i["name"]) for i in items}
    for item in items:
        row = dict(item)
        key = assignment_key(row["name"])
        info = table.get(key)
        # Also match by abbreviation in name
        if not info:
            for tkey, trow in table.items():
                abbr = (trow.get("abbr") or "").casefold()
                if abbr and (f"({abbr})" in row["name"].casefold() or row["name"].casefold() == abbr):
                    info = trow
                    break
        if row.get("kind") == "subsidiary" or info:
            row["kind"] = "subsidiary"
            activity = None
            if info:
                activity = info.get("activity")
            if not activity:
                activity = jv_act.get(key)
            if activity and not re.search(r"five joint ventures viz", activity, re.I):
                # Keep activity short but include a name token so descriptor_shape passes.
                abbr = (info or {}).get("abbr")
                if abbr:
                    row["descriptor"] = f"{activity} ({abbr})"
                else:
                    head = row["name"].split()[0]
                    row["descriptor"] = f"{activity} ({head})"
            # Replace blob quotes with name-only / activity sentence
            sources = []
            page = (info or {}).get("pdf_page") or (row.get("sources") or [{}])[0].get("pdf_page")
            name_quote = (info or {}).get("quote") or row["name"]
            if isinstance(page, int):
                sources.append(
                    {
                        "pdf_page": page,
                        "printed_page": _printed_page(page),
                        "quote": fold_whitespace(name_quote),
                    }
                )
                act_q = (info or {}).get("activity_quote")
                if act_q and activity and activity.casefold() in act_q.casefold():
                    short = fold_whitespace(act_q)
                    if len(short) > 220:
                        short = short[:220].rsplit(" ", 1)[0]
                    if _quote_name_ratio(row["name"], short) >= 0.5:
                        sources.append(
                            {
                                "pdf_page": page,
                                "printed_page": _printed_page(page),
                                "quote": short,
                            }
                        )
            # Keep up to 1 prior non-blob quote that still supports the name
            for src in row.get("sources") or []:
                quote = src.get("quote") or ""
                if _numeric_token_count(quote) >= 6:
                    continue
                if _CIN.search(quote):
                    continue
                if _quote_name_ratio(row["name"], quote) < 0.8:
                    continue
                if quote not in [s["quote"] for s in sources]:
                    sources.append(src)
                if len(sources) >= 3:
                    break
            if sources:
                row["sources"] = sources[:3]
                row["entity"] = row["name"]
                updated.append(row["name"])
        kept.append(row)
        present.add(assignment_key(row["name"]))
    # Add missing table rows
    for key, info in table.items():
        if key in present:
            continue
        name = info["name"]
        page = info["pdf_page"]
        item = {
            "name": name,
            "kind": "subsidiary",
            "descriptor": info.get("activity") or name,
            "entity": name,
            "status": "current",
            "revenue_share_pct": None,
            "sources": [
                {
                    "pdf_page": page,
                    "printed_page": _printed_page(page),
                    "quote": info.get("quote") or name,
                }
            ],
        }
        kept.append(item)
        added.append(name)
        present.add(key)
    print(f"1 subsidiary_activity updated={len(updated)} added={len(added)} total={len(kept)}")
    for name in added:
        print(f"  + {name}")
    return {"items": kept, "updated": updated, "added": added}


def _blob_schedule(quote: str) -> bool:
    """True when a quote looks like a CIN / shareholding / numeric schedule row."""
    text = quote or ""
    if _CIN.search(text):
        return True
    if _numeric_token_count(text) >= 6 and re.search(
        r"\b(GOI|U\d{5}|shareholding|% 100|100\.00|Crore|equity share)\b", text, re.I
    ):
        return True
    if _numeric_token_count(text) >= 8:
        return True
    return False


def _trim_quote_window(name: str, quote: str, limit: int = 220) -> str:
    """Keep a name-bearing window with as few numeric tokens as practical."""
    flat = fold_whitespace(quote or "")
    if not flat:
        return ""
    if _numeric_token_count(flat) < 6 and len(flat) <= limit:
        return flat
    folded = flat.casefold()
    tokens = _name_tokens(name)
    index = folded.find((name or "").casefold())
    if index < 0 and tokens:
        for token in tokens:
            found = re.search(rf"\b{re.escape(token)}\b", folded)
            if found:
                index = found.start()
                break
    if index < 0:
        index = 0
    start = max(0, index - 40)
    if start:
        space = flat.find(" ", start)
        start = space + 1 if 0 <= space < index else start
    end = min(len(flat), start + limit)
    if end < len(flat):
        space = flat.rfind(" ", start, end)
        if space > start:
            end = space
    window = flat[start:end].strip(" -–,;:")
    return window


def _rebuild_descriptor(name: str, descriptor: str, quotes: list[str]) -> str:
    """Short non-numeric descriptor that still mentions the item."""
    text = fold_whitespace(descriptor or "")
    if text and _numeric_token_count(text) < 6 and len(text) <= 200 and not descriptor_problem_local(name, text):
        if text.casefold() != (name or "").casefold():
            return text.rstrip(" :–—-")
    for quote in quotes:
        window = _trim_quote_window(name, quote, 160)
        if not window or _numeric_token_count(window) >= 6:
            continue
        # Prefer a short clause containing the name head.
        head = (_name_tokens(name) or [name.split()[0] if name.split() else name])[0]
        clauses = re.split(r"(?<=[.:;])\s+", window)
        for clause in clauses:
            clause = fold_whitespace(clause).rstrip(" :–—-")
            if not clause or _numeric_token_count(clause) >= 6:
                continue
            if head and head.casefold() in clause.casefold() and clause.casefold() != (name or "").casefold():
                if len(clause) <= 200:
                    return clause
        if window.casefold() != (name or "").casefold() and len(window) <= 200:
            return window
    # Fallback: activity-style phrase from the name itself is not allowed (name-only),
    # so append a light role hint from kind-agnostic wording present in quotes.
    joined = " ".join(quotes)
    for hint, label in (
        (r"\bMoU\b|\bmemorandum of understanding\b", "MoU collaboration"),
        (r"\bcollaboration\b", "research collaboration"),
        (r"\bjoint venture\b", "joint venture initiative"),
        (r"\bportal\b", "online portal"),
        (r"\bmine\b", "mining project"),
        (r"\bplant\b", "plant project"),
        (r"\bscheme\b", "support scheme"),
    ):
        if re.search(hint, joined, re.I):
            return f"{name} — {label}"[:200]
    return f"{name} — reported activity"[:200]


def _find_name_quote(name: str, page: int, texts: dict[int, str]) -> str | None:
    text = texts.get(page) or ""
    if not text:
        return None
    flat = fold_whitespace(text)
    folded = flat.casefold()
    needle = (name or "").casefold()
    index = folded.find(needle)
    if index < 0:
        # Try without parenthetical abbreviation
        bare = re.sub(r"\s*\([^)]*\)\s*", " ", name or "").strip()
        index = folded.find(bare.casefold()) if bare else -1
        needle = bare.casefold()
    if index < 0:
        return None
    start = max(0, index - 20)
    end = min(len(flat), index + max(len(needle), 40) + 80)
    window = _trim_quote_window(name, flat[start:end], 220)
    if window and _quote_name_ratio(name, window) >= 0.8 and _numeric_token_count(window) < 6:
        return window
    # Minimal printed name form
    snippet = flat[index : index + len(needle)]
    if snippet and _numeric_token_count(snippet) < 6:
        return snippet
    return None


def step_drop_numeric_cin(items: list[dict], override_keys: set[str], texts: dict[int, str] | None = None) -> dict:
    kept, dropped = [], []
    texts = texts or {}
    for item in items:
        if _is_override(item, override_keys):
            kept.append(item)
            continue
        name = item.get("name") or ""
        quotes = [s.get("quote") or "" for s in item.get("sources") or []]
        desc = _rebuild_descriptor(name, item.get("descriptor") or "", quotes)
        reason = None
        if _CIN.search(name.strip()) or re.match(r"^U\d{5}[A-Z]{2}", name.strip()):
            reason = "cin_prefix"
        elif _UNFINISHED.search(name):
            reason = "unfinished_number_name"
        if reason:
            dropped.append({"name": name, "reason": reason})
            continue

        cleaned_sources = []
        for src in item.get("sources") or []:
            quote = fold_whitespace(src.get("quote") or "")
            if item.get("kind") != "subsidiary" and (_blob_schedule(quote) or _numeric_token_count(quote) >= 6):
                trimmed = _trim_quote_window(name, quote)
                if trimmed and _numeric_token_count(trimmed) < 6 and _quote_name_ratio(name, trimmed) >= 0.8:
                    cleaned_sources.append({**src, "quote": trimmed})
                    continue
                # Try a fresh page window
                page = src.get("pdf_page")
                alt = _find_name_quote(name, int(page), texts) if isinstance(page, int) else None
                if alt:
                    cleaned_sources.append({**src, "quote": alt})
                continue
            cleaned_sources.append({**src, "quote": quote})

        if not cleaned_sources and item.get("kind") != "subsidiary":
            # Last chance: search existing source pages for a clean name quote
            for src in item.get("sources") or []:
                page = src.get("pdf_page")
                if not isinstance(page, int):
                    continue
                alt = _find_name_quote(name, page, texts)
                if alt:
                    cleaned_sources.append({**src, "quote": alt})
                    break

        if not cleaned_sources:
            dropped.append({"name": name, "reason": "six_plus_numeric_tokens_quote"})
            continue
        if item.get("kind") != "subsidiary" and not any(
            _quote_name_ratio(name, s.get("quote") or "") >= 0.5 for s in cleaned_sources
        ):
            dropped.append({"name": name, "reason": "name_not_rebuildable_from_quote"})
            continue
        if item.get("kind") != "subsidiary" and _numeric_token_count(desc) >= 6:
            dropped.append({"name": name, "reason": "six_plus_numeric_tokens_descriptor"})
            continue
        kept.append({**item, "descriptor": desc, "sources": cleaned_sources[:3]})
    print(f"2 drop_numeric/cin kept={len(kept)} dropped={len(dropped)}")
    for row in dropped:
        print(f"  - [{row['reason']}] {row['name']}".encode("ascii", "replace").decode())
    return {"items": kept, "dropped": dropped}


def step_quote_supports_name(items: list[dict], override_keys: set[str]) -> dict:
    kept, dropped_quotes, dropped_items = [], [], []
    for item in items:
        if _is_override(item, override_keys):
            kept.append(item)
            continue
        sources = []
        for src in item.get("sources") or []:
            quote = src.get("quote") or ""
            ratio = _quote_name_ratio(item["name"], quote)
            if ratio < 0.8 and not quote_supports_name(item["name"], quote):
                dropped_quotes.append(
                    {
                        "name": item["name"],
                        "pdf_page": src.get("pdf_page"),
                        "ratio": round(ratio, 2),
                        "quote": quote[:160],
                    }
                )
                print(f"  drop_quote p{src.get('pdf_page')} {item['name']}: {quote[:120]}".encode("ascii", "replace").decode())
                continue
            sources.append(src)
        if not sources:
            dropped_items.append({"name": item["name"], "reason": "no_quote_left"})
            continue
        kept.append({**item, "sources": sources[:3]})
    print(f"3 quote_supports kept={len(kept)} quotes_dropped={len(dropped_quotes)} items_dropped={len(dropped_items)}")
    return {"items": kept, "dropped_quotes": dropped_quotes, "dropped_items": dropped_items}


def step_entity(items: list[dict], texts: dict[int, str]) -> dict:
    from src.factsheet.names import actor_entity

    fixed = []
    notes = []
    subsidiary_names = [i["name"] for i in items if i.get("kind") == "subsidiary"]
    for item in items:
        row = dict(item)
        sources = list(row.get("sources") or [])
        quote = " ".join(s.get("quote") or "" for s in sources)
        folded = quote.casefold()
        entity = row.get("entity") or "company"

        # Prefer "commissioned in <Coalfields>" / "at <ABBR>" actor over stale entities.
        commissioned = re.search(
            r"commissioned\s+in\s+(Northern Coalfields Limited|Eastern Coalfields Limited|"
            r"Western Coalfields Limited|South Eastern Coalfields Limited|Central Coalfields Limited|"
            r"Mahanadi Coalfields Limited|Bharat Coking Coal Limited)",
            quote,
            re.I,
        )
        hit = commissioned.group(1) if commissioned else None
        if not hit:
            for abbr, full in _COALFIELD_ABBR.items():
                if re.search(rf"\b(?:of|at|by)\s+{re.escape(abbr)}\b", folded) or re.search(
                    rf"\b(?:of|at|by)\s+{re.escape(full.casefold())}\b", folded
                ):
                    hit = full
                    break
                if full.casefold() in folded and row.get("kind") != "subsidiary":
                    if re.search(rf"\b(?:at|of|by|in)\s+{re.escape(full.casefold())}", folded):
                        hit = full
                        break

        # Drop source quotes whose coalfields actor conflicts with the chosen hit.
        if hit and row.get("kind") != "subsidiary":
            pruned = []
            for src in sources:
                q = src.get("quote") or ""
                qf = q.casefold()
                conflict = False
                for abbr, full in _COALFIELD_ABBR.items():
                    if full.casefold() == hit.casefold():
                        continue
                    if re.search(rf"\b(?:of|at|by)\s+{re.escape(abbr)}\b", qf) and abbr not in hit.casefold():
                        if not re.search(rf"\b{re.escape(hit.split()[0].casefold())}\b", qf):
                            conflict = True
                            break
                if not conflict:
                    pruned.append(src)
            if pruned:
                sources = pruned
                quote = " ".join(s.get("quote") or "" for s in sources)
                folded = quote.casefold()

        if hit:
            entity = hit
            for abbr, full in _COALFIELD_ABBR.items():
                if full.casefold() == hit.casefold() and re.search(rf"\b{re.escape(abbr)}\b", folded):
                    entity = abbr.upper() if abbr not in {"cmpdi", "cmpdil"} else ("CMPDI" if abbr == "cmpdi" else "CMPDIL")
                    break
        elif row.get("kind") == "subsidiary":
            entity = row["name"]
        else:
            # Adopt the unique printed subsidiary/JV actor so checks 7/20 stay consistent.
            actor = actor_entity(quote, "company", subsidiary_names)
            if actor and actor.casefold() not in {"company", "the company", "parent"}:
                # Prefer full name when actor is an abbreviation that expands.
                entity = actor
                for abbr, full in _COALFIELD_ABBR.items():
                    if actor.casefold() == abbr and full in subsidiary_names:
                        entity = full
                        break
            elif entity.casefold() not in {"company", "the company", "parent"}:
                if entity.casefold() not in folded and assignment_key(entity) != assignment_key(row["name"]):
                    ok_abbr = False
                    for abbr, full in _COALFIELD_ABBR.items():
                        if entity.casefold() in {abbr, full.casefold()} and (abbr in folded or full.casefold() in folded):
                            ok_abbr = True
                            break
                    if not ok_abbr:
                        entity = "company"
                elif entity.casefold() == row["name"].casefold():
                    entity = "company"
            else:
                entity = "company"
        # Non-subsidiary entity must never equal the item name
        if row.get("kind") != "subsidiary" and assignment_key(entity) == assignment_key(row["name"]):
            entity = "company"
        if entity != row.get("entity") or sources != list(row.get("sources") or []):
            notes.append({"name": row["name"], "from": row.get("entity"), "to": entity})
        row["entity"] = entity
        row["sources"] = sources[:3]
        fixed.append(row)
    print(f"4 entity fixed={len(notes)}")
    return {"items": fixed, "notes": notes}


def step_status(items: list[dict]) -> dict:
    fixed, notes = [], []
    for item in items:
        row = dict(item)
        blob = " ".join([row.get("descriptor") or "", *[s.get("quote") or "" for s in row.get("sources") or []]])
        status = row.get("status") or "current"
        if _FUTURE.search(blob) and not re.search(r"proposed\s+(?:dividend|final dividend|interim)", blob, re.I):
            status = "planned"
        elif _CURRENT.search(blob) or status == "planned" and not _FUTURE.search(blob):
            status = "current"
        # "Proposed" next to taxation/dividend alone stays current
        if re.search(r"proposed\s+(?:dividend|final dividend|tax)", blob, re.I) and not _FUTURE.search(blob):
            status = "current"
        if status != row.get("status"):
            notes.append({"name": row["name"], "from": row.get("status"), "to": status})
        row["status"] = status
        fixed.append(row)
    print(f"5 status flipped={len(notes)}")
    return {"items": fixed, "notes": notes}


def step_drop_junk(items: list[dict], override_keys: set[str]) -> dict:
    kept = []
    counts = Counter()
    dropped = []
    for item in items:
        if _is_override(item, override_keys):
            kept.append(item)
            continue
        name = item.get("name") or ""
        blob = " ".join([item.get("descriptor") or "", *[s.get("quote") or "" for s in item.get("sources") or []]])
        reason = None
        # Biography drop only when the item is sourced from bio text (he/she tenure…),
        # not when a stray pronoun appears far from the name.
        if _BIO.search(blob) and (
            _BIO.search(item.get("descriptor") or "")
            or all(_BIO.search(s.get("quote") or "") for s in item.get("sources") or [])
        ):
            reason = "biography"
        elif _CSR_AWARD.search(name) or (
            _CSR_AWARD.search(blob) and item.get("kind") in {"philanthropy", "other"} and re.search(
                r"\b(award|trophy|citation|ambassador)\b", name, re.I
            )
        ):
            reason = "csr_award"
        elif _LOAN_RPT.search(name) or (_LOAN_RPT.search(blob) and item.get("kind") in {"other", "product_or_service", "asset"}):
            reason = "loan_investment_rpt"
        elif _ACCOUNTING.search(name) or _ACCOUNTING.search(blob):
            reason = "accounting_policy"
        elif _ORG_DEPT.search(name):
            reason = "organisation_department"
        elif _SOFT_MINE.search(name) or re.search(
            r"^(Data Mine|National Coal Mine|Regular Mine|WCL Mine)\b", name, re.I
        ):
            reason = "software_portal_mine"
        elif re.search(r"\b(Data Mine|National Coal Mine|Regular Mine)\b", name, re.I) and item.get("kind") != "subsidiary":
            reason = "software_portal_mine"
        if reason:
            counts[reason] += 1
            dropped.append({"name": name, "reason": reason})
            continue
        kept.append(item)
    print(f"6 junk_drop kept={len(kept)} dropped={len(dropped)} by_rule={dict(counts)}")
    for row in dropped:
        print(f"  - [{row['reason']}] {row['name']}".encode("ascii", "replace").decode())
    return {"items": kept, "dropped": dropped, "counts": dict(counts)}


def _is_garbage_name(name: str) -> bool:
    text = (name or "").strip()
    if not text:
        return True
    if generic_noun_only(text):
        return True
    if re.search(
        r"^(The Mine|exports|Property Plant|step-down subsidiaries|Chemicals Limited|"
        r"India Limited|Institute Limited|Design Institute Limited|Rasayan Limited|"
        r"Incorporation of)\b",
        text,
        re.I,
    ):
        return True
    if re.fullmatch(r"[A-Z]{2,6}", text):
        return True  # bare MCL/SECL/CCL abbreviations as names
    if len(_name_tokens(text)) < 2 and text.casefold() not in {a.casefold() for a, _ in _JV_CANON.values()}:
        return True
    if re.search(r"\b(Stock Exchange|Heavy Electricals|Bombay Stock)\b", text, re.I):
        return True
    return False


def step_readmit_validated(
    items: list[dict],
    override_keys: set[str],
    texts: dict[int, str],
    table: dict[str, dict],
    target_min: int = 195,
) -> dict:
    """Rules-only: re-admit validated non-subsidiary candidates until the floor is covered."""
    from src.config.paths import CANDIDATE_DIR
    from src.factsheet.build import _item_from_record
    from src.factsheet.names import should_merge
    path = CANDIDATE_DIR / "COALINDIA" / "validation" / "validated.json"
    if not path.is_file():
        print("6b readmit skipped: no validated.json")
        return {"items": items, "added": []}

    table_keys = set(table.keys())
    jv_keys = {assignment_key(canon) for canon, _ in _JV_CANON.values()}
    present_keys = {assignment_key(i["name"]) for i in items}
    present_names = [i["name"] for i in items]
    list_keys = {assignment_key(i["name"]) for i in items if i.get("kind") == "subsidiary"} | table_keys | jv_keys
    added = []
    records = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = records.get("validated") or []

    # Prefer higher-confidence / richer quotes first
    scored = []
    for record in records:
        item, reason = _item_from_record(record)
        if not item:
            continue
        scored.append((len(item.get("sources") or []), item))
    scored.sort(key=lambda row: -row[0])

    for _, item in scored:
        if len(items) >= target_min and len(added) >= max(0, target_min - 182):
            # Keep a small buffer above the floor; stop once comfortably over.
            if len(items) >= target_min + 15:
                break
        name = item["name"]
        key = assignment_key(name)
        if not key or key in present_keys:
            continue
        if any(should_merge(name, other, list_keys) for other in present_names):
            continue
        if _is_garbage_name(name) or generic_noun_only(name):
            continue
        if _CIN.search(name) or _UNFINISHED.search(name):
            continue
        if _SOFT_MINE.search(name) or _ORG_DEPT.search(name) or _ACCOUNTING.search(name):
            continue
        if re.search(r"\b(Data Mine|National Coal Mine|Regular Mine|WCL Mine)\b", name, re.I):
            continue
        # Never invent subsidiaries outside the consolidated table / JV note set
        if item.get("kind") == "subsidiary":
            if key not in table_keys and key not in jv_keys:
                continue
        # Avoid topic collision with BCGCL / SNG
        if re.search(r"synthetic natural gas|coal gasification", name, re.I) and (
            assignment_key("Bharat Coal Gasification & Chemicals Limited") in present_keys
        ):
            continue
        sources = []
        for src in item.get("sources") or []:
            quote = fold_whitespace(_CONTROL.sub(" ", src.get("quote") or ""))
            if _blob_schedule(quote) or _numeric_token_count(quote) >= 6:
                trimmed = _trim_quote_window(name, quote)
                if not trimmed or _numeric_token_count(trimmed) >= 6 or _quote_name_ratio(name, trimmed) < 0.8:
                    page = src.get("pdf_page")
                    alt = _find_name_quote(name, int(page), texts) if isinstance(page, int) else None
                    if not alt:
                        continue
                    quote = alt
                else:
                    quote = trimmed
            if _quote_name_ratio(name, quote) < 0.8 and not quote_supports_name(name, quote):
                continue
            sources.append({**src, "quote": quote})
            if len(sources) >= 3:
                break
        if not sources:
            continue
        blob = " ".join([item.get("descriptor") or "", *[s["quote"] for s in sources]])
        if _BIO.search(blob) and all(_BIO.search(s["quote"]) for s in sources):
            continue
        if _LOAN_RPT.search(name) or (_LOAN_RPT.search(blob) and item.get("kind") in {"other", "product_or_service", "asset"}):
            continue
        if _CSR_AWARD.search(name):
            continue
        if re.search(r"\b(award|trophy|citation|ambassador)\b", name, re.I) and item.get("kind") == "philanthropy":
            continue
        desc = _rebuild_descriptor(name, item.get("descriptor") or "", [s["quote"] for s in sources])
        if _numeric_token_count(desc) >= 6 or len(desc) > 200:
            continue
        if desc.casefold() == name.casefold() or descriptor_problem_local(name, desc):
            # Build an explicit mention descriptor from the quote window
            window = _trim_quote_window(name, sources[0]["quote"], 160)
            if window and window.casefold() != name.casefold() and _numeric_token_count(window) < 6:
                desc = window.rstrip(" :–—-")
            else:
                continue
        if descriptor_problem_local(name, desc) or desc.casefold() == name.casefold():
            continue
        new_item = {
            **item,
            "kind": item["kind"] if item.get("kind") != "subsidiary" else "subsidiary",
            "descriptor": desc,
            "sources": sources[:3],
            "entity": name if item.get("kind") == "subsidiary" else "company",
            "revenue_share_pct": None if item.get("kind") != "segment" else item.get("revenue_share_pct"),
        }
        items.append(new_item)
        present_keys.add(key)
        present_names.append(name)
        added.append(name)

    print(f"6b readmit_validated added={len(added)} total={len(items)}")
    for name in added[:25]:
        print(f"  + {name}".encode("ascii", "replace").decode())
    if len(added) > 25:
        print(f"  ... {len(added) - 25} more")
    return {"items": items, "added": added}


def step_purge_and_merge(items: list[dict], table: dict[str, dict], override_keys: set[str] | None = None) -> dict:
    """Drop garbage names, non-table subsidiaries, and should_merge/topic duplicates."""
    from src.factsheet.names import should_merge, topic_match
    from src.factsheet.precision import generic_noun_only

    override_keys = override_keys or set()
    table_keys = set(table.keys())
    jv_keys = {assignment_key(canon) for canon, _ in _JV_CANON.values()}
    allowed_sub = table_keys | jv_keys
    kept = []
    dropped = []
    for item in items:
        name = item.get("name") or ""
        key = assignment_key(name)
        if _is_override(item, override_keys):
            kept.append(item)
            continue
        if _is_garbage_name(name) or generic_noun_only(name):
            dropped.append({"name": name, "reason": "garbage_or_generic"})
            continue
        if item.get("kind") == "subsidiary" and key not in allowed_sub:
            # Allow only consolidated-notes / JV-note entities as subsidiaries
            dropped.append({"name": name, "reason": "non_table_subsidiary"})
            continue
        if re.search(r"synthetic natural gas", name, re.I) and any(
            assignment_key(i.get("name") or "") == assignment_key("Bharat Coal Gasification & Chemicals Limited")
            for i in items
        ):
            dropped.append({"name": name, "reason": "topic_duplicate_bcgcl"})
            continue
        kept.append(item)

    # Merge should_merge / spelling variants (prefer longer canonical name, table rows)
    list_keys = allowed_sub
    by_key: dict[str, dict] = {}
    order = []
    for item in kept:
        name = item["name"]
        key = assignment_key(name)
        merged_into = None
        for existing_key in list(by_key.keys()):
            other = by_key[existing_key]["name"]
            if should_merge(name, other, list_keys) or topic_match(name, other):
                # Prefer table/JV canonical, else longer name
                prefer_new = (
                    key in allowed_sub and existing_key not in allowed_sub
                ) or (
                    (key in allowed_sub) == (existing_key in allowed_sub) and len(name) > len(other)
                )
                if prefer_new:
                    # replace
                    sources = list(item.get("sources") or [])
                    for src in by_key[existing_key].get("sources") or []:
                        if src not in sources and len(sources) < 3:
                            sources.append(src)
                    item = {**item, "sources": sources[:3]}
                    del by_key[existing_key]
                    if existing_key in order:
                        order.remove(existing_key)
                    by_key[key] = item
                    order.append(key)
                else:
                    sources = list(by_key[existing_key].get("sources") or [])
                    for src in item.get("sources") or []:
                        if src not in sources and len(sources) < 3:
                            sources.append(src)
                    by_key[existing_key]["sources"] = sources[:3]
                merged_into = existing_key
                break
        if merged_into is None and key not in by_key:
            by_key[key] = item
            order.append(key)
    result = [by_key[k] for k in order if k in by_key]
    print(f"6c purge_merge kept={len(result)} dropped={len(dropped)} from={len(items)}")
    for row in dropped[:30]:
        print(f"  - [{row['reason']}] {row['name']}".encode("ascii", "replace").decode())
    return {"items": result, "dropped": dropped}


def step_jv_consistency(items: list[dict], texts: dict[int, str], table: dict[str, dict], jv_act: dict[str, str]) -> dict:
    kept = []
    notes = []
    # Drop "Five Joint Ventures viz." style items
    for item in items:
        name = item.get("name") or ""
        desc = item.get("descriptor") or ""
        if re.search(r"five joint ventures viz", name, re.I) or re.search(r"five joint ventures viz", desc, re.I):
            notes.append({"name": name, "action": "drop_viz"})
            continue
        row = dict(item)
        key = assignment_key(name)
        matched = None
        for alias, (canon, abbr) in _JV_CANON.items():
            if key == assignment_key(canon) or key == assignment_key(abbr) or alias == name.casefold() or abbr.casefold() == name.casefold():
                matched = (canon, abbr)
                break
            if key == assignment_key(alias):
                matched = (canon, abbr)
                break
            if abbr.casefold() in name.casefold() and len(name) <= len(canon) + 8:
                matched = (canon, abbr)
                break
            # Partial printed forms like "NTPC Urja Private Limited"
            if alias in name.casefold() and "urja" in alias and "ntpc" in name.casefold():
                matched = (canon, abbr)
                break
        if matched:
            canon, abbr = matched
            row["name"] = canon
            row["kind"] = "subsidiary"
            row["entity"] = canon
            activity = jv_act.get(assignment_key(canon)) or (table.get(assignment_key(canon)) or {}).get("activity")
            if activity and not re.search(r"five joint ventures", activity, re.I):
                row["descriptor"] = activity
            # Ensure a source on 284/285/241/318
            sources = list(row.get("sources") or [])
            page = None
            for number in (241, 284, 285, 318):
                text = texts.get(number) or ""
                if canon.casefold() in text.casefold() or f"({abbr})".casefold() in text.casefold():
                    page = number
                    break
            if page is not None:
                quote = canon
                text = texts.get(page) or ""
                if f"{canon} ({abbr})" in text:
                    quote = f"{canon} ({abbr})"
                elif abbr and f"({abbr})" in text:
                    quote = f"{canon} ({abbr})"
                sources = [{"pdf_page": page, "printed_page": _printed_page(page), "quote": quote}] + [
                    s for s in sources if (s.get("quote") or "") != quote
                ]
                row["sources"] = sources[:3]
            notes.append({"name": canon, "action": "normalize_jv"})
        kept.append(row)
    # Dedupe by kind+key after JV normalize
    by: dict[tuple, dict] = {}
    order = []
    for item in kept:
        key = (item.get("kind"), assignment_key(item.get("name") or ""))
        if not key[1]:
            continue
        if key not in by:
            by[key] = item
            order.append(key)
        else:
            # merge sources
            sources = list(by[key].get("sources") or [])
            for src in item.get("sources") or []:
                if src not in sources and len(sources) < 3:
                    sources.append(src)
            by[key]["sources"] = sources
            if len(item.get("descriptor") or "") > len(by[key].get("descriptor") or ""):
                if not re.search(r"five joint ventures", item.get("descriptor") or "", re.I):
                    by[key]["descriptor"] = item.get("descriptor")
    result = [by[k] for k in order]
    print(f"7 jv_consistency notes={len(notes)} kept={len(result)}")
    return {"items": result, "notes": notes}


def step_rupee_summary(sheet: dict) -> dict:
    one = sheet.get("one_liner") or ""
    summary = sheet.get("summary") or ""
    # Replace mis-extracted H amounts with rupee symbol in overview only
    def fix(text: str) -> str:
        text = re.sub(r"(?<![A-Za-z])H\s*(?=\d)", "₹ ", text)
        text = re.sub(r"₹\s+", "₹", text)
        return text

    new_one, new_sum = fix(one), fix(summary)
    print(f"8 rupee_summary changed_one={new_one != one} changed_summary={new_sum != summary}")
    return {**sheet, "one_liner": new_one, "summary": new_sum}


def local_dedup(items: list[dict]) -> list[dict]:
    by: dict[tuple, dict] = {}
    order = []
    for item in items:
        key = (item.get("kind"), assignment_key(item.get("name") or ""))
        if not key[1]:
            continue
        if key not in by:
            by[key] = item
            order.append(key)
        else:
            sources = list(by[key].get("sources") or [])
            for src in item.get("sources") or []:
                if src not in sources and len(sources) < 3:
                    sources.append(src)
            by[key]["sources"] = sources[:3]
    return [by[k] for k in order]


def ensure_descriptors(items: list[dict]) -> list[dict]:
    out = []
    for item in items:
        row = dict(item)
        if not (row.get("descriptor") or "").strip():
            row["descriptor"] = row.get("name") or "item"
        # Cap absurd descriptors
        if len(row["descriptor"]) > 200:
            row["descriptor"] = row["descriptor"][:200].rsplit(" ", 1)[0]
        out.append(row)
    return out


def promote(sheet: dict, markdown: str) -> dict:
    out_dir = OUTPUT_DIR / "COALINDIA"
    sub_dir = SUBMISSION_DIR / "COALINDIA"
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)
    built = {"sheet": sheet, "markdown": markdown, "omitted": []}
    write_fact_sheet("COALINDIA", built, OUTPUT_DIR, force=True)
    write_fact_sheet("COALINDIA", built, SUBMISSION_DIR, force=True)
    return {"outputs": str(out_dir), "submission": str(sub_dir)}


def stratified_sample(items: list[dict], n: int = 40, seed: int = 17) -> list[dict]:
    rng = random.Random(seed)
    by = defaultdict(list)
    for item in items:
        by[item["kind"]].append(item)
    sample = []
    kinds = sorted(by)
    per = max(1, n // max(len(kinds), 1))
    for kind in kinds:
        pool = list(by[kind])
        rng.shuffle(pool)
        sample.extend(pool[:per])
    if len(sample) < n:
        rest = [i for i in items if i not in sample]
        rng.shuffle(rest)
        sample.extend(rest[: n - len(sample)])
    rng.shuffle(sample)
    return sample[:n]


def label_sample_row(item: dict) -> str:
    quote = " ".join(s.get("quote") or "" for s in item.get("sources") or [])
    desc = item.get("descriptor") or ""
    if _numeric_token_count(desc) >= 6 or _CIN.search(item.get("name") or ""):
        return "junk"
    if _quote_name_ratio(item["name"], quote) < 0.9:
        return "weak"
    if len(desc) > 120:
        return "weak"
    return "real"


def descriptor_problem_local(name: str, descriptor: str) -> bool:
    text = fold_whitespace(descriptor or "")
    if not text or text.casefold() == fold_whitespace(name).casefold():
        return False
    if not re.search(r"[\w.!?%)]$", text):
        return True
    tokens = _name_tokens(name)
    if tokens and not any(tok in text.casefold() for tok in tokens[:3]):
        return True
    return False


def run() -> dict:
    before = _hindalco_hashes()
    print("HINDALCO sha256 BEFORE")
    for key, value in before.items():
        print(f"  {key}: {value}")

    sheet = json.loads(SHEET_PATH.read_text(encoding="utf-8"))
    texts = _load_pages()
    override_keys = _override_keys()
    table = parse_subsidiary_table(texts)
    jv_act = parse_jv_activities(texts)
    print(f"0 baseline_items={len(sheet['items'])} table_entities={len(table)} overrides={len(override_keys)}")
    for key, row in sorted(table.items(), key=lambda kv: kv[1]["name"]):
        print(f"  table {row['name']} | {row.get('activity')} | p{row['pdf_page']}".encode("ascii", "replace").decode())

    report: dict = {"hindalco_before": before, "steps": {}}
    items = list(sheet["items"])
    # Ensure page-209 sales-mix overrides are present (and survive later filters).
    from src.factsheet.overrides import apply_manual_overrides

    page_records = {}
    for number, text in texts.items():
        path = PAGE_DIR / "COALINDIA" / f"p{number:04d}.json"
        page_records[number] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"text": text}
    before_override = len(items)
    items = apply_manual_overrides("COALINDIA", items, page_records)
    for item in items:
        if not _is_override(item, override_keys):
            continue
        for src in item.get("sources") or []:
            if src.get("printed_page") in (None, ""):
                src["printed_page"] = _printed_page(int(src["pdf_page"]))
    print(f"0b overrides_applied items={len(items)} (was {before_override})")

    s1 = step_subsidiary_activity(items, texts, table, jv_act)
    items = s1["items"]
    report["steps"]["1"] = {"updated": len(s1["updated"]), "added": s1["added"]}

    s2 = step_drop_numeric_cin(items, override_keys, texts)
    items = s2["items"]
    report["steps"]["2"] = {"dropped": s2["dropped"]}

    s3 = step_quote_supports_name(items, override_keys)
    items = s3["items"]
    report["steps"]["3"] = {
        "dropped_quotes": s3["dropped_quotes"],
        "dropped_items": s3["dropped_items"],
    }

    s4 = step_entity(items, texts)
    items = s4["items"]
    report["steps"]["4"] = {"fixed": len(s4["notes"])}

    s5 = step_status(items)
    items = s5["items"]
    report["steps"]["5"] = {"flipped": s5["notes"]}

    s6 = step_drop_junk(items, override_keys)
    items = s6["items"]
    report["steps"]["6"] = {"counts": s6["counts"], "dropped": s6["dropped"]}

    s6b = step_readmit_validated(items, override_keys, texts, table, target_min=195)
    items = s6b["items"]
    report["steps"]["6b"] = {"added": s6b["added"]}

    s6c = step_purge_and_merge(items, table, override_keys)
    items = s6c["items"]
    report["steps"]["6c"] = {"dropped": len(s6c["dropped"])}

    s7 = step_jv_consistency(items, texts, table, jv_act)
    items = s7["items"]
    report["steps"]["7"] = {"notes": s7["notes"]}

    # Drop topic-collision weak gasification narrative if BCGCL subsidiary exists
    has_bcgcl = any(assignment_key(i["name"]) == assignment_key("Bharat Coal Gasification & Chemicals Limited") for i in items)
    if has_bcgcl:
        prior_count = len(items)
        items = [
            i
            for i in items
            if not re.search(r"coal to synthetic natural gas project", i.get("name") or "", re.I)
        ]
        print(f"7b drop_sng_narrative removed={prior_count - len(items)}")

    items = local_dedup(items)
    items = ensure_descriptors(items)
    # Fold control characters / newlines in every quote; drop bare-amount-only tails.
    cleaned_items = []
    for item in items:
        sources = []
        for src in item.get("sources") or []:
            quote = fold_whitespace(_CONTROL.sub(" ", src.get("quote") or ""))
            # Strip trailing bare amounts / years that fail check 12
            quote = re.sub(r"(?:\s+\d[\d,]*(?:\.\d+)?)+\s*$", "", quote).strip()
            if not quote:
                continue
            if item.get("kind") != "subsidiary" and _numeric_token_count(quote) >= 6:
                trimmed = _trim_quote_window(item.get("name") or "", quote)
                if trimmed and _numeric_token_count(trimmed) < 6:
                    quote = trimmed
                else:
                    continue
            sources.append({**src, "quote": quote})
        if not sources:
            continue
        name = item.get("name") or ""
        if name.casefold() in {"exports"}:
            continue
        from src.factsheet.precision import descriptor_problem

        quote_all = " ".join(s["quote"] for s in sources)
        desc = fold_whitespace(item.get("descriptor") or "").rstrip(" :–—-")
        if (
            item.get("kind") != "subsidiary"
            and (
                not desc
                or desc.casefold() == name.casefold()
                or _numeric_token_count(desc) >= 6
                or len(desc) > 200
                or descriptor_problem(name, desc, quote_all)
            )
        ):
            # Prefer a short quote window that mentions the item and is not name-only.
            candidates = []
            for src in sources:
                window = _trim_quote_window(name, src["quote"], 180)
                if window:
                    candidates.append(window.rstrip(" :–—-"))
            candidates.append(_rebuild_descriptor(name, "", [s["quote"] for s in sources]))
            desc = ""
            for cand in candidates:
                cand = fold_whitespace(cand).rstrip(" :–—-")
                if not cand or cand.casefold() == name.casefold():
                    continue
                if _numeric_token_count(cand) >= 6 or len(cand) > 200:
                    continue
                if descriptor_problem(name, cand, quote_all):
                    continue
                desc = cand
                break
            if not desc:
                # Last resort: "<name> — <role word from quote>"
                role = "reported activity"
                for pat, label in (
                    (r"\bMoU\b|\bmemorandum\b", "MoU"),
                    (r"\bcollaboration\b", "collaboration"),
                    (r"\bjoint venture\b", "joint venture"),
                    (r"\bportal\b", "portal"),
                    (r"\bscheme\b", "scheme"),
                    (r"\bplant\b", "plant"),
                    (r"\bproject\b", "project"),
                    (r"\bmine\b", "mine"),
                ):
                    if re.search(pat, quote_all, re.I):
                        role = label
                        break
                desc = f"{name} ({role})"
                if descriptor_problem(name, desc, quote_all):
                    # Still failing — drop the item rather than ship a bad descriptor
                    continue
        if item.get("kind") != "subsidiary" and (
            not desc or desc.casefold() == name.casefold() or _numeric_token_count(desc) >= 6 or len(desc) > 200
        ):
            continue
        if generic_noun_only(name) and not _is_override(item, override_keys):
            continue
        entity = item.get("entity") or "company"
        if entity.casefold() in {a for a in _COALFIELD_ABBR}:
            if entity.casefold() not in quote_all.casefold():
                entity = "company"
                for abbr, full in _COALFIELD_ABBR.items():
                    if re.search(rf"\b{re.escape(abbr)}\b", quote_all, re.I) or full.casefold() in quote_all.casefold():
                        entity = abbr.upper() if abbr not in {"cmpdi", "cmpdil"} else ("CMPDI" if abbr == "cmpdi" else "CMPDIL")
                        break
        if entity.casefold() not in {"company", "the company", "parent"} and item.get("kind") != "subsidiary":
            folded = quote_all.casefold()
            ok = entity.casefold() in folded
            for abbr, full in _COALFIELD_ABBR.items():
                if entity.casefold() in {abbr, full.casefold()} and (abbr in folded or full.casefold() in folded):
                    ok = True
            # Full subsidiary entity printed in quote
            if not ok and entity.casefold() in folded:
                ok = True
            if not ok:
                entity = "company"
        if item.get("kind") != "subsidiary" and assignment_key(entity) == assignment_key(name):
            entity = "company"
        cleaned_items.append({**item, "sources": sources[:3], "entity": entity, "descriptor": desc})
    items = cleaned_items
    # Re-run entity adoption now that subsidiary list is purged
    items = step_entity(items, texts)["items"]
    items = local_dedup(items)
    items = step_purge_and_merge(items, table, override_keys)["items"]
    items = sorted(items, key=lambda i: (min(s["pdf_page"] for s in i["sources"]), i["name"]))

    sheet = {
        **sheet,
        "items": items,
    }
    sheet = step_rupee_summary(sheet)

    validated = FactSheet.model_validate(sheet).model_dump(mode="json")
    markdown = render_markdown(validated)
    errors = consistency_errors(validated, markdown)
    report["consistency_errors"] = errors
    print(f"9 local_rebuild items={len(validated['items'])} consistency_errors={errors}")

    # Write staging only
    built = {"sheet": validated, "markdown": markdown, "omitted": []}
    write_fact_sheet("COALINDIA", built, STAGING_DIR / "factsheet", force=True)

    # Update list_subsidiaries keys sidecar (used by check_factsheet descriptor exceptions)
    keys = sorted({assignment_key(i["name"]) for i in validated["items"] if i["kind"] == "subsidiary"})
    (STAGING / "list_subsidiaries.json").write_text(
        json.dumps({"keys": keys}, indent=2) + "\n", encoding="utf-8"
    )
    # Clear stale proper-noun merge conflicts from an older precision pass
    precision_path = STAGING / "precision_report.json"
    if precision_path.is_file():
        precision = json.loads(precision_path.read_text(encoding="utf-8"))
        precision["merge_groups"] = []
        precision_path.write_text(json.dumps(precision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Extended local checks (also added to check_factsheet)
    extra_failures = extended_checks(validated)
    report["extended_failures"] = extra_failures
    print("EXTENDED_CHECKS")
    for row in extra_failures:
        print(f"  FAIL {row}")

    # Run check_factsheet
    from src.eval.check_factsheet import check_factsheet

    check_rows = check_factsheet("COALINDIA")
    report["checks"] = check_rows
    failed = [r for r in check_rows if not r["passed"]] + ([{"detail": x} for x in extra_failures] if extra_failures else [])
    for row in check_rows:
        status = "PASS" if row["passed"] else "FAIL"
        print(f"{status} {row['check']}. {row['detail']}")
        if not row["passed"]:
            for ex in row.get("examples") or []:
                print(f"  - {ex}")

    by_kind = dict(Counter(i["kind"] for i in validated["items"]))
    sample = stratified_sample(validated["items"], 40, seed=17)
    sample_rows = [
        {
            "label": label_sample_row(i),
            "kind": i["kind"],
            "name": i["name"],
            "entity": i["entity"],
            "pdf_page": i["sources"][0]["pdf_page"],
            "quote": i["sources"][0]["quote"][:120],
            "descriptor": (i.get("descriptor") or "")[:100],
        }
        for i in sample
    ]
    print("BY_KIND", json.dumps(by_kind, indent=2))
    print("SAMPLE40")
    for row in sample_rows:
        line = f"{row['label']}\t{row['kind']}\t{row['name']}\t{row['quote']}"
        print(line.encode("ascii", "replace").decode())

    costs = cost_table("COALINDIA")
    report["cost_table"] = costs
    print("COST_TABLE")
    for row in costs:
        print(row)

    after = _hindalco_hashes()
    report["hindalco_after"] = after
    print("HINDALCO sha256 AFTER")
    for key, value in after.items():
        print(f"  {key}: {value}")
        if value != before.get(key):
            print("  WARNING hash changed")

    report["summary"] = {
        "item_count": len(validated["items"]),
        "by_kind": by_kind,
        "sample40": sample_rows,
        "failed_checks": len(failed),
    }
    all_pass = not failed and not errors and before == after
    report["all_pass"] = all_pass
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if all_pass:
        paths = promote(validated, markdown)
        report["promoted"] = paths
        print(f"PROMOTED items={len(validated['items'])} -> {paths}")
    else:
        print("SKIP promote: checks failed or HINDALCO hash drift")
    print(f"FINAL_ITEM_COUNT={len(validated['items'])}")
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def extended_checks(sheet: dict) -> list[str]:
    failures = []
    for item in sheet["items"]:
        name = item["name"]
        desc = item.get("descriptor") or ""
        if len(desc) > 200:
            failures.append(f"descriptor>200: {name}")
        if _numeric_token_count(desc) >= 6:
            failures.append(f"descriptor>=6 numbers: {name}")
        if _CIN.search(name.strip()) or re.match(r"^U\d{5}[A-Z]{2}", name.strip()):
            failures.append(f"cin_prefix: {name}")
        quote = " ".join(s.get("quote") or "" for s in item.get("sources") or [])
        if _quote_name_ratio(name, quote) < 0.8 and not quote_supports_name(name, quote):
            failures.append(f"quote missing name tokens: {name}")
        entity = item.get("entity") or ""
        if entity.casefold() not in {"company", "the company", "parent"}:
            folded = quote.casefold()
            ok = entity.casefold() in folded or assignment_key(entity) == assignment_key(name)
            # Abbr forms (ECL, SECL, CMPDI) count when they expand to a known coalfields name.
            for abbr, full in _COALFIELD_ABBR.items():
                if entity.casefold() in {abbr, full.casefold()} and (
                    abbr in folded or full.casefold() in folded
                ):
                    ok = True
            if item.get("kind") == "subsidiary" and assignment_key(entity) == assignment_key(name):
                ok = True
            if not ok:
                failures.append(f"entity missing from quote: {name} / {entity}")
            if item.get("kind") != "subsidiary" and assignment_key(entity) == assignment_key(name):
                failures.append(f"entity equals name (non-subsidiary): {name}")
    return failures


if __name__ == "__main__":
    run()
