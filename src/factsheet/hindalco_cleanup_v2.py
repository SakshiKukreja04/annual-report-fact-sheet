"""Tighten HINDALCO sheet from the current output + cache. No Groq. Copy-first."""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from rapidfuzz import fuzz

from src.config.paths import CANDIDATE_DIR, INPUT_DIR, OUTPUT_DIR, PROJECT_ROOT
from src.dedup.normalize import normalize
from src.extract.evidence import pages_for_extraction
from src.extract.run import load_saved_company
from src.extract.schema import KINDS, STATUSES
from src.extract.select import page_text
from src.factsheet.build import consistency_errors, render_markdown
from src.factsheet.hindalco_final_cache import (
    _atomic,
    _atomic_text,
    _grounded,
    _is_prose_sentence,
    _pdf_layer_meta,
    _prose_overview,
    load_cached_candidates,
)
from src.factsheet.postprocess import _kind_rank, _name_in_quote_or_sentence, _printed_for_quote
from src.factsheet.schema import FactSheet
from src.ingestion.audit import locate_pdfs
from src.tables.digital import extract_digital_page
from src.validate.quotes import fold_whitespace, locate_quote, name_tokens, repair_named_quote

COPY_DIR = PROJECT_ROOT / "eval" / "hindalco_cleanup_v2"
OUT_JSON = OUTPUT_DIR / "HINDALCO" / "HINDALCO.json"
OUT_MD = OUTPUT_DIR / "HINDALCO" / "HINDALCO.md"

_LIST_HEADING = re.compile(
    r"form\s*aoc-?\s*1|statement containing salient features|"
    r"investment in subsidiaries|interests in other entities|"
    r"subsidiaries,\s*associates and joint|name of the (?:holding\s*/\s*)?subsidiary|"
    r"financial information regarding subsidiaries|"
    r"group information|investment in equity shares.*subsidiar",
    re.I,
)
_BAD_SUB_PAGE = re.compile(
    r"equity.?investments?\s+(?:measured\s+)?at\s+fair value|investments?\s+in\s+equity\s+instruments|"
    r"shares held by promoters|promoter.?group|shareholding pattern|"
    r"relationship with struck off|gross amount set\s*off|master netting|"
    r"registrar|depository|scrutinizer|independent auditor",
    re.I,
)
_FALSE_SUB = re.compile(
    r"\b(stock exchange|registrar|depository|auditor|consultant|customer|supplier|"
    r"utility|electricity|power corporation|vidyut|nigam|bank\b|sebi|"
    r"chartered accountant|cost accountant)\b",
    re.I,
)
_FRAGMENT_START = re.compile(
    r"^(private|pvt\.?|ltd\.?|limited|finance|infrastructure|minerals|industries|retail|technology|company)\b",
    re.I,
)
_BAD_PREFIX = re.compile(r"^(at|work|adopt|for|our|c)\b", re.I)
_SUB_SITE_END = re.compile(r"\b(unit|plant|mine|division)\s*$", re.I)
_GLOSSARY = re.compile(r"^(cetp|etp|twru|rcmu|pmcp|potf|ual)\b", re.I)
_SOLAR = re.compile(r"\b(solar|wind|hybrid)\b.*\b(project|plant|capacity|park)\b|\b(project|plant)\b.*\b(solar|wind|hybrid)\b", re.I)
_LAB = re.compile(
    r"abstc|kennesaw|sierre|spokane|innovation centr|science and technology|research (?:and|&) development centr|r&d centr|laboratory|lab\b",
    re.I,
)
_HR_DROP = re.compile(
    r"\b(hr\b|training|leave|engagement|recognition|grievance|pra\b|consultation|"
    r"wage|beneficiary|coursera|helpline|edi\b|platform|risk|safety|digital|"
    r"biodiversity|hrdd|human rights due diligence|"
    r"esg|metric|target|framework|policy|committee|energy audit|carbon pric|"
    r"waste|agm|depository|instavote|dividend|"
    r"occupational health|curiosity carnival|long-?term settlement|"
    r"internal audits?|learning and development|industrial relations)\b",
    re.I,
)
_CSR_KEEP = re.compile(r"\b(foundation|csr|community development|tamrodaya|education|hospital|school)\b", re.I)
_CHANNEL_KEEP = re.compile(r"\b(distributor|distribution|dealer|retail|sales channel|b2b|b2c|export)\b", re.I)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\t]")


def nearest_heading(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for ln in lines[:15]:
        if len(ln) >= 8:
            return ln[:160]
    return lines[0][:160] if lines else ""


def identify_list_pages(pages: list[dict]) -> dict:
    texts = {int(p["pdf_page"]): page_text(p) for p in pages}
    list_pages = set()
    equity_pages = set()
    promoter_pages = set()
    for number, text in texts.items():
        head = nearest_heading(text)
        blob = f"{head}\n{(text or '')[:2500]}"
        if _LIST_HEADING.search(blob) and not _BAD_SUB_PAGE.search(blob[:800]):
            list_pages.add(number)
        if re.search(r"equity.?investments?\s+(?:measured\s+)?at\s+fair value|investments?\s+in\s+equity\s+instruments", blob, re.I):
            equity_pages.add(number)
        if re.search(r"shares held by promoters|promoter.?group|shareholding pattern", blob, re.I):
            promoter_pages.add(number)
    # Force-known good pages from inspection / user hints.
    for n in (174, 179, 234, 248, 251, 252, 253, 261, 262, 263, 264, 265, 333, 343, 346, 356, 357, 391, 392, 393, 394):
        if n in texts and _LIST_HEADING.search(texts[n][:3000] + nearest_heading(texts[n])):
            list_pages.add(n)
        elif n in texts and re.search(r"subsidiary|joint venture|associate", texts[n][:2000], re.I):
            if not _BAD_SUB_PAGE.search(texts[n][:1000]):
                list_pages.add(n)
    # Explicit exclusions.
    for n in (180, 199, 200, 201, 202, 203, 204, 205, 233, 293, 331, 370, 403):
        list_pages.discard(n)
    return {
        "list_pages": sorted(list_pages),
        "equity_pages": sorted(equity_pages),
        "promoter_pages": sorted(promoter_pages),
        "page_331": {"heading": nearest_heading(texts.get(331, "")), "preview": (texts.get(331, "") or "")[:400]},
        "page_403": {"heading": nearest_heading(texts.get(403, "")), "preview": (texts.get(403, "") or "")[:400]},
        "texts": texts,
    }


def full_legal_names_on_lists(texts: dict[int, str], list_pages: list[int]) -> set[str]:
    """Collect full legal corporate names appearing on subsidiary list pages."""
    corp = re.compile(
        r"\b([A-Z][A-Za-z0-9&.'’\-]*(?:\s+(?:&|[A-Z][A-Za-z0-9&.'’\-]*)){0,12}"
        r"\s+(?:Limited|Ltd\.?|Inc\.?|LLC|GmbH|Private Limited|Pvt\.?\s*Limited|Company Limited))\b"
    )
    names = set()
    for number in list_pages:
        text = texts.get(number, "") or ""
        for match in corp.finditer(text):
            name = " ".join(match.group(1).split())
            names.add(fold_whitespace(name).casefold())
            names.add(normalize(name))
        # Also keep Novelis Inc / Novelis Korea etc. lines from tables without capturing junk.
        for line in text.splitlines():
            if re.search(r"\b(subsidiary|associate|joint venture|wholly[- ]owned)\b", line, re.I) or re.search(
                r"\b(Limited|Ltd\.?|Inc\.?)\b", line
            ):
                for match in corp.finditer(line):
                    name = " ".join(match.group(1).split())
                    names.add(fold_whitespace(name).casefold())
                    names.add(normalize(name))
    # Explicit core parent allowed when full legal form is on a list page.
    if any("novelis inc" in (texts.get(n, "") or "").casefold() for n in list_pages):
        names.add("novelis inc")
        names.add(normalize("Novelis Inc"))
    return names


def step1_subsidiaries(items: list[dict], meta: dict) -> dict:
    texts = meta["texts"]
    list_pages = set(meta["list_pages"])
    legal = full_legal_names_on_lists(texts, meta["list_pages"])
    kept, dropped, provenance = [], [], []
    for item in items:
        if item.get("kind") != "subsidiary":
            kept.append(item)
            continue
        pages = [s.get("pdf_page") for s in item.get("sources") or [] if isinstance(s.get("pdf_page"), int)]
        heads = []
        for n in pages:
            heads.append({"pdf_page": n, "heading": nearest_heading(texts.get(n, ""))})
        provenance.append({"name": item.get("name"), "pages": heads})
        name = item.get("name") or ""
        key = fold_whitespace(name).casefold()
        norm = normalize(name)
        # Full legal name must appear on a real list page.
        on_list = key in legal or norm in legal
        # Allow exact line presence of full name on list pages.
        if not on_list:
            for n in list_pages:
                if name and name in (texts.get(n, "") or ""):
                    on_list = True
                    break
                if fold_whitespace(name).casefold() in fold_whitespace(texts.get(n, "") or "").casefold() and re.search(
                    r"\b(Limited|Ltd\.?|Inc\.?|LLC|GmbH)\b", name, re.I
                ):
                    on_list = True
                    break
        if pages and all(n in {331, 403} for n in pages):
            # Only keep if heading shows subsidiary context.
            if not any(_LIST_HEADING.search(texts.get(n, "") or "") for n in pages):
                dropped.append(_drop(item, "subsidiary_page_331_or_403_not_list"))
                continue
        if any(n in meta["equity_pages"] or n in {199, 200, 201, 202, 203, 204, 205} for n in pages) and not on_list:
            dropped.append(_drop(item, "subsidiary_equity_or_director_page"))
            continue
        if any(n in meta["promoter_pages"] or n in {293, 370} for n in pages) and not on_list:
            dropped.append(_drop(item, "subsidiary_promoter_shareholding_page"))
            continue
        if _FALSE_SUB.search(name) or _FALSE_SUB.search(item.get("descriptor") or ""):
            dropped.append(_drop(item, "subsidiary_false_entity"))
            continue
        if not on_list:
            dropped.append(_drop(item, "subsidiary_not_on_list_pages"))
            continue
        if not re.search(r"\b(Limited|Ltd\.?|Inc\.?|LLC|GmbH|Private Limited)\b", name, re.I):
            # Require full legal name form.
            dropped.append(_drop(item, "subsidiary_not_full_legal_name"))
            continue
        # Source pages must include at least one real list page (not only BRSR product/equity pages).
        if pages and not any(n in list_pages for n in pages):
            dropped.append(_drop(item, "subsidiary_source_not_list_page"))
            continue
        kept.append(item)
    return {"items": kept, "dropped": dropped, "provenance": provenance, "legal_name_count": len(legal)}


def step2_fragments(items: list[dict]) -> dict:
    kept, dropped = [], []
    for item in items:
        name = (item.get("name") or "").strip()
        reason = None
        if _GLOSSARY.search(name):
            reason = "glossary_abbrev"
        elif _FRAGMENT_START.search(name):
            reason = "name_fragment_stem"
        elif _BAD_PREFIX.search(name):
            reason = "name_bad_prefix"
        elif item.get("kind") == "subsidiary" and _SUB_SITE_END.search(name):
            reason = "subsidiary_site_ending"
        if reason:
            dropped.append(_drop(item, reason))
            continue
        kept.append(item)
    return {"items": kept, "dropped": dropped}


def step3_allowlist(items: list[dict]) -> dict:
    kept, dropped = [], []
    for item in items:
        kind = item.get("kind")
        name = item.get("name") or ""
        desc = item.get("descriptor") or ""
        quote = " ".join(s.get("quote") or "" for s in item.get("sources") or [])
        blob = f"{name} {desc} {quote}"
        if kind == "other":
            if _SOLAR.search(blob):
                kept.append({**item, "kind": "asset"})
            else:
                dropped.append(_drop(item, "drop_other"))
            continue
        if kind == "research_platform":
            if _LAB.search(blob) and not _HR_DROP.search(name):
                kept.append(item)
            else:
                dropped.append(_drop(item, "research_not_lab"))
            continue
        if kind == "philanthropy":
            if _CSR_KEEP.search(blob) and not re.search(r"grievance|pra\b|consultation|wage|beneficiary", blob, re.I):
                kept.append(item)
            else:
                dropped.append(_drop(item, "philanthropy_not_named_csr"))
            continue
        if kind == "channel":
            if _CHANNEL_KEEP.search(blob) and not re.search(r"coursera|edi\b|helpline|platform", blob, re.I):
                kept.append(item)
            else:
                dropped.append(_drop(item, "channel_not_sales"))
            continue
        if kind == "segment":
            # Filtered again in step5; keep candidates for now if reportable-looking.
            if re.search(r"novelis|aluminium upstream|aluminum upstream|aluminium downstream|aluminum downstream|\bcopper\b|others|power", name, re.I):
                kept.append(item)
            else:
                dropped.append(_drop(item, "segment_not_reportable"))
            continue
        if kind in {"business_line", "product_or_service", "asset", "brand"}:
            if _HR_DROP.search(blob) and not re.search(r"\b(aluminium|aluminum|copper|alumina|novelis|plant|smelter|refinery|brand|foil|extrusion)\b", blob, re.I):
                dropped.append(_drop(item, f"{kind}_hr_esg_policy"))
                continue
            if re.search(r"\b(agm|depository|instavote|dividend|committee|framework|policy|carbon pric|energy audit)\b", blob, re.I):
                dropped.append(_drop(item, f"{kind}_admin_esg"))
                continue
            kept.append(item)
            continue
        kept.append(item)
    return {"items": kept, "dropped": dropped}


def step4_restore_and_dedup(items: list[dict], pages_by: dict, texts: dict[int, str]) -> dict:
    """Restore core items from cache; fix Novelis/Copper Business dedup loss."""
    cache = load_cached_candidates("HINDALCO")
    by_name = {fold_whitespace(i.get("name") or "").casefold(): i for i in items}
    restored = []
    notes = []

    def add_or_replace(candidate: dict, *, reason: str):
        nonlocal items
        shaped = _shape(candidate)
        if not shaped:
            notes.append({"name": candidate.get("name"), "reason": "restore_failed_shape"})
            return
        key = fold_whitespace(shaped["name"]).casefold()
        # Remove fuzzy-wrong merges that swallowed this name.
        items = [i for i in items if not _ate_core(i.get("name") or "", shaped["name"])]
        by_name[key] = shaped
        items.append(shaped)
        restored.append({"name": shaped["name"], "reason": reason, "kind": shaped["kind"]})

    # Novelis / Novelis Inc → one subsidiary with full legal name (avoids clash with segment "Novelis").
    novelis = _find_cache(cache["all"], name_equals="Novelis", prefer_page=44) or _find_cache(cache["all"], name_equals="Novelis", prefer_page=38)
    novelis_inc = _find_cache(cache["all"], name_contains="Novelis Inc")
    p44 = _find_cache(cache["all"], name_equals="Novelis", prefer_page=44)
    base = novelis_inc or novelis
    if base:
        descriptor = (p44 or novelis or base).get("descriptor") or "wholly owned subsidiary; sustainable aluminium solutions provider"
        sources = (p44 or novelis or base).get("sources") or base.get("sources")
        # Prefer a Novelis Inc quote from list pages when present.
        if novelis_inc and novelis_inc.get("sources"):
            sources = list(sources or []) + list(novelis_inc.get("sources") or [])
        restored_novelis = {
            **base,
            "kind": "subsidiary",
            "name": "Novelis Inc",
            "descriptor": descriptor,
            "entity": "Novelis Inc",
            "sources": sources,
        }
        # Drop short "Novelis" subsidiary rows so they merge into Novelis Inc.
        items = [i for i in items if not (i.get("kind") == "subsidiary" and fold_whitespace(i.get("name") or "").casefold() in {"novelis", "novelis inc", "novelis inc."})]
        add_or_replace(restored_novelis, reason="restore_Novelis_Inc")
        notes.append(
            {
                "bug": "Novelis lost because fuzz.token_set_ratio merged short 'Novelis' into longer 'Novelis UK Ltd' (token subset). Fixed by forbidding proper-subset merges; restored as Novelis Inc (full legal name) with p44 descriptor."
            }
        )

    restores = [
        ("Copper Business", "business_line", "Copper Business"),
        ("Alumina Business", "business_line", "Alumina Business"),
        ("AL:sust", "brand", "AL:sust"),
    ]
    for target, kind, nice in restores:
        found = _find_cache(cache["all"], name_equals=target) or _find_cache(cache["all"], name_contains=target)
        if not found:
            continue
        item = {**found, "name": nice, "kind": kind}
        add_or_replace(item, reason=f"restore_{nice}")
    # Aluminium upstream comes from segment note rebuild in step5; avoid duplicate restore here.

    # Pakhajan e-waste
    pak = _find_cache(cache["all"], name_contains="Pakhajan") or _find_cache(cache["all"], name_contains="e-waste recycling project")
    if pak:
        item = {
            **pak,
            "name": "Pakhajan e-waste recycling project",
            "kind": "asset",
            "entity": "company",
            "descriptor": pak.get("descriptor") or "e-waste recycling project of 50 KTPA at Pakhajan, Gujarat",
        }
        add_or_replace(item, reason="restore_Pakhajan")

    # Split Eternia and Totalis
    combo = next((i for i in items if "eternia" in (i.get("name") or "").casefold() and "totalis" in (i.get("name") or "").casefold()), None)
    cache_combo = _find_cache(cache["all"], name_contains="Eternia and Totalis")
    source_item = combo or cache_combo
    if source_item:
        items = [i for i in items if i is not combo]
        quote = ((source_item.get("sources") or [{}])[0].get("quote")) or "Eternia windows and façades, and Totalis"
        page = ((source_item.get("sources") or [{}])[0].get("pdf_page")) or 65
        for brand in ("Eternia", "Totalis"):
            add_or_replace(
                {
                    "name": brand,
                    "kind": "brand",
                    "descriptor": f"{brand} home brand",
                    "entity": "company",
                    "status": "current",
                    "sources": [{"pdf_page": page, "printed_page": None, "quote": quote if brand.casefold() in quote.casefold() else brand}],
                },
                reason=f"split_{brand}",
            )

    # Copper e-waste plant entity = company
    for item in items:
        if re.search(r"copper.*e-?waste|e-?waste.*copper|pakhajan", (item.get("name") or ""), re.I):
            item["entity"] = "company"
            item["kind"] = "asset" if item.get("kind") in {"other", "asset", "product_or_service"} else item.get("kind")

    # Dedup with fixed rule.
    deduped, dropped = dedupe_fixed(items)
    return {"items": deduped, "dropped": dropped, "restored": restored, "notes": notes}


def _ate_core(existing: str, core: str) -> bool:
    """True if existing is a longer Novelis-* that wrongly replaced core Novelis."""
    a = fold_whitespace(existing).casefold()
    b = fold_whitespace(core).casefold()
    if a == b:
        return True
    if b == "novelis" and a.startswith("novelis ") and a != "novelis inc" and a != "novelis inc.":
        # Keep Novelis country subs; only remove exact duplicate Novelis rows.
        return a == "novelis"
    return False


def dedupe_fixed(items: list[dict]) -> tuple[list[dict], list[dict]]:
    groups: list[list[dict]] = []
    dropped = []
    for item in items:
        placed = False
        for group in groups:
            if same_item_fixed(group[0], item):
                group.append(item)
                placed = True
                break
        if not placed:
            groups.append([item])
    kept = []
    for group in groups:
        if len(group) == 1:
            kept.append(group[0])
            continue
        kept.append(merge_group(group))
        for extra in group[1:]:
            dropped.append(_drop(extra, "dedup_merged_fixed"))
    # Collapse exact casefold duplicates.
    by_key: dict[str, dict] = {}
    order = []
    final_dropped = list(dropped)
    for item in kept:
        key = (item.get("name") or "").casefold()
        if key not in by_key:
            by_key[key] = item
            order.append(key)
        else:
            by_key[key] = merge_group([by_key[key], item])
            final_dropped.append(_drop(item, "duplicate_name_collapsed"))
    return [by_key[k] for k in order], final_dropped


def same_item_fixed(left: dict, right: dict) -> bool:
    if (left.get("kind") or "") != (right.get("kind") or ""):
        return False
    a = _key(left.get("name") or "")
    b = _key(right.get("name") or "")
    if not a or not b:
        return False
    if a == b:
        return True
    ta, tb = a.split(), b.split()
    sa, sb = set(ta), set(tb)
    # Dedup bug fix: do not merge when one token set is a proper subset of the other
    # (Novelis ⊂ Novelis UK Ltd). Allow only legal-suffix-only differences.
    if sa < sb or sb < sa:
        shorter, longer = (sa, sb) if sa < sb else (sb, sa)
        extras = longer - shorter
        legal = {"limited", "ltd", "inc", "incorporated", "private", "pvt", "llc", "gmbh"}
        return bool(extras) and extras <= legal
    score_set = fuzz.token_set_ratio(a, b)
    score_sort = fuzz.token_sort_ratio(a, b)
    return score_set >= 90 and score_sort >= 90


def _key(name: str) -> str:
    text = re.sub(r"[™®©]", "", name or "")
    text = text.casefold().replace("&", " and ")
    text = re.sub(r"task\s*force", "taskforce", text)
    text = re.sub(r"\b(limited|ltd|pvt|private|inc|incorporated)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    tokens = []
    for word in text.split():
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        tokens.append(word)
    return " ".join(tokens)  # preserve order (sort was making subset merges worse for display only; comparison uses sets above)


def merge_group(members: list[dict]) -> dict:
    base = dict(members[0])
    sources = []
    for member in members:
        if _kind_rank(member.get("kind")) > _kind_rank(base.get("kind")):
            base["kind"] = member.get("kind")
        if len(member.get("descriptor") or "") > len(base.get("descriptor") or ""):
            base["descriptor"] = member.get("descriptor")
        # Prefer shorter core name for Novelis parent over Novelis Inc spelling variants only.
        if normalize(member.get("name") or "") == normalize(base.get("name") or "") and len(member.get("name") or "") < len(base.get("name") or ""):
            base["name"] = member["name"]
        for source in member.get("sources") or []:
            if source not in sources:
                sources.append(source)
            if len(sources) >= 3:
                break
    base["sources"] = sources[:3]
    return base


def step5_segments(items: list[dict], pages: list[dict]) -> dict:
    pdf = locate_pdfs(INPUT_DIR).get("HINDALCO")
    tables = []
    if pdf:
        for number in (319, 320, 321):
            try:
                tables.extend(extract_digital_page(Path(pdf), "HINDALCO", number))
            except Exception as exc:
                tables.append({"error": str(exc), "pdf_page": number})
    allowed = {
        "novelis": "Novelis",
        "aluminium upstream": "Aluminium upstream",
        "aluminum upstream": "Aluminium upstream",
        "aluminium downstream": "Aluminium downstream",
        "aluminum downstream": "Aluminium downstream",
        "copper": "Copper",
        "others": "Others",
        "power": "Power",
    }
    revenue = {}
    total = None
    for table in tables:
        for row in table.get("rows") or []:
            cells = [fold_whitespace(c or "") for c in (row.get("cells") or [])]
            if not cells:
                continue
            label = cells[0].casefold()
            for key, nice in allowed.items():
                if key in label and re.search(rf"\b{re.escape(key)}\b", label):
                    nums = []
                    for cell in cells[1:]:
                        m = re.search(r"([\d,]+(?:\.\d+)?)", cell.replace(",", ""))
                        if m:
                            try:
                                nums.append(float(m.group(1).replace(",", "")))
                            except ValueError:
                                pass
                    if nums:
                        revenue[nice] = nums[0]
            if "total" in label and "segment" in label or label.strip() in {"total", "total revenue"}:
                nums = []
                for cell in cells[1:]:
                    m = re.search(r"([\d,]+(?:\.\d+)?)", cell.replace(",", ""))
                    if m:
                        nums.append(float(m.group(1).replace(",", "")))
                if nums:
                    total = nums[0]
    # Fallback from page text if tables thin.
    texts = {int(p["pdf_page"]): page_text(p) for p in pages}
    blob = "\n".join(texts.get(n, "") for n in (319, 320, 321))
    for key, nice in allowed.items():
        if nice in revenue:
            continue
        m = re.search(rf"{re.escape(key)}[^\n]{{0,40}}?([\d,]{{2,}}(?:\.\d+)?)", blob, re.I)
        if m:
            revenue[nice] = float(m.group(1).replace(",", ""))
    if total is None:
        m = re.search(r"total\s+(?:segment\s+)?revenue[^\n]{0,40}?([\d,]{2,}(?:\.\d+)?)", blob, re.I)
        if m:
            total = float(m.group(1).replace(",", ""))

    # Remove non-segment items with kind=segment.
    kept = [i for i in items if i.get("kind") != "segment"]
    dropped = [_drop(i, "segment_removed_non_reportable") for i in items if i.get("kind") == "segment"]
    segment_rows = []
    for nice in ("Novelis", "Aluminium upstream", "Aluminium downstream", "Copper", "Others", "Power"):
        if nice not in revenue and nice in {"Others", "Power"}:
            continue
        if nice not in revenue and nice not in {"Novelis", "Aluminium upstream", "Aluminium downstream", "Copper"}:
            continue
        # Always include the four core if mentioned in note text even without parsed revenue.
        if nice not in revenue and not re.search(nice.casefold(), blob.casefold()):
            if nice in {"Others", "Power"}:
                continue
        share = None
        if total and nice in revenue and total > 0:
            share = round(100.0 * float(revenue[nice]) / float(total), 2)
        segment_rows.append(
            {
                "name": nice,
                "revenue": revenue.get(nice),
                "unit": "INR crore" if nice in revenue else None,
                "share_pct": share,
            }
        )
        # Ensure matching item exists.
        quote = _segment_quote(nice, blob)
        kept.append(
            {
                "name": nice,
                "kind": "segment",
                "descriptor": f"{nice} reportable segment",
                "entity": "company",
                "status": "current",
                "revenue_share_pct": share,
                "sources": [
                    {
                        "pdf_page": 320 if nice.casefold() in (texts.get(320, "") or "").casefold() else 319,
                        "printed_page": None,
                        "quote": quote,
                    }
                ],
            }
        )
    return {"items": kept, "dropped": dropped, "segments": segment_rows, "tables_found": len([t for t in tables if t.get("rows")]), "revenue": revenue, "total": total}


def _segment_quote(name: str, blob: str) -> str:
    patterns = {
        "Novelis": r"Novelis segment.{0,40}",
        "Aluminium upstream": r"Aluminium upstream segment.{0,40}",
        "Aluminium downstream": r"Aluminium downstream.{0,40}",
        "Copper": r"copper segment.{0,40}",
        "Others": r"Others.{0,30}",
        "Power": r"Power segment.{0,30}",
    }
    pat = patterns.get(name) or re.escape(name)
    match = re.search(pat, blob, re.I)
    if match:
        return _trim_quote(fold_whitespace(match.group(0)), name, 15)
    return _trim_quote(name, name, 15)


def step6_header(texts: dict[int, str]) -> dict:
    pages = [33, 38, 44, 67]
    blob = "\n".join(texts.get(n, "") for n in pages)
    ranked = []
    for s in re.split(r"(?<=[.!?])\s+", blob):
        flat = fold_whitespace(s)
        if not flat or flat.count("?") > 2:
            continue
        if re.search(r"market capitalisation|statutory reports|financial statements|operating cash flow|integrated annual report|page \d+", flat, re.I):
            continue
        if not _is_prose_sentence(flat):
            continue
        if not re.search(r"aluminium|aluminum|copper|novelis|hindalco", flat, re.I):
            continue
        if not _grounded(flat, blob):
            continue
        score = 0
        if re.search(r"hindalco industries", flat, re.I):
            score += 5
        if re.search(r"aluminium", flat, re.I) and re.search(r"copper", flat, re.I):
            score += 3
        if re.search(r"novelis", flat, re.I):
            score += 2
        if re.search(r"largest|integrated|producer|operates|subsidiary", flat, re.I):
            score += 1
        ranked.append((score, flat if flat.endswith(".") else flat + "."))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    if not ranked:
        one_liner = "Hindalco Industries Limited, the metals flagship of the Aditya Birla Group, is India’s largest fully integrated aluminium player."
        summary = one_liner
        # Only keep if grounded.
        if not _grounded(one_liner, blob):
            # Fall back to any grounded aluminium/copper sentence even if slightly short.
            for s in re.split(r"(?<=[.!?])\s+", blob):
                flat = fold_whitespace(s)
                if len(flat.split()) >= 8 and _grounded(flat, blob) and re.search(r"hindalco|aluminium|novelis", flat, re.I):
                    one_liner = flat if flat.endswith(".") else flat + "."
                    summary = one_liner
                    break
    else:
        one_liner = ranked[0][1]
        summary = " ".join(sentence for _score, sentence in ranked[:3])
    parts = [fold_whitespace(p) for p in re.split(r"(?<=[.!?])\s+", summary) if p.strip()]
    parts = [p if p.endswith(".") else p + "." for p in parts if _grounded(p, blob)]
    summary = " ".join(parts) if parts else one_liner
    return {
        "one_liner": one_liner,
        "summary": summary,
        "pages": pages,
        "grounded_one": _grounded(one_liner, blob),
        "grounded_sum": all(_grounded(p, blob) for p in parts) if parts else _grounded(summary, blob),
    }


def step7_quotes(items: list[dict], pages_by: dict, texts: dict[int, str], pdf_meta: dict) -> dict:
    kept, dropped = [], []
    qcounts = Counter()
    for item in items:
        sources = []
        for source in item.get("sources") or []:
            number = source.get("pdf_page")
            page = pages_by.get(number)
            stored = texts.get(number, "") if isinstance(number, int) else ""
            original = _CONTROL.sub(" ", source.get("quote") or "")
            original = fold_whitespace(original)
            # Prefer a single text-block span containing the name.
            block_quote = _quote_from_blocks(page, item.get("name") or "", original) if page else None
            candidate = block_quote or original
            repaired = repair_named_quote(candidate, stored, item.get("name") or "")
            quote = repaired.get("quote")
            if not quote:
                qcounts["dropped"] += 1
                continue
            quote = fold_whitespace(_CONTROL.sub(" ", quote))
            if "\n" in quote or "\t" in quote:
                quote = fold_whitespace(quote)
            words = quote.split()
            if len(words) > 15:
                # Trim to <=15 words while keeping name if possible.
                quote = _trim_quote(quote, item.get("name") or "", 15)
                words = quote.split()
            if len(words) > 15 or not quote:
                qcounts["dropped"] += 1
                continue
            if not _name_in_quote_or_sentence(item.get("name") or "", quote, stored):
                qcounts["dropped"] += 1
                continue
            # PDF layer verify when present.
            meta = pdf_meta.get(number) if isinstance(number, int) else None
            if meta and meta.get("has_layer"):
                kind = locate_quote(quote, meta.get("text") or "")
                if not kind:
                    alt = repair_named_quote(quote, meta.get("text") or "", item.get("name") or "")
                    if alt.get("quote") and locate_quote(alt["quote"], meta.get("text") or ""):
                        quote = fold_whitespace(alt["quote"])
                        qcounts["repaired"] += 1
                    else:
                        qcounts["dropped"] += 1
                        continue
                elif kind == "exact":
                    qcounts["strict"] += 1
                else:
                    qcounts["normalized"] += 1
            else:
                kind = locate_quote(quote, stored)
                if kind == "exact":
                    qcounts["strict"] += 1
                elif kind:
                    qcounts["normalized"] += 1
                else:
                    qcounts["dropped"] += 1
                    continue
            sources.append({**source, "quote": quote})
        if not sources:
            dropped.append(_drop(item, "no_valid_short_quote"))
            continue
        kept.append({**item, "sources": sources[:3]})
    return {"items": kept, "dropped": dropped, "quotes": dict(qcounts)}


def step8_printed(items: list[dict], pages: list[dict]) -> list[dict]:
    by = {int(p["pdf_page"]): p for p in pages}
    # Expected printed ≈ linear with pdf_page using neighbors.
    pairs = []
    for page in pages:
        printed = page.get("printed_page")
        num = _first_int(printed)
        if num is not None:
            pairs.append((int(page["pdf_page"]), num))
    updated = []
    for item in items:
        sources = []
        for source in item.get("sources") or []:
            number = source.get("pdf_page")
            page = by.get(number)
            printed = _printed_for_quote(page, source.get("quote") or "") if page else source.get("printed_page")
            value = _first_int(printed)
            expected = _expected_printed(number, pairs)
            if value is None or value == 0:
                printed = None
            elif expected is not None and abs(value - expected) > 3:
                printed = None
            else:
                printed = str(value)
            sources.append({**source, "printed_page": printed})
        updated.append({**item, "sources": sources})
    return updated


def run() -> dict:
    COPY_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {"steps": {}}
    # Prefer the pre-cleanup 503-item copy when present so counts cover the full pass.
    baseline = PROJECT_ROOT / "eval" / "hindalco_final_build" / "HINDALCO.copy.json"
    source_path = baseline if baseline.is_file() else OUT_JSON
    original = json.loads(source_path.read_text(encoding="utf-8"))
    items = list(original.get("items") or [])
    report["source_path"] = str(source_path)
    # Track origin roughly.
    for item in items:
        item.setdefault("_origin", "sheet")
    report["steps"]["start"] = {"items": len(items), "kinds": dict(Counter(i.get("kind") for i in items))}
    print(f"START items={len(items)}")

    pages = pages_for_extraction(load_saved_company("HINDALCO"))
    pages_by = {int(p["pdf_page"]): p for p in pages}
    meta = identify_list_pages(pages)
    report["steps"]["list_pages"] = {
        "list_pages": meta["list_pages"],
        "equity_pages": meta["equity_pages"],
        "promoter_pages": meta["promoter_pages"],
        "page_331": meta["page_331"],
        "page_403": meta["page_403"],
    }
    print("LIST_PAGES", meta["list_pages"])
    print("PAGE_331", meta["page_331"]["heading"].encode("ascii", "replace").decode("ascii"))
    print("PAGE_403", meta["page_403"]["heading"].encode("ascii", "replace").decode("ascii"))

    # 1 subsidiaries
    s1 = step1_subsidiaries(items, meta)
    # Write subsidiary provenance sample
    (COPY_DIR / "subsidiary_provenance.json").write_text(json.dumps(s1["provenance"], indent=2, ensure_ascii=False), encoding="utf-8")
    items = s1["items"]
    dropped_all = list(s1["dropped"])
    report["steps"]["1_subsidiaries"] = {
        "kept": len(items),
        "dropped": len(s1["dropped"]),
        "legal_names": s1["legal_name_count"],
        "dropped_by": dict(Counter(d["reason"] for d in s1["dropped"])),
    }
    print(f"1 subsidiaries kept={len(items)} dropped={len(s1['dropped'])} {report['steps']['1_subsidiaries']['dropped_by']}")

    # 2 fragments
    s2 = step2_fragments(items)
    items = s2["items"]
    dropped_all.extend(s2["dropped"])
    report["steps"]["2_fragments"] = {"kept": len(items), "dropped": len(s2["dropped"]), "dropped_by": dict(Counter(d["reason"] for d in s2["dropped"]))}
    print(f"2 fragments kept={len(items)} dropped={len(s2['dropped'])}")

    # 3 allowlist
    s3 = step3_allowlist(items)
    items = s3["items"]
    dropped_all.extend(s3["dropped"])
    report["steps"]["3_allowlist"] = {"kept": len(items), "dropped": len(s3["dropped"]), "dropped_by": dict(Counter(d["reason"] for d in s3["dropped"]))}
    print(f"3 allowlist kept={len(items)} dropped={len(s3['dropped'])} {report['steps']['3_allowlist']['dropped_by']}")

    # 4 restore + dedup fix
    s4 = step4_restore_and_dedup(items, pages_by, meta["texts"])
    items = s4["items"]
    dropped_all.extend(s4["dropped"])
    report["steps"]["4_restore_dedup"] = {
        "kept": len(items),
        "dropped": len(s4["dropped"]),
        "restored": s4["restored"],
        "notes": s4["notes"],
    }
    print(f"4 restore/dedup kept={len(items)} restored={s4['restored']}")

    # 5 segments
    s5 = step5_segments(items, pages)
    items = s5["items"]
    dropped_all.extend(s5["dropped"])
    report["steps"]["5_segments"] = {
        "kept_items": len(items),
        "segments": s5["segments"],
        "tables_found": s5["tables_found"],
        "revenue": s5["revenue"],
        "total": s5["total"],
        "dropped": len(s5["dropped"]),
    }
    print(f"5 segments rows={len(s5['segments'])} tables={s5['tables_found']} revenue={s5['revenue']}")

    # 6 header
    header = step6_header(meta["texts"])
    report["steps"]["6_header"] = header
    print("6 header", header["one_liner"].encode("ascii", "replace").decode("ascii")[:160])

    # 8 printed before quote trim uses page records; do after quotes too.
    # 7 quotes
    cited = {s["pdf_page"] for i in items for s in i.get("sources") or [] if isinstance(s.get("pdf_page"), int)}
    pdf_meta = _pdf_layer_meta(cited)
    s7 = step7_quotes(items, pages_by, meta["texts"], pdf_meta)
    items = s7["items"]
    dropped_all.extend(s7["dropped"])
    report["steps"]["7_quotes"] = {"kept": len(items), "dropped": len(s7["dropped"]), "quotes": s7["quotes"]}
    print(f"7 quotes kept={len(items)} dropped={len(s7['dropped'])} {s7['quotes']}")

    # 8 printed_page
    items = step8_printed(items, pages)
    report["steps"]["8_printed"] = {"kept": len(items)}
    print(f"8 printed kept={len(items)}")

    # Shape + validate
    shaped = []
    for item in items:
        row = _shape(item)
        if row:
            shaped.append(row)
    shaped = sorted(shaped, key=lambda i: (min(s["pdf_page"] for s in i["sources"]), i["name"]))
    sheet = {
        "ticker": "HINDALCO",
        "period": "FY25",
        "one_liner": header["one_liner"],
        "summary": header["summary"],
        "segments": s5["segments"],
        "items": shaped,
    }
    validated = FactSheet.model_validate(sheet).model_dump(mode="json")
    markdown = render_markdown(validated)
    errors = consistency_errors(validated, markdown)
    names = [i["name"] for i in validated["items"]]
    dupes = [n for n, c in Counter(n.casefold() for n in names).items() if c > 1]
    quote_bad = [i["name"] for i in validated["items"] if not (1 <= len(i["sources"]) <= 3)]
    ok = not dupes and not quote_bad and not errors
    report["steps"]["9_validate"] = {"ok": ok, "items": len(validated["items"]), "dupes": dupes, "quote_bad": quote_bad, "errors": errors}
    print(f"9 validate ok={ok} items={len(validated['items'])}")

    # Counts
    final = validated["items"]
    subs = [i for i in final if i["kind"] == "subsidiary"]
    with_desc = [
        i
        for i in subs
        if fold_whitespace(i.get("descriptor") or "").casefold() != fold_whitespace(i["name"]).casefold()
    ]
    # Source estimate from cache names.
    cache = load_cached_candidates("HINDALCO")
    origin = {}
    for c in cache["all"]:
        origin.setdefault(normalize(c.get("name") or ""), c.get("source_batch"))
    by_source = Counter(origin.get(normalize(i["name"]), "unknown") for i in final)
    report["summary"] = {
        "by_kind": dict(Counter(i["kind"] for i in final)),
        "by_source": dict(by_source),
        "subsidiaries_total": len(subs),
        "subsidiaries_with_descriptor": len(with_desc),
        "subsidiaries_without_descriptor": len(subs) - len(with_desc),
        "total_items": len(final),
    }
    dropped_by = defaultdict(list)
    for row in dropped_all:
        dropped_by[row["reason"]].append(row)
    report["dropped_by_rule"] = {k: v for k, v in sorted(dropped_by.items())}
    report["dropped_counts"] = {k: len(v) for k, v in sorted(dropped_by.items())}

    # CSV of all items
    csv_path = COPY_DIR / "items.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "kind", "entity", "pdf_page", "quote", "descriptor"])
        writer.writeheader()
        for item in final:
            src = item["sources"][0]
            writer.writerow(
                {
                    "name": item["name"],
                    "kind": item["kind"],
                    "entity": item["entity"],
                    "pdf_page": src["pdf_page"],
                    "quote": src["quote"],
                    "descriptor": item["descriptor"],
                }
            )

    # Stratified sample of 60
    sample = stratified_sample(final, 60, seed=42)
    report["sample_60"] = [
        {
            "name": i["name"],
            "kind": i["kind"],
            "entity": i["entity"],
            "pdf_page": i["sources"][0]["pdf_page"],
            "quote": i["sources"][0]["quote"],
            "descriptor": i["descriptor"],
        }
        for i in sample
    ]

    # Copy outputs first
    _atomic(COPY_DIR / "HINDALCO.copy.json", json.dumps(validated, indent=2, ensure_ascii=False) + "\n")
    _atomic_text(COPY_DIR / "HINDALCO.copy.md", markdown)
    _atomic(COPY_DIR / "report.json", json.dumps(_jsonable(report), indent=2, ensure_ascii=False) + "\n")

    wrote = False
    if ok:
        _atomic(OUT_JSON, json.dumps(validated, indent=2, ensure_ascii=False) + "\n")
        _atomic_text(OUT_MD, markdown)
        wrote = True
        print(f"WROTE {OUT_JSON} items={len(final)}")
    else:
        print("SKIP overwrite: validation failed")
    report["wrote_ok"] = wrote
    _atomic(COPY_DIR / "report.json", json.dumps(_jsonable(report), indent=2, ensure_ascii=False) + "\n")

    # Print removed per rule (names)
    print("\n=== DROPPED COUNTS ===")
    print(json.dumps(report["dropped_counts"], indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(report["summary"], indent=2))
    print("\n=== SAMPLE 60 ===")
    for row in report["sample_60"]:
        line = f"{row['pdf_page']}\t{row['kind']}\t{row['name']}\t{row['quote'][:100]}"
        print(line.encode("ascii", "replace").decode("ascii"))
    return report


def stratified_sample(items: list[dict], n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    by_kind = defaultdict(list)
    for item in items:
        by_kind[item["kind"]].append(item)
    kinds = sorted(by_kind)
    if not kinds:
        return []
    per = max(1, n // len(kinds))
    sample = []
    for kind in kinds:
        pool = list(by_kind[kind])
        rng.shuffle(pool)
        sample.extend(pool[:per])
    if len(sample) < n:
        rest = [i for i in items if i not in sample]
        rng.shuffle(rest)
        sample.extend(rest[: n - len(sample)])
    rng.shuffle(sample)
    return sample[:n]


def _find_cache(candidates: list[dict], name_equals: str | None = None, name_contains: str | None = None, prefer_page: int | None = None):
    hits = []
    for c in candidates:
        name = c.get("name") or ""
        if name_equals and fold_whitespace(name).casefold() != fold_whitespace(name_equals).casefold():
            continue
        if name_contains and name_contains.casefold() not in name.casefold():
            continue
        hits.append(c)
    if not hits:
        return None
    if prefer_page is not None:
        for c in hits:
            pages = [s.get("pdf_page") for s in c.get("sources") or []]
            if prefer_page in pages:
                return c
    return hits[0]


def _quote_from_blocks(page: dict | None, name: str, fallback: str) -> str | None:
    if not page:
        return None
    blocks = [b for b in page.get("text_blocks") or [] if (b.get("text") or "").strip()]
    blocks.sort(key=lambda b: b.get("reading_order", 0))
    name_cf = (name or "").casefold()
    for block in blocks:
        text = fold_whitespace(block.get("text") or "")
        if name_cf and name_cf not in text.casefold():
            continue
        # Prefer fallback if it sits inside this block.
        if fallback and fold_whitespace(fallback).casefold() in text.casefold():
            return _trim_quote(fallback, name, 15)
        # Build a short window around the name.
        return _trim_quote(text, name, 15)
    return None


def _trim_quote(quote: str, name: str, limit: int) -> str:
    words = fold_whitespace(quote).split()
    if len(words) <= limit:
        return " ".join(words)
    name_words = fold_whitespace(name).split()
    # Find name start.
    start = 0
    for i in range(len(words) - len(name_words) + 1):
        if [w.casefold() for w in words[i : i + len(name_words)]] == [w.casefold() for w in name_words]:
            start = i
            break
    end = min(len(words), start + limit)
    start = max(0, end - limit)
    return " ".join(words[start:end])


def _first_int(value) -> int | None:
    if value is None:
        return None
    text = str(value)
    m = re.search(r"\d+", text)
    return int(m.group(0)) if m else None


def _expected_printed(pdf_page: int | None, pairs: list[tuple[int, int]]) -> int | None:
    if not isinstance(pdf_page, int) or not pairs:
        return None
    # Use nearest known mapping.
    best = min(pairs, key=lambda pair: abs(pair[0] - pdf_page))
    delta = best[1] - best[0]
    return pdf_page + delta


def _shape(item: dict) -> dict | None:
    kind = item.get("kind") if item.get("kind") in KINDS else "other"
    status = item.get("status") if item.get("status") in STATUSES else "current"
    name = (item.get("name") or "").strip()
    descriptor = (item.get("descriptor") or name).strip()
    sources = []
    for source in (item.get("sources") or [])[:3]:
        quote = fold_whitespace(_CONTROL.sub(" ", source.get("quote") or ""))
        page = source.get("pdf_page")
        if not quote or not isinstance(page, int):
            continue
        printed = source.get("printed_page")
        sources.append(
            {
                "pdf_page": page,
                "printed_page": None if printed in (None, "") else str(printed),
                "quote": quote,
            }
        )
    if not name or not sources:
        return None
    share = item.get("revenue_share_pct")
    return {
        "name": name,
        "kind": kind,
        "descriptor": descriptor,
        "entity": item.get("entity") or "company",
        "status": status,
        "revenue_share_pct": share,
        "sources": sources,
    }


def _drop(item: dict, reason: str) -> dict:
    sources = item.get("sources") or []
    return {
        "name": item.get("name"),
        "kind": item.get("kind"),
        "reason": reason,
        "pdf_page": sources[0].get("pdf_page") if sources else None,
        "quote": sources[0].get("quote") if sources else None,
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    run()
