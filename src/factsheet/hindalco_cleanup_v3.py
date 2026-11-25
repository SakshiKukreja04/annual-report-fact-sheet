"""HINDALCO cleanup v3. No Groq. Copy-first under eval/hindalco_cleanup_v2/work/; promote last."""

from __future__ import annotations

import csv
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from rapidfuzz import fuzz

from src.config.paths import INPUT_DIR, OUTPUT_DIR, PROJECT_ROOT, STAGING_DIR, SUBMISSION_DIR
from src.dedup.normalize import normalize
from src.extract.evidence import pages_for_extraction
from src.extract.run import load_saved_company
from src.extract.schema import KINDS, STATUSES
from src.extract.select import page_text
from src.factsheet.build import consistency_errors, render_markdown
from src.factsheet.hindalco_cleanup_v2 import (
    _CONTROL,
    _expected_printed,
    _first_int,
    _jsonable,
    _key,
    _segment_quote,
    _shape,
    _trim_quote,
    identify_list_pages,
    nearest_heading,
    same_item_fixed,
    step6_header,
    step7_quotes,
    step8_printed,
)
from src.factsheet.hindalco_final_cache import _atomic, _atomic_text, _pdf_layer_meta, load_cached_candidates
from src.factsheet.postprocess import _kind_rank, _printed_for_quote
from src.factsheet.schema import FactSheet
from src.ingestion.audit import locate_pdfs
from src.tables.digital import extract_digital_page
from src.validate.quotes import fold_whitespace, locate_quote, repair_named_quote

WORK = PROJECT_ROOT / "eval" / "hindalco_cleanup_v2" / "work"
BASE_COPY = PROJECT_ROOT / "eval" / "hindalco_cleanup_v2" / "HINDALCO.copy.json"
OUT_JSON = OUTPUT_DIR / "HINDALCO" / "HINDALCO.json"
OUT_MD = OUTPUT_DIR / "HINDALCO" / "HINDALCO.md"
REPORT_V2 = PROJECT_ROOT / "eval" / "hindalco_cleanup_v2" / "report.json"

_GENERIC = {
    "plant", "unit", "mine", "division", "power", "recycling", "rolling", "green", "anode",
    "coal", "open", "cast", "captive", "cogeneration", "extrusion", "floating", "solar",
    "spin", "flash", "drying", "common", "effluent", "treatment", "water", "connected",
    "our", "optimising", "optimizing", "the", "and", "for", "at", "of", "in", "a", "an",
}
_NEVER_RESTORE = {
    "power plant", "optimising plant", "optimizing plant", "our connected plant",
    "water treatment plant", "common effluent treatment plant", "potf", "cetp", "etp",
    "twru", "rcmu", "pmcp", "cash generating unit", "quantity unit", "mr. senthil nath unit",
    "for kathautia coal mine",
}
# Exact restore names only (casefold). No substring matching.
_RESTORE_SITES = {
    "guthrie recycling plant": ("Guthrie Recycling Plant", "asset", "company"),
    "pindamonhangaba plant": ("Pindamonhangaba Plant", "asset", "company"),
    "nachterstedt rolling plant": ("Nachterstedt Rolling Plant", "asset", "company"),
    "yeongju plant": ("YeongJu Plant", "asset", "company"),
    "zhenjiang plant": ("Zhenjiang Plant", "asset", "company"),
    "mahan green anode plant": ("Mahan Green Anode Plant", "asset", "company"),
    "floating solar plant": ("Floating Solar Plant", "asset", "company"),
    "cogeneration power plant": ("Cogeneration Power Plant", "asset", "company"),
    "extrusion plant": ("Extrusion Plant", "asset", "company"),
    "spin flash drying unit": ("Spin Flash Drying Unit", "asset", "company"),
    "aditya unit": ("Aditya Unit", "asset", "company"),
    "hirakud unit": ("Hirakud Unit", "asset", "company"),
    "mahan unit": ("Mahan Unit", "asset", "company"),
    "renukoot unit": ("Renukoot Unit", "asset", "company"),
    "renusagar unit": ("Renusagar Unit", "asset", "company"),
    "kuppam unit": ("Kuppam Unit", "asset", "company"),
    "mouda unit": ("Mouda Unit", "asset", "company"),
    "kathautia coal mine": ("Kathautia Coal Mine", "asset", "company"),
    "kathautia open cast coal mine": ("Kathautia Open Cast Coal Mine", "asset", "company"),
    "captive power plant": ("Captive Power Plant", "asset", "company"),
    "hindalco captive power plant": ("Hindalco Captive Power Plant", "asset", "company"),
    "copper division": ("Copper Division", "asset", "company"),
    "the copper division": ("Copper Division", "asset", "company"),
    "mines division": ("Mines Division", "asset", "company"),
    "birla copper division": ("Birla Copper Division", "asset", "company"),
}
_RESTORE_SPECIAL = {
    "hindalco-almex aerospace limited": ("Hindalco-Almex Aerospace Limited", "subsidiary", "Hindalco-Almex Aerospace Limited"),
    "hindalco- almex aerospace limited": ("Hindalco-Almex Aerospace Limited", "subsidiary", "Hindalco-Almex Aerospace Limited"),
    "hindalco almex aerospace limited": ("Hindalco-Almex Aerospace Limited", "subsidiary", "Hindalco-Almex Aerospace Limited"),
    "almex aerospace limited": ("Almex Aerospace Limited", "subsidiary", "Almex Aerospace Limited"),
    "specialty alumina business": ("Specialty Alumina business", "business_line", "company"),
    "power station": ("Power Station", "business_line", "company"),
    "project varun": ("Project Varun", "asset", "company"),
    "ppt ath plant in belagavi": ("PPT ATH plant in Belagavi", "asset", "company"),
}
_DROP_EXACT = {
    "cost audit of cost records", "foundation of purposeful growth",
    "value enhancement through portfolio enrichment sp",
    "value-enhancing growth/double-down on upstream capacities sp",
    "value?enhancing growth/double?down on upstream capacities sp",
    "community development & relations", "manufacturing plants",
    "30 new products and applications developed", "16 plants", "190 mw", "downstream ebitda",
    "world's first metal manufacturer to utilise its bauxite residue",
    "group hospitals", "group schools and educational institutes",
    "gyanodaya leadership and learning centre", "digital experience centre",
    "taproot", "crm software", "customer relationship management (crm) software",
    "logistics analytics dashboard", "oracle transport management",
    "blockchains enabling tolling management system", "cyber defence command centre",
    "ro unit", "piezometers", "rainwater harvesting structures", "rainwater harvesting ponds",
    "advanced wastewater treatment technologies", "terre haute closed-loop water system",
    "tertiary water recycling unit", "tailings storage facilities", "tailings storage facilities (tsfs)",
    "mechanical vapour recompression", "advanced filtration system", "bag house",
    "furnace automation control system", "regenerative furnace burners",
    "selective catalytic reduction", "renewable energy certificates",
    "cathode design modification", "cathode design modification at aditya",
    "cathode quality improvement", "optimisation of pre-calcination circuit",
    "optimisation of pre?calcination circuit", "optimisation of precalcination circuit",
    "cold rolling mill productivity enhancement",
    "cold rolling mill productivity enhancement at mouda",
    "empowerpreneur", "fly", "long-term settlements", "corporate logistics team",
    "tailoring units", "bse limited", "erstwhile ryker base pvt ltd",
    "biannual water survey", "renewable energy capacity commissioned",
}
_DROP_FRAGMENTS = (
    "foundation of purposeful growth", "community development & relations",
    "manufacturing plants", "30 new products", "group hospitals", "group schools",
    "gyanodaya", "digital experience centre", "crm software", "customer relationship management",
    "logistics analytics", "oracle transport", "blockchains enabling", "cyber defence",
    "rainwater harvesting", "terre haute", "tailings storage", "renewable energy certificates",
    "cathode design", "cathode quality", "pre-calcination", "pre?calcination", "precalcination",
    "cold rolling mill productivity", "long-term settlements", "bse limited",
    "erstwhile ryker", "biannual water", "renewable energy capacity commissioned",
    "value-enhancing growth", "value?enhancing growth", "valueenhancing growth",
    "double-down on upstream", "doubledown on upstream", "portfolio enrichment",
    "world's first metal manufacturer to utilise", "taproot", "piezometers",
    "mechanical vapour", "bag house", "furnace automation", "regenerative furnace",
    "selective catalytic", "empowerpreneur", "corporate logistics", "tailoring units",
    "advanced wastewater", "advanced filtration", "tertiary water recycling",
    "16 plants", "190 mw", "downstream ebitda", "ro unit",
    "cost audit of cost records", "fly (future leader",
)
_BRAND_MERGE = {
    "hindalco masterbrand", "hindalco new branding", "bold new brand identity",
    "engineering better futures",
}
_KEEP_BRANDS = {"eternia", "totalis", "freshwrapp", "ecoedge g", "ecoedge c", "al:sust", "hic aluminium"}


def _cf(name: str) -> str:
    return fold_whitespace(name or "").casefold().replace("–", "-").replace("—", "-")


def has_proper_stem(name: str) -> bool:
    key = _cf(name)
    if key in _NEVER_RESTORE:
        return False
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9']*", name or "")
    if not tokens:
        return False
    # Drop trailing Plant/Unit/Mine/Division and inspect remaining.
    while tokens and tokens[-1].casefold() in {"plant", "unit", "mine", "division"}:
        tokens = tokens[:-1]
    stems = [t for t in tokens if t.casefold() not in _GENERIC]
    return bool(stems)


def step0_verify() -> dict:
    sheet = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    count = len(sheet["items"])
    one = sheet.get("one_liner") or ""
    print(f"0 outputs count={count}")
    print(f"0 one_liner={one[:180].encode('ascii','replace').decode()}")
    restored = False
    if count != 191:
        # Restore from cleanup_v2 copy if it is the 191 sheet, else refuse.
        if BASE_COPY.is_file():
            copy = json.loads(BASE_COPY.read_text(encoding="utf-8"))
            if len(copy.get("items") or []) == 191:
                _atomic(OUT_JSON, json.dumps(copy, indent=2, ensure_ascii=False) + "\n")
                md = PROJECT_ROOT / "eval" / "hindalco_cleanup_v2" / "HINDALCO.copy.md"
                if md.is_file():
                    _atomic_text(OUT_MD, md.read_text(encoding="utf-8"))
                restored = True
                sheet = copy
                count = 191
                print("0 restored 191-item sheet from eval/hindalco_cleanup_v2/HINDALCO.copy.json")
            else:
                raise RuntimeError(f"outputs has {count} items and copy has {len(copy.get('items') or [])}; need the 191-item sheet")
        else:
            raise RuntimeError(f"outputs has {count} items; expected 191")
    # Stage/factsheet overwrite note
    note = (
        "Pipeline stage_factsheet previously wrote to outputs/ (or .continued beside it). "
        "slice_final/assemble could also write under runs/. "
        "stage_factsheet + factsheet.run now write only to staging/factsheet/; "
        "promote to outputs/ and submission/ only from this final-assembly step."
    )
    print(f"0 overwrite_note={note}")
    return {"count": count, "one_liner": one, "restored": restored, "note": note, "sheet": sheet}


def load_dropped_index() -> dict[str, dict]:
    report = json.loads(REPORT_V2.read_text(encoding="utf-8")) if REPORT_V2.is_file() else {}
    index = {}
    for rows in (report.get("dropped_by_rule") or {}).values():
        for row in rows:
            key = _cf(row.get("name") or "")
            if key and key not in index:
                index[key] = row
    # Also search cache for quotes.
    cache = load_cached_candidates("HINDALCO")
    for item in cache["all"]:
        key = _cf(item.get("name") or "")
        if key and key not in index:
            src = (item.get("sources") or [{}])[0]
            index[key] = {
                "name": item.get("name"),
                "kind": item.get("kind"),
                "pdf_page": src.get("pdf_page"),
                "quote": src.get("quote"),
                "descriptor": item.get("descriptor"),
                "sources": item.get("sources"),
                "entity": item.get("entity"),
            }
        elif key:
            # Prefer richer cache record.
            if not index[key].get("sources") and item.get("sources"):
                index[key]["sources"] = item.get("sources")
                index[key]["descriptor"] = item.get("descriptor") or index[key].get("descriptor")
    return index


def step1_restore(items: list[dict], dropped_index: dict) -> dict:
    kept = list(items)
    present = {_cf(i["name"]) for i in kept}
    restored = []
    skipped = []

    def add(name: str, kind: str, entity: str, row: dict, reason: str):
        nonlocal kept
        key = _cf(name)
        if key in present or any(_cf(i["name"]) == key for i in kept):
            skipped.append({"name": name, "reason": "already_present"})
            return
        if key in _NEVER_RESTORE:
            skipped.append({"name": name, "reason": "never_restore"})
            return
        sources = row.get("sources")
        if not sources:
            page = row.get("pdf_page")
            quote = row.get("quote") or name
            if not isinstance(page, int):
                skipped.append({"name": name, "reason": "no_page"})
                return
            sources = [{"pdf_page": page, "printed_page": None, "quote": quote}]
        # Prefer page quote that still mentions the name.
        quote0 = fold_whitespace((sources[0] or {}).get("quote") or "")
        if name.casefold() not in quote0.casefold() and quote0:
            sources = [{"pdf_page": sources[0].get("pdf_page"), "printed_page": sources[0].get("printed_page"), "quote": name}]
        item = {
            "name": name,
            "kind": kind,
            "descriptor": row.get("descriptor") or name,
            "entity": entity,
            "status": "current",
            "revenue_share_pct": None,
            "sources": sources[:3],
        }
        kept.append(item)
        present.add(key)
        restored.append({"name": name, "kind": kind, "reason": reason, "pdf_page": sources[0].get("pdf_page")})

    wanted = {**_RESTORE_SITES, **_RESTORE_SPECIAL}
    for key, (nice, kind, entity) in wanted.items():
        row = dropped_index.get(key)
        if not row:
            # Exact name match only among dropped (no substring).
            row = next((v for k, v in dropped_index.items() if _cf(v.get("name") or "") == key), None)
        if not row:
            skipped.append({"name": nice, "reason": "not_in_dropped_index"})
            continue
        raw_name = row.get("name") or nice
        if key in _RESTORE_SITES and not has_proper_stem(nice) and key not in {
            "copper division", "the copper division", "mines division", "birla copper division",
            "captive power plant", "floating solar plant", "cogeneration power plant",
            "extrusion plant", "spin flash drying unit",
        }:
            skipped.append({"name": nice, "reason": "no_proper_stem"})
            continue
        add(nice, kind, entity, row, "restore_exact")

    # PPT ATH Belagavi from Project Varun quote if not already restored
    if _cf("PPT ATH plant in Belagavi") not in present:
        row = dropped_index.get("project varun") or next(
            (v for k, v in dropped_index.items() if "project varun" in k or "belagavi" in k),
            None,
        )
        if row:
            quote = row.get("quote") or "Project Varun evolved into a fully operational superfine PPT ATH plant in Belagavi"
            add(
                "PPT ATH plant in Belagavi",
                "asset",
                "company",
                {**row, "quote": quote, "descriptor": "superfine PPT ATH plant in Belagavi", "sources": row.get("sources") or [{"pdf_page": row.get("pdf_page") or 93, "printed_page": None, "quote": quote}]},
                "restore_PPT_ATH_Belagavi",
            )

    print(f"1 restored={len(restored)} skipped={len(skipped)} total={len(kept)}")
    for row in restored:
        print(f"  + {row['kind']}\t{row['name']}\tp{row['pdf_page']}".encode("ascii", "replace").decode())
    for row in skipped:
        print(f"  skip {row['reason']}\t{row['name']}".encode("ascii", "replace").decode())
    return {"items": kept, "restored": restored, "skipped": skipped}


def step2_drop(items: list[dict]) -> dict:
    kept, dropped = [], []
    for item in items:
        key = _cf(item.get("name") or "")
        # ASCII-fold odd punctuation from PDF/encoding (soft hyphen, en-dash, replacement chars)
        key2 = key.encode("ascii", "replace").decode("ascii").casefold()
        key2 = re.sub(r"[?\u2013\u2014\-]+", "-", key2)
        key2 = re.sub(r"[^a-z0-9&/.\- ]+", "", key2)
        key2 = re.sub(r"\s+", " ", key2).strip()
        hit = None
        for drop in _DROP_EXACT:
            d = re.sub(r"[?\u2013\u2014]", "-", drop)
            if key == drop or key2 == d or key2.startswith(d) or d in key2:
                hit = drop
                break
        if not hit:
            for frag in _DROP_FRAGMENTS:
                f = re.sub(r"[?\u2013\u2014]", "-", frag)
                if f in key2 or frag in key:
                    hit = frag
                    break
        # exact short names that must not substring-match longer good names
        if not hit and key2 in {"fly", "ro unit", "16 plants", "190 mw", "taproot", "bag house", "piezometers"}:
            hit = key2
        if hit:
            dropped.append({"name": item.get("name"), "kind": item.get("kind"), "reason": f"drop_named:{hit}", "pdf_page": (item.get("sources") or [{}])[0].get("pdf_page")})
            continue
        kept.append(item)
    print(f"2 drop kept={len(kept)} dropped={len(dropped)}")
    for row in dropped:
        print(f"  - {row['name']}".encode("ascii", "replace").decode())
    return {"items": kept, "dropped": dropped}


def step3_merge_brands(items: list[dict]) -> dict:
    merge_members = []
    kept = []
    for item in items:
        key = _cf(item.get("name") or "")
        if any(b in key for b in _BRAND_MERGE) or key in _BRAND_MERGE:
            merge_members.append(item)
            continue
        # normalize AL:sust / ecoEDGE variants stay separate
        kept.append(item)
    merged = None
    if merge_members:
        sources = []
        for member in merge_members:
            for src in member.get("sources") or []:
                if src not in sources:
                    sources.append(src)
                if len(sources) >= 3:
                    break
            if len(sources) >= 3:
                break
        merged = {
            "name": "Hindalco Masterbrand",
            "kind": "brand",
            "descriptor": max((m.get("descriptor") or "" for m in merge_members), key=len) or "Hindalco brand identity",
            "entity": "company",
            "status": "current",
            "revenue_share_pct": None,
            "sources": sources[:3],
        }
        kept.append(merged)
    print(f"3 brand_merge members={len(merge_members)} kept_items={len(kept)} merged={'yes' if merged else 'no'}")
    if merged:
        print(f"  merged_quotes={len(merged['sources'])} names={[m['name'] for m in merge_members]}".encode("ascii", "replace").decode())
    return {"items": kept, "merged_from": [m.get("name") for m in merge_members], "merged": merged}


def step4_dedup_fix(items: list[dict], dropped_index: dict) -> dict:
    """Never merge Renuka vs Renukeshwar; restore Renukeshwar; fix entities."""
    # Fix Renuka descriptor if it wrongly holds Renukeshwar.
    fixed = []
    notes = []
    for item in items:
        name = item.get("name") or ""
        desc = item.get("descriptor") or ""
        if _cf(name) == "renuka investments & finance limited" and "renukeshwar" in desc.casefold():
            item = {**item, "descriptor": "Investments and finance subsidiary"}
            notes.append("fixed_Renuka_descriptor")
        # entity typo Renewables
        ent = item.get("entity") or ""
        if re.search(r"aditya birla renewable utkal", ent, re.I) and "renewables" not in ent.casefold():
            item = {**item, "entity": re.sub(r"Renewable Utkal", "Renewables Utkal", ent, flags=re.I)}
            notes.append("fixed_entity_Renewables_Utkal")
        if re.search(r"aditya birla renewable utkal", name, re.I) and "renewables" not in name.casefold():
            item = {**item, "name": re.sub(r"Renewable Utkal", "Renewables Utkal", name, flags=re.I)}
            notes.append("fixed_name_Renewables_Utkal")
        # AL:sust entity should be Novelis or company — leave
        fixed.append(item)

    # Restore Renukeshwar if missing
    present = {_cf(i["name"]) for i in fixed}
    restored = []
    if "renukeshwar investments & finance limited" not in present:
        row = dropped_index.get("renukeshwar investments & finance limited")
        if not row:
            row = next((v for k, v in dropped_index.items() if "renukeshwar" in k), None)
        if row:
            sources = row.get("sources") or [{"pdf_page": row.get("pdf_page") or 261, "printed_page": None, "quote": row.get("quote") or "Renukeshwar Investments & Finance Limited"}]
            fixed.append(
                {
                    "name": "Renukeshwar Investments & Finance Limited",
                    "kind": "subsidiary",
                    "descriptor": "Investments and finance subsidiary",
                    "entity": "Renukeshwar Investments & Finance Limited",
                    "status": "current",
                    "revenue_share_pct": None,
                    "sources": sources[:3],
                }
            )
            restored.append("Renukeshwar Investments & Finance Limited")

    # Dedup with stricter rule: no merge if content tokens differ beyond legal suffix/punct.
    deduped, dropped = dedupe_strict(fixed)
    print(f"4 dedup kept={len(deduped)} dropped={len(dropped)} restored={restored} notes={notes}")
    return {"items": deduped, "dropped": dropped, "restored": restored, "notes": notes}


def dedupe_strict(items: list[dict]) -> tuple[list[dict], list[dict]]:
    groups: list[list[dict]] = []
    dropped = []
    for item in items:
        placed = False
        for group in groups:
            if same_item_strict(group[0], item):
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
        base = dict(group[0])
        sources = []
        for member in group:
            if _kind_rank(member.get("kind")) > _kind_rank(base.get("kind")):
                base["kind"] = member.get("kind")
            if len(member.get("descriptor") or "") > len(base.get("descriptor") or ""):
                base["descriptor"] = member.get("descriptor")
            for src in member.get("sources") or []:
                if src not in sources:
                    sources.append(src)
        base["sources"] = sources[:3]
        kept.append(base)
        for extra in group[1:]:
            dropped.append({"name": extra.get("name"), "reason": "dedup_strict", "pdf_page": (extra.get("sources") or [{}])[0].get("pdf_page")})
    # casefold exact collapse
    by: dict[str, dict] = {}
    order = []
    for item in kept:
        key = (item.get("name") or "").casefold()
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
            dropped.append({"name": item.get("name"), "reason": "duplicate_name", "pdf_page": (item.get("sources") or [{}])[0].get("pdf_page")})
    return [by[k] for k in order], dropped


def same_item_strict(left: dict, right: dict) -> bool:
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
    legal = {"limited", "ltd", "inc", "incorporated", "private", "pvt", "llc", "gmbh", "company"}
    if sa < sb or sb < sa:
        shorter, longer = (sa, sb) if sa < sb else (sb, sa)
        extras = longer - shorter
        return bool(extras) and extras <= legal
    # Differ by more than legal suffix/punctuation → never merge (Renuka vs Renukeshwar).
    if sa != sb:
        return False
    return fuzz.token_sort_ratio(a, b) >= 95


_CORP_END = re.compile(
    r"(?:"
    r"Private\s+Limited|Pvt\.?\s*Ltd\.?|Limited|Ltd\.?|Inc\.?|LLC|GmbH|AG|B\.V\.|"
    r"N\.V\.|S\.A\.|SAS|SpA|SPA|Ltda\.?|Corporation|Company|"
    r"S\.A\.\s*de\s*C\.V\.|Unlimited\s+Company|Co\.,?\s*Ltd\.?"
    r")\.?$",
    re.I,
)
_ACTIVITY_START = re.compile(
    r"^(Manufacturing|Mining|Investment|Holding|Cargo|Engineering|Dormant|Distribution|"
    r"Import|Management|Trading|Recycling|Intermediate|Services|Sales|Cash)\b",
    re.I,
)
_COUNTRY = re.compile(
    r"^(India|Canada|Brazil|USA|US|United States|China|Korea|Germany|France|Italy|Mexico|"
    r"Switzerland|UK|United Kingdom|Netherlands|Netherland|British Virgin|Singapore|Japan|"
    r"Vietnam|Australia|Spain|Belgium|Barbados)$",
    re.I,
)


def _glue_parts(parts: list[str]) -> str:
    out: list[str] = []
    for cell in parts:
        cell = fold_whitespace(cell)
        if not cell:
            continue
        if not out:
            out.append(cell)
            continue
        prev = out[-1]
        # Mid-word wrap, suffix splinters (SA+S -> SAS, Sp+A -> SpA, N+.V. -> N.V.), or punct glue.
        splinter = (
            len(cell) <= 3
            and re.match(r"^[A-Za-z.]+$", cell)
            and re.search(r"[A-Za-z]$", prev)
            and not prev.endswith(("Limited", "Company", "Inc", "LLC", "GmbH"))
        )
        if (
            prev.endswith(("-", "(", "/", ","))
            or cell.startswith((".", ")", ",", "-"))
            or (cell[:1].islower() and not prev.endswith(" "))
            or (re.search(r"[A-Za-z]$", prev) and re.match(r"^[a-z]", cell))
            or splinter
        ):
            out[-1] = prev + cell
        else:
            out.append(cell)
    raw = fold_whitespace(" ".join(out))
    raw = re.sub(r"\s+\.", ".", raw)
    raw = re.sub(r"\s+-\s*\(iii\)\s*$", "", raw, flags=re.I)
    raw = re.sub(r"\s*-\s*$", "", raw)
    raw = re.sub(r"\s+", " ", raw).strip(" ,;-")
    return raw


def _clean_entity_name(raw: str) -> str | None:
    name = fold_whitespace(raw)
    name = re.sub(r"\s*-\s*\(iii\)\s*$", "", name, flags=re.I)
    # Strip short S.No only (1-3 digits), keep 4260848-style company numbers.
    name = re.sub(r"^\d{1,3}\s+(?=[A-Za-z])", "", name)
    name = re.sub(
        r"^(Creating And Sustaining Value|Strategic Overview|Corporate Overview|Statutory Reports|"
        r"Business Responsibility|Awards and Recognition|Notes forming|Annexure)\s+",
        "",
        name,
        flags=re.I,
    )
    # Keep trailing periods so S.A. / B.V. / N.V. / C.V. still match.
    name = fold_whitespace(name).strip(" ,;-")
    name = re.sub(r"([a-z])(Limited|Ltd)\b", r"\1 \2", name)
    name = re.sub(r"\s+", " ", name)
    if len(name.split()) < 2 and not re.match(r"^\d{5,}\s", name):
        return None
    if not _CORP_END.search(name):
        return None
    if re.match(r"^(Private|Pvt|Ltd|Limited|Company|Associates?|Subsidiaries|Parent|Canada Inc)\b", name, re.I):
        return None
    if re.search(r"subsidiaries associates|name of the|principal activity|key managerial|contingent liabilit", name, re.I):
        return None
    # Allow digit-leading legal names (4260848 Canada Inc) but reject % / bare decimals
    if re.match(r"^[.%]", name):
        return None
    # "12 4260848 Canada Inc" -> "4260848 Canada Inc"
    name = re.sub(r"^(\d{1,2})\s+(?=\d{4,})", "", name)
    if re.match(r"^\d{1,3}\s+[A-Za-z]", name):
        name = re.sub(r"^\d{1,3}\s+", "", name)
    if len(re.findall(r"\bLimited\b|\bInc\b|\bGmbH\b|\bLtd\b", name, re.I)) > 1:
        return None
    if len(name) > 100:
        return None
    if re.match(r"^(Private|Pvt|Ltd|Limited|Company|Associates?|Subsidiaries|Parent|Canada Inc)\b", name, re.I):
        return None
    return name


def _join_name_cells(cells: list[str]) -> tuple[str | None, str | None]:
    """Join wrapped name cells; return (name, activity)."""
    nonempty = [fold_whitespace(c) for c in cells if fold_whitespace(c or "")]
    if not nonempty:
        return None, None
    if re.fullmatch(r"\d+", nonempty[0]):
        nonempty = nonempty[1:]
    # Split name vs activity/country: activity starts with known verb/noun; country is a place.
    split_at = None
    for i, cell in enumerate(nonempty):
        if _COUNTRY.match(cell):
            split_at = i
            break
        # Activity labels like "Holding Company" / "Management Company" also match _CORP_END;
        # treat them as activity when the prior cells already look like a name fragment.
        # Do NOT split on "Trading Co., Ltd" — that is part of the legal name.
        if i > 0 and _ACTIVITY_START.match(cell) and not _CORP_END.search(cell):
            prior = _glue_parts(nonempty[:i])
            if _CORP_END.search(prior) or len(prior) >= 8:
                split_at = i
                break
        if i > 0 and re.match(
            r"^(Holding Company|Management Company|Intermediate subsidiary|Manufacturing|"
            r"Investment|Dormant|Engineering|Import and export|Cash management|Sales Office|"
            r"Distribution Services|Cargo services)$",
            cell,
            re.I,
        ):
            split_at = i
            break
    name_cells = nonempty if split_at is None else nonempty[:split_at]
    rest = [] if split_at is None else nonempty[split_at:]
    # Strip short row numbers only; keep 4260848-style company numbers.
    if name_cells and re.match(r"^\d{1,3}\s+[A-Za-z]", name_cells[0]):
        name_cells = [re.sub(r"^\d{1,3}\s+", "", name_cells[0])] + name_cells[1:]
    # Drop trailing "(iii)" / roman fragments from name cells
    cleaned_cells = []
    for c in name_cells:
        c = re.sub(r"\s*-\s*\(iii\)\s*$", "", c, flags=re.I).strip()
        c = re.sub(r"^\(iii\)$", "", c, flags=re.I).strip()
        if c:
            cleaned_cells.append(c)
    name_cells = cleaned_cells
    raw = _glue_parts(name_cells)
    # Strip leftover " - (iii)" after glue
    raw = re.sub(r"\s*-\s*\(iii\)\s*$", "", raw, flags=re.I).strip()
    # If activity text was glued onto the name, peel it off.
    for label in (
        "Holding Company", "Management Company", "Intermediate subsidiary",
        "Manufacturing", "Investment", "Dormant", "Engineering",
        "Import and export aluminum", "Cash management service provider", "Sales Office",
        "Distribution Services", "Cargo services",
    ):
        if raw.casefold().endswith(label.casefold()):
            raw = raw[: -len(label)].rstrip(" -")
            if not rest:
                rest = [label]
            break
    name = _clean_entity_name(raw)
    activity = None
    if rest:
        # country is last-ish; activity is everything before country that isn't a country
        act_parts = []
        for cell in rest:
            if _COUNTRY.match(cell):
                break
            act_parts.append(cell)
        if act_parts:
            activity = _glue_parts(act_parts)
            if not _ACTIVITY_START.match(activity):
                activity = None
    return name, activity


def step5_subsidiaries_tables(items: list[dict], pages: list[dict], meta: dict) -> dict:
    pdf = locate_pdfs(INPUT_DIR).get("HINDALCO")
    list_pages = meta["list_pages"]
    texts = meta["texts"]
    entities: dict[str, dict] = {}
    activities: dict[str, str] = {}
    table_count = 0

    def ingest(name: str | None, page: int, activity: str | None = None) -> None:
        if not name:
            return
        if re.search(r"princ|related int|me of the|name of the|joint control|ventures \(jo", name, re.I):
            return
        # Canonical Renewables spelling
        name = re.sub(r"\bAditya Birla Renewables?\s+(Utkal|Solar|Subsidiary)\b", r"Aditya Birla Renewables \1", name, flags=re.I)
        name = re.sub(r"\bA\.V\.\s+Minerals\b", "AV Minerals", name, flags=re.I)
        name = fold_whitespace(name).rstrip(".")
        if name.casefold().endswith(" ltd"):
            name = name + "."  # keep Ltd. form optional; normalize below
        # Normalize trailing punctuation on common suffixes
        name = re.sub(r"\b(Inc|Ltd|Ltda|SPA)\.$", r"\1", name)
        name = re.sub(r"\b(Inc|Ltd|Ltda)$", r"\1", name)
        key = _cf(name)
        # Drop shorter duplicates that are punctuation/legal-suffix variants only
        for existing in list(entities.keys()):
            if existing == key:
                break
            a, b = existing, key
            if a in b or b in a:
                shorter, longer = (a, b) if len(a) < len(b) else (b, a)
                if len(longer) - len(shorter) <= 12:
                    if longer == key:
                        entities.pop(existing, None)
                        activities.pop(existing, None)
                    else:
                        return
        ent = entities.setdefault(key, {"name": name, "pages": []})
        if page not in ent["pages"]:
            ent["pages"].append(page)
        if len(name) > len(ent["name"]):
            ent["name"] = name
        if activity:
            activities[key] = activity

    if pdf:
        for number in list_pages:
            try:
                tables = extract_digital_page(Path(pdf), "HINDALCO", number)
            except Exception:
                tables = []
            for table in tables:
                headers = table.get("headers") or []
                header_blob = " ".join(
                    fold_whitespace(" ".join(h) if isinstance(h, list) else str(h or "")) for h in headers
                ).casefold()
                rows = table.get("rows") or []
                # Only subsidiary / AOC / entity-list shaped tables
                looks_entity = any(
                    k in header_blob
                    for k in ("name of the", "subsidiary", "entity", "s. no", "principal act", "reporting currency")
                )
                # Or rows that look like numbered entity lines
                sample = " ".join(
                    fold_whitespace(" ".join(c or "" for c in (r.get("cells") or [])[:4])) for r in rows[:5]
                )
                if not looks_entity and not re.search(r"\bLimited\b|\bInc\b|\bGmbH\b", sample, re.I):
                    continue
                if re.search(r"independent auditor|key audit matter|internal financial", sample.casefold()):
                    continue
                table_count += 1
                # Headers sometimes embed early numbered entities (e.g. Minerals & Renukeshwar).
                if header_blob:
                    for match in re.finditer(
                        r"\d+\s+([A-Z][^0-9]{3,80}?(?:Limited|Ltd\.?|Inc\.?|GmbH|N\.V\.))",
                        " ".join(
                            fold_whitespace(" ".join(h) if isinstance(h, list) else str(h or ""))
                            for h in headers
                        ),
                        re.I,
                    ):
                        ingest(_clean_entity_name(match.group(1)), number, None)
                pending_cells: list[str] = []
                for row in rows:
                    cells = [fold_whitespace(c or "") for c in (row.get("cells") or [])]
                    if not any(cells):
                        continue
                    # Continuation row: no leading S.No and previous name incomplete
                    leading = next((c for c in cells if c), "")
                    is_new = bool(re.match(r"^\d+\b", leading)) or (
                        pending_cells == [] and bool(re.search(r"[A-Z]", leading[:1] or ""))
                    )
                    if pending_cells and not re.match(r"^\d+\b", leading) and not _CORP_END.search(
                        fold_whitespace(" ".join(pending_cells))
                    ):
                        pending_cells.extend(cells)
                        name, activity = _join_name_cells(pending_cells)
                        if name:
                            ingest(name, number, activity)
                            pending_cells = []
                        continue
                    if pending_cells:
                        name, activity = _join_name_cells(pending_cells)
                        ingest(name, number, activity)
                        pending_cells = []
                    name, activity = _join_name_cells(cells)
                    if name:
                        ingest(name, number, activity)
                    else:
                        # maybe incomplete wrap — hold
                        if any(c for c in cells if re.search(r"[A-Za-z]", c)):
                            pending_cells = list(cells)
                if pending_cells:
                    name, activity = _join_name_cells(pending_cells)
                    ingest(name, number, activity)

    distinct = sorted(entities.keys())
    print(f"5 table_parse tables={table_count} distinct_entities={len(distinct)}")
    for name in sorted({entities[k]["name"] for k in distinct}):
        print(f"  entity: {name}".encode("ascii", "replace").decode())

    verify = {}
    for needle in ("Kobe Steel Ltd", "Minerals & Minerals Limited", "Minerals and Minerals Limited"):
        contexts = []
        for number in list_pages:
            text = texts.get(number, "") or ""
            needle_cf = needle.casefold().replace(" and ", " & ")
            text_cf = text.casefold().replace(" and ", " & ")
            if "kobe" in needle_cf:
                hit = "kobe steel" in text_cf
            else:
                hit = "minerals" in text_cf and ("minerals limited" in text_cf or "minerals & minerals" in text_cf)
            if hit:
                idx = text_cf.find("kobe" if "kobe" in needle_cf else "minerals")
                contexts.append(
                    {
                        "pdf_page": number,
                        "heading": nearest_heading(text),
                        "snippet": fold_whitespace(text[max(0, idx - 100) : idx + 200]),
                    }
                )
        verify[needle] = contexts
        print(f"5 verify {needle}: hits={len(contexts)}")
        for c in contexts[:3]:
            print(f"  p{c['pdf_page']} [{c['heading'][:50]}] {c['snippet'][:170]}".encode("ascii", "replace").decode())

    keep_kobe = False
    for c in verify.get("Kobe Steel Ltd") or []:
        if re.search(r"unrelated party", c["snippet"], re.I):
            print("5 Kobe Steel Ltd: UNRELATED PARTY — drop")
            keep_kobe = False
            break
        if re.search(r"\b(subsidiar|joint venture|associate)\b", c["snippet"], re.I):
            keep_kobe = True
    keep_minerals = True  # listed under wholly-owned subsidiaries with Mining
    print(f"5 keep_kobe={keep_kobe} keep_minerals={keep_minerals}")

    non_subs = [i for i in items if i.get("kind") != "subsidiary"]
    preserve = [
        i
        for i in items
        if i.get("kind") == "subsidiary"
        and any(x in _cf(i.get("name") or "") for x in ("almex", "renukeshwar", "renuka investments"))
    ]
    subs = []
    for key, info in entities.items():
        name = info["name"]
        if "kobe" in key:
            continue
        if "erstwhile ryker" in key or "bse limited" in key:
            continue
        page = min(info["pages"])
        activity = activities.get(key)
        descriptor = activity if activity else None
        # Soft-verify name appears on one of its pages
        soft = re.sub(r"[^a-z0-9]", "", name.casefold())
        # Renewables pages often print "Renewable" without the trailing s.
        soft_alt = soft.replace("renewables", "renewable")
        ok = False
        for p in info["pages"]:
            page_soft = re.sub(r"[^a-z0-9]", "", (texts.get(p, "") or "").casefold())
            if (soft[:12] and soft[:12] in page_soft) or (soft_alt[:12] and soft_alt[:12] in page_soft):
                ok = True
                break
            # Also accept if core tokens appear (aditya+birla+renewable+utkal)
            tokens = [t for t in re.findall(r"[a-z]{4,}", name.casefold()) if t not in {"limited", "private", "company"}]
            if tokens and all(t in page_soft for t in tokens[:3]):
                ok = True
                break
        if not ok:
            continue
        # Stated activity only; never fall back to the bare name here.
        if descriptor and fold_whitespace(descriptor).casefold() == fold_whitespace(name).casefold():
            descriptor = None
        if key.startswith("minerals") and "hydromine" not in key:
            descriptor = descriptor or "Mining"
        quote = _page_visible_quote(name, texts.get(page, "") or "") or name
        subs.append(
            {
                "name": name,
                "kind": "subsidiary",
                "descriptor": descriptor,
                "entity": name,
                "status": "current",
                "revenue_share_pct": None,
                "sources": [{"pdf_page": page, "printed_page": None, "quote": quote}],
            }
        )
    present = {_cf(s["name"]) for s in subs}
    for item in preserve:
        if _cf(item["name"]) not in present:
            if "renuka investments" in _cf(item["name"]) and "renukeshwar" in (item.get("descriptor") or "").casefold():
                item = {**item, "descriptor": "Investment"}
            subs.append(item)
            present.add(_cf(item["name"]))
    # Ensure Minerals kept under canonical name
    if keep_minerals and not any("minerals" in _cf(s["name"]) and "hydromine" not in _cf(s["name"]) and "av minerals" not in _cf(s["name"]) for s in subs):
        subs.append(
            {
                "name": "Minerals & Minerals Limited",
                "kind": "subsidiary",
                "descriptor": "Mining",
                "entity": "Minerals & Minerals Limited",
                "status": "current",
                "revenue_share_pct": None,
                "sources": [{"pdf_page": 261, "printed_page": None, "quote": "Minerals & Minerals Limited"}],
            }
        )
    # Ensure Almex Aerospace Limited (distinct from Hindalco-Almex) if restored earlier
    if "almex aerospace limited" not in present:
        for item in items:
            if _cf(item.get("name") or "") == "almex aerospace limited":
                subs.append(item)
                present.add("almex aerospace limited")
                break
        else:
            # Still missing — add from list-page evidence if present in text.
            for number in list_pages:
                text = texts.get(number, "") or ""
                if re.search(r"\bAlmex Aerospace Limited\b", text) and not re.search(
                    r"Hindalco[\s\-]*Almex Aerospace Limited", text
                ):
                    subs.append(
                        {
                            "name": "Almex Aerospace Limited",
                            "kind": "subsidiary",
                            "descriptor": None,
                            "entity": "Almex Aerospace Limited",
                            "status": "current",
                            "revenue_share_pct": None,
                            "sources": [{"pdf_page": number, "printed_page": None, "quote": "Almex Aerospace Limited"}],
                        }
                    )
                    present.add("almex aerospace limited")
                    break
                if "almex aerospace limited" in text.casefold():
                    # Page mentions it; keep even beside Hindalco-Almex context if restored name exists in dropped.
                    subs.append(
                        {
                            "name": "Almex Aerospace Limited",
                            "kind": "subsidiary",
                            "descriptor": None,
                            "entity": "Almex Aerospace Limited",
                            "status": "current",
                            "revenue_share_pct": None,
                            "sources": [{"pdf_page": number, "printed_page": None, "quote": "Almex Aerospace Limited"}],
                        }
                    )
                    present.add("almex aerospace limited")
                    break
    # Drop fragment names only when the longer name adds nothing but a legal suffix
    # (Limited/Ltd/Inc…). Distinct proper-noun prefixes (Hindalco-Almex vs Almex) are kept.
    survivors = []
    legal = {"limited", "ltd", "inc", "private", "pvt", "llc", "gmbh", "company", "and", "the"}
    tokenized = [(_cf(s["name"]), set(_key(s["name"]).split()), s) for s in subs]
    for key, tokens, s in tokenized:
        drop = False
        for other_key, other_tokens, _ in tokenized:
            if key == other_key or not (tokens < other_tokens):
                continue
            extras = other_tokens - tokens
            if extras and extras <= legal:
                drop = True
                break
        if not drop:
            survivors.append(s)
    subs, _ = dedupe_strict(survivors)
    with_desc = sum(1 for s in subs if s.get("descriptor"))
    without = len(subs) - with_desc
    print(f"5 subsidiaries_kept={len(subs)} with_descriptor={with_desc} without={without}")
    for s in sorted(subs, key=lambda x: x["name"].casefold()):
        print(f"  sub desc={s.get('descriptor')!r}\t{s['name']}".encode("ascii", "replace").decode())
    return {
        "items": non_subs + subs,
        "distinct_entities": len(distinct),
        "subsidiaries": len(subs),
        "with_descriptor": with_desc,
        "without_descriptor": without,
        "verify": verify,
        "sample_entities": sorted({entities[k]["name"] for k in distinct}),
        "keep_kobe": keep_kobe,
        "keep_minerals": keep_minerals,
    }


def _page_visible_quote(name: str, text: str) -> str | None:
    """Return a short quote that actually appears on the page (handles Renewable vs Renewables)."""
    if not text:
        return None
    folded = fold_whitespace(text)
    if name in folded:
        return name
    idx = folded.casefold().find(name.casefold())
    if idx >= 0:
        return folded[idx : idx + len(name)]
    pattern = re.escape(name)
    pattern = pattern.replace("Renewables", "Renewables?")
    pattern = pattern.replace(r"\&", r"(?:\&|and)")
    match = re.search(pattern, folded, re.I)
    if match:
        return fold_whitespace(match.group(0))
    tokens = [t for t in re.findall(r"[A-Za-z0-9&]+", name) if t.casefold() not in {"limited", "ltd", "inc"}]
    if len(tokens) >= 2:
        pat = r".{0,5}".join(re.escape(t) for t in tokens[:4])
        match = re.search(pat, folded, re.I)
        if match:
            return _trim_quote(fold_whitespace(match.group(0)), name, 15)
    return None


def step6_segments(items: list[dict], pages: list[dict]) -> dict:
    pdf = locate_pdfs(INPUT_DIR).get("HINDALCO")
    texts = {int(p["pdf_page"]): page_text(p) for p in pages}
    blob = "\n".join(texts.get(n, "") for n in (319, 320, 321))
    tables = []
    if pdf:
        for number in (319, 320):
            try:
                tables.extend(extract_digital_page(Path(pdf), "HINDALCO", number))
            except Exception as exc:
                print(f"6 table error p{number}: {exc}")

    print("6 SEGMENT REVENUE TABLE (pdf 319-320)")
    print("  columns: Segment | Segment revenue | Inter-segment | Diff.accounting | External | ...")
    revenue = {}
    inter_by = {}
    total = None
    inter_total = None
    printed_rows = []
    for table in tables:
        for row in table.get("rows") or []:
            cells = [fold_whitespace(c or "") for c in (row.get("cells") or [])]
            if not cells:
                continue
            label = (cells[0] or "").strip()
            label_cf = label.casefold()
            nums = []
            for cell in cells[1:]:
                m = re.search(r"(-?[\d,]+(?:\.\d+)?)", (cell or "").replace(",", ""))
                if m:
                    try:
                        nums.append(float(m.group(1).replace(",", "")))
                    except ValueError:
                        pass
            mapped = None
            for key, nice in (
                ("novelis", "Novelis"),
                ("aluminium upstream", "Aluminium upstream"),
                ("aluminum upstream", "Aluminium upstream"),
                ("aluminium downstream", "Aluminium downstream"),
                ("aluminum downstream", "Aluminium downstream"),
                ("copper", "Copper"),
                ("others", "Others"),
                ("power", "Power"),
            ):
                if re.fullmatch(rf"{re.escape(key)}", label_cf) or label_cf == key:
                    mapped = nice
                    break
            if mapped and nums:
                revenue[mapped] = nums[0]
                if len(nums) >= 2:
                    # Second numeric is often inter-segment (may be '-')
                    raw_inter = cells[2] if len(cells) > 2 else ""
                    if raw_inter.strip() not in {"", "-", "–"}:
                        inter_by[mapped] = nums[1] if nums[1] < 0 or "(" in (cells[2] or "") else nums[1]
                printed_rows.append(cells)
                print(f"  ROW {label} | {cells[1:6]}".encode("ascii", "replace").decode())
            if label_cf == "total" and nums:
                total = nums[0]
                raw_inter = cells[2] if len(cells) > 2 else ""
                m_inter = re.search(r"\(?-?[\d,]+(?:\.\d+)?\)?", raw_inter.replace(",", ""))
                if m_inter:
                    val = float(re.sub(r"[()]", "", m_inter.group(0)))
                    inter_total = -abs(val) if "(" in raw_inter else val
                printed_rows.append(cells)
                print(f"  TOTAL {cells[:6]}".encode("ascii", "replace").decode())
            if "others" in label_cf and nums and "Others" not in revenue:
                if re.search(r"\bothers\b", label_cf):
                    revenue["Others"] = nums[0]
                    printed_rows.append(cells)
                    print(f"  ROW Others | {cells[1:6]}".encode("ascii", "replace").decode())
            if re.fullmatch(r"power", label_cf) and nums:
                revenue["Power"] = nums[0]
                printed_rows.append(cells)
                print(f"  ROW Power | {cells[1:6]}".encode("ascii", "replace").decode())

    # Fallback parse from page text if table grid missed rows
    if "Novelis" not in revenue:
        for nice, pat in (
            ("Novelis", r"Novelis\s+([\d,]+)"),
            ("Aluminium upstream", r"Aluminium Upstream\s+([\d,]+)"),
            ("Aluminium downstream", r"Aluminium Downstream\s+([\d,]+)"),
            ("Copper", r"Copper\s+([\d,]+)"),
        ):
            m = re.search(pat, texts.get(320, "") or "", re.I)
            if m:
                revenue[nice] = float(m.group(1).replace(",", ""))
        m = re.search(r"Total\s+([\d,]+)\s+\(?([\d,]+)\)?", texts.get(320, "") or "", re.I)
        if m:
            total = float(m.group(1).replace(",", ""))
            inter_total = -float(m.group(2).replace(",", ""))

    # Inter-segment profit line on p319
    m = re.search(r"Inter-segment profit/\s*\(loss\)\s*elimination[^\d\-]*\(?([\d,]+)\)?", blob, re.I)
    inter_pl = float(m.group(1).replace(",", "")) * -1 if m and "(" in (m.group(0)) else (float(m.group(1).replace(",", "")) if m else None)
    print(f"6 parsed revenue={revenue}")
    print(f"6 printed_total={total} inter_segment_col_total={inter_total} inter_segment_PL={inter_pl}")
    print(f"6 Others_printed={'Others' in revenue} Power_printed={'Power' in revenue}")

    non_seg = [i for i in items if i.get("kind") != "segment"]
    segment_rows = []
    for nice in ("Novelis", "Aluminium upstream", "Aluminium downstream", "Copper", "Others", "Power"):
        if nice in {"Others", "Power"} and nice not in revenue:
            continue  # keep only if printed
        if nice not in revenue and nice not in {"Novelis", "Aluminium upstream", "Aluminium downstream", "Copper"}:
            continue
        if nice not in revenue:
            continue
        share = round(100.0 * float(revenue[nice]) / float(total), 2) if total and total > 0 else None
        segment_rows.append(
            {
                "name": nice,
                "revenue": revenue.get(nice),
                "unit": "INR crore",
                "share_pct": share,
            }
        )
        print(f"  share {nice}: revenue={revenue.get(nice)} inter_segment={inter_by.get(nice)} share_pct={share}")
        non_seg.append(
            {
                "name": nice,
                "kind": "segment",
                "descriptor": f"{nice} reportable segment",
                "entity": "company",
                "status": "current",
                "revenue_share_pct": share,
                "sources": [{"pdf_page": 320, "printed_page": None, "quote": _segment_quote(nice, blob)}],
            }
        )
    print(f"6 segments_kept={len(segment_rows)} shares={[ (r['name'], r['share_pct']) for r in segment_rows ]}")
    return {
        "items": non_seg,
        "segments": segment_rows,
        "total": total,
        "inter_segment": inter_total,
        "inter_segment_PL": inter_pl,
        "revenue": revenue,
        "printed_rows": printed_rows[:40],
    }


def step7_quote_review(items: list[dict], pages_by: dict, texts: dict[int, str], list_pages: set[int]) -> dict:
    cited = {s["pdf_page"] for i in items for s in i.get("sources") or [] if isinstance(s.get("pdf_page"), int)}
    pdf_meta = _pdf_layer_meta(cited)
    # Pre-fix subsidiary quotes to page-visible forms before snap.
    fixed = []
    for item in items:
        if item.get("kind") == "subsidiary":
            sources = []
            for src in item.get("sources") or []:
                page = src.get("pdf_page")
                text = texts.get(page, "") if isinstance(page, int) else ""
                visible = _page_visible_quote(item.get("name") or "", text or "")
                if visible:
                    sources.append({**src, "quote": visible})
                else:
                    sources.append(src)
            item = {**item, "sources": sources or item.get("sources")}
        fixed.append(item)
    snapped = step7_quotes(fixed, pages_by, texts, pdf_meta)
    # Rescue list-page subsidiaries dropped only for spelling drift (Renewable vs Renewables).
    kept_names = {_cf(i["name"]) for i in snapped["items"]}
    dropped_names = {_cf(d.get("name") or "") for d in (snapped.get("dropped") or [])}
    rescued = []
    for item in fixed:
        if item.get("kind") != "subsidiary":
            continue
        key = _cf(item.get("name") or "")
        if key in kept_names or key not in dropped_names:
            continue
        name = item.get("name") or ""
        page = (item.get("sources") or [{}])[0].get("pdf_page")
        if page not in list_pages:
            page = next((p for p in list_pages if _page_visible_quote(name, texts.get(p, "") or "")), None)
        if not isinstance(page, int):
            continue
        visible = _page_visible_quote(name, texts.get(page, "") or "")
        if not visible:
            continue
        rescued.append(
            {
                **item,
                "sources": [{"pdf_page": page, "printed_page": None, "quote": visible}],
            }
        )
        kept_names.add(key)
        print(f"  rescued_sub {name} p{page}".encode("ascii", "replace").decode())
    items = snapped["items"] + rescued
    name_only = []
    for item in items:
        for src in item.get("sources") or []:
            quote = fold_whitespace(src.get("quote") or "")
            name = fold_whitespace(item.get("name") or "")
            page = src.get("pdf_page")
            if quote.casefold() == name.casefold() and page not in list_pages:
                # skip if page looks like a table-heavy notes page
                text = texts.get(page, "") or ""
                if not re.search(r"form\s*aoc|investment in subsidiar|name of the entity|principal activity", text[:1500], re.I):
                    name_only.append(
                        {
                            "name": item.get("name"),
                            "kind": item.get("kind"),
                            "pdf_page": page,
                            "quote": quote,
                        }
                    )
                break
    print(f"7 quotes kept={len(items)} dropped={len(snapped['dropped'])} name_only_review={len(name_only)} {snapped['quotes']}")
    for row in name_only[:40]:
        print(f"  name_only p{row['pdf_page']}\t{row['kind']}\t{row['name']}".encode("ascii", "replace").decode())
    return {"items": items, "dropped": snapped["dropped"], "quotes": snapped["quotes"], "name_only_review": name_only}


def promote(sheet: dict, markdown: str) -> dict:
    staging = STAGING_DIR / "final" / "HINDALCO"
    staging.mkdir(parents=True, exist_ok=True)
    _atomic(staging / "HINDALCO.json", json.dumps(sheet, indent=2, ensure_ascii=False) + "\n")
    _atomic_text(staging / "HINDALCO.md", markdown)
    # Promote to outputs + submission
    out_dir = OUTPUT_DIR / "HINDALCO"
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic(out_dir / "HINDALCO.json", json.dumps(sheet, indent=2, ensure_ascii=False) + "\n")
    _atomic_text(out_dir / "HINDALCO.md", markdown)
    sub = SUBMISSION_DIR / "HINDALCO"
    sub.mkdir(parents=True, exist_ok=True)
    _atomic(sub / "HINDALCO.json", json.dumps(sheet, indent=2, ensure_ascii=False) + "\n")
    _atomic_text(sub / "HINDALCO.md", markdown)
    # Refresh work copy
    _atomic(WORK / "HINDALCO.json", json.dumps(sheet, indent=2, ensure_ascii=False) + "\n")
    _atomic_text(WORK / "HINDALCO.md", markdown)
    return {"staging": str(staging), "outputs": str(out_dir), "submission": str(sub)}


def run() -> dict:
    WORK.mkdir(parents=True, exist_ok=True)
    report: dict = {"steps": {}}

    s0 = step0_verify()
    report["steps"]["0"] = {k: v for k, v in s0.items() if k != "sheet"}
    sheet0 = s0["sheet"]
    # Work on a copy under work/
    items = list(sheet0.get("items") or [])
    _atomic(WORK / "baseline.json", json.dumps(sheet0, indent=2, ensure_ascii=False) + "\n")

    pages = pages_for_extraction(load_saved_company("HINDALCO"))
    pages_by = {int(p["pdf_page"]): p for p in pages}
    meta = identify_list_pages(pages)
    dropped_index = load_dropped_index()

    s1 = step1_restore(items, dropped_index)
    items = s1["items"]
    report["steps"]["1"] = {"kept": len(items), "restored": s1["restored"], "skipped": s1["skipped"][:30]}

    s2 = step2_drop(items)
    items = s2["items"]
    report["steps"]["2"] = {"kept": len(items), "dropped": s2["dropped"]}

    s3 = step3_merge_brands(items)
    items = s3["items"]
    # Drop junk AL:sust alloy-code variants; keep plain AL:sust.
    cleaned_brands = []
    for item in items:
        key = _cf(item.get("name") or "")
        if key.startswith("al:sust") and key != "al:sust":
            continue
        cleaned_brands.append(item)
    items = cleaned_brands
    report["steps"]["3"] = {"kept": len(items), "merged_from": s3["merged_from"]}

    s4 = step4_dedup_fix(items, dropped_index)
    items = s4["items"]
    report["steps"]["4"] = {"kept": len(items), "restored": s4["restored"], "notes": s4["notes"], "dropped": len(s4["dropped"])}

    s5 = step5_subsidiaries_tables(items, pages, meta)
    items = s5["items"]
    report["steps"]["5"] = {
        "distinct_entities": s5["distinct_entities"],
        "subsidiaries": s5["subsidiaries"],
        "with_descriptor": s5["with_descriptor"],
        "without_descriptor": s5["without_descriptor"],
        "verify_kobe_hits": len(s5["verify"].get("Kobe Steel Ltd") or []),
        "verify_minerals_hits": len(s5["verify"].get("Minerals & Minerals Limited") or []) + len(s5["verify"].get("Minerals and Minerals Limited") or []),
        "sample_entities": s5["sample_entities"],
        "kept_items": len(items),
    }

    s6 = step6_segments(items, pages)
    items = s6["items"]
    report["steps"]["6"] = {
        "segments": s6["segments"],
        "total": s6["total"],
        "inter_segment": s6["inter_segment"],
        "revenue": s6["revenue"],
        "kept_items": len(items),
    }

    header = step6_header(meta["texts"])
    # Prefer existing good header from 191 sheet if still grounded.
    if sheet0.get("one_liner") and "hindalco industries limited" in sheet0["one_liner"].casefold():
        header = {
            "one_liner": sheet0["one_liner"],
            "summary": sheet0.get("summary") or sheet0["one_liner"],
            "pages": [33, 38, 44, 67],
        }
    report["steps"]["header"] = {"one_liner": header["one_liner"], "summary": (header.get("summary") or "")[:240]}
    print(f"header: {header['one_liner'][:160].encode('ascii','replace').decode()}")

    s7 = step7_quote_review(items, pages_by, meta["texts"], set(meta["list_pages"]))
    items = s7["items"]
    report["steps"]["7"] = {
        "kept": len(items),
        "dropped": len(s7["dropped"]),
        "quotes": s7["quotes"],
        "name_only_review": s7["name_only_review"],
    }

    items = step8_printed(items, pages)
    report["steps"]["8"] = {"kept": len(items)}
    print(f"8 printed kept={len(items)}")

    shaped = []
    for item in items:
        # Allow null descriptor for subsidiaries → schema requires non-empty descriptor.
        if not (item.get("descriptor") or "").strip():
            item = {**item, "descriptor": item.get("name") or "subsidiary"}
        row = _shape(item)
        if row:
            shaped.append(row)
    shaped = sorted(shaped, key=lambda i: (min(s["pdf_page"] for s in i["sources"]), i["name"]))

    final_sheet = {
        "ticker": "HINDALCO",
        "period": "FY25",
        "one_liner": header["one_liner"],
        "summary": header.get("summary") or header["one_liner"],
        "segments": s6["segments"],
        "items": shaped,
    }
    validated = FactSheet.model_validate(final_sheet).model_dump(mode="json")
    markdown = render_markdown(validated)
    errors = consistency_errors(validated, markdown)
    dupes = [n for n, c in Counter(i["name"].casefold() for i in validated["items"]).items() if c > 1]
    quote_bad = [i["name"] for i in validated["items"] if not (1 <= len(i["sources"]) <= 3)]
    ok = not errors and not dupes and not quote_bad
    report["steps"]["validate"] = {"ok": ok, "items": len(validated["items"]), "dupes": dupes, "errors": errors, "quote_bad": quote_bad}
    print(f"validate ok={ok} items={len(validated['items'])} dupes={dupes}")

    # Origin / kind counts
    cache = load_cached_candidates("HINDALCO")
    origin = {}
    for c in cache["all"]:
        origin.setdefault(normalize(c.get("name") or ""), c.get("source_batch"))
    by_source = Counter(origin.get(normalize(i["name"]), "unknown") for i in validated["items"])
    subs = [i for i in validated["items"] if i["kind"] == "subsidiary"]
    with_desc = sum(
        1
        for i in subs
        if fold_whitespace(i.get("descriptor") or "").casefold() != fold_whitespace(i["name"]).casefold()
    )
    report["summary"] = {
        "by_kind": dict(Counter(i["kind"] for i in validated["items"])),
        "by_source": dict(by_source),
        "total_items": len(validated["items"]),
        "subsidiaries_total": len(subs),
        "subsidiaries_with_descriptor": with_desc,
        "subsidiaries_without_descriptor": len(subs) - with_desc,
    }
    print("SUMMARY", json.dumps(report["summary"], indent=2))

    # CSV
    csv_path = WORK / "items.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "kind", "entity", "pdf_page", "quote", "descriptor"])
        writer.writeheader()
        for item in validated["items"]:
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

    # New random sample of 60 (different seed)
    sample = stratified_sample(validated["items"], 60, seed=99)
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
    print("SAMPLE60")
    for row in report["sample_60"]:
        line = f"{row['pdf_page']}\t{row['kind']}\t{row['name']}\t{row['quote'][:100]}"
        print(line.encode("ascii", "replace").decode())

    # Write staging always; promote only if ok
    staging_dir = STAGING_DIR / "final" / "HINDALCO"
    staging_dir.mkdir(parents=True, exist_ok=True)
    _atomic(staging_dir / "HINDALCO.json", json.dumps(validated, indent=2, ensure_ascii=False) + "\n")
    _atomic_text(staging_dir / "HINDALCO.md", markdown)
    _atomic(WORK / "report.json", json.dumps(_jsonable(report), indent=2, ensure_ascii=False) + "\n")

    if ok:
        paths = promote(validated, markdown)
        report["promoted"] = paths
        report["wrote_ok"] = True
        print(f"PROMOTED items={len(validated['items'])} -> outputs + submission")
    else:
        report["wrote_ok"] = False
        print("SKIP promote: validation failed")

    _atomic(WORK / "report.json", json.dumps(_jsonable(report), indent=2, ensure_ascii=False) + "\n")
    print(f"FINAL_ITEM_COUNT={len(validated['items'])}")
    return report


def stratified_sample(items: list[dict], n: int, seed: int = 99) -> list[dict]:
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


if __name__ == "__main__":
    run()
