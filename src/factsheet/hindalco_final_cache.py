"""Build HINDALCO.json from cached candidates only. No Groq calls."""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from rapidfuzz import fuzz

from src.config.paths import CANDIDATE_DIR, INPUT_DIR, OUTPUT_DIR, PAGE_DIR, PROJECT_ROOT
from src.dedup.normalize import normalize
from src.extract.evidence import pages_for_extraction
from src.extract.rules import extract_listed
from src.extract.run import load_saved_company
from src.extract.schema import KINDS, STATUSES
from src.extract.select import page_text, select_pages
from src.factsheet.build import consistency_errors, describe_company, render_markdown
from src.factsheet.postprocess import (
    _COMPANY,
    _kind_rank,
    _name_in_quote_or_sentence,
    _name_key,
    _printed_for_quote,
    _text,
    assign_printed_pages,
)
from src.factsheet.schema import FactSheet
from src.factsheet.stage13_audit import company_overview, overview_text
from src.validate.quotes import fold_whitespace, locate_quote, name_tokens, repair_named_quote

# Expanded junk filter (names / descriptors / page-topic cues).
_JUNK = re.compile(
    r"\b(agm|e-?\s*voting|instavote|instameet|dividend|book closure|iepf|"
    r"remuneration|auditor|re-?\s*appointment|appointment of|"
    r"material topic|m\d+\s*:|"
    r"(?:financial|human|intellectual|manufactured|natural|social(?: and relationship)?)\s+capital|"
    r"six capitals?|risk management|\bhedg\w*|"
    r"\bghg\b|scope\s*[123]\b|water[- ]?(saving|positivity|intensity)|zero[- ]waste|"
    r"waste\s+target|emission\s+target|"
    r"\bhr\b|training programme|training program|engagement programme|engagement program|"
    r"stakeholder engagement|awareness programme|awareness program|"
    r"business ethics|integrity framework|"
    r"committee|task\s*force|job title|chief\s+\w+\s+officer|\bdirector\b|"
    r"\bpolicy\b|framework|charter|"
    r"membership|association|chamber of commerce|"
    r"\bawards?\b|certification|esg\s+metric|brsr|"
    r"future[- ]?ready growth)\b",
    re.I,
)
_ACTIVITY = re.compile(
    r"\b(produc|manufactur|operat|smelt|refin|mining|mine|plant|subsidiary|joint venture|"
    r"alumina|aluminium|aluminum|copper|rolled|downstream|upstream|facility|site|"
    r"acquired|owns|holding|invested|commissioned)\b",
    re.I,
)
_TM = re.compile(r"[™®©]", re.I)
FUZZY_MERGE = 90


def load_cached_candidates(ticker: str = "HINDALCO") -> dict:
    """Rules + successful LLM batches from data/processed/candidates only."""
    parsed = CANDIDATE_DIR / ticker / "parsed"
    rules_items = []
    llm_items = []
    llm_pages = set()
    for path in sorted(parsed.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        is_rules = record.get("source") == "rules" or "_rules_" in path.name
        if is_rules:
            for index, candidate in enumerate(record.get("candidates") or []):
                rules_items.append({**candidate, "candidate_id": f"rules:{index}", "source_batch": "rules"})
            continue
        if record.get("source") == "truncated_salvage":
            continue
        if record.get("request_status") != "success":
            continue
        pages = [int(p) for p in (record.get("target_pages") or [])]
        llm_pages.update(pages)
        for index, candidate in enumerate(record.get("candidates") or []):
            llm_items.append(
                {
                    **candidate,
                    "candidate_id": f"{path.stem}:{index}",
                    "source_batch": "llm",
                }
            )
    return {
        "rules": rules_items,
        "llm": llm_items,
        "llm_pages": sorted(llm_pages),
        "all": rules_items + llm_items,
    }


def junk_filter(items: list[dict], pages_by: dict[int, dict]) -> tuple[list[dict], list[dict]]:
    kept, dropped = [], []
    for item in items:
        reason = _junk_reason(item, pages_by)
        if reason:
            dropped.append(_drop_row(item, reason))
            continue
        kept.append(item)
    return kept, dropped


def name_only_filter(items: list[dict], texts: dict[int, str]) -> dict:
    kept = []
    dropped = []
    all_subs = 0
    subs_with_activity = 0
    for item in items:
        kind = item.get("kind")
        name = (item.get("name") or "").strip()
        descriptor = (item.get("descriptor") or "").strip()
        quotes = [ (s.get("quote") or "").strip() for s in item.get("sources") or [] ]
        if kind == "other" and quotes and all(fold_whitespace(q).casefold() == fold_whitespace(name).casefold() for q in quotes if q):
            dropped.append(_drop_row(item, "name_only_other"))
            continue
        if kind == "subsidiary":
            all_subs += 1
            page_nos = [s.get("pdf_page") for s in item.get("sources") or [] if isinstance(s.get("pdf_page"), int)]
            page_blob = "\n".join(texts.get(n, "") for n in page_nos)
            desc = descriptor or name
            desc_on_page = bool(desc) and (
                locate_quote(desc, page_blob) is not None
                or fold_whitespace(desc).casefold() in fold_whitespace(page_blob).casefold()
                or any(locate_quote(q, page_blob) for q in quotes if q)
            )
            richer = fold_whitespace(desc).casefold() != fold_whitespace(name).casefold()
            activity = desc_on_page and (
                richer
                or bool(_ACTIVITY.search(desc))
                or bool(_ACTIVITY.search(" ".join(quotes)))
            )
            if not desc_on_page:
                dropped.append(_drop_row(item, "subsidiary_no_page_descriptor"))
                continue
            if activity:
                subs_with_activity += 1
            kept.append({**item, "descriptor": desc, "_has_activity": bool(activity)})
            continue
        kept.append(item)
    return {
        "items": kept,
        "dropped": dropped,
        "all_subsidiaries": all_subs,
        "subsidiaries_with_activity": subs_with_activity,
        "subsidiaries_kept": sum(1 for i in kept if i.get("kind") == "subsidiary"),
    }


def dedupe_items(items: list[dict]) -> tuple[list[dict], list[dict], int]:
    """Normalize + fuzzy >= 0.9 within entity. One item per entity group, <=3 quotes, one kind."""
    # Bucket by entity first.
    by_entity: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        entity = _entity_norm(item.get("entity") or "company")
        by_entity[entity].append(item)

    kept = []
    dropped = []
    merged_away = 0
    for entity, members in by_entity.items():
        groups: list[list[dict]] = []
        for item in members:
            placed = False
            for group in groups:
                if _same_entity_item(group[0], item):
                    group.append(item)
                    placed = True
                    break
            if not placed:
                groups.append([item])
        for group in groups:
            if len(group) == 1:
                cleaned = _merge_group(group)
                kept.append(cleaned)
                continue
            merged_away += len(group) - 1
            for extra in group[1:]:
                dropped.append(_drop_row(extra, "dedup_merged"))
            kept.append(_merge_group(group))
    return kept, dropped, merged_away


def fill_entity(items: list[dict], subsidiaries: list[str], pages: list[dict]) -> list[dict]:
    exact = {_name_key(name): name for name in subsidiaries}
    filled = []
    for item in items:
        entity = (item.get("entity") or "").strip()
        key = _name_key(entity)
        name_key = _name_key(item.get("name") or "")
        if entity.casefold() in _COMPANY or key in {"company", ""}:
            chosen = "company"
        elif key in exact:
            chosen = exact[key]
        elif item.get("kind") == "subsidiary" and name_key in exact:
            chosen = exact[name_key]
        elif item.get("kind") == "subsidiary":
            chosen = item.get("name") or "company"
        else:
            chosen = "company"
        filled.append({**item, "entity": chosen})
    return assign_printed_pages(filled, pages)


def resnap_quotes(items: list[dict], texts: dict[int, str]) -> dict:
    kept, dropped = [], []
    quote_counts = Counter()
    for item in items:
        sources = []
        for source in item.get("sources") or []:
            number = source.get("pdf_page")
            stored = texts.get(number, "") if isinstance(number, int) else ""
            original = source.get("quote") or ""
            repaired = repair_named_quote(original, stored, item.get("name") or "")
            quote = repaired.get("quote")
            if repaired["outcome"] == "dropped" or not quote:
                quote_counts["dropped_resnap"] += 1
                continue
            if not _name_in_quote_or_sentence(item.get("name") or "", quote, stored):
                quote_counts["dropped_name_not_in_quote"] += 1
                continue
            kind = locate_quote(quote, stored)
            if kind is None:
                quote_counts["dropped_not_on_page"] += 1
                continue
            if quote == original and kind == "exact":
                quote_counts["strict"] += 1
            elif quote == original:
                quote_counts["normalized"] += 1
            else:
                quote_counts["repaired"] += 1
            sources.append({**source, "quote": quote})
        if not sources:
            dropped.append(_drop_row(item, "no_valid_quote_after_resnap"))
            continue
        kept.append({**item, "sources": sources[:3]})
    return {"items": kept, "dropped": dropped, "quotes": dict(quote_counts)}


def verify_pdf_layer(items: list[dict], pdf_texts: dict[int, dict]) -> dict:
    """pdf_texts[page] = {'text': str, 'has_layer': bool}."""
    kept, dropped = [], []
    counts = Counter()
    for item in items:
        sources = []
        for source in item.get("sources") or []:
            number = source.get("pdf_page")
            quote = source.get("quote") or ""
            meta = pdf_texts.get(number) if isinstance(number, int) else None
            if not meta or not meta.get("has_layer"):
                counts["skipped_no_layer"] += 1
                sources.append(source)
                continue
            kind = locate_quote(quote, meta.get("text") or "")
            if kind == "exact":
                counts["strict"] += 1
                sources.append(source)
            elif kind:
                counts["normalized"] += 1
                sources.append(source)
            else:
                # Try repair against PDF layer.
                repaired = repair_named_quote(quote, meta.get("text") or "", item.get("name") or "")
                if repaired.get("quote") and locate_quote(repaired["quote"], meta.get("text") or ""):
                    counts["repaired"] += 1
                    sources.append({**source, "quote": repaired["quote"]})
                else:
                    counts["dropped"] += 1
        if not sources:
            dropped.append(_drop_row(item, "pdf_layer_no_valid_quote"))
            continue
        kept.append({**item, "sources": sources[:3]})
    return {"items": kept, "dropped": dropped, "quotes": dict(counts)}


def build_header(pages: list[dict], texts: dict[int, str], items: list[dict]) -> dict:
    selection = select_pages(pages)
    overview_pages = [
        row["pdf_page"]
        for row in selection["pages"]
        if row.get("section") == "business_overview" and row.get("decision") == "keep"
    ]
    narrative_pages = []
    for number in overview_pages:
        text = texts.get(number, "")
        if _looks_like_toc(text):
            continue
        narrative_pages.append(number)
    # Pull MDA + directors prose as well when business_overview pages are layout-heavy.
    for row in selection["pages"]:
        if row.get("section") in {"mda", "directors_report"} and row.get("decision") == "keep":
            number = int(row["pdf_page"])
            text = texts.get(number, "")
            if text and not _looks_like_toc(text) and number not in narrative_pages:
                narrative_pages.append(number)
    blob = "\n".join(texts.get(n, "")[:1200] for n in narrative_pages[:16])
    one_liner, summary = _prose_overview(blob)
    source = "business_overview_prose"
    check_blob = blob
    if not one_liner or not _grounded(one_liner, blob):
        overview = overview_text(texts, pages)
        one_liner, summary = _prose_overview(overview)
        source = "overview_prose"
        check_blob = overview
    if not one_liner or not _is_prose_sentence(one_liner):
        # Last resort: stage13 ranker, then reject non-prose.
        overview = overview_text(texts, pages)
        cand_one, cand_sum = company_overview(overview)
        if _is_prose_sentence(cand_one) and _grounded(cand_one, overview):
            one_liner, summary = cand_one, cand_sum
            source = "overview_ranked"
            check_blob = overview
        else:
            # Keep a short grounded fragment if present.
            fragment = _first_grounded_fragment(overview or blob)
            one_liner = fragment or "Hindalco Industries Limited produces aluminium and copper products."
            summary = fragment or one_liner
            source = "overview_fragment_or_fallback"
            check_blob = overview or blob
    listed = extract_listed(pages)
    segments = _segments_from_listed(listed, items)
    return {
        "one_liner": one_liner,
        "summary": summary,
        "segments": segments,
        "overview_pages": overview_pages,
        "header_source": source,
        "listed_subsidiaries": len(listed.get("subsidiaries") or []),
        "listed_segments": len(listed.get("segments") or []),
        "grounded_one_liner": _grounded(one_liner, check_blob) if check_blob else False,
        "grounded_summary": _grounded(summary, check_blob) if check_blob else False,
    }


def _looks_like_toc(text: str) -> bool:
    head = (text or "")[:800].casefold()
    if head.count("\n") >= 12 and ("contents" in head or "corporate overview" in head):
        return True
    digits = sum(ch.isdigit() for ch in head)
    return digits >= 50 and "aluminium" not in head and "copper" not in head


def _is_prose_sentence(text: str) -> bool:
    raw = (text or "").strip()
    if not raw or raw.count("\n") > 1:
        return False
    if "+" in raw or re.search(r"production capacity|at a glance|operations\s*\(", raw, re.I):
        return False
    words = raw.split()
    if len(words) < 8 or len(words) > 60:
        return False
    return bool(re.search(r"\b(is|are|has|have|produces|produce|operates|operate|manufactures|provides)\b", raw, re.I))


def _prose_overview(source_text: str) -> tuple[str, str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", source_text or "") if s.strip()]
    ranked = []
    for sentence in sentences:
        flat = fold_whitespace(sentence)
        if not _is_prose_sentence(flat):
            continue
        folded = flat.casefold()
        if "aluminium" not in folded and "aluminum" not in folded and "copper" not in folded:
            continue
        score = 0
        if folded.startswith("hindalco") or "hindalco industries" in folded:
            score += 5
        if "aluminium" in folded and "copper" in folded:
            score += 3
        if "novelis" in folded or "downstream" in folded or "upstream" in folded:
            score += 2
        cleaned = flat if flat.endswith(".") else flat + "."
        ranked.append((score, cleaned))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    if not ranked:
        return "", ""
    one_liner = ranked[0][1]
    summary = " ".join(sentence for _score, sentence in ranked[:3])
    return one_liner, summary


def _first_grounded_fragment(source_text: str) -> str | None:
    for sentence in re.split(r"(?<=[.!?])\s+", source_text or ""):
        flat = fold_whitespace(sentence)
        if _is_prose_sentence(flat) and ("aluminium" in flat.casefold() or "copper" in flat.casefold()):
            return flat if flat.endswith(".") else flat + "."
    return None


def validate_sheet(sheet: dict) -> dict:
    validated = FactSheet.model_validate(sheet).model_dump(mode="json")
    names = [item["name"] for item in validated["items"]]
    dupes = [name for name, count in Counter(normalize(n) for n in names).items() if count > 1]
    # Prefer exact-name duplicates for schema rule "no duplicate names".
    exact_dupes = [name for name, count in Counter(n.casefold() for n in names).items() if count > 1]
    quote_violations = [item["name"] for item in validated["items"] if len(item["sources"]) > 3 or len(item["sources"]) < 1]
    markdown = render_markdown(validated)
    errors = consistency_errors(validated, markdown)
    ok = not exact_dupes and not quote_violations and not errors
    return {
        "ok": ok,
        "sheet": validated,
        "markdown": markdown,
        "duplicate_names": exact_dupes,
        "normalized_duplicate_names": dupes,
        "quote_violations": quote_violations,
        "consistency_errors": errors,
    }


def run() -> dict:
    report: dict = {"steps": {}}
    cached = load_cached_candidates("HINDALCO")
    report["steps"]["load"] = {
        "rules_candidates": len(cached["rules"]),
        "llm_candidates": len(cached["llm"]),
        "llm_pages": len(cached["llm_pages"]),
        "total": len(cached["all"]),
    }
    print(f"LOAD rules={len(cached['rules'])} llm={len(cached['llm'])} llm_pages={len(cached['llm_pages'])} total={len(cached['all'])}")

    pages = pages_for_extraction(load_saved_company("HINDALCO", PAGE_DIR))
    pages_by = {int(p["pdf_page"]): p for p in pages}
    texts = {int(p["pdf_page"]): page_text(p) for p in pages}

    # Tag source for later reporting.
    items = []
    for item in cached["all"]:
        items.append({**item, "_origin": item.get("source_batch") or "unknown"})

    # 2a junk
    items, dropped_junk = junk_filter(items, pages_by)
    report["steps"]["2a_junk"] = {"kept": len(items), "dropped": len(dropped_junk)}
    print(f"2a junk kept={len(items)} dropped={len(dropped_junk)}")

    # 2b name-only / subsidiary descriptor
    name_only = name_only_filter(items, texts)
    items = name_only["items"]
    dropped_name = name_only["dropped"]
    report["steps"]["2b_name_only"] = {
        "kept": len(items),
        "dropped": len(dropped_name),
        "all_subsidiaries": name_only["all_subsidiaries"],
        "subsidiaries_with_activity": name_only["subsidiaries_with_activity"],
        "subsidiaries_kept": name_only["subsidiaries_kept"],
    }
    print(
        f"2b name-only kept={len(items)} dropped={len(dropped_name)} "
        f"all_subs={name_only['all_subsidiaries']} with_activity={name_only['subsidiaries_with_activity']} "
        f"subs_kept={name_only['subsidiaries_kept']}"
    )

    # 2c dedup
    items, dropped_dedup, merged_away = dedupe_items(items)
    report["steps"]["2c_dedup"] = {"kept": len(items), "dropped": len(dropped_dedup), "merged_away": merged_away}
    print(f"2c dedup kept={len(items)} merged_away={merged_away}")

    # 2d entity + printed_page
    listed = extract_listed(pages)
    sub_names = [item["name"] for item in listed.get("subsidiaries") or []]
    items = fill_entity(items, sub_names, pages)
    report["steps"]["2d_entity"] = {"kept": len(items)}
    print(f"2d entity filled={len(items)}")

    # 2e resnap
    resnap = resnap_quotes(items, texts)
    items = resnap["items"]
    dropped_resnap = resnap["dropped"]
    report["steps"]["2e_resnap"] = {"kept": len(items), "dropped": len(dropped_resnap), "quotes": resnap["quotes"]}
    print(f"2e resnap kept={len(items)} dropped={len(dropped_resnap)} quotes={resnap['quotes']}")

    # 2f PDF layer
    cited = {s["pdf_page"] for item in items for s in item.get("sources") or [] if isinstance(s.get("pdf_page"), int)}
    pdf_texts = _pdf_layer_meta(cited)
    pdf = verify_pdf_layer(items, pdf_texts)
    items = pdf["items"]
    dropped_pdf = pdf["dropped"]
    report["steps"]["2f_pdf"] = {"kept": len(items), "dropped": len(dropped_pdf), "quotes": pdf["quotes"]}
    print(f"2f pdf kept={len(items)} dropped={len(dropped_pdf)} quotes={pdf['quotes']}")

    # Collapse remaining case-insensitive duplicate names (schema: no duplicate names).
    items, dropped_dup_names = _collapse_duplicate_names(items)
    report["steps"]["2c_dup_names"] = {"kept": len(items), "dropped": len(dropped_dup_names)}
    print(f"dup-name collapse kept={len(items)} dropped={len(dropped_dup_names)}")

    # Clean internal flags / finalize item shape.
    cleaned_items = []
    origin_by_name = {}
    for item in items:
        shaped = _shape_item(item)
        if not shaped:
            continue
        origin_by_name[normalize(shaped["name"])] = item.get("_origin") or origin_by_name.get(normalize(shaped["name"]))
        cleaned_items.append(shaped)
    report["steps"]["shaped"] = {"kept": len(cleaned_items)}
    print(f"shaped={len(cleaned_items)}")

    # 2g header
    header = build_header(pages, texts, cleaned_items)
    report["steps"]["2g_header"] = {
        "overview_pages": header["overview_pages"],
        "header_source": header["header_source"],
        "segments": len(header["segments"]),
        "one_liner": header["one_liner"],
    }
    print(f"2g header source={header['header_source']} segments={len(header['segments'])}")

    sheet = {
        "ticker": "HINDALCO",
        "period": "FY25",
        "one_liner": header["one_liner"],
        "summary": header["summary"],
        "segments": header["segments"],
        "items": sorted(cleaned_items, key=lambda item: (min(s["pdf_page"] for s in item["sources"]), item["name"])),
    }

    # 2h validate
    validated = validate_sheet(sheet)
    report["steps"]["2h_validate"] = {
        "ok": validated["ok"],
        "item_count": len(validated["sheet"]["items"]),
        "duplicate_names": validated["duplicate_names"],
        "quote_violations": validated["quote_violations"],
        "consistency_errors": validated["consistency_errors"],
    }
    print(f"2h validate ok={validated['ok']} items={len(validated['sheet']['items'])}")

    # Counts for report
    final_items = validated["sheet"]["items"]
    by_kind = Counter(item["kind"] for item in final_items)
    for item in cached["all"]:
        origin_by_name.setdefault(normalize(item.get("name") or ""), item.get("source_batch"))
    by_source = Counter(origin_by_name.get(normalize(item["name"]), "unknown") for item in final_items)

    final_subs = [item for item in final_items if item["kind"] == "subsidiary"]
    name_only_subs = sum(
        1
        for item in final_subs
        if fold_whitespace(item.get("descriptor") or "").casefold() == fold_whitespace(item["name"]).casefold()
        and not _ACTIVITY.search(" ".join(s.get("quote") or "" for s in item.get("sources") or []))
    )
    with_activity_subs = len(final_subs) - name_only_subs

    all_dropped = dropped_junk + dropped_name + dropped_dedup + dropped_dup_names + dropped_resnap + dropped_pdf
    dropped_by_rule = defaultdict(list)
    for row in all_dropped:
        dropped_by_rule[row["reason"]].append(row)

    sample = random.Random(42).sample(final_items, min(40, len(final_items)))
    sample_rows = [
        {
            "name": item["name"],
            "kind": item["kind"],
            "entity": item["entity"],
            "pdf_page": item["sources"][0]["pdf_page"],
            "quote": item["sources"][0]["quote"],
        }
        for item in sample
    ]

    report["summary"] = {
        "by_kind": dict(by_kind),
        "by_source": dict(by_source),
        "total_items": len(final_items),
        "total_with_name_only_subsidiaries": len(final_items),
        "total_without_name_only_subsidiaries": len(final_items) - name_only_subs,
        "name_only_subsidiaries_in_final": name_only_subs,
        "subsidiaries_with_activity_in_final": with_activity_subs,
        "all_subsidiaries_seen_2b": name_only["all_subsidiaries"],
        "subsidiaries_with_activity_2b": name_only["subsidiaries_with_activity"],
    }
    report["sample_40"] = sample_rows
    report["dropped_by_rule"] = {rule: rows for rule, rows in sorted(dropped_by_rule.items())}
    report["dropped_counts"] = {rule: len(rows) for rule, rows in sorted(dropped_by_rule.items())}

    # Work on copy first.
    copy_dir = PROJECT_ROOT / "eval" / "hindalco_final_build"
    copy_dir.mkdir(parents=True, exist_ok=True)
    _atomic(copy_dir / "HINDALCO.copy.json", json.dumps(validated["sheet"], indent=2, ensure_ascii=False) + "\n")
    _atomic_text(copy_dir / "HINDALCO.copy.md", validated["markdown"])
    report["copy_paths"] = {
        "json": str(copy_dir / "HINDALCO.copy.json"),
        "md": str(copy_dir / "HINDALCO.copy.md"),
        "report": str(copy_dir / "report.json"),
    }

    wrote = False
    if validated["ok"]:
        out_dir = OUTPUT_DIR / "HINDALCO"
        out_dir.mkdir(parents=True, exist_ok=True)
        _atomic(out_dir / "HINDALCO.json", json.dumps(validated["sheet"], indent=2, ensure_ascii=False) + "\n")
        _atomic_text(out_dir / "HINDALCO.md", validated["markdown"])
        wrote = True
        report["wrote"] = {"json": str(out_dir / "HINDALCO.json"), "md": str(out_dir / "HINDALCO.md")}
        print(f"WROTE outputs/HINDALCO/HINDALCO.json and .md items={len(final_items)}")
    else:
        report["wrote"] = None
        print("SKIP write: validation failed")

    report["wrote_ok"] = wrote
    _atomic(copy_dir / "report.json", json.dumps(_jsonable(report), indent=2, ensure_ascii=False) + "\n")
    return report


def _junk_reason(item: dict, pages_by: dict[int, dict]) -> str | None:
    name = item.get("name") or ""
    descriptor = item.get("descriptor") or ""
    quotes = " ".join(s.get("quote") or "" for s in item.get("sources") or [])
    blob = f"{name} {descriptor} {quotes}"
    if _JUNK.search(_TM.sub("", blob)):
        return "junk_filter"
    # Page-level AGM/governance cue when every source page opens as notice/e-voting.
    pages = [s.get("pdf_page") for s in item.get("sources") or [] if isinstance(s.get("pdf_page"), int)]
    if pages and all(_page_is_admin(pages_by.get(p)) for p in pages):
        return "junk_admin_page"
    return None


def _page_is_admin(page: dict | None) -> bool:
    if not page:
        return False
    head = _text(page)[:1200]
    return bool(re.search(r"notice of the .{0,40}annual general meeting|remote e-?\s*voting|instavote|book closure", head, re.I))


def _same_entity_item(left: dict, right: dict) -> bool:
    left_name = _dedupe_key(left.get("name") or "")
    right_name = _dedupe_key(right.get("name") or "")
    if not left_name or not right_name:
        return False
    if left_name == right_name:
        return True
    score = fuzz.token_set_ratio(left_name, right_name)
    return score >= FUZZY_MERGE


def _dedupe_key(name: str) -> str:
    text = _TM.sub("", name or "")
    text = text.casefold().replace("&", " and ")
    text = re.sub(r"task\s*force", "taskforce", text)
    text = re.sub(r"\b(limited|ltd|pvt|private|inc|incorporated)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    tokens = []
    for word in text.split():
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        tokens.append(word)
    # word-order invariant
    return " ".join(sorted(tokens))


def _collapse_duplicate_names(items: list[dict]) -> tuple[list[dict], list[dict]]:
    groups: dict[str, list[dict]] = {}
    order = []
    for item in items:
        key = (item.get("name") or "").casefold()
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(item)
    kept = []
    dropped = []
    for key in order:
        members = groups[key]
        if len(members) == 1:
            kept.append(members[0])
            continue
        kept.append(_merge_group(members))
        for extra in members[1:]:
            dropped.append(_drop_row(extra, "duplicate_name_collapsed"))
    return kept, dropped


def _merge_group(members: list[dict]) -> dict:
    base = dict(members[0])
    sources = []
    for member in members:
        if _kind_rank(member.get("kind")) > _kind_rank(base.get("kind")):
            base["kind"] = member.get("kind")
        if len(member.get("descriptor") or "") > len(base.get("descriptor") or ""):
            base["descriptor"] = member.get("descriptor")
        if len(member.get("name") or "") > len(base.get("name") or ""):
            base["name"] = member.get("name")
        for source in member.get("sources") or []:
            if source not in sources:
                sources.append(source)
            if len(sources) >= 3:
                break
        if len(sources) >= 3:
            # still scan for better kind/descriptor above
            continue
    origins = {m.get("_origin") for m in members if m.get("_origin")}
    base["sources"] = sources[:3]
    base["_origin"] = "both" if len(origins) > 1 else (next(iter(origins)) if origins else base.get("_origin"))
    return base


def _entity_norm(entity: str) -> str:
    key = _name_key(entity)
    if entity.casefold() in _COMPANY or key in {"", "company"}:
        return "company"
    return key or "company"


def _shape_item(item: dict) -> dict | None:
    kind = item.get("kind") if item.get("kind") in KINDS else "other"
    status = item.get("status") if item.get("status") in STATUSES else "current"
    name = (item.get("name") or "").strip()
    descriptor = (item.get("descriptor") or name).strip()
    sources = []
    for source in (item.get("sources") or [])[:3]:
        quote = (source.get("quote") or "").strip()
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
    return {
        "name": name,
        "kind": kind,
        "descriptor": descriptor,
        "entity": item.get("entity") or "company",
        "status": status,
        "revenue_share_pct": None,
        "sources": sources,
    }


def _segments_from_listed(listed: dict, items: list[dict]) -> list[dict]:
    revenue = listed.get("revenue") or {}
    seen = set()
    rows = []
    for item in items:
        if item.get("kind") != "segment":
            continue
        key = normalize(item["name"])
        if key in seen:
            continue
        seen.add(key)
        money = revenue.get(item["name"].casefold()) or {}
        share = money.get("share_pct")  # only from printed revenue
        rows.append(
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
        rows.append(
            {
                "name": item["name"],
                "revenue": money.get("revenue"),
                "unit": money.get("unit"),
                "share_pct": money.get("share_pct"),
            }
        )
    for key, money in revenue.items():
        name = money.get("name") or key
        norm = normalize(name)
        if norm in seen:
            continue
        seen.add(norm)
        rows.append(
            {
                "name": name,
                "revenue": money.get("revenue"),
                "unit": money.get("unit"),
                "share_pct": money.get("share_pct"),
            }
        )
    return rows


def _grounded(text: str, source: str) -> bool:
    words = fold_whitespace(text).casefold().split()
    folded = fold_whitespace(source).casefold()
    if len(words) < 4:
        return False
    return any(" ".join(words[index : index + 4]) in folded for index in range(len(words) - 3))


def _pdf_layer_meta(numbers: set[int]) -> dict[int, dict]:
    folder = INPUT_DIR / "HINDALCO"
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        return {}
    import pymupdf

    document = pymupdf.open(pdfs[0])
    layer = {}
    for number in numbers:
        if 1 <= number <= document.page_count:
            text = document[number - 1].get_text("text") or ""
            # A usable text layer has more than a handful of letters.
            letters = sum(ch.isalpha() for ch in text)
            layer[number] = {"text": text, "has_layer": letters >= 40}
    document.close()
    return layer


def _drop_row(item: dict, reason: str) -> dict:
    sources = item.get("sources") or []
    return {
        "name": item.get("name"),
        "kind": item.get("kind"),
        "reason": reason,
        "pdf_page": sources[0].get("pdf_page") if sources else None,
        "quote": (sources[0].get("quote") if sources else None),
        "origin": item.get("_origin") or item.get("source_batch"),
    }


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    _atomic(path, text)


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def main() -> None:
    report = run()
    print("\n=== SUMMARY ===")
    print(json.dumps(report.get("summary"), indent=2))
    print("\n=== DROPPED COUNTS ===")
    print(json.dumps(report.get("dropped_counts"), indent=2))
    print("\n=== SAMPLE 40 ===")
    for row in report.get("sample_40") or []:
        quote = (row.get("quote") or "")[:160]
        line = f"{row['pdf_page']}\t{row['kind']}\t{row['entity']}\t{row['name']}\t{quote}"
        print(line.encode("ascii", "replace").decode("ascii"))
    print("\n=== DROPPED BY RULE (names) ===")
    for rule, rows in (report.get("dropped_by_rule") or {}).items():
        print(f"\n[{rule}] {len(rows)}")
        for row in rows[:30]:
            line = f"  p{row.get('pdf_page')}\t{row.get('kind')}\t{row.get('name')}"
            print(line.encode("ascii", "replace").decode("ascii"))
        if len(rows) > 30:
            print(f"  ... +{len(rows) - 30} more")


if __name__ == "__main__":
    main()
