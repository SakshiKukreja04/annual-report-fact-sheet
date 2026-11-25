"""Extract candidates in page batches. Unsent pages are listed, not dropped quietly."""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

from src.config.paths import CANDIDATE_DIR, LOG_DIR, PAGE_DIR, STAGING_DIR
from src.extract.client import extract_batch
from src.extract.compact import pack_pages, paragraphs_for_page, render_new, repeated_lines
from src.extract.evidence import pages_for_extraction
from src.extract.keys import fit_daily_cap, key_pool
from src.extract.known import relevant_known_names
from src.extract.ledger import Ledger, estimate_tokens, legacy_input_tokens, planned_input_tokens
from src.extract.prompt import PROMPT_VERSION, SYSTEM_PROMPT
from src.extract.rules import extract_rules, short_names
from src.extract.select import select_pages, write_selection
from src.extract.settings import text_settings
from src.utils.logging_config import configure_logging
from src.vision.usage import record_attempt, tokens_used, usage_path

LOGGER = logging.getLogger("factsheet")


def _page_input_estimate(page: dict, by_page: dict, ticker: str) -> int:
    packet, _fresh = render_new(by_page.get(int(page["pdf_page"])) or [], set())
    user = (
        f"Ticker: {ticker}. Target pdf pages: [{int(page['pdf_page'])}]. "
        f"Already found: none. Extract only new items.\n\n{packet}"
    )
    return estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(user)


def _token_prefix(
    ranked_pages: list[dict],
    by_page: dict,
    ticker: str,
    max_queue_tokens: int,
) -> tuple[list[dict], int]:
    """Walk the ranked queue until cumulative estimated input reaches the budget."""
    chosen = []
    cumulative = 0
    for page in ranked_pages:
        estimate = _page_input_estimate(page, by_page, ticker)
        if max_queue_tokens and chosen and cumulative + estimate > max_queue_tokens:
            break
        if max_queue_tokens and not chosen and estimate > max_queue_tokens:
            break
        chosen.append(page)
        cumulative += estimate
    return chosen, cumulative


def budget_preview(ticker: str, pages: list[dict], max_queue_tokens: int) -> dict:
    """Offline queue cut. Does not call a model or write candidates."""
    selection = select_pages(pages)
    score_of = {row["pdf_page"]: int(row.get("score") or 0) for row in selection["pages"]}
    reason_of = {row["pdf_page"]: str(row.get("reason") or "") for row in selection["pages"]}
    section_of = {row["pdf_page"]: str(row.get("section") or "") for row in selection["pages"]}
    kept = [page for page in pages if int(page["pdf_page"]) in set(selection["kept"])]
    kept.sort(key=lambda page: (-score_of.get(int(page["pdf_page"]), 0), int(page["pdf_page"])))
    page_records = {int(page["pdf_page"]): page for page in kept}
    banners = repeated_lines(kept)
    by_page = {int(page["pdf_page"]): paragraphs_for_page(page, banners) for page in kept}
    llm_pages = [
        page
        for page in kept
        if not _is_rules_only_page(
            int(page["pdf_page"]),
            reason=reason_of.get(int(page["pdf_page"]), ""),
            section=section_of.get(int(page["pdf_page"]), ""),
            page=page_records[int(page["pdf_page"])],
        )
    ]
    prefix, cumulative = _token_prefix(llm_pages, by_page, ticker, max_queue_tokens)
    return {
        "max_queue_tokens": int(max_queue_tokens or 0),
        "page_count": len(prefix),
        "cumulative_tokens": cumulative,
        "ranked_llm_pages": len(llm_pages),
        "pdf_pages": [int(page["pdf_page"]) for page in prefix],
    }


def _maybe_yield(ticker: str, cumulative_input: int, pages_done: set[int], candidates: int) -> None:
    """Append one row each time billed input crosses another 50k."""
    path = STAGING_DIR / "pilot" / f"{ticker}_yield.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing.append(json.loads(line))
    last_mark = 0
    for row in existing:
        last_mark = max(last_mark, int(row.get("cumulative_tokens") or 0))
    mark = ((last_mark // 50000) + 1) * 50000
    while cumulative_input >= mark:
        payload = {
            "cumulative_tokens": mark,
            "billed_input_tokens": cumulative_input,
            "pages_done": len(pages_done),
            "candidates": candidates,
            "validated_estimate": candidates,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        mark += 50000


def batch_id(ticker: str, target_pages: list[int], model: str) -> str:
    slug = model.replace("/", "_")
    first, last = target_pages[0], target_pages[-1]
    return f"{ticker}_p{first:04d}-p{last:04d}_{slug}_{PROMPT_VERSION}"


def run_pages(
    ticker: str,
    pages: list[dict],
    *,
    output_dir: Path | None = None,
    settings: dict | None = None,
    create=None,
    sleep=time.sleep,
    force: bool = False,
    reextract_version: bool = False,
    dry_run_batches: int = 0,
    pilot_selection: bool = False,
    redo_pages: set[int] | None = None,
    max_queue_tokens: int = 0,
) -> dict:
    output_dir = output_dir or CANDIDATE_DIR
    settings = settings or text_settings()
    company_dir = output_dir / ticker
    selection = select_pages(pages)
    if not dry_run_batches:
        # Limited-page pilot runs must not overwrite the full ranked queue file.
        selection_name = "selected.pilot.json" if pilot_selection else "selected.json"
        write_selection(company_dir / selection_name, ticker, selection)
    baseline = legacy_input_tokens(pages, ticker)
    LOGGER.info(
        "%s baseline estimated input tokens %s across %s pages before filtering",
        ticker,
        baseline["input_tokens"],
        baseline["pages"],
    )
    kept_numbers = set(selection["kept"])
    score_of = {row["pdf_page"]: int(row.get("score") or 0) for row in selection["pages"]}
    reason_of = {row["pdf_page"]: str(row.get("reason") or "") for row in selection["pages"]}
    section_of = {row["pdf_page"]: str(row.get("section") or "") for row in selection["pages"]}
    scope_pages = {row["pdf_page"] for row in selection["pages"] if str(row["reason"]).endswith("_in_scope")}
    kept_pages = [page for page in pages if int(page["pdf_page"]) in kept_numbers]
    kept_pages.sort(key=lambda page: (-score_of.get(int(page["pdf_page"]), 0), int(page["pdf_page"])))
    page_records = {int(page["pdf_page"]): page for page in pages}
    banners = repeated_lines(kept_pages)
    by_page = {
        int(page["pdf_page"]): paragraphs_for_page(page, banners, scope_only=int(page["pdf_page"]) in scope_pages)
        for page in kept_pages
    }
    # Rules mine every usable page; the LLM queue stays limited to uncovered kept pages.
    rules = extract_rules(pages)
    if not dry_run_batches:
        _write_rules(company_dir, ticker, rules, settings)
    known = short_names(rules)
    budget = int(settings.get("batch_input_tokens") or 24000)
    covered = set() if force else _covered_pages(company_dir, reextract_older_than=PROMPT_VERSION if reextract_version else None)
    if redo_pages:
        covered -= {int(number) for number in redo_pages}
    # Protected notes_other / AOC-1 / subsidiaries pages are rules-only — drop from LLM queue.
    rules_only_pages = [
        page
        for page in kept_pages
        if _is_rules_only_page(
            int(page["pdf_page"]),
            reason=reason_of.get(int(page["pdf_page"]), ""),
            section=section_of.get(int(page["pdf_page"]), ""),
            page=page,
        )
    ]
    rules_only_numbers = {int(page["pdf_page"]) for page in rules_only_pages}
    llm_kept_pages = [page for page in kept_pages if int(page["pdf_page"]) not in rules_only_numbers]
    removed_from_queue = sorted(
        number
        for number in rules_only_numbers
        if number not in covered and number in kept_numbers
    )
    prefix, prefix_tokens = _token_prefix(llm_kept_pages, by_page, ticker, max_queue_tokens)
    if max_queue_tokens:
        allowed = {int(page["pdf_page"]) for page in prefix}
        llm_kept_pages = [page for page in llm_kept_pages if int(page["pdf_page"]) in allowed]
        LOGGER.info(
            "%s queue budget %s tokens keeps %s ranked pages (cumulative estimate %s)",
            ticker,
            max_queue_tokens,
            len(prefix),
            prefix_tokens,
        )
    uncovered_pages = [page for page in llm_kept_pages if int(page["pdf_page"]) not in covered]
    # Dry-run prefers MDA / business-narrative pages among the ranked unsent queue.
    dry_run_source = uncovered_pages
    if dry_run_batches > 0:
        narrative = {
            "mda",
            "business_overview",
            "directors_report",
        }
        preferred = [
            page
            for page in uncovered_pages
            if section_of.get(int(page["pdf_page"]), "") in narrative
        ]
        if preferred:
            dry_run_source = preferred
            LOGGER.info(
                "%s dry-run using %s MDA/business-narrative unsent pages (of %s LLM queue)",
                ticker,
                len(preferred),
                len(uncovered_pages),
            )
    LOGGER.info(
        "%s rules-only dequeue removed %s protected notes/AOC/subsidiaries pages; LLM queue size %s",
        ticker,
        len(removed_from_queue),
        len(uncovered_pages),
    )
    # Plan and pack only uncovered pages, preserving rank order.
    planned = planned_input_tokens(uncovered_pages, by_page, ticker, budget, known)
    planned_all_known = planned_input_tokens(uncovered_pages, by_page, ticker, budget, known, filter_known=False)
    batches = pack_pages(dry_run_source if dry_run_batches > 0 else uncovered_pages, by_page, budget, preserve_order=True)
    if dry_run_batches > 0:
        batches = batches[:dry_run_batches]
    ledger = Ledger(company_dir / "ledger" / "tokens.jsonl")
    if not dry_run_batches:
        ledger.add(ticker=ticker, step="baseline", pdf_pages=[], input_tokens=baseline["input_tokens"], output_tokens=0, billed=False)
        ledger.add(
            ticker=ticker,
            step="planned_input",
            pdf_pages=[int(page["pdf_page"]) for page in uncovered_pages],
            input_tokens=planned,
            output_tokens=0,
            billed=False,
        )
    LOGGER.info(
        "%s planned input tokens %s on %s uncovered / %s kept pages, %s gap pages, %s batches",
        ticker,
        planned,
        len(uncovered_pages),
        len(selection["kept"]),
        len(selection["gaps"]),
        len(batches),
    )
    completed = []
    not_sent = []
    dry_rows = []
    stop_reason = None
    billed_input = [0]
    candidate_total = [0]
    seen: set[str] = set()
    log_path = usage_path(output_dir)
    LOGGER.info("%s extract %s uncovered pages in %s batches", ticker, len(uncovered_pages), len(batches))

    def run_group(group: list[dict], index_label: str) -> str | None:
        nonlocal stop_reason, log_path
        target_pages = [int(page["pdf_page"]) for page in group]
        paragraphs = []
        for page in group:
            paragraphs.extend(by_page.get(int(page["pdf_page"]), []))
        packet, fresh = render_new(paragraphs, seen)
        if not packet:
            parsed = _empty_success(ticker, target_pages, settings)
            if not dry_run_batches:
                _store_batch(company_dir, parsed, None)
            completed.append(parsed)
            seen.update(fresh)
            covered.update(target_pages)
            if not dry_run_batches:
                ledger.add(ticker=ticker, step="llm", pdf_pages=target_pages, input_tokens=0, output_tokens=0, billed=False)
            LOGGER.info("%s extract finished %s, pages %s, duplicate text not sent", ticker, index_label, _span(target_pages))
            return None
        limited = relevant_known_names(known, packet, limit=40)
        names = ", ".join(limited) if limited else "none"
        user = (
            f"Ticker: {ticker}. Target pdf pages: {target_pages}. "
            f"Already found: {names}. Extract only new items.\n\n{packet}"
        )
        estimate = estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(user) + int(settings["max_output_tokens"])
        if not dry_run_batches:
            before = key_pool().label()
            fits, active_path = fit_daily_cap(output_dir, estimate, int(settings["daily_token_cap"]))
            log_path = active_path
            if key_pool().label() != before:
                LOGGER.info(
                    "%s local daily cap reached on %s (%s tokens). Switched to %s.",
                    ticker,
                    before,
                    settings["daily_token_cap"],
                    key_pool().label(),
                )
            if not fits:
                stop_reason = "daily_cap"
                not_sent.extend(target_pages)
                LOGGER.warning(
                    "%s local daily cap reached on every key (%s tokens). Finished pages are not sent again.",
                    ticker,
                    settings["daily_token_cap"],
                )
                return "daily_cap"
        LOGGER.info("%s extract starting batch %s, pages %s", ticker, index_label, _span(target_pages))
        result = extract_batch(
            packet,
            ticker=ticker,
            target_pages=target_pages,
            allowed_pages=set(target_pages),
            settings=settings,
            create=create,
            sleep=sleep,
            known_names=list(known),
            page_records=page_records,
        )
        usage = result.get("usage") or {}
        accepted = len(result.get("candidates") or [])
        rejected = len(result.get("rejected") or [])
        total_items = accepted + rejected
        survive = round(accepted / total_items, 3) if total_items else None
        dry_rows.append(
            {
                "batch": index_label,
                "target_pages": target_pages,
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "status": result.get("request_status"),
                "error": result.get("error"),
                "finish_reason": result.get("finish_reason"),
                "accepted": accepted,
                "rejected": rejected,
                "survive_fraction": survive,
                "already_found_count": len(limited),
                "estimated_input_tokens": estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(user),
                "http_413_or_429": bool(result.get("error") and ("413" in str(result.get("error")) or "429" in str(result.get("error")))),
                "truncated": bool(result.get("finish_reason") == "length" or (result.get("error") and "truncated" in str(result.get("error")).casefold())),
                "salvaged_items": result.get("salvaged_items") or 0,
                "lost_items_estimate": result.get("lost_items_estimate") or 0,
                "quote_valid_accepted": sum(
                    1
                    for item in (result.get("candidates") or [])
                    if any((source.get("quote") or "").strip() for source in item.get("sources") or [])
                ),
            }
        )
        if result.get("request_status") == "daily_cap":
            stop_reason = "daily_cap"
            not_sent.extend(target_pages)
            return "daily_cap"
        if dry_run_batches:
            # Split truncated/malformed multi-page batches even in dry-run reporting.
            if _should_split(result, target_pages):
                LOGGER.info(
                    "%s dry-run salvage=%s lost~%s then split pages %s",
                    ticker,
                    result.get("salvaged_items") or 0,
                    result.get("lost_items_estimate") or 0,
                    _span(target_pages),
                )
                middle = len(group) // 2
                left = run_group(group[:middle], f"{index_label}a")
                if left == "daily_cap":
                    return "daily_cap"
                return run_group(group[middle:], f"{index_label}b")
            completed.append({key: value for key, value in result.items() if key != "raw_response"})
            LOGGER.info(
                "%s dry-run batch %s pages %s in %s out %s survive %s finish=%s salvaged=%s",
                ticker,
                index_label,
                _span(target_pages),
                usage.get("input_tokens"),
                usage.get("output_tokens"),
                survive,
                result.get("finish_reason"),
                result.get("salvaged_items") or 0,
            )
            _pace(result, settings["published_tpm"], sleep)
            return None
        log_path = usage_path(output_dir, api_key=key_pool().current() if key_pool().keys else None)
        record_attempt(log_path, {**result, "half": None, "pdf_page": target_pages[0]})
        ledger.add(
            ticker=ticker,
            step="llm",
            pdf_pages=target_pages,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            billed=True,
            provider=str(settings.get("provider") or "groq"),
            key_label=str(result.get("key_label") or key_pool().label()),
        )
        billed_input[0] += int(usage.get("input_tokens") or 0)
        candidate_total[0] += len(result.get("candidates") or [])
        _maybe_yield(ticker, billed_input[0], covered | set(target_pages), candidate_total[0])
        # Keep salvaged complete items before any split/re-send.
        # Do not mark target pages covered (empty target_pages + partial status).
        if result.get("truncated_salvaged") and (result.get("candidates") or []) and _should_split(result, target_pages):
            salvage_copy = {
                **result,
                "request_status": "partial_salvage",
                "context_pages": [],
                "source": "truncated_salvage",
                "salvaged_from_pages": list(target_pages),
                "target_pages": [],
            }
            identity = batch_id(ticker, target_pages, settings["model"]) + "_salvage"
            salvage_copy["batch_id"] = identity
            _atomic_json(company_dir / "parsed" / f"{identity}.json", {k: v for k, v in salvage_copy.items() if k != "raw_response"})
            if result.get("raw_response") is not None:
                _atomic_json(
                    company_dir / "raw" / f"{identity}.json",
                    {"batch_id": identity, "raw_response": result.get("raw_response"), "usage": result.get("usage")},
                )
            for candidate in result.get("candidates") or []:
                name = (candidate.get("name") or "").strip()
                if name and name not in known:
                    known.append(name)
            LOGGER.info(
                "%s salvaged %s items (lost~%s) from truncated batch %s before split",
                ticker,
                result.get("salvaged_items") or len(result.get("candidates") or []),
                result.get("lost_items_estimate") or 0,
                _span(target_pages),
            )
        if result.get("truncated_salvaged") and result.get("request_status") == "success":
            LOGGER.info(
                "%s keep %s salvaged items (lost~%s) finish=length pages %s",
                ticker,
                result.get("salvaged_items") or len(result.get("candidates") or []),
                result.get("lost_items_estimate") or 0,
                _span(target_pages),
            )
        if _should_split(result, target_pages):
            LOGGER.warning("%s extract split pages %s after invalid/truncated JSON", ticker, _span(target_pages))
            middle = len(group) // 2
            left = run_group(group[:middle], f"{index_label}a")
            if left == "daily_cap":
                not_sent.extend(int(page["pdf_page"]) for page in group[middle:])
                return "daily_cap"
            return run_group(group[middle:], f"{index_label}b")
        # Single-page truncation: keep salvaged candidates but do NOT mark complete.
        truncated = bool(
            result.get("finish_reason") == "length"
            or result.get("truncated_salvaged")
            or (result.get("error") and "truncated" in str(result.get("error")).casefold())
        )
        if truncated and len(target_pages) == 1 and result.get("request_status") in {"success", "partial_salvage", "malformed_response"}:
            result = {
                **result,
                "request_status": "needs_rerun",
                "truncated": True,
                "context_pages": [],
            }
            parsed = _store_batch(company_dir, result, result.get("raw_response"))
            completed.append(parsed)
            for candidate in result.get("candidates") or []:
                name = (candidate.get("name") or "").strip()
                if name and name not in known:
                    known.append(name)
            LOGGER.warning(
                "%s extract needs_rerun pages %s finish=%s salvaged=%s (not marked covered)",
                ticker,
                _span(target_pages),
                result.get("finish_reason"),
                result.get("salvaged_items") or len(result.get("candidates") or []),
            )
            _pace(result, settings["published_tpm"], sleep)
            return None
        result["context_pages"] = []
        parsed = _store_batch(company_dir, result, result.get("raw_response"))
        completed.append(parsed)
        if result.get("request_status") == "success":
            seen.update(fresh)
            covered.update(target_pages)
            for candidate in result.get("candidates") or []:
                name = (candidate.get("name") or "").strip()
                if name and name not in known:
                    known.append(name)
        LOGGER.info(
            "%s extract finished %s, pages %s, saved %s candidates, tokens in %s out %s",
            ticker,
            index_label,
            _span(target_pages),
            len(result.get("candidates") or []),
            usage.get("input_tokens") or 0,
            usage.get("output_tokens") or 0,
        )
        _pace(result, settings["published_tpm"], sleep)
        return None

    for index, batch in enumerate(batches):
        outcome = run_group(batch["pages"], f"{index + 1}/{len(batches)}")
        if outcome == "daily_cap":
            for later in batches[index + 1 :]:
                not_sent.extend(later["target_pages"])
            remaining = sorted(set(not_sent))
            LOGGER.warning(
                "%s daily token limit reached on every key (%s tokens). Finished %s/%s batches. %s pages were not sent, starting at page %s. Run the same command again after the limit resets. Finished batches are not sent again.",
                ticker,
                settings["daily_token_cap"],
                len(completed),
                len(batches),
                len(remaining),
                remaining[0] if remaining else "none",
            )
            break
    if dry_run_batches:
        accepted_total = sum(row["accepted"] for row in dry_rows)
        rejected_total = sum(row["rejected"] for row in dry_rows)
        total_items = accepted_total + rejected_total
        return {
            "ticker": ticker,
            "mode": "dry_run",
            "prompt_version": PROMPT_VERSION,
            "settings": {
                "batch_input_tokens": settings["batch_input_tokens"],
                "max_output_tokens": settings["max_output_tokens"],
                "sum": settings["batch_input_tokens"] + settings["max_output_tokens"],
                "prompt_reserve_tokens": settings.get("prompt_reserve_tokens"),
                "total_with_prompt_reserve": settings["batch_input_tokens"]
                + settings["max_output_tokens"]
                + int(settings.get("prompt_reserve_tokens") or 0),
                "tpm_safe_budget": settings.get("tpm_safe_budget"),
                "batch_input_lowered_for_tpm": settings.get("batch_input_lowered_for_tpm"),
                "token_estimate_factor": settings.get("token_estimate_factor"),
            },
            "rules_only_removed_count": len(removed_from_queue),
            "rules_only_removed_pages": removed_from_queue[:40],
            "llm_queue_size": len(uncovered_pages),
            "uncovered_pages": len(uncovered_pages),
            "queue_batches_total": len(pack_pages(uncovered_pages, by_page, budget, preserve_order=True)),
            "planned_input_tokens_filtered_known": planned,
            "planned_input_tokens_all_known": planned_all_known,
            "avg_estimated_input_before_known_filter": round(planned_all_known / max(len(pack_pages(uncovered_pages, by_page, budget, preserve_order=True)), 1), 1),
            "avg_estimated_input_after_known_filter": round(planned / max(len(pack_pages(uncovered_pages, by_page, budget, preserve_order=True)), 1), 1),
            "dry_run_batches": dry_rows,
            "accepted_total": accepted_total,
            "rejected_total": rejected_total,
            "survive_fraction": round(accepted_total / total_items, 3) if total_items else None,
            "target_survive_fraction": 0.70,
            "stop_reason": "dry_run_complete",
        }
    done = [number for number in selection["kept"] if number in covered]
    queue = [int(page["pdf_page"]) for page in uncovered_pages if int(page["pdf_page"]) not in covered]
    queue_batches = pack_pages(
        [page for page in uncovered_pages if int(page["pdf_page"]) not in covered],
        by_page,
        budget,
        preserve_order=True,
    )
    queue_tokens = planned_input_tokens(
        [page for page in uncovered_pages if int(page["pdf_page"]) not in covered],
        by_page,
        ticker,
        budget,
        known,
    )
    days_left = round(queue_tokens / max(int(settings["daily_token_cap"]), 1), 2)
    used_today = tokens_used(log_path)
    LOGGER.info(
        "%s pages done %s/%s, tokens used today %s, tokens remaining in ranked queue %s, estimated days %s",
        ticker,
        len(done),
        len(selection["kept"]),
        used_today,
        queue_tokens,
        days_left,
    )
    manifest = {
        "ticker": ticker,
        "model": settings["model"],
        "prompt_version": PROMPT_VERSION,
        "page_count": len(pages),
        "kept_pages": len(kept_pages),
        "gap_count": len(selection["gaps"]),
        "gaps": selection["gaps"],
        "baseline_input_tokens": baseline["input_tokens"],
        "planned_input_tokens": planned,
        "rule_candidate_count": len(rules),
        "batches_completed": len(completed),
        "batches_total": len(batches) if stop_reason != "daily_cap" else len(completed) + len(queue_batches),
        "pages_not_sent": not_sent or queue,
        "pages_done": len(done),
        "tokens_used_today": used_today,
        "queue_tokens": queue_tokens,
        "max_queue_tokens": int(max_queue_tokens or 0),
        "budget_page_count": len(prefix),
        "budget_cumulative_tokens": prefix_tokens,
        "estimated_days": days_left,
        "provider": settings.get("provider") or "groq",
        "stop_reason": stop_reason or "completed",
        "candidate_count": sum(len(item.get("candidates") or []) for item in completed) + len(rules),
        "deduplicated": False,
        "quotes_verified": False,
    }
    _atomic_json(company_dir / "manifest.json", manifest)
    return manifest


def _span(pages: list[int]) -> str:
    if not pages:
        return "none"
    if pages[0] == pages[-1]:
        return str(pages[0])
    return f"{pages[0]}-{pages[-1]}"


def _is_rules_only_page(pdf_page: int, *, reason: str, section: str, page: dict) -> bool:
    """Protected notes_other / AOC-1 / subsidiaries pages stay on rules, not the LLM queue."""
    if reason != "protected_entities":
        return False
    section_key = (section or "").casefold()
    if section_key in {"notes_other", "subsidiaries"}:
        return True
    from src.extract.select import page_text

    text = page_text(page) or ""
    if re.search(r"form\s*aoc-?\s*1|statement containing salient features|list of material unlisted subsidiar", text, re.I):
        return True
    return False


def _should_split(result: dict, target_pages: list[int]) -> bool:
    if len(target_pages) < 2:
        return False
    if result.get("needs_split"):
        return True
    if result.get("request_status") not in {"malformed_response", "failed"}:
        return False
    if result.get("finish_reason") == "length":
        return True
    error = (result.get("error") or "").lower()
    return "json" in error or "truncated" in error or "candidates" in error or "validate" in error


def _write_rules(company_dir: Path, ticker: str, rules: list[dict], settings: dict) -> None:
    record = {
        "ticker": ticker,
        "target_pages": sorted({source["pdf_page"] for item in rules for source in item.get("sources") or []}),
        "model": None,
        "prompt_version": PROMPT_VERSION,
        "request_status": "success",
        "source": "rules",
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "candidates": rules,
        "rejected": [],
        "error": None,
        "tools_enabled": False,
    }
    _atomic_json(company_dir / "parsed" / f"{ticker}_rules_{PROMPT_VERSION}.json", record)


def _empty_success(ticker: str, target_pages: list[int], settings: dict) -> dict:
    return {
        "ticker": ticker,
        "target_pages": target_pages,
        "model": settings["model"],
        "prompt_version": PROMPT_VERSION,
        "request_status": "success",
        "source": "duplicate_text",
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "candidates": [],
        "rejected": [],
        "error": None,
        "tools_enabled": False,
        "context_pages": [],
    }


def _store_batch(company_dir: Path, result: dict, raw: str | None) -> dict:
    parsed = {key: value for key, value in result.items() if key != "raw_response"}
    identity = parsed.get("batch_id") or batch_id(parsed.get("ticker") or "TICKER", parsed.get("target_pages") or [0], parsed.get("model") or "rules")
    parsed["batch_id"] = identity
    if raw is not None:
        _atomic_json(company_dir / "raw" / f"{identity}.json", {"batch_id": identity, "raw_response": raw, "usage": parsed.get("usage")})
    _atomic_json(company_dir / "parsed" / f"{identity}.json", parsed)
    for number in parsed.get("target_pages") or []:
        page_items = [
            item for item in parsed.get("candidates") or []
            if any(source.get("pdf_page") == number for source in item.get("sources") or [])
        ]
        status = parsed.get("request_status")
        _atomic_json(
            company_dir / "state" / f"p{int(number):04d}.json",
            {
                "pdf_page": int(number),
                "request_status": status,
                "prompt_version": parsed.get("prompt_version") or PROMPT_VERSION,
                "finish_reason": parsed.get("finish_reason"),
                "needs_rerun": status == "needs_rerun",
                "candidates": page_items,
            },
        )
    return parsed


def _version_is_current(version: str | None, required: str) -> bool:
    if not version:
        return False
    return str(version) == str(required)


def _covered_pages(company_dir: Path, reextract_older_than: str | None = None) -> set[int]:
    """Pages already extracted successfully. Optional reextract skips older prompt versions."""
    covered = set()
    state_versions: dict[int, str | None] = {}
    state = company_dir / "state"
    if state.is_dir():
        for path in state.glob("p*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            status = record.get("request_status")
            # Truncated salvage is incomplete until a non-truncated pass finishes.
            if status != "success" or record.get("needs_rerun"):
                continue
            number = int(record["pdf_page"])
            state_versions[number] = record.get("prompt_version")
            if reextract_older_than and not _version_is_current(record.get("prompt_version"), reextract_older_than):
                continue
            covered.add(number)
    parsed = company_dir / "parsed"
    if parsed.is_dir():
        for path in parsed.glob("*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("source") in {"rules", "truncated_salvage"} or record.get("request_status") != "success":
                continue
            version = record.get("prompt_version")
            for number in record.get("target_pages") or []:
                page = int(number)
                if page in state_versions:
                    continue
                if reextract_older_than and not _version_is_current(version, reextract_older_than):
                    continue
                covered.add(page)
    return covered


def _saved_batch(company_dir: Path, target_pages: list[int]) -> dict:
    parsed = company_dir / "parsed"
    if parsed.is_dir():
        for path in parsed.glob("*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("source") == "rules" or record.get("request_status") != "success":
                continue
            if list(record.get("target_pages") or []) == list(target_pages):
                return record
    return {"target_pages": target_pages, "request_status": "success", "candidates": [], "source": "cache"}


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _page_span(batch: dict) -> str:
    pages = batch["target_pages"]
    if not pages:
        return "none"
    if pages[0] == pages[-1]:
        return str(pages[0])
    return f"{pages[0]}-{pages[-1]}"


def _pace(result: dict, tpm: int, sleep) -> None:
    usage = result.get("usage") or {}
    total = int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
    if total <= 0 or tpm <= 0:
        return
    wait = total / tpm * 60
    if wait > 0:
        sleep(wait)


def load_saved_company(ticker: str, page_dir: Path | None = None) -> dict:
    """Read cached page JSON. This does not open the PDF."""
    folder = (page_dir or PAGE_DIR) / ticker
    candidate = []
    image_only = []
    for path in sorted(folder.glob("p*.json")):
        page = json.loads(path.read_text(encoding="utf-8"))
        status = page.get("content_status")
        if status == "recoverable":
            candidate.append(page)
        elif status == "image_only_visual":
            image_only.append(page)
    return {"ticker": ticker, "candidate_pages": candidate, "image_only_pages": image_only}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract business-item candidates for one company.")
    parser.add_argument("ticker")
    parser.add_argument("--pages", default="", help="Comma-separated PDF pages or a range such as 60-62. Default: all usable pages.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--reextract-version",
        action="store_true",
        help="Re-run pages whose state/parsed prompt_version is missing or older than the current prompt.",
    )
    parser.add_argument(
        "--redo-pages",
        default="",
        help="Comma-separated PDF pages to re-extract even if state marks them complete.",
    )
    parser.add_argument(
        "--dry-run-batches",
        type=int,
        default=0,
        help="Call the model for N highest-ranked unsent batches without writing candidates/manifest.",
    )
    args = parser.parse_args()
    configure_logging(LOG_DIR / "extract.log")
    ticker = args.ticker.upper()
    settings = text_settings()
    print(json.dumps({"text_settings": settings}, indent=2))
    loaded = load_saved_company(ticker, PAGE_DIR)
    pages = pages_for_extraction(loaded)
    wanted = _parse_pages(args.pages)
    if wanted:
        pages = [page for page in pages if page["pdf_page"] in wanted]
    manifest = run_pages(
        ticker,
        pages,
        force=args.force,
        reextract_version=args.reextract_version,
        dry_run_batches=args.dry_run_batches,
        settings=settings,
        pilot_selection=bool(wanted),
        redo_pages=_parse_pages(args.redo_pages),
    )
    print(json.dumps(manifest, indent=2))


def _parse_pages(text: str) -> set[int]:
    wanted = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            wanted.update(range(int(start), int(end) + 1))
        else:
            wanted.add(int(part))
    return wanted


if __name__ == "__main__":
    main()
