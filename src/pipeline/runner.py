"""Run one company from the PDF through the fact sheet. A failed stage stops the rest."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.config.paths import (
    AUDIT_DIR,
    CANDIDATE_DIR,
    CLASSIFICATION_DIR,
    INPUT_DIR,
    LOG_DIR,
    OCR_DIR,
    OUTPUT_DIR,
    PAGE_DIR,
    PREPROCESS_CACHE_DIR,
    TABLE_DIR,
    VISION_DIR,
)
from src.pipeline.companies import COMPANIES, STAGES, TEXT_RATES, VISION_RATES

LOGGER = logging.getLogger("factsheet")


@dataclass
class RunContext:
    ticker: str
    force: bool = False
    dry_run: bool = False
    resume: bool = True
    limit_pages: set[int] = field(default_factory=set)
    redo_pages: set[int] = field(default_factory=set)
    max_queue_tokens: int = 0
    ocr_engine: str = "tesseract"
    ocr_pages: set[int] = field(default_factory=set)
    mistral_halves: bool = False
    input_dir: Path = INPUT_DIR
    audit_dir: Path = AUDIT_DIR
    classification_dir: Path = CLASSIFICATION_DIR
    ocr_dir: Path = OCR_DIR
    table_dir: Path = TABLE_DIR
    page_dir: Path = PAGE_DIR
    vision_dir: Path = VISION_DIR
    candidate_dir: Path = CANDIDATE_DIR
    output_dir: Path = OUTPUT_DIR
    log_dir: Path = LOG_DIR
    preprocess_dir: Path = PREPROCESS_CACHE_DIR
    pdf_path: Path | None = None


def run_company(context: RunContext, stages: dict | None = None) -> dict:
    """Run the stages in order. Completed stages are not repeated unless force is set."""
    if context.ticker not in COMPANIES:
        raise KeyError(f"Unknown company {context.ticker}. Expected one of {', '.join(COMPANIES)}.")
    chosen = stages or STAGE_FUNCTIONS
    state_path = context.log_dir / "runs" / f"{context.ticker}.json"
    state = _read_json(state_path) if context.resume and not context.force else {}
    completed = list(state.get("completed_stages") or [])
    report = {
        "ticker": context.ticker,
        "period": COMPANIES[context.ticker]["period"],
        "dry_run": context.dry_run,
        "stages": {},
        "failed_stage": None,
        "error": None,
    }
    for name in STAGES:
        if name not in chosen:
            continue
        if context.resume and not context.force and name in completed and not context.dry_run:
            report["stages"][name] = {"status": "skipped_completed"}
            LOGGER.info("%s %s skipped_completed", context.ticker, name)
            continue
        LOGGER.info("%s %s start", context.ticker, name)
        try:
            result = chosen[name](context)
        except Exception as exc:
            failure = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            report["stages"][name] = failure
            report["failed_stage"] = name
            report["error"] = failure["error"]
            _write_failure(context, report)
            _write_state(state_path, context.ticker, completed, name, failure["error"])
            LOGGER.exception("%s %s failed", context.ticker, name)
            return report
        report["stages"][name] = result
        if result.get("status") == "daily_cap":
            for later in _STAGES_AFTER_EXTRACT:
                if later in completed:
                    completed.remove(later)
        extract_stopped = report["stages"].get("extract", {}).get("status") == "daily_cap"
        hold_for_resume = name in _STAGES_AFTER_EXTRACT and extract_stopped
        if hold_for_resume and name in completed:
            completed.remove(name)
        if (
            not context.dry_run
            and not context.limit_pages
            and not hold_for_resume
            and result.get("status") in {"ok", "cached", "completed", "unchanged", "replaced"}
        ):
            if name not in completed:
                completed.append(name)
            _write_state(state_path, context.ticker, completed, None, None)
        elif name in completed and result.get("status") == "completed_with_gaps":
            completed.remove(name)
            _write_state(state_path, context.ticker, completed, None, None)
        elif result.get("status") == "daily_cap" or hold_for_resume:
            _write_state(state_path, context.ticker, completed, None, None)
        LOGGER.info("%s %s %s", context.ticker, name, result.get("status"))
    if not context.dry_run and not context.limit_pages:
        _write_state(state_path, context.ticker, completed, None, None)
    return report


def stage_discover(context: RunContext) -> dict:
    folder = context.input_dir / context.ticker
    pdfs = sorted(folder.glob("*.pdf")) if folder.is_dir() else []
    if len(pdfs) != 1:
        raise FileNotFoundError(f"{context.ticker} needs one PDF in {folder}")
    context.pdf_path = pdfs[0]
    return {"status": "ok", "pdf": str(pdfs[0]), "bytes": pdfs[0].stat().st_size}


def stage_inspect(context: RunContext) -> dict:
    path = context.audit_dir / f"{context.ticker}_pages.jsonl"
    expected = COMPANIES[context.ticker]["expected_pages"]
    if path.is_file() and not context.force:
        count = _jsonl_count(path)
        if count != expected:
            raise RuntimeError(f"{context.ticker} audit has {count} pages; expected {expected}.")
        return {"status": "cached", "pages": count, "path": str(path)}
    if context.dry_run:
        return {"status": "dry_run", "would_audit": str(path), "expected_pages": expected}
    from src.ingestion.audit import audit_document, summarize_company, write_jsonl

    pages, metadata = audit_document(context.pdf_path, context.ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, pages)
    summary = summarize_company(context.ticker, context.pdf_path, pages, metadata)
    if summary["page_count"] != expected:
        raise RuntimeError(f"{context.ticker} PDF has {summary['page_count']} pages; expected {expected}.")
    return {"status": "completed", "pages": summary["page_count"], "path": str(path)}


def stage_classify(context: RunContext) -> dict:
    path = context.classification_dir / f"{context.ticker}_pages.jsonl"
    expected = COMPANIES[context.ticker]["expected_pages"]
    if path.is_file() and not context.force:
        count = _jsonl_count(path)
        if count != expected:
            raise RuntimeError(f"{context.ticker} classification has {count} pages; expected {expected}.")
        return {"status": "cached", "pages": count, "path": str(path)}
    if context.dry_run:
        return {"status": "dry_run", "would_classify": str(path)}
    from src.classification.classify import classify_document, write_jsonl

    rows = classify_document(context.pdf_path, context.ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, rows)
    if len(rows) != expected:
        raise RuntimeError(f"{context.ticker} classification wrote {len(rows)} pages; expected {expected}.")
    return {"status": "completed", "pages": len(rows), "path": str(path)}


def stage_ocr(context: RunContext) -> dict:
    """OCR scanned pages that have no saved record. Digital pages are left on their text layer."""
    if (context.ocr_engine or "tesseract") == "mistral":
        return _stage_ocr_mistral(context)
    needed = _ocr_work(context)
    if context.dry_run:
        return {"status": "dry_run", "halves": len(needed), "pages": sorted({item["pdf_page"] for item in needed})}
    if not needed:
        return {"status": "cached", "halves": 0, "failed": []}
    processed = []
    failed = []
    touched = set()
    for item in needed:
        LOGGER.info("OCR %s pdf %s %s", context.ticker, item["pdf_page"], item["half"])
        try:
            _ocr_one(context, item)
            processed.append(item)
            touched.add(item["pdf_page"])
        except Exception as exc:
            failed.append({**item, "error": f"{type(exc).__name__}: {exc}"})
            LOGGER.exception("OCR failed %s %s", item["pdf_page"], item["half"])
    for pdf_page in sorted(touched):
        try:
            _merge_ocr_page(context, pdf_page)
        except Exception as exc:
            failed.append({"pdf_page": pdf_page, "half": None, "error": f"{type(exc).__name__}: {exc}"})
            LOGGER.exception("OCR page merge failed %s", pdf_page)
    gaps = _ocr_gaps(context, needed, failed)
    status = "completed_with_gaps" if failed or gaps else "completed"
    return {
        "status": status,
        "processed": len(processed),
        "failed": failed,
        "gaps": gaps,
        "pages_updated": sorted(touched),
    }


def _stage_ocr_mistral(context: RunContext) -> dict:
    """Document OCR via Mistral. Does not replace Tesseract/Paddle caches."""
    from src.ocr_mistral import infer_printed_folios, merge_mistral_into_pages, ocr_pdf_pages

    expected = COMPANIES[context.ticker]["expected_pages"]
    if context.ocr_pages:
        pages = sorted(context.ocr_pages)
    elif context.limit_pages:
        pages = sorted(context.limit_pages)
    else:
        pages = list(range(1, expected + 1))
    if context.dry_run:
        return {"status": "dry_run", "engine": "mistral", "pages": pages, "halves": context.mistral_halves}
    summary = ocr_pdf_pages(
        context.ticker,
        pages,
        force=context.force,
        pdf_path=context.pdf_path,
        halves=context.mistral_halves,
    )
    merge = merge_mistral_into_pages(context.ticker, pages)
    folios = infer_printed_folios(context.ticker, pages)
    return {
        "status": "completed",
        "engine": "mistral",
        "cached": summary.get("cached") or [],
        "written": summary.get("written") or [],
        "merge": merge,
        "folios": folios,
    }


def stage_tables(context: RunContext) -> dict:
    """Extract digital tables onto pages that do not already have a saved table."""
    needed = _table_work(context)
    if context.dry_run:
        return {"status": "dry_run", "pages": needed}
    if not needed:
        return {"status": "cached", "pages": []}
    written = []
    failed = []
    from src.tables.digital import extract_digital_page

    destination = context.table_dir / context.ticker
    destination.mkdir(parents=True, exist_ok=True)
    for pdf_page in needed:
        try:
            tables = extract_digital_page(context.pdf_path, context.ticker, pdf_page)
            for table in tables:
                (destination / f"{table['table_id']}.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
            (destination / f"p{pdf_page:04d}.done.json").write_text(
                json.dumps({"pdf_page": pdf_page, "table_count": len(tables)}),
                encoding="utf-8",
            )
            written.append({"pdf_page": pdf_page, "table_count": len(tables)})
        except Exception as exc:
            failed.append({"pdf_page": pdf_page, "error": f"{type(exc).__name__}: {exc}"})
            LOGGER.error("Table extraction failed on PDF page %s: %s", pdf_page, exc)
    status = "completed_with_gaps" if failed else "completed"
    return {"status": status, "written": written, "failed": failed}


def stage_pages(context: RunContext) -> dict:
    expected = COMPANIES[context.ticker]["expected_pages"]
    folder = context.page_dir / context.ticker
    existing = len(list(folder.glob("p*.json"))) if folder.is_dir() else 0
    refreshed = set() if context.force else _pages_with_new_ocr(context)
    if context.limit_pages:
        refreshed = {page for page in refreshed if page in context.limit_pages}
    if not context.force and not _pymupdf_imports():
        if not context.limit_pages and existing == expected and not refreshed:
            return {"status": "cached", "pages": existing, "rebuild": "ocr_merged_without_pymupdf"}
        if context.limit_pages and all((folder / f"p{pdf_page:04d}.json").is_file() for pdf_page in context.limit_pages) and not refreshed:
            return {"status": "cached", "pages": existing, "rebuild": "ocr_merged_without_pymupdf"}
    if context.limit_pages:
        missing = [
            pdf_page for pdf_page in context.limit_pages
            if not (folder / f"p{pdf_page:04d}.json").is_file()
        ]
        if not context.force and not missing and not refreshed:
            return {"status": "cached", "pages": existing}
    elif not context.force and existing == expected and not refreshed:
        return {"status": "cached", "pages": existing}
    if context.dry_run:
        return {"status": "dry_run", "cached_pages": existing, "expected_pages": expected, "rebuild_pages": sorted(refreshed)}
    from src.pages.build import build_company

    if context.limit_pages:
        pages = sorted(context.limit_pages)
    elif context.force or existing != expected:
        pages = None
    elif refreshed:
        pages = sorted(refreshed)
    else:
        pages = None
    # A stale OCR file or an empty OCR count must rewrite the page that is already on disk.
    resume = not context.force and not bool(refreshed)
    result = build_company(
        context.ticker,
        pages=pages,
        resume=resume,
        output_dir=context.page_dir,
        pdf_path=context.pdf_path,
        classification_path=context.classification_dir / f"{context.ticker}_pages.jsonl",
        ocr_dir=context.ocr_dir / "sample",
        table_dir=context.table_dir,
    )
    return {"status": "completed", **result}


def stage_vision(context: RunContext) -> dict:
    from src.extract.run import load_saved_company
    from src.vision.route import route_page, vision_cap
    from src.vision.settings import vision_settings

    loaded = load_saved_company(context.ticker, context.page_dir)
    pages = loaded["candidate_pages"] + loaded["image_only_pages"]
    if context.limit_pages:
        pages = [page for page in pages if int(page["pdf_page"]) in context.limit_pages]
    cap = vision_cap(COMPANIES[context.ticker]["expected_pages"], vision_settings()["max_page_percent"])
    eligible = []
    for page in pages:
        decision = route_page(page)
        if decision["eligible"] and len(eligible) >= cap:
            continue
        if decision["eligible"]:
            eligible.append({"pdf_page": page["pdf_page"], "routing_reason": decision["routing_reason"]})
    if context.dry_run:
        return {"status": "dry_run", "eligible_pages": [item["pdf_page"] for item in eligible], "cap": cap, "decisions": len(pages)}
    from src.vision.pilot import cache_path, image_path, run_target

    calls = []
    for item in eligible:
        page = next(page for page in pages if page["pdf_page"] == item["pdf_page"])
        halves = sorted({block.get("half") or "FULL" for block in page.get("text_blocks") or []}) or ["FULL"]
        for half in halves:
            if not image_path(context.ticker, int(item["pdf_page"]), half).is_file() and not cache_path(context.vision_dir, context.ticker, int(item["pdf_page"]), half).is_file():
                calls.append({"pdf_page": item["pdf_page"], "half": half, "request_status": "no_image"})
                continue
            try:
                calls.append(run_target(context.ticker, int(item["pdf_page"]), half, output_dir=context.vision_dir, page=page, force=context.force))
            except Exception as exc:
                calls.append({"pdf_page": item["pdf_page"], "half": half, "request_status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                LOGGER.exception("Vision failed %s %s", item["pdf_page"], half)
    return {
        "status": "completed",
        "eligible_pages": [item["pdf_page"] for item in eligible],
        "calls": [{"pdf_page": item.get("pdf_page"), "half": item.get("half"), "request_status": item.get("request_status")} for item in calls],
    }


def stage_extract(context: RunContext) -> dict:
    from src.extract.run import load_saved_company, run_pages
    from src.extract.evidence import pages_for_extraction
    from src.vision.usage import tokens_used, usage_path

    blocked = _ocr_blocks_extract(context)
    if blocked is not None:
        return blocked
    loaded = load_saved_company(context.ticker, context.page_dir)
    pages = pages_for_extraction(loaded)
    outside = []
    if context.limit_pages:
        outside = [int(page["pdf_page"]) for page in pages if int(page["pdf_page"]) not in context.limit_pages]
        pages = [page for page in pages if int(page["pdf_page"]) in context.limit_pages]
    if context.dry_run:
        from src.extract.run import budget_preview

        preview = budget_preview(context.ticker, pages, int(context.max_queue_tokens or 0))
        LOGGER.info(
            "%s dry-run budget page_count=%s cumulative_tokens=%s max_queue_tokens=%s",
            context.ticker,
            preview["page_count"],
            preview["cumulative_tokens"],
            preview["max_queue_tokens"],
        )
        return {
            "status": "dry_run",
            "usable_pages": [int(page["pdf_page"]) for page in pages],
            "pages_outside_limit": outside,
            "page_count": preview["page_count"],
            "cumulative_tokens": preview["cumulative_tokens"],
            "max_queue_tokens": preview["max_queue_tokens"],
            "ranked_llm_pages": preview["ranked_llm_pages"],
        }
    log_path = usage_path(context.candidate_dir)
    before = tokens_used(log_path)
    manifest = run_pages(
        context.ticker,
        pages,
        output_dir=context.candidate_dir,
        force=context.force,
        pilot_selection=bool(context.limit_pages),
        redo_pages=set(context.redo_pages or ()),
        max_queue_tokens=int(context.max_queue_tokens or 0),
    )
    manifest["tokens_added"] = tokens_used(log_path) - before
    manifest["pages_outside_limit"] = outside
    manifest["status"] = "daily_cap" if manifest["stop_reason"] == "daily_cap" else "completed"
    return manifest


def stage_dedup(context: RunContext) -> dict:
    if context.dry_run:
        return {"status": "dry_run"}
    from src.dedup.resolve import write_report
    from src.factsheet.table_items import write_table_candidates

    tables = write_table_candidates(context.ticker, pdf_path=context.pdf_path)
    result = write_report(context.ticker, context.candidate_dir / context.ticker)
    return {
        "status": "completed",
        "merged_count": result["merged_count"],
        "review_count": result["review_count"],
        "unchanged": len(result["unchanged"]),
        "table_candidates": tables["count"],
        "language_model_used": False,
    }


def stage_validate(context: RunContext) -> dict:
    if context.dry_run:
        return {"status": "dry_run"}
    from src.validate.run import build_context, load_candidates, validate_many, write_reports

    candidates = load_candidates(context.ticker, context.candidate_dir / context.ticker)
    cited = sorted({
        int(source["pdf_page"])
        for item in candidates
        for source in item.get("sources") or []
        if isinstance(source.get("pdf_page"), int) and not isinstance(source.get("pdf_page"), bool)
    })
    validation_context = build_context(context.ticker, cited, context.pdf_path)
    report = validate_many(candidates, validation_context)
    report["pdf_reader"] = validation_context["pdf_reader"]
    destination = write_reports(context.ticker, report, context.candidate_dir / context.ticker / "validation")
    return {
        "status": "completed",
        "validated_count": report["validated_count"],
        "rejected_count": report["rejected_count"],
        "review_count": report["review_count"],
        "pdf_reader": validation_context["pdf_reader"],
        "report": str(destination),
    }


def stage_factsheet(context: RunContext) -> dict:
    from src.config.paths import INPUT_DIR, STAGING_DIR
    from src.extract.run import load_saved_company
    from src.extract.evidence import pages_for_extraction
    from src.factsheet.build import build_fact_sheet, consistency_errors, load_validated, write_fact_sheet
    from src.factsheet.cost import record_stage, stage_pages_from_parsed
    from src.factsheet.overview import build_overview, record_overview, write_overview_sidecar
    from src.factsheet.schema import FactSheet
    from src.factsheet.segments import locate_segment_tables, write_segment_report

    records = load_validated(context.candidate_dir / context.ticker / "validation" / "validated.json")
    loaded = load_saved_company(context.ticker, context.page_dir)
    pages = pages_for_extraction(loaded)
    from src.factsheet.subsidiary_clean import clean_subsidiary_records, write_clean_report

    records, clean = clean_subsidiary_records(records, pages)
    from src.factsheet.precision import apply_precision, write_precision_report

    records, precision = apply_precision(records, pages, ticker=context.ticker)
    pdfs = sorted((INPUT_DIR / context.ticker).glob("*.pdf"))
    pdf_path = context.pdf_path or (pdfs[0] if len(pdfs) == 1 else None)
    segment_report = locate_segment_tables(context.ticker, pages, pdf_path)
    staging_root = STAGING_DIR / "factsheet"
    if not context.dry_run:
        write_clean_report(context.ticker, clean, staging_root)
        write_precision_report(context.ticker, precision, staging_root)
    write_segment_report(context.ticker, segment_report, staging_root)
    overview = build_overview(context.ticker, pages, allow_model=not context.dry_run)
    write_overview_sidecar(context.ticker, overview, staging_root)
    if not context.dry_run and not overview.get("cache_hit"):
        record_overview(context.ticker, overview)
    if not context.dry_run:
        parsed = stage_pages_from_parsed(context.ticker)
        for step in ("rules", "tables"):
            record_stage(
                context.ticker,
                step,
                parsed[step]["pages"],
                input_tokens=parsed[step]["input_tokens"],
                output_tokens=parsed[step]["output_tokens"],
                billed=False,
            )
    built = build_fact_sheet(
        context.ticker,
        records,
        segments=segment_report["segments"],
        overview=overview,
        apply_overrides=True,
    )
    if not context.dry_run:
        precision["merge_groups"] = built.get("merge_groups") or []
        write_precision_report(context.ticker, precision, staging_root)
    sheet = FactSheet.model_validate(built["sheet"]).model_dump(mode="json")
    errors = consistency_errors(sheet, built["markdown"])
    errors.extend(_citation_errors(context, sheet))
    if errors:
        raise RuntimeError("Fact sheet failed validation: " + "; ".join(errors))
    if context.dry_run:
        return {
            "status": "dry_run",
            "item_count": len(sheet["items"]),
            "segment_count": len(sheet["segments"]),
            "overview_fallback": bool(overview.get("fallback")),
        }
    # Staging only. Promotion into outputs/ and submission/ is a separate command.
    json_path = staging_root / context.ticker / f"{context.ticker}.json"
    markdown_path = staging_root / context.ticker / f"{context.ticker}.md"
    written = write_fact_sheet(context.ticker, built, staging_root, force=True)
    written["status"] = "completed_staging"
    written["staging"] = True
    written["json"] = str(json_path)
    written["markdown"] = str(markdown_path)
    written["overview_fallback"] = bool(overview.get("fallback"))
    written["segment_count"] = len(sheet["segments"])
    return written


def stage_evaluate(context: RunContext) -> dict:
    pages = _page_outcomes(context)
    extract = {}
    manifest_path = context.candidate_dir / context.ticker / "manifest.json"
    if manifest_path.is_file():
        extract = _read_json(manifest_path)
    report = {
        "status": "completed",
        "ticker": context.ticker,
        "expected_pages": COMPANIES[context.ticker]["expected_pages"],
        "page_records": pages["records"],
        "illegible_pages": pages["illegible"],
        "pending_ocr_pages": pages["pending_ocr"],
        "image_only_pages": pages["image_only"],
        "recoverable_pages": len(pages["recoverable"]),
        "missing_page_records": pages["missing"],
        "pages_not_sent": extract.get("pages_not_sent") or [],
        "extract_stop_reason": extract.get("stop_reason"),
        "extract_page_count": extract.get("page_count"),
        "candidate_count": extract.get("candidate_count"),
        "cost": _cost(context),
        "outputs": {
            "json": str(context.output_dir / context.ticker / f"{context.ticker}.json"),
            "markdown": str(context.output_dir / context.ticker / f"{context.ticker}.md"),
        },
        "assignment_complete": False,
    }
    report["assignment_complete"] = _assignment_complete(report)
    if context.dry_run:
        report["status"] = "dry_run"
        return report
    destination = context.log_dir / "evaluation" / f"{context.ticker}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["path"] = str(destination)
    return report


def _ocr_work(context: RunContext) -> list[dict]:
    path = context.classification_dir / f"{context.ticker}_pages.jsonl"
    if not path.is_file():
        return []
    needed = []
    for row in _jsonl_rows(path):
        pdf_page = int(row["pdf_page"])
        if context.limit_pages and pdf_page not in context.limit_pages:
            continue
        labels = set(row.get("classifications") or [])
        # OCR digital-layer failures too: image-only visuals and garbled text layers.
        needs_ocr = (
            ("SCANNED_IMAGE" in labels and "DIGITAL_TEXT" not in labels)
            or "IMAGE_ONLY_VISUAL" in labels
            or "GARBLED_OR_LOW_QUALITY_TEXT" in labels
        )
        if not needs_ocr:
            continue
        halves = ("LEFT", "RIGHT") if "TWO_PAGE_SPREAD" in labels else ("FULL",)
        for half in halves:
            name = f"{context.ticker}_p{pdf_page:04d}_{half}.json"
            if (context.ocr_dir / context.ticker / name).is_file() or (context.ocr_dir / "sample" / name).is_file():
                continue
            needed.append({"pdf_page": pdf_page, "half": half, "classifications": sorted(labels)})
    return needed


def _ocr_one(context: RunContext, item: dict) -> None:
    import cv2

    from src.ocr.fast import assess_half_fast
    from src.ocr.sample import confidence_threshold
    from src.preprocessing.pipeline import process_rendered_fast, render_embedded_native

    image = context.preprocess_dir / context.ticker / f"p{item['pdf_page']:04d}_{item['half']}" / "full.png"
    if not image.is_file():
        color, width_pt, height_pt, _metrics = _render_page_native(context, item["pdf_page"])
        process_rendered_fast(
            color,
            width_pt,
            height_pt,
            item["pdf_page"],
            context.ticker,
            context.preprocess_dir,
        )
    gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Missing preprocessed image: {image}")
    record = assess_half_fast(
        gray,
        item.get("classifications") or [],
        confidence_threshold(),
        context.ticker,
        item["pdf_page"],
        item["half"],
    )
    record["source_image"] = image.as_posix()
    destination = context.ocr_dir / context.ticker / f"{context.ticker}_p{item['pdf_page']:04d}_{item['half']}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    record["renderer"] = context.__dict__.get("page_renderer", "pdfplumber_embedded_image")
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(destination)


def _render_page_native(context: RunContext, pdf_page: int):
    """Native embedded pixels preferred for the fast OCR path."""
    from src.preprocessing.pipeline import render_embedded_native

    try:
        color, width_pt, height_pt, metrics = render_embedded_native(context.pdf_path, pdf_page)
        context.__dict__["page_renderer"] = "pdfplumber_embedded_native"
        return color, width_pt, height_pt, metrics
    except Exception as exc:
        LOGGER.info("Native embedded render failed %s p%s: %s", context.ticker, pdf_page, type(exc).__name__)
        color, width_pt, height_pt = _render_page(context, pdf_page)
        return color, width_pt, height_pt, {}


def _render_page(context: RunContext, pdf_page: int):
    """Prefer PyMuPDF. The embedded page image is the fallback when it is blocked."""
    try:
        import pymupdf

        from src.preprocessing.pipeline import render_page

        with pymupdf.open(context.pdf_path) as document:
            page = document[pdf_page - 1]
            context.__dict__["page_renderer"] = "pymupdf"
            return render_page(page), float(page.rect.width), float(page.rect.height)
    except Exception as exc:
        LOGGER.info("PyMuPDF render unavailable for %s page %s: %s", context.ticker, pdf_page, type(exc).__name__)
        from src.preprocessing.pipeline import render_embedded_page

        context.__dict__["page_renderer"] = "pdfplumber_embedded_image"
        return render_embedded_page(context.pdf_path, pdf_page)


def _merge_ocr_page(context: RunContext, pdf_page: int) -> None:
    """Put new OCR text into the saved page record without opening PyMuPDF."""
    from src.pages.assemble import assemble_page
    from src.pages.extract import ocr_blocks_from_record

    path = context.page_dir / context.ticker / f"p{pdf_page:04d}.json"
    if not path.is_file():
        return
    current = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for folder in (context.ocr_dir / context.ticker, context.ocr_dir / "sample"):
        if not folder.is_dir():
            continue
        for record_path in sorted(folder.glob(f"{context.ticker}_p{pdf_page:04d}_*.json")):
            records.append(json.loads(record_path.read_text(encoding="utf-8")))
    blocks = []
    captions = []
    recoverability = {}
    for record in records:
        ocr_blocks, caption_rows, status = ocr_blocks_from_record(record)
        blocks.extend(ocr_blocks)
        captions.extend(caption_rows)
        if status:
            recoverability[str(record.get("half") or "FULL")] = status
    from src.ocr_mistral import mistral_blocks_for_page

    digital = [
        block for block in current.get("text_blocks") or []
        if block.get("source") not in {"tesseract", "paddleocr", "ocr_mistral"}
    ]
    mistral_blocks = mistral_blocks_for_page(context.ticker, pdf_page)
    if mistral_blocks:
        blocks.extend(mistral_blocks)
        recoverability.setdefault("FULL", "RECOVERABLE")
    updated = assemble_page(
        ticker=context.ticker,
        pdf_page=pdf_page,
        printed_page=current.get("printed_page"),
        page_type=list(current.get("page_type") or []),
        digital_blocks=digital,
        ocr_blocks=blocks,
        captions=captions,
        images=list(current.get("images") or []),
        tables=list(current.get("tables") or []),
        ocr_recoverability=recoverability,
        garbled=bool((current.get("quality_metadata") or {}).get("garbled")),
        classification_confidence=(current.get("quality_metadata") or {}).get("classification_confidence"),
        recommended_method=(current.get("quality_metadata") or {}).get("recommended_method"),
    )
    if mistral_blocks:
        meta = updated.setdefault("quality_metadata", {})
        meta["mistral_char_count"] = sum(len(b.get("text") or "") for b in mistral_blocks)
        meta["mistral_weak"] = meta["mistral_char_count"] < 80
    path.write_text(json.dumps(updated, indent=2), encoding="utf-8")


def _ocr_blocks_extract(context: RunContext) -> dict | None:
    """PowerGrid extraction waits until scanned gaps are closed or listed in accepted_gaps.json."""
    if context.ticker != "POWERGRID":
        return None
    needed = _ocr_work(context)
    accepted_path = context.log_dir / "accepted_gaps.json"
    accepted = set()
    if accepted_path.is_file():
        import json as _json
        payload = _json.loads(accepted_path.read_text(encoding="utf-8"))
        for row in payload.get("gaps") or []:
            accepted.add((int(row.get("pdf_page")), row.get("half")))
    open_gaps = [
        {"pdf_page": item["pdf_page"], "half": item["half"], "reason": "pending_ocr"}
        for item in needed
        if (item["pdf_page"], item["half"]) not in accepted
    ]
    if not open_gaps:
        return None
    return {"status": "blocked_on_ocr", "gaps": open_gaps, "gap_count": len(open_gaps)}


def _ocr_gaps(context: RunContext, needed: list[dict], failed: list[dict]) -> list[dict]:
    """Pages that did not produce usable OCR, with empty / refused / timeout."""
    failed_keys = {(item.get("pdf_page"), item.get("half")) for item in failed}
    gaps = []
    for item in failed:
        gaps.append({
            "pdf_page": item.get("pdf_page"),
            "half": item.get("half"),
            "reason": _failure_label(item.get("error") or ""),
            "detail": item.get("error"),
        })
    for item in needed:
        key = (item["pdf_page"], item["half"])
        if key in failed_keys:
            continue
        path = context.ocr_dir / context.ticker / f"{context.ticker}_p{item['pdf_page']:04d}_{item['half']}.json"
        if not path.is_file():
            gaps.append({"pdf_page": item["pdf_page"], "half": item["half"], "reason": "empty", "detail": "No OCR file was written."})
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("recoverability") == "IMAGE_ONLY_VISUAL":
            continue
        if record.get("recoverability") == "RECOVERABLE" and record.get("use_ocr_text"):
            continue
        gaps.append({
            "pdf_page": item["pdf_page"],
            "half": item["half"],
            "reason": "empty",
            "detail": record.get("failure_reason") or record.get("recoverability"),
        })
    return gaps


def _failure_label(error: str) -> str:
    lowered = error.lower()
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if error:
        return "refused"
    return "empty"


def _table_work(context: RunContext) -> list[int]:
    path = context.classification_dir / f"{context.ticker}_pages.jsonl"
    if not path.is_file():
        return []
    needed = []
    for row in _jsonl_rows(path):
        pdf_page = int(row["pdf_page"])
        if context.limit_pages and pdf_page not in context.limit_pages:
            continue
        labels = set(row.get("classifications") or [])
        if "TABLE_HEAVY" not in labels or "SCANNED_IMAGE" in labels:
            continue
        if _table_cached(context, pdf_page):
            continue
        needed.append(pdf_page)
    return needed


def _table_cached(context: RunContext, pdf_page: int) -> bool:
    marker = context.table_dir / context.ticker / f"p{pdf_page:04d}.done.json"
    if marker.is_file():
        return True
    pattern = f"{context.ticker}-p{pdf_page:04d}*.json"
    return any(context.table_dir.rglob(pattern))


def _citation_errors(context: RunContext, sheet: dict) -> list[str]:
    accepted = set()
    quote_path = context.candidate_dir / context.ticker / "validation" / "quote_report.json"
    if quote_path.is_file():
        for row in _read_json(quote_path):
            if row.get("accepted"):
                accepted.add((row.get("pdf_page"), row.get("quote")))
    errors = []
    for item in sheet["items"]:
        sources = item["sources"]
        if not sources or len(sources) > 3:
            errors.append(f"{item['name']} has {len(sources)} sources")
        for source in sources:
            page = source["pdf_page"]
            if not isinstance(page, int) or page < 1:
                errors.append(f"{item['name']} cites PDF page {page}")
                continue
            if not (context.page_dir / context.ticker / f"p{page:04d}.json").is_file():
                errors.append(f"{item['name']} cites PDF page {page}, which has no page record")
            if accepted and (page, source["quote"]) not in accepted and not _quote_on_page(context, page, source["quote"]):
                errors.append(f"{item['name']} quote was not accepted on PDF page {page}")
    return errors


def _quote_on_page(context: RunContext, pdf_page: int, quote: str) -> bool:
    path = context.page_dir / context.ticker / f"p{pdf_page:04d}.json"
    if not path.is_file() or not quote:
        return False
    from src.extract.select import page_text
    from src.validate.quotes import snap_quote

    text = page_text(_read_json(path))
    if " ".join(quote.split()).casefold() in " ".join(text.split()).casefold():
        return True
    snapped = snap_quote(quote, text, minimum_ratio=0.9)
    return bool(snapped.get("quote"))


def _page_outcomes(context: RunContext) -> dict:
    expected = COMPANIES[context.ticker]["expected_pages"]
    folder = context.page_dir / context.ticker
    present = set()
    illegible = []
    pending = []
    visual = []
    recoverable = []
    if folder.is_dir():
        for path in folder.glob("p*.json"):
            page = json.loads(path.read_text(encoding="utf-8"))
            number = int(page["pdf_page"])
            present.add(number)
            status = page.get("content_status")
            if status == "illegible":
                illegible.append(number)
            elif status == "pending_ocr":
                pending.append(number)
            elif status == "image_only_visual":
                visual.append(number)
            elif status == "recoverable":
                recoverable.append(number)
    missing = [number for number in range(1, expected + 1) if number not in present]
    return {
        "records": len(present),
        "illegible": illegible,
        "pending_ocr": pending,
        "image_only": visual,
        "recoverable": recoverable,
        "missing": missing,
    }


def _cost(context: RunContext) -> dict:
    text = _usage_cost(context.candidate_dir / "usage", TEXT_RATES)
    vision = _usage_cost(context.vision_dir / "usage", VISION_RATES)
    return {"text": text, "vision": vision, "source": "response.usage logs"}


def _usage_cost(folder: Path, rates: dict) -> dict:
    logs = sorted(folder.glob("*.json")) if folder.is_dir() else []
    if not logs:
        return {"input_tokens": 0, "output_tokens": 0, "computed_usd": 0.0, "model": rates["model"], "log": None}
    latest = json.loads(logs[-1].read_text(encoding="utf-8"))
    incoming = int(latest.get("input_tokens") or 0)
    outgoing = int(latest.get("output_tokens") or 0)
    cost = incoming / 1_000_000 * rates["input"] + outgoing / 1_000_000 * rates["output"]
    return {
        "input_tokens": incoming,
        "output_tokens": outgoing,
        "computed_usd": round(cost, 6),
        "model": latest.get("model") or rates["model"],
        "requests": latest.get("requests"),
        "cache_hits": latest.get("cache_hits"),
        "malformed": latest.get("malformed"),
        "log": str(logs[-1]),
    }


def _assignment_complete(report: dict) -> bool:
    """Complete means every page is represented and extraction covered the recoverable pages."""
    extracted = report.get("extract_page_count")
    return (
        report["page_records"] == report["expected_pages"]
        and not report["missing_page_records"]
        and not report["pending_ocr_pages"]
        and not report["pages_not_sent"]
        and report.get("extract_stop_reason") == "completed"
        and isinstance(extracted, int)
        and extracted >= report.get("recoverable_pages", 0)
        and report.get("recoverable_pages", 0) > 0
    )


def _pymupdf_imports() -> bool:
    try:
        import pymupdf  # noqa: F401
    except Exception:
        return False
    return True


def _pages_with_new_ocr(context: RunContext) -> set[int]:
    """Pages to rebuild because OCR landed after the page file, or the page stored no OCR chars."""
    folder = context.ocr_dir / context.ticker
    page_folder = context.page_dir / context.ticker
    if not folder.is_dir():
        return set()
    by_page: dict[int, list[Path]] = {}
    prefix = f"{context.ticker}_p"
    for path in folder.glob(f"{prefix}*.json"):
        number = path.stem.split("_p")[-1].split("_")[0]
        if number.isdigit():
            by_page.setdefault(int(number), []).append(path)
    stale = set()
    for pdf_page, ocr_paths in by_page.items():
        page_path = page_folder / f"p{pdf_page:04d}.json"
        if not page_path.is_file():
            stale.add(pdf_page)
            continue
        newest_ocr = max(path.stat().st_mtime for path in ocr_paths)
        if newest_ocr > page_path.stat().st_mtime:
            stale.add(pdf_page)
            continue
        try:
            record = json.loads(page_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            stale.add(pdf_page)
            continue
        ocr_chars = int((record.get("quality_metadata") or {}).get("ocr_char_count") or 0)
        if ocr_chars == 0:
            stale.add(pdf_page)
    return stale


def _items_changed(path: Path, sheet: dict) -> bool:
    if not path.is_file():
        return True
    try:
        current = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return True
    if not isinstance(current, dict):
        return True
    return current.get("items") != sheet["items"]


def _write_failure(context: RunContext, report: dict) -> None:
    if context.dry_run:
        return
    destination = context.log_dir / "failures" / f"{context.ticker}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _write_state(path: Path, ticker: str, completed: list[str], failed: str | None, error: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "ticker": ticker,
        "completed_stages": completed,
        "failed_stage": failed,
        "error": error,
    }, indent=2), encoding="utf-8")


def _jsonl_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _jsonl_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_json(path: Path):
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


_STAGES_AFTER_EXTRACT = ("dedup", "validate", "factsheet", "evaluate")


STAGE_FUNCTIONS = {
    "discover": stage_discover,
    "inspect": stage_inspect,
    "classify": stage_classify,
    "ocr": stage_ocr,
    "tables": stage_tables,
    "pages": stage_pages,
    "vision": stage_vision,
    "extract": stage_extract,
    "dedup": stage_dedup,
    "validate": stage_validate,
    "factsheet": stage_factsheet,
    "evaluate": stage_evaluate,
}
