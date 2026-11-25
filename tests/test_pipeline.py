"""One-command pipeline: company config, resume, citations, and rate limits."""

import json
from pathlib import Path

from src.extract.client import _retryable
from src.extract.run import run_pages
from src.factsheet.schema import FactSheet
from src.main import main
from src.pipeline.companies import COMPANIES, STAGES
from src.pipeline.runner import RunContext, _citation_errors, run_company, stage_ocr, stage_pages


def test_each_company_has_a_page_count_and_the_period():
    assert set(COMPANIES) == {"POWERGRID", "HINDALCO", "COALINDIA"}
    assert COMPANIES["POWERGRID"]["expected_pages"] == 364
    assert COMPANIES["HINDALCO"]["expected_pages"] == 413
    assert COMPANIES["COALINDIA"]["expected_pages"] == 371
    for company in COMPANIES.values():
        assert company["period"] == "FY25"
    assert "extract" in STAGES and "factsheet" in STAGES and "evaluate" in STAGES


def test_a_failed_stage_stops_the_run_and_resume_keeps_finished_work(tmp_path):
    calls = []

    def discover(context):
        calls.append("discover")
        return {"status": "ok"}

    def inspect(context):
        calls.append("inspect")
        raise RuntimeError("page inspection stopped")

    def classify(context):
        calls.append("classify")
        return {"status": "ok"}

    context = RunContext(ticker="HINDALCO", log_dir=tmp_path)
    report = run_company(context, {"discover": discover, "inspect": inspect, "classify": classify})
    assert report["failed_stage"] == "inspect"
    assert calls == ["discover", "inspect"]
    failure = json.loads((tmp_path / "failures" / "HINDALCO.json").read_text(encoding="utf-8"))
    assert failure["error"].startswith("RuntimeError")

    def inspect_ok(context):
        calls.append("inspect-ok")
        return {"status": "ok"}

    second = run_company(context, {"discover": discover, "inspect": inspect_ok, "classify": classify})
    assert second["failed_stage"] is None
    assert calls == ["discover", "inspect", "inspect-ok", "classify"]
    assert second["stages"]["discover"]["status"] == "skipped_completed"


def test_a_stage_with_gaps_is_run_again(tmp_path):
    calls = {"ocr": 0}

    def ocr(context):
        calls["ocr"] += 1
        if calls["ocr"] == 1:
            return {"status": "completed_with_gaps", "processed": 0}
        return {"status": "completed", "processed": 1}

    def discover(context):
        return {"status": "ok"}

    context = RunContext(ticker="POWERGRID", log_dir=tmp_path)
    first = run_company(context, {"discover": discover, "ocr": ocr})
    assert first["stages"]["ocr"]["status"] == "completed_with_gaps"
    second = run_company(context, {"discover": discover, "ocr": ocr})
    assert second["stages"]["discover"]["status"] == "skipped_completed"
    assert second["stages"]["ocr"]["status"] == "completed"
    assert calls["ocr"] == 2


def test_stage_pages_rebuilds_a_stale_ocr_page_without_touching_other_outputs(tmp_path, monkeypatch):
    import os

    ticker = "POWERGRID"
    page_dir = tmp_path / "pages"
    ocr_dir = tmp_path / "ocr"
    page_folder = page_dir / ticker
    ocr_folder = ocr_dir / ticker
    page_folder.mkdir(parents=True)
    ocr_folder.mkdir(parents=True)
    for number in range(1, 365):
        chars = 0 if number == 10 else 120
        (page_folder / f"p{number:04d}.json").write_text(
            json.dumps({"pdf_page": number, "quality_metadata": {"ocr_char_count": chars}}),
            encoding="utf-8",
        )
    ocr_10 = ocr_folder / "POWERGRID_p0010_LEFT.json"
    ocr_11 = ocr_folder / "POWERGRID_p0011_LEFT.json"
    ocr_12 = ocr_folder / "POWERGRID_p0012_LEFT.json"
    ocr_10.write_text("{}", encoding="utf-8")
    ocr_11.write_text("{}", encoding="utf-8")
    ocr_12.write_text("{}", encoding="utf-8")
    now = 1_800_000_000
    os.utime(page_folder / "p0010.json", (now - 100, now - 100))
    os.utime(ocr_10, (now - 10, now - 10))
    os.utime(ocr_11, (now - 50, now - 50))
    os.utime(page_folder / "p0011.json", (now, now))
    os.utime(page_folder / "p0012.json", (now - 100, now - 100))
    os.utime(ocr_12, (now - 5, now - 5))
    seen = {}

    def fake_build(company, pages=None, resume=True, **kwargs):
        seen["pages"] = pages
        seen["resume"] = resume
        seen["output_dir"] = kwargs.get("output_dir")
        return {"written": 0 if pages is None else len(pages), "reused": 0}

    monkeypatch.setattr("src.pages.build.build_company", fake_build)
    context = RunContext(
        ticker=ticker,
        page_dir=page_dir,
        ocr_dir=ocr_dir,
        log_dir=tmp_path / "logs",
        output_dir=tmp_path / "outputs",
        candidate_dir=tmp_path / "candidates",
    )
    result = stage_pages(context)
    assert result["status"] == "completed"
    assert seen["resume"] is False
    assert seen["output_dir"] == page_dir
    assert set(seen["pages"]) == {10, 12}


def test_a_token_cap_resumes_extract_without_repeating_finished_stages(tmp_path):
    calls = []

    def discover(context):
        calls.append("discover")
        return {"status": "ok"}

    def extract(context):
        calls.append("extract")
        if calls.count("extract") == 1:
            return {"status": "daily_cap", "pages_not_sent": [40], "stop_reason": "daily_cap"}
        return {"status": "completed", "pages_not_sent": [], "stop_reason": "completed"}

    def dedup(context):
        calls.append("dedup")
        return {"status": "completed"}

    def factsheet(context):
        calls.append("factsheet")
        return {"status": "replaced"}

    stages = {"discover": discover, "extract": extract, "dedup": dedup, "factsheet": factsheet}
    context = RunContext(ticker="HINDALCO", log_dir=tmp_path)
    first = run_company(context, stages)
    assert first["stages"]["extract"]["status"] == "daily_cap"
    saved = json.loads((tmp_path / "runs" / "HINDALCO.json").read_text(encoding="utf-8"))
    assert saved["completed_stages"] == ["discover"]

    second = run_company(context, stages)
    assert second["stages"]["discover"]["status"] == "skipped_completed"
    assert second["stages"]["extract"]["status"] == "completed"
    assert calls == ["discover", "extract", "dedup", "factsheet", "extract", "dedup", "factsheet"]


def test_unknown_company_is_rejected(tmp_path):
    try:
        run_company(RunContext(ticker="OTHER", log_dir=tmp_path))
    except KeyError as exc:
        assert "OTHER" in str(exc)
    else:
        raise AssertionError("An unknown company should be rejected.")


def test_scanned_pages_are_listed_for_ocr_and_digital_pages_are_not(tmp_path):
    classification = tmp_path / "classification"
    classification.mkdir()
    rows = [
        {"pdf_page": 1, "classifications": ["DIGITAL_TEXT"]},
        {"pdf_page": 2, "classifications": ["SCANNED_IMAGE", "TWO_PAGE_SPREAD"]},
    ]
    (classification / "POWERGRID_pages.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )
    context = RunContext(
        ticker="POWERGRID",
        dry_run=True,
        classification_dir=classification,
        ocr_dir=tmp_path / "ocr",
        log_dir=tmp_path / "logs",
    )
    result = stage_ocr(context)
    assert result["status"] == "dry_run"
    assert result["pages"] == [2]
    assert result["halves"] == 2


def test_a_quote_must_cite_a_real_page_record(tmp_path):
    page_dir = tmp_path / "pages" / "HINDALCO"
    page_dir.mkdir(parents=True)
    (page_dir / "p0013.json").write_text(json.dumps({"pdf_page": 13, "content_status": "recoverable"}), encoding="utf-8")
    validation = tmp_path / "candidates" / "HINDALCO" / "validation"
    validation.mkdir(parents=True)
    quote = "to conduct the audit of the cost records of the Company for FY 2025-26"
    (validation / "quote_report.json").write_text(json.dumps([
        {"pdf_page": 13, "quote": quote, "accepted": True},
    ]), encoding="utf-8")
    context = RunContext(ticker="HINDALCO", page_dir=tmp_path / "pages", candidate_dir=tmp_path / "candidates")
    sheet = {
        "items": [{
            "name": "Cost audit of cost records",
            "sources": [{"pdf_page": 13, "printed_page": "12", "quote": quote}],
        }],
    }
    assert _citation_errors(context, sheet) == []
    sheet["items"][0]["sources"][0]["pdf_page"] = 99
    assert _citation_errors(context, sheet)


def test_fact_sheet_schema_keeps_pdf_and_printed_pages_apart():
    sheet = FactSheet.model_validate({
        "ticker": "COALINDIA",
        "period": "FY25",
        "one_liner": "COALINDIA FY25 has no validated items.",
        "summary": "No validated items.",
        "segments": [{"name": "Coal", "revenue": None, "unit": None, "share_pct": None}],
        "items": [{
            "name": "Coal",
            "kind": "product_or_service",
            "descriptor": "coal",
            "entity": "company",
            "status": "current",
            "revenue_share_pct": None,
            "sources": [{"pdf_page": 209, "printed_page": "377", "quote": "Coal"}],
        }],
    })
    assert sheet.items[0].sources[0].pdf_page == 209
    assert sheet.items[0].sources[0].printed_page == "377"
    assert sheet.segments[0].revenue is None


def test_rate_limit_is_retried_and_a_cached_success_is_not_sent_again(tmp_path, monkeypatch):
    from src.extract.keys import key_pool

    class Limited(Exception):
        status_code = 429

    assert _retryable(Limited())
    calls = {"count": 0}

    def create(payload):
        calls["count"] += 1
        if calls["count"] == 1:
            raise Limited("slow down")
        return type("Response", (), {
            "usage": type("Usage", (), {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})(),
            "choices": [type("Choice", (), {"message": type("Message", (), {"content": '{"candidates": []}'})()})()],
        })()

    page = {"ticker": "HINDALCO", "pdf_page": 13, "printed_page": "12", "content_status": "recoverable", "text_blocks": [
        {"selected": True, "text": "The company produces aluminium.", "reading_order": 0, "half": "FULL"},
    ], "tables": [], "captions": [], "warnings": []}
    monkeypatch.delenv("GROQ_API_KEYS", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-pipeline-a")
    key_pool(reset=True)
    settings = {
        "model": "openai/gpt-oss-120b",
        "max_output_tokens": 20,
        "batch_chars": 12000,
        "daily_token_cap": 50000,
        "max_attempts": 3,
        "timeout_seconds": 5,
        "published_tpm": 8000,
        "prompt_version": "extract-candidates-v1",
    }
    first = run_pages("HINDALCO", [page], output_dir=tmp_path, settings=settings, create=create, sleep=lambda _delay: None)
    assert first["stop_reason"] == "completed"
    assert calls["count"] == 2
    run_pages("HINDALCO", [page], output_dir=tmp_path, settings=settings, create=create, sleep=lambda _delay: None)
    assert calls["count"] == 2


def test_a_malformed_batch_is_not_treated_as_a_cached_success(tmp_path, monkeypatch):
    from src.extract.keys import key_pool

    calls = {"count": 0}

    def create(payload):
        calls["count"] += 1
        text = "not json" if calls["count"] == 1 else '{"candidates": []}'
        return type("Response", (), {
            "usage": type("Usage", (), {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})(),
            "choices": [type("Choice", (), {"message": type("Message", (), {"content": text})()})()],
        })()

    page = {"ticker": "HINDALCO", "pdf_page": 14, "printed_page": "13", "content_status": "recoverable", "text_blocks": [
        {"selected": True, "text": "The company produces copper.", "reading_order": 0, "half": "FULL"},
    ], "tables": [], "captions": [], "warnings": []}
    monkeypatch.delenv("GROQ_API_KEYS", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-pipeline-b")
    key_pool(reset=True)
    settings = {
        "model": "openai/gpt-oss-120b",
        "max_output_tokens": 20,
        "batch_chars": 12000,
        "daily_token_cap": 50000,
        "max_attempts": 1,
        "timeout_seconds": 5,
        "published_tpm": 8000,
        "prompt_version": "extract-candidates-v1",
    }
    first = run_pages("HINDALCO", [page], output_dir=tmp_path, settings=settings, create=create, sleep=lambda _delay: None)
    assert first["batches_completed"] == 1
    saved = json.loads(next((tmp_path / "HINDALCO" / "parsed").glob("*.json")).read_text(encoding="utf-8"))
    assert saved["request_status"] == "malformed_response"
    second = run_pages("HINDALCO", [page], output_dir=tmp_path, settings=settings, create=create, sleep=lambda _delay: None)
    assert calls["count"] == 2
    assert second["stop_reason"] == "completed"


def test_dry_run_command_rejects_nothing_for_a_known_company(monkeypatch, tmp_path, capsys):
    def fake_run(context, stages=None):
        assert context.ticker == "COALINDIA"
        assert context.dry_run is True
        return {"ticker": "COALINDIA", "failed_stage": None, "stages": {}}

    monkeypatch.setattr("src.main.run_company", fake_run)
    assert main(["--company", "COALINDIA", "--dry-run"]) == 0
    assert "COALINDIA" in capsys.readouterr().out
