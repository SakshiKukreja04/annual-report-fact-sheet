"""A missing half is logged and retried once. It does not stop the other halves."""

from src.ocr.pool import run_pool


def test_pool_retries_a_missing_image_and_keeps_going(monkeypatch, tmp_path):
    calls = []

    def fake_pending(ticker, limit=None):
        rows = [{"pdf_page": 61, "half": "LEFT"}, {"pdf_page": 64, "half": "RIGHT"}]
        if limit is None:
            return []
        return rows[:limit]

    def fake_write(ticker, item):
        calls.append((item["pdf_page"], item["half"]))
        if item["half"] == "LEFT" and calls.count((61, "LEFT")) == 1:
            raise FileNotFoundError("Missing preprocessed image: p0061_LEFT/full.png")
        return {"pdf_page": item["pdf_page"], "half": item["half"], "status": "written"}

    monkeypatch.setattr("src.ocr.pool.pending", fake_pending)
    monkeypatch.setattr("src.ocr.pool._write_record", fake_write)
    monkeypatch.setattr("src.ocr.pool.PROJECT_ROOT", tmp_path)
    report = run_pool("POWERGRID", limit=5, workers=0)
    assert report["written"] == 2
    assert report["failed"] == []
    assert calls.count((61, "LEFT")) == 2
    assert (64, "RIGHT") in calls


def test_pool_reports_a_half_that_fails_twice(monkeypatch, tmp_path):
    def fake_pending(ticker, limit=None):
        rows = [{"pdf_page": 61, "half": "LEFT"}]
        return [] if limit is None else rows

    def fake_write(ticker, item):
        raise FileNotFoundError("Missing preprocessed image: p0061_LEFT/full.png")

    monkeypatch.setattr("src.ocr.pool.pending", fake_pending)
    monkeypatch.setattr("src.ocr.pool._write_record", fake_write)
    monkeypatch.setattr("src.ocr.pool.PROJECT_ROOT", tmp_path)
    report = run_pool("POWERGRID", limit=1, workers=0)
    assert report["written"] == 0
    assert report["failed"][0]["pdf_page"] == 61
    assert "Missing preprocessed image" in report["failed"][0]["error"]
