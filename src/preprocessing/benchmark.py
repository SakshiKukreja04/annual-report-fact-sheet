"""Run the preprocessing benchmark on a fixed set of representative pages.

Super-resolution models are optional. Lanczos remains the method that always runs.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

from src.config.paths import LOG_DIR, PREPROCESS_CACHE_DIR, PROJECT_ROOT
from src.ingestion.audit import locate_pdfs
from src.preprocessing.pipeline import process_page
from src.utils.logging_config import configure_logging

LOGGER = logging.getLogger("factsheet")

# Fixed pages so a rerun writes the same folders.
BENCHMARK_PAGES = {
    "POWERGRID": [25, 26, 50, 80, 100, 150, 200, 250, 300, 364],
    "COALINDIA": [20, 38, 40, 82, 108, 314],
    "HINDALCO": [45, 67, 180, 245, 259],
}


def benchmark_page_count() -> int:
    return sum(len(pages) for pages in BENCHMARK_PAGES.values())


def super_resolution_status() -> dict:
    """Report optional enhancers. None of them replace Lanczos."""
    status = {
        "realesrgan": False,
        "swinir": False,
        "opencv_dnn_superres": False,
        "used": False,
        "fallback": "lanczos",
    }
    try:
        import realesrgan  # type: ignore
        status["realesrgan"] = realesrgan is not None
    except Exception:
        status["realesrgan"] = False
    try:
        import swinir  # type: ignore
        status["swinir"] = swinir is not None
    except Exception:
        status["swinir"] = False
    try:
        from cv2 import dnn_superres  # type: ignore
        status["opencv_dnn_superres"] = dnn_superres is not None
    except Exception:
        status["opencv_dnn_superres"] = False
    return status


def _relative(path: str) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def run_benchmark(output_dir: Path | None = None) -> dict:
    destination = output_dir or PREPROCESS_CACHE_DIR
    pdfs = locate_pdfs()
    records = []
    for ticker, pages in BENCHMARK_PAGES.items():
        with pymupdf.open(pdfs[ticker]) as document:
            for pdf_page in pages:
                LOGGER.info("%s preprocessing pdf_page %s", ticker, pdf_page)
                page_records = process_page(document[pdf_page - 1], ticker, destination)
                for record in page_records:
                    record["output_image_path"] = _relative(record["output_image_path"])
                    record["variant_paths"] = {
                        name: _relative(path) for name, path in record["variant_paths"].items()
                    }
                    records.append(record)
                    LOGGER.info(
                        "%s p%s %s ops=%s duration=%ss",
                        ticker,
                        record["pdf_page"],
                        record["half"],
                        ",".join(record["applied_operations"]),
                        record["processing_duration_sec"],
                    )
    summary = {
        "stage": "preprocessing_benchmark",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "page_count": benchmark_page_count(),
        "half_count": len(records),
        "super_resolution": super_resolution_status(),
        "notes": [
            "Variant A is the rendered half before enhancement.",
            "Variant B is 5x Lanczos.",
            "Variant C adds fastNlMeansDenoising.",
            "Variant D adds CLAHE and a light unsharp mask.",
            "full.png is the image intended for later OCR. It matches D unless a 3x3 median or a small deskew was applied.",
            "Spreads are split before upscaling. Portrait pages stay FULL.",
            "The grayscale images are not binarized and pages are not perspective-warped.",
            "Lanczos is the upscaler. Optional super-resolution models were not required.",
        ],
        "records": records,
    }
    destination.mkdir(parents=True, exist_ok=True)
    index_path = destination / "benchmark.json"
    index_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s halves=%s", index_path, len(records))
    return summary


def main() -> int:
    configure_logging(LOG_DIR / "preprocessing.log")
    run_benchmark()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
