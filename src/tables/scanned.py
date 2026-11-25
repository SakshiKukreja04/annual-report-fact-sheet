"""Scanned table path. PaddleOCR PP-Structure reads the grayscale image."""

from __future__ import annotations

import os

import numpy as np

from src.tables.grid import grid_score, html_to_grid, prepare_grid, split_header

os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

_ENGINE = None
_ENGINE_ERROR: str | None = None


def structure_status() -> dict:
    try:
        from paddleocr import TableRecognitionPipelineV2  # noqa: F401
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"available": _ENGINE_ERROR is None, "error": _ENGINE_ERROR}


def _engine():
    global _ENGINE, _ENGINE_ERROR
    if _ENGINE is not None or _ENGINE_ERROR is not None:
        return _ENGINE
    try:
        from paddleocr import TableRecognitionPipelineV2

        _ENGINE = TableRecognitionPipelineV2(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            enable_mkldnn=False,
        )
    except Exception as exc:
        _ENGINE_ERROR = f"{type(exc).__name__}: {exc}"
        return None
    return _ENGINE


def _walk(value, html_blocks: list[str], scores: list[float], boxes: list) -> None:
    if isinstance(value, dict):
        for key in ("pred_html", "html", "table_html"):
            if isinstance(value.get(key), str) and "<table" in value[key].lower():
                html_blocks.append(value[key])
        for key in ("rec_scores", "scores"):
            raw = value.get(key)
            if isinstance(raw, list):
                for item in raw:
                    try:
                        number = float(item)
                    except (TypeError, ValueError):
                        continue
                    scores.append(number * 100 if number <= 1 else number)
        if "cell_box_list" in value and value["cell_box_list"] is not None:
            boxes.append(value["cell_box_list"])
        for child in value.values():
            _walk(child, html_blocks, scores, boxes)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _walk(child, html_blocks, scores, boxes)
    elif hasattr(value, "json"):
        try:
            _walk(value.json, html_blocks, scores, boxes)
        except Exception:
            return


def extract_scanned_image(
    image: np.ndarray,
    ticker: str,
    pdf_page: int,
    printed_page_value: str | None = None,
    half: str | None = None,
) -> dict:
    """Read tables from a grayscale page image. The image is not binarized."""
    if image.ndim != 2:
        raise ValueError("Scanned table input must be a grayscale image")
    # The detector expects three channels. Repeating the gray plane does not threshold it.
    image = np.stack([image, image, image], axis=-1)
    engine = _engine()
    if engine is None:
        return {
            "available": False,
            "error": _ENGINE_ERROR,
            "tables": [],
        }
    try:
        prediction = engine.predict(
            image,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
    except Exception as exc:
        cause = exc.__cause__ or exc
        return {
            "available": False,
            "error": f"{type(cause).__name__}: {cause}",
            "tables": [],
        }
    html_blocks: list[str] = []
    scores: list[float] = []
    boxes: list = []
    _walk(prediction, html_blocks, scores, boxes)
    tables = []
    seen = set()
    for html in html_blocks:
        if html in seen:
            continue
        seen.add(html)
        grid = prepare_grid(html_to_grid(html))
        if grid_score(grid) < 0.45:
            continue
        headers, body = split_header(grid)
        confidence = round(sum(scores) / len(scores) / 100, 2) if scores else grid_score(grid)
        index = len(tables) + 1
        suffix = f"-{half}" if half else ""
        tables.append(
            {
                "ticker": ticker,
                "pdf_page": pdf_page,
                "printed_page": printed_page_value,
                "half": half,
                "table_id": f"{ticker}-p{pdf_page:04d}-s{index:02d}{suffix}",
                "headers": headers,
                "rows": [
                    {"pdf_page": pdf_page, "printed_page": printed_page_value, "cells": row}
                    for row in body
                ],
                "units": None,
                "footnotes": [],
                "extraction_method": "paddleocr_pp_structure",
                "confidence": confidence,
                "continuation_from": None,
                "continuation_to": None,
                "bbox": None,
                "cell_boxes_present": bool(boxes),
            }
        )
    return {"available": True, "error": None, "tables": tables}
