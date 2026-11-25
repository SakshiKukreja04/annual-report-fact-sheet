"""PaddleOCR path. The image stays grayscale; it is not binarized."""

from __future__ import annotations

import os

import numpy as np

# oneDNN on this CPU raises ConvertPirAttribute2RuntimeAttribute. Use plain CPU inference.
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

_ENGINE = None
_ENGINE_ERROR: str | None = None


def prepare_paddle_image(gray: np.ndarray) -> np.ndarray:
    """Repeat the gray plane into three channels. Pixel values are unchanged."""
    if gray.ndim != 2:
        raise ValueError("Paddle input must be a grayscale image")
    return np.stack([gray, gray, gray], axis=-1)


def paddle_status() -> dict:
    try:
        import paddleocr  # noqa: F401
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"available": True, "error": None}


def _engine():
    global _ENGINE, _ENGINE_ERROR
    if _ENGINE is not None or _ENGINE_ERROR is not None:
        return _ENGINE
    try:
        from paddleocr import PaddleOCR

        _ENGINE = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
    except TypeError:
        from paddleocr import PaddleOCR

        _ENGINE = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
    except Exception as exc:
        _ENGINE_ERROR = f"{type(exc).__name__}: {exc}"
        return None
    return _ENGINE


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _parse_prediction(prediction) -> list[dict]:
    """Read text, score, and polygon from either the 3.x or 2.x result shape."""
    lines = []
    if prediction is None:
        return lines
    payload = prediction
    if isinstance(prediction, list) and prediction and not isinstance(prediction[0], dict):
        # 2.x: [[box, (text, score)], ...] or a one-page wrapper around that list.
        if prediction and isinstance(prediction[0], list):
            payload = prediction[0] if prediction[0] and isinstance(prediction[0][0], (list, tuple)) else prediction
        for item in payload:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            box, rec = item[0], item[1]
            text = str(rec[0]) if isinstance(rec, (list, tuple)) else str(rec)
            score = float(rec[1]) if isinstance(rec, (list, tuple)) and len(rec) > 1 else None
            lines.append({"text": text, "confidence": score, "bbox": _box_from_polygon(box)})
        return lines

    rows = prediction if isinstance(prediction, list) else [prediction]
    for row in rows:
        data = row
        if hasattr(row, "json"):
            try:
                data = row.json
            except Exception:
                data = getattr(row, "res", row)
        if isinstance(data, dict) and "res" in data and isinstance(data["res"], dict):
            data = data["res"]
        if not isinstance(data, dict):
            continue
        texts = _as_list(data.get("rec_texts"))
        scores = _as_list(data.get("rec_scores"))
        polygons = _as_list(data.get("rec_polys") or data.get("dt_polys") or data.get("rec_boxes"))
        for index, text in enumerate(texts):
            score = scores[index] if index < len(scores) else None
            polygon = polygons[index] if index < len(polygons) else None
            lines.append(
                {
                    "text": str(text),
                    "confidence": None if score is None else round(float(score) * (100 if float(score) <= 1 else 1), 2),
                    "bbox": _box_from_polygon(polygon),
                }
            )
    return lines


def _box_from_polygon(polygon) -> list[int]:
    points = _as_list(polygon)
    if not points:
        return [0, 0, 0, 0]
    if len(points) == 4 and all(isinstance(value, (int, float, np.integer, np.floating)) for value in points):
        x0, y0, x1, y1 = [float(value) for value in points]
        return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
    pairs = []
    for point in points:
        coords = _as_list(point)
        if len(coords) >= 2:
            pairs.append((float(coords[0]), float(coords[1])))
    if not pairs:
        return [0, 0, 0, 0]
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    return [int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys))]


def run_paddle(gray: np.ndarray) -> dict:
    """OCR the unbinarized grayscale image. Failure is recorded, not hidden."""
    image = prepare_paddle_image(gray)
    engine = _engine()
    if engine is None:
        return {
            "available": False,
            "engine": "paddleocr",
            "error": _ENGINE_ERROR or paddle_status()["error"],
            "processing_method": "grayscale_no_binarization",
            "raw_text": "",
            "lines": [],
        }
    try:
        if hasattr(engine, "predict"):
            prediction = engine.predict(image)
        else:
            prediction = engine.ocr(image, cls=False)
        lines = _parse_prediction(prediction)
    except Exception as exc:
        return {
            "available": False,
            "engine": "paddleocr",
            "error": f"{type(exc).__name__}: {exc}",
            "processing_method": "grayscale_no_binarization",
            "raw_text": "",
            "lines": [],
        }
    return {
        "available": True,
        "engine": "paddleocr",
        "error": None,
        "processing_method": "grayscale_no_binarization",
        "raw_text": "\n".join(line["text"] for line in lines),
        "lines": lines,
    }
