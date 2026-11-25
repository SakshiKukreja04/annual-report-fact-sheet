"""Render a PDF page, split real spreads, then enhance grayscale images.

Binarization and perspective warp are intentionally not part of this pipeline.
The saved grayscale image stays gray so later OCR can choose its own threshold.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.classification.classify import SPREAD_MIN_WIDTH_PT
from src.config.ocr import OCR_TARGET_DPI

HALF_LEFT = "LEFT"
HALF_RIGHT = "RIGHT"
HALF_FULL = "FULL"

LANCZOS_SCALE = 5
TARGET_DPI = OCR_TARGET_DPI
DENOISE_H = 10
DENOISE_TEMPLATE = 7
DENOISE_SEARCH = 15
SPECKLE_RATIO_MIN = 0.008
DESKEW_MIN_DEGREES = 0.5
DESKEW_MAX_DEGREES = 8.0


def is_two_page_spread(width_pt: float, height_pt: float) -> bool:
    """A spread is wide enough to be two portrait pages, not merely landscape."""
    if height_pt <= 0 or width_pt < SPREAD_MIN_WIDTH_PT:
        return False
    return (width_pt / height_pt) >= 1.25


def find_gutter_x(gray: np.ndarray) -> tuple[int, str]:
    """Return a split column and whether a bright central gutter was found."""
    height, width = gray.shape[:2]
    midpoint = width // 2
    if width < 20:
        return midpoint, "midpoint"
    x0 = int(width * 0.46)
    x1 = int(width * 0.54)
    if x1 <= x0 + 2:
        return midpoint, "midpoint"
    band = gray[:, x0:x1]
    column_mean = band.mean(axis=0)
    kernel = np.ones(5, dtype=np.float64) / 5
    smoothed = np.convolve(column_mean, kernel, mode="same")
    peak = int(np.argmax(smoothed))
    baseline = float(np.median(smoothed))
    if float(smoothed[peak]) >= baseline + 8:
        return x0 + peak, "gutter"
    return midpoint, "midpoint"


def speckle_ratio(gray: np.ndarray) -> float:
    """Share of near-black or near-white pixels that disagree with a 3x3 median."""
    sample = gray
    if gray.shape[1] > 900:
        scale = 900 / gray.shape[1]
        sample = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    median = cv2.medianBlur(sample, 3)
    difference = cv2.absdiff(sample, median)
    extreme = (sample < 12) | (sample > 243)
    mask = ((difference > 35) & extreme).astype(np.uint8)
    # A ruled line has dark neighbors. A speckle does not. Keep only isolated pixels.
    neighbor_kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    neighbors = cv2.filter2D(mask, -1, neighbor_kernel)
    isolated = (mask == 1) & (neighbors == 0)
    return float(isolated.mean())


def measured_skew_degrees(gray: np.ndarray) -> float:
    """Median angle of near-horizontal lines. Zero when the page has too few lines."""
    scale = 0.35 if gray.shape[1] > 1200 else 1.0
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(small, 50, 150)
    min_length = max(30, small.shape[1] // 5)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=min_length, maxLineGap=8)
    if lines is None:
        return 0.0
    lines = np.asarray(lines)
    if lines.ndim == 3:
        lines = lines[:, 0]
    angles: list[float] = []
    for x1, y1, x2, y2 in lines:
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle > 90:
            angle -= 180
        if angle < -90:
            angle += 180
        if abs(angle) <= 15:
            angles.append(angle)
    # A handful of decorative lines is not a page skew. Require many lines that agree.
    if len(angles) < 20:
        return 0.0
    low, high = np.percentile(angles, [25, 75])
    if float(high - low) > 1.0:
        return 0.0
    return float(np.median(angles))


def should_deskew(angle_degrees: float) -> bool:
    return DESKEW_MIN_DEGREES < abs(angle_degrees) <= DESKEW_MAX_DEGREES


def deskew(gray: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate by a small angle onto a white canvas. This is not perspective warp."""
    height, width = gray.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    new_width = int(height * sine + width * cosine)
    new_height = int(height * cosine + width * sine)
    matrix[0, 2] += new_width / 2 - center[0]
    matrix[1, 2] += new_height / 2 - center[1]
    return cv2.warpAffine(
        gray,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def upscale_lanczos(gray: np.ndarray, scale: int = LANCZOS_SCALE) -> np.ndarray:
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, None, DENOISE_H, DENOISE_TEMPLATE, DENOISE_SEARCH)


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def unsharp_mask(gray: np.ndarray) -> np.ndarray:
    """Light sharpening. A wide blur would eat thin strokes, so sigma stays at 1."""
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.addWeighted(gray, 1.25, blurred, -0.25, 0)


def _largest_embedded(page) -> tuple[Image.Image, dict]:
    if not page.images:
        raise RuntimeError(f"page {page.page_number} has no embedded image.")
    chosen = max(page.images, key=lambda item: float(item["width"]) * float(item["height"]))
    import io

    picture = Image.open(io.BytesIO(chosen["stream"].get_data())).convert("RGB")
    return picture, chosen


def embedded_image_metrics(pdf_path: Path, pdf_page: int) -> dict:
    """Pixel size of the largest embedded image vs the page size in points."""
    import pdfplumber

    with pdfplumber.open(pdf_path) as document:
        page = document.pages[pdf_page - 1]
        picture, chosen = _largest_embedded(page)
        width_pt = float(page.width)
        height_pt = float(page.height)
        pixel_w, pixel_h = picture.size
    dpi_x = round(pixel_w * 72.0 / max(width_pt, 1e-6), 2)
    dpi_y = round(pixel_h * 72.0 / max(height_pt, 1e-6), 2)
    return {
        "pdf_page": pdf_page,
        "page_width_pt": round(width_pt, 2),
        "page_height_pt": round(height_pt, 2),
        "embedded_pixel_width": int(pixel_w),
        "embedded_pixel_height": int(pixel_h),
        "embedded_display_width": round(float(chosen.get("width") or 0), 2),
        "embedded_display_height": round(float(chosen.get("height") or 0), 2),
        "effective_dpi_x": dpi_x,
        "effective_dpi_y": dpi_y,
        "effective_dpi": round((dpi_x + dpi_y) / 2.0, 2),
    }


def render_embedded_native(pdf_path: Path, pdf_page: int) -> tuple[np.ndarray, float, float, dict]:
    """Return the largest embedded image at its native pixel size (no 1pt resize)."""
    import pdfplumber

    with pdfplumber.open(pdf_path) as document:
        page = document.pages[pdf_page - 1]
        picture, chosen = _largest_embedded(page)
        width_pt = float(page.width)
        height_pt = float(page.height)
        metrics = {
            "embedded_pixel_width": picture.size[0],
            "embedded_pixel_height": picture.size[1],
            "effective_dpi_x": round(picture.size[0] * 72.0 / max(width_pt, 1e-6), 2),
            "effective_dpi_y": round(picture.size[1] * 72.0 / max(height_pt, 1e-6), 2),
            "embedded_display_width": round(float(chosen.get("width") or 0), 2),
            "embedded_display_height": round(float(chosen.get("height") or 0), 2),
        }
        return np.array(picture), width_pt, height_pt, metrics


def render_embedded_page(pdf_path: Path, pdf_page: int) -> tuple[np.ndarray, float, float]:
    """Use the page's own embedded image when PyMuPDF cannot load.

    The image is scaled to one pixel per PDF point, which is the same grid
    the PyMuPDF renderer uses. No extra detail is invented.
    """
    color, width_pt, height_pt, _metrics = render_embedded_native(pdf_path, pdf_page)
    width = max(int(round(width_pt)), 1)
    height = max(int(round(height_pt)), 1)
    if color.shape[1] != width or color.shape[0] != height:
        picture = Image.fromarray(color, mode="RGB").resize((width, height), Image.Resampling.LANCZOS)
        color = np.array(picture)
    return color, width_pt, height_pt


def scale_to_target_dpi(gray: np.ndarray, width_pt: float, height_pt: float, target_dpi: int = TARGET_DPI) -> tuple[np.ndarray, float, dict]:
    """Resize a half so its long edge matches ~target_dpi for its point size."""
    target_w = max(1, int(round(width_pt * target_dpi / 72.0)))
    target_h = max(1, int(round(height_pt * target_dpi / 72.0)))
    source_h, source_w = gray.shape[:2]
    scale = min(target_w / max(source_w, 1), target_h / max(source_h, 1))
    # Prefer the geometric mean of axis scales when the crop already matches aspect.
    scale_x = target_w / max(source_w, 1)
    scale_y = target_h / max(source_h, 1)
    scale = float((scale_x + scale_y) / 2.0)
    if abs(scale - 1.0) < 0.02:
        return gray, 1.0, {"target_dpi": target_dpi, "scale": 1.0, "target_size": [source_w, source_h]}
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interpolation)
    return resized, scale, {
        "target_dpi": target_dpi,
        "scale": round(scale, 4),
        "target_size": [int(resized.shape[1]), int(resized.shape[0])],
        "source_size": [source_w, source_h],
    }


def build_fast_half(color_half: np.ndarray, width_pt: float, height_pt: float, target_dpi: int = TARGET_DPI) -> dict:
    """Native-resolution half → ~300 DPI gray full.png only. No 5x Lanczos."""
    timers: dict[str, float] = {}
    operations = ["grayscale"]

    started = time.perf_counter()
    gray = cv2.cvtColor(color_half, cv2.COLOR_RGB2GRAY)
    timers["grayscale"] = round(time.perf_counter() - started, 3)

    started = time.perf_counter()
    scaled, scale, scale_meta = scale_to_target_dpi(gray, width_pt, height_pt, target_dpi)
    timers["scale_to_dpi"] = round(time.perf_counter() - started, 3)
    operations.append(f"scale_to_{target_dpi}dpi({scale:.3f}x)")

    started = time.perf_counter()
    speckles = speckle_ratio(scaled)
    timers["speckle_ratio"] = round(time.perf_counter() - started, 3)

    denoise_applied = speckles >= SPECKLE_RATIO_MIN
    working = scaled
    if denoise_applied:
        started = time.perf_counter()
        working = denoise(working)
        working = cv2.medianBlur(working, 3)
        timers["denoise"] = round(time.perf_counter() - started, 3)
        operations.extend([f"fastNlMeansDenoising(h={DENOISE_H})", "median_3x3"])
    else:
        timers["denoise"] = 0.0

    started = time.perf_counter()
    enhanced = unsharp_mask(apply_clahe(working))
    timers["clahe_unsharp"] = round(time.perf_counter() - started, 3)
    operations.extend(["clahe", "unsharp"])

    started = time.perf_counter()
    skew = measured_skew_degrees(enhanced)
    deskew_applied = should_deskew(skew)
    full = deskew(enhanced, skew) if deskew_applied else enhanced
    timers["deskew"] = round(time.perf_counter() - started, 3)
    if deskew_applied:
        operations.append(f"deskew({skew:.2f}deg)")

    duration = round(sum(timers.values()), 3)
    return {
        "full": full,
        "meta": {
            "applied_operations": operations,
            "scale_factor": round(scale, 4),
            "target_dpi": target_dpi,
            "original_dimensions": [int(color_half.shape[1]), int(color_half.shape[0])],
            "processed_dimensions": [int(full.shape[1]), int(full.shape[0])],
            "processing_duration_sec": duration,
            "timers": timers,
            "speckle_ratio": round(speckles, 6),
            "median_applied": denoise_applied,
            "denoise_applied": denoise_applied,
            "skew_degrees": round(skew, 3),
            "deskew_applied": deskew_applied,
            "scale_meta": scale_meta,
        },
    }


def process_rendered_fast(
    color: np.ndarray,
    width_pt: float,
    height_pt: float,
    pdf_page: int,
    ticker: str,
    output_dir: Path,
    target_dpi: int = TARGET_DPI,
) -> list[dict]:
    """Write only full.png per half at ~target DPI. Benchmark A-D variants are skipped."""
    spread = is_two_page_spread(width_pt, height_pt)
    gutter_x = None
    gutter_method = None
    if spread:
        pieces, gutter_x, gutter_method = split_spread(color)
        half_width_pt = width_pt / 2.0
    else:
        pieces = [(HALF_FULL, color)]
        half_width_pt = width_pt

    records = []
    for half, piece in pieces:
        if piece.size == 0 or piece.shape[1] < 2 or piece.shape[0] < 2:
            continue
        # For spreads, apportion width by the cut column when available.
        if spread and color.shape[1] > 0:
            frac = piece.shape[1] / color.shape[1]
            piece_width_pt = width_pt * frac
        else:
            piece_width_pt = half_width_pt
        built = build_fast_half(piece, piece_width_pt, height_pt, target_dpi=target_dpi)
        folder = output_dir / ticker / f"p{pdf_page:04d}_{half}"
        folder.mkdir(parents=True, exist_ok=True)
        full_path = folder / "full.png"
        save_array(full_path, built["full"])
        meta = built["meta"]
        records.append(
            {
                "ticker": ticker,
                "pdf_page": pdf_page,
                "half": half,
                "is_spread": spread,
                "page_width_pt": round(width_pt, 2),
                "page_height_pt": round(height_pt, 2),
                "half_width_pt": round(piece_width_pt, 2),
                "gutter_x": gutter_x,
                "gutter_method": gutter_method,
                "original_dimensions": meta["original_dimensions"],
                "processed_dimensions": meta["processed_dimensions"],
                "scale_factor": meta["scale_factor"],
                "target_dpi": meta["target_dpi"],
                "applied_operations": meta["applied_operations"],
                "processing_duration_sec": meta["processing_duration_sec"],
                "timers": meta["timers"],
                "output_image_path": full_path.as_posix(),
                "variant_paths": {"full": full_path.as_posix()},
                "speckle_ratio": meta["speckle_ratio"],
                "median_applied": meta["median_applied"],
                "denoise_applied": meta["denoise_applied"],
                "skew_degrees": meta["skew_degrees"],
                "deskew_applied": meta["deskew_applied"],
                "path": "fast",
            }
        )
    return records


def render_page(page: pymupdf.Page) -> np.ndarray:
    """Render at 1 PDF point per pixel so a low-resolution scan is not pre-enlarged."""
    pixmap = page.get_pixmap(alpha=False)
    channels = pixmap.n
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, channels)
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    if channels == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


def split_spread(color: np.ndarray) -> tuple[list[tuple[str, np.ndarray]], int, str]:
    gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
    cut, method = find_gutter_x(gray)
    cut = int(np.clip(cut, int(color.shape[1] * 0.4), int(color.shape[1] * 0.6)))
    halves = (
        (HALF_LEFT, color[:, :cut]),
        (HALF_RIGHT, color[:, cut:]),
    )
    return list(halves), cut, method


def build_variants(color_half: np.ndarray) -> dict[str, np.ndarray | dict]:
    """Create benchmark images A-D and the full grayscale pipeline image."""
    timers: dict[str, float] = {}
    started = time.perf_counter()
    gray = cv2.cvtColor(color_half, cv2.COLOR_RGB2GRAY)
    timers["grayscale"] = round(time.perf_counter() - started, 3)
    operations = ["grayscale"]

    started = time.perf_counter()
    skew = measured_skew_degrees(gray)
    speckles = speckle_ratio(gray)
    timers["measure"] = round(time.perf_counter() - started, 3)

    started = time.perf_counter()
    lanczos = upscale_lanczos(gray)
    timers["upscale"] = round(time.perf_counter() - started, 3)
    operations.append("lanczos_5x")

    started = time.perf_counter()
    denoised = denoise(lanczos)
    timers["denoise"] = round(time.perf_counter() - started, 3)
    operations.append(f"fastNlMeansDenoising(h={DENOISE_H})")
    # Variant C keeps this image so the benchmark can show denoise before CLAHE.
    variant_c = denoised
    median_applied = speckles >= SPECKLE_RATIO_MIN
    if median_applied:
        started = time.perf_counter()
        denoised = cv2.medianBlur(denoised, 3)
        timers["median"] = round(time.perf_counter() - started, 3)
        operations.append("median_3x3")
    else:
        timers["median"] = 0.0

    started = time.perf_counter()
    enhanced = unsharp_mask(apply_clahe(denoised))
    timers["clahe_unsharp"] = round(time.perf_counter() - started, 3)
    operations.extend(["clahe", "unsharp"])
    deskew_applied = should_deskew(skew)
    started = time.perf_counter()
    full = deskew(enhanced, skew) if deskew_applied else enhanced
    timers["deskew"] = round(time.perf_counter() - started, 3)
    if deskew_applied:
        operations.append(f"deskew({skew:.2f}deg)")
    variant_d = unsharp_mask(apply_clahe(variant_c))
    duration = round(sum(timers.values()), 3)
    return {
        "A_original": color_half,
        "B_lanczos5": lanczos,
        "C_lanczos5_denoise": variant_c,
        "D_lanczos5_clahe_sharp": variant_d,
        "full": full,
        "meta": {
            "applied_operations": operations,
            "scale_factor": LANCZOS_SCALE,
            "original_dimensions": [int(color_half.shape[1]), int(color_half.shape[0])],
            "processed_dimensions": [int(full.shape[1]), int(full.shape[0])],
            "processing_duration_sec": duration,
            "timers": timers,
            "speckle_ratio": round(speckles, 6),
            "median_applied": median_applied,
            "skew_degrees": round(skew, 3),
            "deskew_applied": deskew_applied,
        },
    }


def save_array(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 2:
        picture = Image.fromarray(image, mode="L")
    else:
        picture = Image.fromarray(image, mode="RGB")
    picture.save(path, format="PNG")


def process_page(page: pymupdf.Page, ticker: str, output_dir: Path) -> list[dict]:
    """Render one PDF page and write preprocessing outputs for each half."""
    return process_rendered(
        render_page(page),
        float(page.rect.width),
        float(page.rect.height),
        page.number + 1,
        ticker,
        output_dir,
    )


def process_rendered(
    color: np.ndarray,
    width_pt: float,
    height_pt: float,
    pdf_page: int,
    ticker: str,
    output_dir: Path,
) -> list[dict]:
    """Write preprocessing outputs for a page image that is already rendered."""
    spread = is_two_page_spread(width_pt, height_pt)
    gutter_x = None
    gutter_method = None
    if spread:
        pieces, gutter_x, gutter_method = split_spread(color)
    else:
        pieces = [(HALF_FULL, color)]

    records = []
    for half, piece in pieces:
        if piece.size == 0 or piece.shape[1] < 2 or piece.shape[0] < 2:
            continue
        variants = build_variants(piece)
        folder = output_dir / ticker / f"p{pdf_page:04d}_{half}"
        paths = {
            "A_original": folder / "A_original.png",
            "B_lanczos5": folder / "B_lanczos5.png",
            "C_lanczos5_denoise": folder / "C_lanczos5_denoise.png",
            "D_lanczos5_clahe_sharp": folder / "D_lanczos5_clahe_sharp.png",
            "full": folder / "full.png",
        }
        for name, image_path in paths.items():
            save_array(image_path, variants[name])
        meta = variants["meta"]
        record = {
            "ticker": ticker,
            "pdf_page": pdf_page,
            "half": half,
            "is_spread": spread,
            "page_width_pt": round(width_pt, 2),
            "page_height_pt": round(height_pt, 2),
            "gutter_x": gutter_x,
            "gutter_method": gutter_method,
            "original_dimensions": meta["original_dimensions"],
            "processed_dimensions": meta["processed_dimensions"],
            "scale_factor": meta["scale_factor"],
            "applied_operations": meta["applied_operations"],
            "processing_duration_sec": meta["processing_duration_sec"],
            "output_image_path": paths["full"].as_posix(),
            "variant_paths": {name: path.as_posix() for name, path in paths.items()},
            "speckle_ratio": meta["speckle_ratio"],
            "median_applied": meta["median_applied"],
            "skew_degrees": meta["skew_degrees"],
            "deskew_applied": meta["deskew_applied"],
        }
        records.append(record)
    return records
