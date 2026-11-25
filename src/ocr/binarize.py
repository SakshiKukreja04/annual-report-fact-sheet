"""Thresholds used only on the Tesseract path. The Paddle image stays gray."""

from __future__ import annotations

import cv2
import numpy as np

from src.ocr.settings import WHITE_BORDER_PX


def illumination_std(gray: np.ndarray) -> float:
    """Spread of the slowly varying background. Text itself is blurred away."""
    blur = cv2.GaussianBlur(gray, (0, 0), 21)
    return float(blur.std())


def illumination_is_uniform(gray: np.ndarray, max_std: float = 28.0) -> bool:
    return illumination_std(gray) <= max_std


def _border(binary: np.ndarray) -> np.ndarray:
    return cv2.copyMakeBorder(
        binary,
        WHITE_BORDER_PX,
        WHITE_BORDER_PX,
        WHITE_BORDER_PX,
        WHITE_BORDER_PX,
        cv2.BORDER_CONSTANT,
        value=255,
    )


def _close(binary: np.ndarray) -> np.ndarray:
    kernel = np.ones((2, 2), dtype=np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def adaptive_gaussian(gray: np.ndarray) -> np.ndarray:
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )
    return _border(_close(binary))


def sauvola(gray: np.ndarray, window: int = 31, k: float = 0.2, r: float = 128.0) -> np.ndarray:
    """Local Sauvola threshold. The window is odd so it matches the Gaussian path."""
    if window % 2 == 0:
        window += 1
    values = gray.astype(np.float32)
    mean = cv2.boxFilter(values, -1, (window, window), normalize=True)
    mean_sq = cv2.boxFilter(values * values, -1, (window, window), normalize=True)
    deviation = np.sqrt(np.maximum(mean_sq - mean * mean, 0))
    threshold = mean * (1.0 + k * (deviation / r - 1.0))
    binary = np.where(values > threshold, 255, 0).astype(np.uint8)
    return _border(_close(binary))


def otsu(gray: np.ndarray) -> np.ndarray | None:
    """Return an Otsu image only when the page lighting is even."""
    if not illumination_is_uniform(gray):
        return None
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _border(_close(binary))


def tesseract_variants(gray: np.ndarray) -> list[tuple[str, np.ndarray]]:
    variants = [
        ("adaptive_gaussian", adaptive_gaussian(gray)),
        ("sauvola", sauvola(gray)),
    ]
    otsu_image = otsu(gray)
    if otsu_image is not None:
        variants.append(("otsu", otsu_image))
    return variants
