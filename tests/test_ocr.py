"""Stage 4 quality gate. OCR engines are checked with a short synthetic line."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.ocr.assess import (
    ILLEGIBLE,
    IMAGE_ONLY_VISUAL,
    RECOVERABLE,
    character_error_rate,
    kept_word_runs,
    recoverability,
)
from src.ocr.binarize import adaptive_gaussian, otsu, sauvola, tesseract_variants
from src.ocr.paddle_engine import prepare_paddle_image
from src.ocr.fast import choose_binarization
from src.ocr.sample import SAMPLE_HALVES
from src.ocr.settings import WORD_CONFIDENCE_THRESHOLD
from src.ocr.tesseract_engine import psm_for_page, recognize_words, tesseract_available
from src.preprocessing.pipeline import TARGET_DPI, scale_to_target_dpi


def _word(text: str, confidence: float, line_id: str = "1-1-1") -> dict:
    return {"text": text, "confidence": confidence, "bbox": [0, 0, 10, 10], "line_id": line_id}


def test_character_error_rate_counts_each_edit():
    assert character_error_rate("revenue", "revenue") == 0.0
    # one substitution in seven characters
    assert character_error_rate("revenue", "revenve") == round(100 * 1 / 7, 2)
    assert character_error_rate("", "revenue") is None


def test_isolated_confident_word_is_not_a_quote():
    words = [
        _word("Grid", 95),
        _word("xx", 10),
        _word("Station", 99),
    ]
    assert kept_word_runs(words, threshold=60) == []


def test_a_medium_confidence_junk_run_is_not_evidence():
    words = [_word("sere", 67), _word("ford", 73), _word("be", 71)]
    assert kept_word_runs(words, threshold=60, minimum=3, min_run_confidence=80) == []


def test_punctuation_is_not_a_quote():
    words = [_word("-", 95), _word("a", 99), _word("-", 95)]
    assert kept_word_runs(words, threshold=60) == []


def test_three_adjacent_words_form_one_line():
    words = [
        _word("Power", 80),
        _word("Grid", 70),
        _word("Limited", 90),
        _word("noise", 12),
    ]
    runs = kept_word_runs(words, threshold=60)
    assert len(runs) == 1
    assert runs[0]["text"] == "Power Grid Limited"
    assert runs[0]["word_count"] == 3


def test_run_breaks_when_the_line_changes():
    words = [
        _word("Power", 90, "1-1-1"),
        _word("Grid", 90, "1-1-1"),
        _word("Limited", 90, "1-1-2"),
        _word("India", 90, "1-1-2"),
    ]
    assert kept_word_runs(words, threshold=60, minimum=3) == []


def test_illegible_page_is_kept_and_not_sent_onward():
    decision = recoverability(["SCANNED_IMAGE"], document_like=True, runs=[])
    assert decision["recoverability"] == ILLEGIBLE
    assert decision["failure_reason"] == "illegible_low_dpi"
    assert decision["pass_to_extraction"] is False


def test_photograph_with_a_caption_is_not_a_failed_text_page():
    runs = [{"text": "Substation at night", "word_count": 3, "line_confidence": 88.0}]
    decision = recoverability(["SCANNED_IMAGE"], document_like=False, runs=runs)
    assert decision["recoverability"] == IMAGE_ONLY_VISUAL
    assert decision["failure_reason"] is None
    assert decision["caption_text"] == "Substation at night"
    assert decision["pass_to_extraction"] is False


def test_digital_page_without_an_ocr_line_is_not_called_illegible():
    decision = recoverability(["DIGITAL_TEXT", "TABLE_HEAVY"], document_like=True, runs=[])
    assert decision["recoverability"] == RECOVERABLE
    assert decision["failure_reason"] is None
    assert decision["use_ocr_text"] is False
    assert decision["pass_to_extraction"] is True


def test_readable_text_page_stays_recoverable():
    runs = [{"text": "Revenue from operations", "word_count": 3, "line_confidence": 80.0}]
    decision = recoverability(["SCANNED_IMAGE"], document_like=True, runs=runs)
    assert decision["recoverability"] == RECOVERABLE
    assert decision["pass_to_extraction"] is True


def test_binarization_is_not_applied_to_the_paddle_image():
    gray = np.tile(np.arange(256, dtype=np.uint8), (8, 1))
    paddle = prepare_paddle_image(gray)
    assert paddle.shape == (8, 256, 3)
    assert len(np.unique(paddle)) > 2
    assert np.array_equal(paddle[:, :, 0], gray)


def test_otsu_is_skipped_when_illumination_is_uneven():
    gray = np.zeros((80, 80), dtype=np.uint8)
    gray[:, 40:] = 255
    assert otsu(gray) is None
    names = [name for name, _image in tesseract_variants(gray)]
    assert names == ["adaptive_gaussian", "sauvola"]


def test_even_page_gets_a_border_and_otsu():
    gray = np.full((60, 60), 240, dtype=np.uint8)
    gray[20:40, 10:50] = 20
    binary = adaptive_gaussian(gray)
    assert binary.shape[0] == gray.shape[0] + 40
    assert binary.shape[1] == gray.shape[1] + 40
    assert binary[0, 0] == 255
    assert otsu(gray) is not None
    assert sauvola(gray).shape == binary.shape


def test_a_text_page_tries_sauvola_before_the_other_thresholds():
    from src.ocr.sample import paddle_needed, threshold_order

    assert threshold_order(["SCANNED_IMAGE"], document_like=True) == ("sauvola", "adaptive_gaussian", "otsu")
    assert threshold_order(["SCANNED_IMAGE"], document_like=False) == ("sauvola",)
    assert paddle_needed([], ["SCANNED_IMAGE"], document_like=True) is True
    assert paddle_needed([{"text": "kept"}], ["SCANNED_IMAGE"], document_like=True) is False
    assert paddle_needed([], ["SCANNED_IMAGE"], document_like=False) is False


def test_psm_follows_the_page_kind():
    assert psm_for_page(["SCANNED_IMAGE"], document_like=True) == 3
    assert psm_for_page(["IMAGE_ONLY_VISUAL"], document_like=True) == 11
    assert psm_for_page(["SCANNED_IMAGE"], document_like=False) == 11


def test_sample_covers_all_three_companies():
    tickers = {ticker for ticker, _page, _half in SAMPLE_HALVES}
    assert tickers == {"POWERGRID", "COALINDIA", "HINDALCO"}


def test_starting_threshold_is_configurable_and_not_treated_as_final():
    assert WORD_CONFIDENCE_THRESHOLD == 50
    from src.ocr.sample import confidence_threshold

    assert confidence_threshold() == 50


def test_stage_does_not_import_groq():
    root = Path(__file__).resolve().parents[1] / "src" / "ocr"
    allowed_qwen_flag = {"sample.py", "fast.py"}
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        assert "groq" not in text
        # sample/fast only record that Qwen rescue was not used.
        assert "qwen" not in text or path.name in allowed_qwen_flag
        if path.name in allowed_qwen_flag:
            assert "from groq" not in text and "import groq" not in text


@pytest.mark.skipif(not tesseract_available(), reason="Tesseract binary is not installed")
def test_tesseract_keeps_word_confidence_on_a_synthetic_line():
    image = Image.new("L", (900, 140), 255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 48)
    draw.text((30, 40), "Revenue from operations", font=font, fill=0)
    gray = np.array(image)
    binary = adaptive_gaussian(gray)
    recognized = recognize_words(binary, psm=6)
    text = recognized["raw_text"].casefold()
    assert "revenue" in text
    assert "operations" in text
    assert all("confidence" in word and "bbox" in word for word in recognized["words"])
    assert all(word["confidence"] >= 0 for word in recognized["words"])


def test_fast_binarization_follows_illumination_std():
    assert choose_binarization(10.0) == "otsu"
    assert choose_binarization(40.0) == "sauvola"


def test_scale_to_target_dpi_uses_three_hundred_not_five_x():
    gray = np.full((100, 200), 200, dtype=np.uint8)
    # 200px wide half that is 72pt wide → native ~200 DPI; target 300 → ~1.5x
    scaled, scale, meta = scale_to_target_dpi(gray, width_pt=72.0, height_pt=36.0, target_dpi=TARGET_DPI)
    assert TARGET_DPI == 300
    assert meta["target_dpi"] == 300
    assert 1.2 <= scale <= 1.8
    assert scaled.shape[1] > gray.shape[1]
