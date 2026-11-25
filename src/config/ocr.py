"""OCR raster and gate chosen on a mixed-page sample.

The sample kept the cached half images (about 300–360 DPI). A fresh 300 DPI
re-render kept less text, so this target stays the preprocess size and the
re-run reads the cached PNG. Document pages use PSM 3. A kept run needs two
words at confidence 50 or more and a run mean of at least 50.
"""

# Long edge of a half, in dots per inch of the PDF point size.
OCR_TARGET_DPI = 300
# Tesseract page segmentation for a text page. Photographs stay on PSM 11.
OCR_PSM_DOCUMENT = 3
# Word confidence is 0-100. A kept run also needs the mean and the word count.
OCR_WORD_CONFIDENCE = 50
OCR_MIN_CONTIGUOUS_WORDS = 2
OCR_MIN_RUN_CONFIDENCE = 50
