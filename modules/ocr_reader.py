import os
import cv2
import numpy as np

# Try to locate Tesseract on Windows if not in PATH
_WIN_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def _preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # enhance contrast
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # adaptive threshold helps under uneven lighting
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 10)
    # slight morphology to clean noise
    kernel = np.ones((2,2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    return th

def read_text(frame):
    # Prefer pytesseract; fallback to easyocr if available
    try:
        import pytesseract
        if os.name == "nt":
            # Set default Windows path if present
            if os.path.exists(_WIN_TESSERACT):
                pytesseract.pytesseract.tesseract_cmd = _WIN_TESSERACT
        proc = _preprocess(frame)
        config = "--oem 3 --psm 6"
        txt = pytesseract.image_to_string(proc, config=config)
        return txt.strip()
    except Exception as e:
        # Try EasyOCR fallback
        try:
            import easyocr  # pip install easyocr
            reader = easyocr.Reader(['en'], gpu=False)
            result = reader.readtext(frame)
            text = " ".join([r[1] for r in result])
            return text.strip()
        except Exception as e2:
            return f"OCR Error: {e2}"
