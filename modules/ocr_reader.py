"""
OCR Reader module for text detection and recognition
"""

import cv2
import numpy as np
import pytesseract
import easyocr
from PIL import Image
import os
from modules.utils import clean_text, validate_image_format, limit_text_length

# -------------------- Windows Tesseract path --------------------
# Update this path if your Tesseract is installed elsewhere
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class OCRReader:
    def __init__(self):
        """Initialize OCR reader with multiple engines"""
        self.tesseract_config = '--oem 3 --psm 11'
        self.easyocr_reader = None

        # Try EasyOCR
        try:
            self.easyocr_reader = easyocr.Reader(['en'])
            print("[OCRReader] EasyOCR initialized successfully")
        except Exception as e:
            print(f"[OCRReader] EasyOCR initialization failed: {e}")
            print("[OCRReader] Falling back to Tesseract only")

    def read_text_from_image(self, image_path, preprocess=True):
        if not os.path.exists(image_path):
            return "Image file not found"
        if not validate_image_format(image_path):
            return "Unsupported image format"

        try:
            image = cv2.imread(image_path)
            if image is None:
                return "Failed to load image"
            return self.read_text_from_frame(image, preprocess)
        except Exception as e:
            return f"Error reading image: {str(e)}"

    def read_text_from_frame(self, frame, preprocess=True):
        if frame is None or frame.size == 0:
            return ""

        try:
            if preprocess:
                processed_frame = self._preprocess_image(frame)
            else:
                processed_frame = frame

            text = ""

            # --- EasyOCR first ---
            if self.easyocr_reader:
                try:
                    results = self.easyocr_reader.readtext(processed_frame)
                    easyocr_text = " ".join([r[1] for r in results if r[2] > 0.5])
                    if easyocr_text.strip():
                        text = easyocr_text
                except Exception as e:
                    print(f"[OCRReader] EasyOCR failed: {e}")

            # --- Tesseract fallback ---
            if not text.strip():
                try:
                    pil_image = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
                    tesseract_text = pytesseract.image_to_string(pil_image, config=self.tesseract_config)
                    text = tesseract_text
                except Exception as e:
                    print(f"[OCRReader] Tesseract failed: {e}")
                    return ""

            cleaned_text = clean_text(text)
            return limit_text_length(cleaned_text, max_length=500)

        except Exception as e:
            print(f"[OCRReader] Error in OCR processing: {e}")
            return ""

    def _preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Slight sharpening
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0,-1,0]])
        sharp = cv2.filter2D(gray, -1, kernel)
        blur = cv2.GaussianBlur(sharp, (3,3), 0)
        return blur

    def detect_text_regions(self, frame):
        if frame is None:
            return []

        try:
            if self.easyocr_reader:
                results = self.easyocr_reader.readtext(frame)
                return [r[0] for r in results if r[2] > 0.3]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            text_regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    text_regions.append([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
            return text_regions

        except Exception as e:
            print(f"[OCRReader] Error detecting text regions: {e}")
            return []

# Global OCR instance
ocr_reader = OCRReader()

# Convenience functions
def read_text_from_image(image_path, preprocess=True):
    return ocr_reader.read_text_from_image(image_path, preprocess)

def read_text_from_frame(frame, preprocess=True):
    return ocr_reader.read_text_from_frame(frame, preprocess)

def detect_text_regions(frame):
    return ocr_reader.detect_text_regions(frame)

def read_text(frame):
    return ocr_reader.read_text_from_frame(frame)
