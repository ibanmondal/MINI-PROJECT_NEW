#!/usr/bin/env python3
# Simple test to check imports

try:
    import cv2
    print("✓ OpenCV imported successfully")
except ImportError as e:
    print(f"✗ OpenCV import failed: {e}")

try:
    import pytesseract
    print("✓ Pytesseract imported successfully")
except ImportError as e:
    print(f"✗ Pytesseract import failed: {e}")

try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))
    import ocr_reader
    print("✓ OCR reader imported successfully")
except Exception as e:
    print(f"✗ OCR reader import failed: {e}")

print("Import test completed.")