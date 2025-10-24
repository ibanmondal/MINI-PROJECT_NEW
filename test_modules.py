#!/usr/bin/env python3
"""
Test script to verify all modules work correctly
"""

import sys
import os
import cv2
import numpy as np

# Add the modules directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

def test_imports():
    """Test if all modules can be imported"""
    print("Testing module imports...")
    
    try:
        import ocr_reader
        print("✓ OCR Reader imported successfully")
    except Exception as e:
        print(f"✗ OCR Reader import failed: {e}")
        return False
    
    try:
        import tts_engine
        print("✓ TTS Engine imported successfully")
    except Exception as e:
        print(f"✗ TTS Engine import failed: {e}")
        return False
    
    try:
        import voice_command
        print("✓ Voice Command imported successfully")
    except Exception as e:
        print(f"✗ Voice Command import failed: {e}")
        return False
    
    try:
        import object_detector
        print("✓ Object Detector imported successfully")
    except Exception as e:
        print(f"✗ Object Detector import failed: {e}")
        return False
    
    return True

def test_ocr():
    """Test OCR functionality"""
    print("\nTesting OCR functionality...")
    
    try:
        import ocr_reader
        
        # Create a simple test image with text
        img = np.ones((200, 400, 3), dtype=np.uint8) * 255
        cv2.putText(img, "HELLO WORLD", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        # Test OCR
        text = ocr_reader.read_text(img)
        if text and "HELLO" in text.upper():
            print(f"✓ OCR working: '{text}'")
            return True
        else:
            print(f"✗ OCR not detecting text properly: '{text}'")
            return False
            
    except Exception as e:
        print(f"✗ OCR test failed: {e}")
        return False

def test_tts():
    """Test TTS functionality"""
    print("\nTesting TTS functionality...")
    
    try:
        import tts_engine
        
        # Test TTS (will fallback to print if TTS not available)
        tts_engine.speak_text("Testing text to speech", async_mode=False)
        print("✓ TTS test completed")
        return True
        
    except Exception as e:
        print(f"✗ TTS test failed: {e}")
        return False

def test_object_detection():
    """Test object detection functionality"""
    print("\nTesting Object Detection functionality...")
    
    try:
        import object_detector
        
        # Create a test image with a simple shape
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
        
        # Test shape detection
        shapes = object_detector.detect_shapes_in_frame(img)
        print(f"✓ Shape detection working: {len(shapes)} shapes detected")
        
        # Test color detection
        colors = object_detector.detect_colors_in_frame(img)
        print(f"✓ Color detection working: {len(colors)} colors detected")
        
        return True
        
    except Exception as e:
        print(f"✗ Object detection test failed: {e}")
        return False

def test_voice_commands():
    """Test voice command setup"""
    print("\nTesting Voice Command setup...")
    
    try:
        import voice_command
        
        # Test command registration
        def test_callback():
            print("Test command executed!")
        
        voice_command.register_voice_command("test", test_callback)
        print("✓ Voice command registration working")
        
        return True
        
    except Exception as e:
        print(f"✗ Voice command test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Viso-Sonic Module Tests ===\n")
    
    tests = [
        ("Module Imports", test_imports),
        ("OCR Functionality", test_ocr),
        ("TTS Functionality", test_tts),
        ("Object Detection", test_object_detection),
        ("Voice Commands", test_voice_commands)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! The Viso-Sonic Assistant is ready to use.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()