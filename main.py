import cv2
import sys
import os

# Add the modules directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

import ocr_reader

def main():
    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break

        # Run OCR on the current frame
        text = ocr_reader.read_text(frame)

        if text.strip():
            print(f"OCR Output: {text}")

        # Display webcam feed
        cv2.imshow("Viso-Sonic OCR", frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()