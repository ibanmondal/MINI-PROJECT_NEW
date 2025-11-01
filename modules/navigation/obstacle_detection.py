import cv2
from ultralytics import YOLO
from modules.tts_engine import speak_text

model = YOLO("models/yolov8n.pt")  # use your trained or default YOLOv8 model

def detect_obstacles_realtime():
    """Detect obstacles and give warnings."""
    cap = cv2.VideoCapture(0)
    obstacle_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf > 0.6:
                    obstacle_detected = True
                    speak_text("Obstacle ahead. Please move carefully.")
                    break

        if obstacle_detected:
            break

    cap.release()
    cv2.destroyAllWindows()
