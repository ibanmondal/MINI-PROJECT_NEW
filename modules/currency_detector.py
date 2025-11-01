import os
import cv2
import numpy as np
from ultralytics import YOLO

# Path to your YOLOv8 model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best.pt')
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"YOLOv8 model not found at: {MODEL_PATH}")

# Load YOLO model
model = YOLO(MODEL_PATH)

# Class mapping (matches your YAML exactly)
CURRENCY_CLASSES = ["0", "10", "100", "20", "200", "5", "50", "500"]

# Prediction history for smoothing
prediction_history = []
MAX_HISTORY = 5

# Confidence threshold
CONF_THRESHOLD = 0.5

def detect_currency_in_frame(frame):
    global prediction_history

    # Run YOLO prediction
    results = model.predict(frame, verbose=False)[0]

    # Initialize empty counts
    counts = np.zeros(len(CURRENCY_CLASSES), dtype=np.float32)

    # Aggregate results
    for box in results.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        if conf >= CONF_THRESHOLD:
            counts[cls] += conf

    # Normalize counts to sum=1 (like probability)
    if counts.sum() > 0:
        counts /= counts.sum()

    # Add to history for smoothing
    prediction_history.append(counts)
    if len(prediction_history) > MAX_HISTORY:
        prediction_history.pop(0)

    avg_preds = np.mean(prediction_history, axis=0)
    class_idx = int(np.argmax(avg_preds))
    confidence = float(np.max(avg_preds))

    detected = confidence > 0.55
    denomination = CURRENCY_CLASSES[class_idx] if detected else None

    return {
        "currency_detected": detected,
        "denomination": denomination,
        "confidence": confidence
    }

def get_currency_guidance_text(result):
    if result["currency_detected"]:
        return f"Detected ₹{result['denomination']} note with confidence {result['confidence']:.2f}"
    else:
        return "No currency detected. Please hold the note steady and ensure good lighting."

# Optional: visualize detections (for debugging)
def draw_currency_boxes(frame, results):
    for box in results.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        if conf >= CONF_THRESHOLD:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{CURRENCY_CLASSES[cls]} ₹ {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return frame
