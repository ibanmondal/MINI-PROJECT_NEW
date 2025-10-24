import cv2
import os
import numpy as np
from ultralytics import YOLO

_models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
_model_path = os.path.join(_models_dir, "yolov8n.pt")  # Using yolov8n.pt

_yolo_ready = False
_model = None

if os.path.isfile(_model_path):
    try:
        _model = YOLO(_model_path)
        _yolo_ready = True
        print("[object_detector] YOLOv8 loaded.")
    except Exception as e:
        print(f"[object_detector] YOLO load failed: {e}")
else:
    print("[object_detector] YOLOv8 model not found in 'models/'. Fallback active.")

def detect_objects_in_frame(frame, conf_threshold=0.5, nms_threshold=0.4):
    """
    Returns: [{'label': str, 'confidence': float, 'box': (x,y,w,h)}]
    """
    results = []
    if not _yolo_ready:
        return results

    # Run YOLOv8 inference
    results_yolo = _model.predict(frame, conf=conf_threshold, iou=nms_threshold)

    for result in results_yolo:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            label = _model.names[cls]

            # Convert to (x, y, w, h)
            x = int(x1)
            y = int(y1)
            w = int(x2 - x1)
            h = int(y2 - y1)

            results.append({
                "label": label,
                "confidence": float(conf),
                "box": (x, y, w, h)
            })
    return results

def draw_detections(frame, detections, show_conf=True):
    for d in detections:
        x,y,w,h = d["box"]
        x = max(0, x); y = max(0, y)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        label = d["label"]
        if show_conf:
            label += f" {int(d['confidence']*100)}%"
        cv2.putText(frame, label, (x, max(0,y-7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

# ---- Fallback simple shapes so the app never breaks ----
def detect_shapes_in_frame(frame):
    shapes = []
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 1000:
                continue
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            vertices = len(approx)
            if vertices == 3:
                name = "triangle"
            elif vertices == 4:
                name = "rectangle"
            elif vertices > 8:
                area = cv2.contourArea(cnt)
                peri = cv2.arcLength(cnt, True)
                circ = 4*np.pi*area/(peri*peri + 1e-6)
                name = "circle" if circ > 0.7 else f"{vertices}-sided shape"
            else:
                name = f"{vertices}-sided shape"
            x,y,w,h = cv2.boundingRect(approx)
            shapes.append({
                "name": name,
                "center": (x + w//2, y + h//2),
                "area": float(cv2.contourArea(cnt)),
                "vertices": vertices,
                "bounding_box": (x,y,w,h)
            })
    except Exception:
        pass
    return shapes
