import cv2
import numpy as np
import os
from ultralytics import YOLO

class SceneDescriptor:
    def __init__(self, yolo_model="models/yolov8n.pt"):
        """Initialize scene descriptor with YOLO object detection"""
        self.brightness_thresholds = {
            'very_dark': 30,
            'dark': 80,
            'normal': 180,
            'bright': 220
        }

        # Load YOLOv8
        if os.path.isfile(yolo_model):
            try:
                self.model = YOLO(yolo_model)
                print("[scene_description] YOLOv8 loaded.")
            except Exception as e:
                print(f"[scene_description] YOLO load failed: {e}")
                self.model = None
        else:
            print("[scene_description] YOLOv8 model not found.")
            self.model = None

    def describe_scene(self, frame):
        """Generate full analysis of the scene"""
        description = {
            'lighting': self._analyze_lighting(frame),
            'colors': self._analyze_colors(frame),
            'objects': self._detect_objects(frame),
            'text_present': self._detect_text(frame),
            'overall_description': ''
        }

        description['overall_description'] = self._generate_description(description)
        return description

    def _analyze_lighting(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        brightness_std = np.std(gray)

        if avg_brightness < self.brightness_thresholds['very_dark']:
            condition = 'very dark'
        elif avg_brightness < self.brightness_thresholds['dark']:
            condition = 'dimly lit'
        elif avg_brightness < self.brightness_thresholds['normal']:
            condition = 'well-lit'
        elif avg_brightness < self.brightness_thresholds['bright']:
            condition = 'bright'
        else:
            condition = 'very bright'

        evenness = 'even' if brightness_std < 40 else 'uneven'
        return {'condition': condition, 'evenness': evenness, 'brightness_value': avg_brightness}

    def _analyze_colors(self, frame):
        """Find dominant colors using HSV histogram"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        dominant_hues = np.argsort(hist.ravel())[::-1][:3]

        color_names = []
        for hue in dominant_hues:
            if 0 <= hue <= 10 or 160 <= hue <= 180:
                color_names.append("red")
            elif 10 < hue <= 25:
                color_names.append("orange")
            elif 25 < hue <= 35:
                color_names.append("yellow")
            elif 35 < hue <= 85:
                color_names.append("green")
            elif 85 < hue <= 125:
                color_names.append("blue")
            elif 125 < hue <= 160:
                color_names.append("purple")

        return {'dominant_colors': list(set(color_names))}

    def _detect_objects(self, frame):
        """Detect real-world objects with YOLOv8"""
        if self.model is None:
            return {'detected': []}

        # Run YOLOv8 inference
        results = self.model.predict(frame, conf=0.5, iou=0.4)

        detected_objects = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0].cpu().numpy())
                label = self.model.names[cls]
                detected_objects.append(label)

        return {'detected': list(set(detected_objects))}

    def _detect_text(self, frame):
        """Quick check for text using edges (can be upgraded to EAST)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        text_pixels = np.sum(edges > 0)
        return {'likely_text_present': text_pixels > 5000}

    def _generate_description(self, analysis):
        parts = []

        # Lighting
        lighting = analysis['lighting']
        parts.append(f"The scene is {lighting['condition']} with {lighting['evenness']} lighting")

        # Colors
        if analysis['colors']['dominant_colors']:
            parts.append("Dominant colors include " + ", ".join(analysis['colors']['dominant_colors']))

        # Objects
        if analysis['objects']['detected']:
            parts.append("I can see: " + ", ".join(analysis['objects']['detected']))
        else:
            parts.append("No clear objects detected")

        # Text
        if analysis['text_present']['likely_text_present']:
            parts.append("There might be some text in the scene")

        return ". ".join(parts) + "."

# Global instance
scene_descriptor = SceneDescriptor()

def describe_scene_for_blind_user(frame):
    return scene_descriptor.describe_scene(frame)

def get_quick_scene_description(frame):
    return scene_descriptor.describe_scene(frame)['overall_description']
