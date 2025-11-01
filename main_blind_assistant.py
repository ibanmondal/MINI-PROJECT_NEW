import cv2
import sys
import os
import time
import numpy as np

# Add the modules directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

import ocr_reader
import tts_engine
import voice_command
import object_detector
import scene_description

# ---- Currency detection ----
from currency_detector import detect_currency_in_frame, get_currency_guidance_text

class BlindAssistantReader:
    def __init__(self):
        """Initialize the Blind Assistant Reader"""
        self.cap = None
        self.is_running = False
        self.current_mode = "document"
        self.last_announcement = time.time()
        self.announcement_interval = 2
        self.reading_speed = "normal"
        self.auto_read = True
        self.last_read_text = ""
        self.last_detections = []

        self.document_text_buffer = []
        self.last_stable_text = ""
        self.text_stability_count = 0
        self.stability_threshold = 2

        self.setup_blind_voice_commands()

        tts_engine.speak_text("Blind Assistant Reader initializing. Please wait.", async_mode=False)
        print("Blind Assistant Reader initialized!")
        self.announce_available_modes()

    # ---------------- Voice commands ----------------
    def setup_blind_voice_commands(self):
        # Document commands
        voice_command.register_voice_command("read document", self.switch_to_document)
        voice_command.register_voice_command("read text", self.switch_to_document)
        voice_command.register_voice_command("read page", self.read_full_page)
        voice_command.register_voice_command("repeat", self.repeat_last_reading)
        voice_command.register_voice_command("stop reading", self.stop_reading)
        voice_command.register_voice_command("pause", self.pause_reading)
        voice_command.register_voice_command("continue", self.continue_reading)

        # Navigation & scene
        voice_command.register_voice_command("navigation mode", self.switch_to_navigation)
        voice_command.register_voice_command("describe scene", self.switch_to_scene)
        voice_command.register_voice_command("what do you see", self.describe_current_scene)

        # Object detection
        voice_command.register_voice_command("object detection", self.switch_to_objects)
        voice_command.register_voice_command("object mode", self.switch_to_objects)
        voice_command.register_voice_command("detect objects", self.switch_to_objects)
        voice_command.register_voice_command("find objects", self.switch_to_objects)

        # Currency detection
        voice_command.register_voice_command("identify money", self.switch_to_currency)
        voice_command.register_voice_command("check currency", self.switch_to_currency)

        # Speed
        voice_command.register_voice_command("speak slower", self.set_slow_speed)
        voice_command.register_voice_command("speak faster", self.set_fast_speed)
        voice_command.register_voice_command("normal speed", self.set_normal_speed)

        # Auto read
        voice_command.register_voice_command("auto read on", self.enable_auto_read)
        voice_command.register_voice_command("auto read off", self.disable_auto_read)

        # Help & exit
        voice_command.register_voice_command("help", self.announce_help)
        voice_command.register_voice_command("what can you do", self.announce_capabilities)
        voice_command.register_voice_command("quit", self.quit_application)
        voice_command.register_voice_command("exit", self.quit_application)

    # ---------------- Announcements ----------------
    def announce_available_modes(self):
        modes_text = ("Available modes: Document reading, Navigation, Scene description, "
                      "Currency identification, and Object detection.")
        tts_engine.speak_text(modes_text)

    def announce_capabilities(self):
        capabilities = ("I can read documents, help you navigate by detecting obstacles, "
                        "describe the scene, identify Indian currency, and detect objects like people, vehicles, "
                        "animals and more from the COCO list.")
        tts_engine.speak_text(capabilities)

    def announce_help(self):
        help_text = ("Say 'viso' then: read document, read page, repeat, stop reading, "
                     "navigation mode, describe scene, object detection, identify money, "
                     "speak slower, speak faster, normal speed, auto read on or off, help, or quit. "
                     "Keyboard: SPACE to read, R to repeat, 1-5 to change modes, Q to quit.")
        tts_engine.speak_text(help_text)

    # ---------------- Mode switching ----------------
    def switch_to_document(self):
        self.current_mode = "document"
        self.document_text_buffer.clear()
        self.last_stable_text = ""
        self.text_stability_count = 0
        tts_engine.speak_text("Document reading mode. Hold your document steady in good light.")
        print("Mode: Document")

    def switch_to_navigation(self):
        self.current_mode = "navigation"
        tts_engine.speak_text("Navigation mode. I will warn you about obstacles ahead.")
        print("Mode: Navigation")

    def switch_to_scene(self):
        self.current_mode = "scene"
        tts_engine.speak_text("Scene description mode.")
        print("Mode: Scene")

    def switch_to_currency(self):
        self.current_mode = "currency"
        tts_engine.speak_text("Currency identification mode.")
        print("Mode: Currency")

    def switch_to_objects(self):
        self.current_mode = "objects"
        self.last_detections = []
        tts_engine.speak_text("Object detection mode. Looking for all COCO classes.")
        print("Mode: Objects")

    # ---------------- Reading speed ----------------
    def set_slow_speed(self):
        self.reading_speed = "slow"
        tts_engine.set_rate(120)
        tts_engine.speak_text("Reading speed set to slow")

    def set_normal_speed(self):
        self.reading_speed = "normal"
        tts_engine.set_rate(150)
        tts_engine.speak_text("Reading speed set to normal")

    def set_fast_speed(self):
        self.reading_speed = "fast"
        tts_engine.set_rate(200)
        tts_engine.speak_text("Reading speed set to fast")

    def enable_auto_read(self):
        self.auto_read = True
        tts_engine.speak_text("Automatic reading enabled")

    def disable_auto_read(self):
        self.auto_read = False
        tts_engine.speak_text("Automatic reading disabled. Press space to read manually.")

    # ---------------- Reading ----------------
    def read_full_page(self):
        if self.document_text_buffer:
            full_text = " ".join(self.document_text_buffer)
            tts_engine.speak_text(f"Reading full page: {full_text}")
        else:
            tts_engine.speak_text("No text detected on page")

    def repeat_last_reading(self):
        if self.last_read_text:
            tts_engine.speak_text(f"Repeating: {self.last_read_text}")
        else:
            tts_engine.speak_text("No previous text to repeat")

    def stop_reading(self):
        tts_engine.stop_speaking()
        tts_engine.speak_text("Reading stopped")

    def pause_reading(self):
        self.stop_reading()

    def continue_reading(self):
        tts_engine.speak_text("Continuing")

    def describe_current_scene(self):
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            self.process_scene_description(self.current_frame, force_announce=True)
        else:
            tts_engine.speak_text("No image available to describe")

    def quit_application(self):
        tts_engine.speak_text("Goodbye! Thank you for using Blind Assistant Reader.")
        self.is_running = False
        print("Quitting application...")

    # ---------------- Start camera & main loop ----------------
    def start(self):
        for cam_id in (0,1,2):
            self.cap = cv2.VideoCapture(cam_id)
            if self.cap.isOpened():
                break
        if not self.cap or not self.cap.isOpened():
            msg = "Error: Could not access camera. Please check your camera."
            print(msg)
            tts_engine.speak_text(msg)
            return

        if voice_command.start_voice_listening():
            tts_engine.speak_text("Voice commands activated. Say 'viso' followed by a command.")
        else:
            print("Voice command not available; keyboard controls active.")
            tts_engine.speak_text("Voice command not available. Use the keyboard.")

        self.is_running = True
        tts_engine.speak_text("Blind Assistant Reader started in document reading mode. Hold text steady.")
        print("=== Blind Assistant Reader ===")
        print("SPACE: Manual Read   R: Repeat   1: Document   2: Navigation   3: Scene   4: Currency   5: Objects   Q: Quit")

        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to grab frame")
                tts_engine.speak_text("Camera error occurred")
                break

            self.current_frame = frame.copy()
            self.process_frame(frame)

            display_frame = self.annotate_frame(frame.copy())
            cv2.imshow("Blind Assistant Reader - Visual", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.manual_read_trigger(frame)
            elif key == ord('r'):
                self.repeat_last_reading()
            elif key == ord('1'):
                self.switch_to_document()
            elif key == ord('2'):
                self.switch_to_navigation()
            elif key == ord('3'):
                self.switch_to_scene()
            elif key == ord('4'):
                self.switch_to_currency()
            elif key == ord('5'):
                self.switch_to_objects()

        self.cleanup()

    # ---------------- Frame processing ----------------
    def process_frame(self, frame):
        now = time.time()
        try:
            if self.current_mode == "document":
                self.process_document_reading(frame)
            elif self.current_mode == "navigation":
                self.process_navigation_assistance(frame, now)
            elif self.current_mode == "scene":
                self.process_scene_description(frame, now)
            elif self.current_mode == "currency":
                self.process_currency_identification(frame, now)
            elif self.current_mode == "objects":
                self.process_object_detection(frame, now)
        except Exception as e:
            print(f"Processing error: {e}")

    # ---------------- Document Reading ----------------
    def process_document_reading(self, frame):
        h, w = frame.shape[:2]
        roi = frame[int(h*0.2):int(h*0.8), int(w*0.1):int(w*0.9)]
        cv2.rectangle(frame, (int(w*0.1), int(h*0.2)), (int(w*0.9), int(h*0.8)), (0,255,0), 2)
        text = ocr_reader.read_text(roi)
        if text.strip():
            self.last_read_text = text
            self.document_text_buffer.append(text)
            if self.auto_read:
                tts_engine.speak_text(text)

    # ---------------- Navigation ----------------
    def process_navigation_assistance(self, frame, now):
        if now - self.last_announcement < self.announcement_interval * 1.5:
            return

        detections = object_detector.detect_objects_in_frame(frame)
        obstacle_labels = {
            "person","bicycle","car","motorbike","bus","truck","train","bench",
            "chair","sofa","pottedplant","diningtable","tvmonitor","bird","cat","dog",
            "horse","sheep","cow","elephant","bear","zebra","giraffe","umbrella","handbag"
        }

        announced = False
        if detections:
            H, W = frame.shape[:2]
            obstacles = [d for d in detections if d["label"] in obstacle_labels]
            obstacles = sorted(obstacles, key=lambda d: d['box'][2]*d['box'][3], reverse=True)[:5]

            for d in obstacles:
                x, y, w, h = d['box']
                center_x = x + w // 2
                direction = "straight ahead"
                if center_x < W * 0.4: direction = "on your left"
                elif center_x > W * 0.6: direction = "on your right"

                dist_ratio = (w*h)/(W*H)
                dist = "ahead"
                if dist_ratio > 0.18: dist = "very close"
                elif dist_ratio > 0.08: dist = "nearby"

                msg = f"{d['label']} {direction}, {dist}"
                print(f"Navigation: {msg}")
                tts_engine.speak_text(msg)
                time.sleep(0.3)

            self.last_announcement = now
            announced = True

        if not announced:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            large = [c for c in contours if cv2.contourArea(c) > 8000]
            if large:
                x, y, w, h = cv2.boundingRect(large[0])
                center_x = x + w//2
                frame_center = frame.shape[1] // 2
                pos = "straight ahead"
                if center_x < frame_center - 60: pos = "on your left"
                elif center_x > frame_center + 60: pos = "on your right"
                msg = f"Obstacle {pos}."
                print(f"Navigation: {msg}")
                tts_engine.speak_text(msg)
                self.last_announcement = now

    # ---------------- Scene Description ----------------
    def process_scene_description(self, frame, now=None, force_announce=False):
        scene = scene_description.describe_scene_for_blind_user(frame)
        description = scene.get('overall_description', '')
        if description:
            print(f"Scene: {description}")
            tts_engine.speak_text(description)
            if now:
                self.last_announcement = now

    # ---------------- Currency ----------------
    def process_currency_identification(self, frame, now):
        results = detect_currency_in_frame(frame)
        guidance = get_currency_guidance_text(results)
        if guidance:
            print(f"Currency: {guidance}")
            tts_engine.speak_text(guidance)
            self.last_announcement = now

    # ---------------- Objects ----------------
    def process_object_detection(self, frame, now=None):
        detections = object_detector.detect_objects_in_frame(frame)
        self.last_detections = detections
        if detections:
            labels = [d['label'] for d in detections[:3]]
            msg = "Detected: " + ", ".join(labels)
            print(f"Objects: {msg}")
            tts_engine.speak_text(msg)
            if now:
                self.last_announcement = now

    # ---------------- Manual read ----------------
    def manual_read_trigger(self, frame):
        text = ocr_reader.read_text(frame)
        if text.strip():
            print(f"Manual Read: {text}")
            tts_engine.speak_text(text)
            self.last_read_text = text

    # ---------------- Annotate frame ----------------
    def annotate_frame(self, frame):
        cv2.putText(frame, f"Mode: {self.current_mode.upper()}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
        status = "AUTO READ ON" if self.auto_read else "AUTO READ OFF"
        cv2.putText(frame, status, (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),2)
        cv2.putText(frame, f"Speed: {self.reading_speed.upper()}", (10,110), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
        instructions = "SPACE: Read  R: Repeat  1:Doc 2:Nav 3:Scene 4:Currency 5:Objects  Q:Quit"
        cv2.putText(frame, instructions, (10,frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)
        if self.current_mode == "objects":
            object_detector.draw_detections(frame, self.last_detections)
        return frame

    # ---------------- Cleanup ----------------
    def cleanup(self):
        print("Cleaning up...")
        try: voice_command.stop_voice_listening()
        except: pass
        try: tts_engine.stop_speaking()
        except: pass
        if self.cap: self.cap.release()
        cv2.destroyAllWindows()
        print("Blind Assistant Reader stopped.")

def main():
    assistant = BlindAssistantReader()
    assistant.start()

if __name__ == "__main__":
    main()
