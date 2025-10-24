import cv2
import sys
import os
import time
import threading

# Add the modules directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

import ocr_reader
import tts_engine
import voice_command
import object_detector

class VisoSonicAssistant:
    def __init__(self):
        """Initialize the Viso-Sonic Assistant"""
        self.cap = None
        self.is_running = False
        self.current_mode = "ocr"  # ocr, object, color, voice
        self.background_frame = None
        self.last_announcement = time.time()
        self.announcement_interval = 3  # seconds
        
        # Initialize voice commands
        self.setup_voice_commands()
        
        print("Viso-Sonic Assistant initialized!")
        print("Available modes: OCR, Object Detection, Color Detection, Voice Commands")
        print("Voice commands: 'read text', 'detect objects', 'detect colors', 'help', 'quit'")
    
    def setup_voice_commands(self):
        """Setup voice command callbacks"""
        voice_command.register_voice_command("read text", self.switch_to_ocr)
        voice_command.register_voice_command("detect objects", self.switch_to_objects)
        voice_command.register_voice_command("detect colors", self.switch_to_colors)
        voice_command.register_voice_command("help", self.announce_help)
        voice_command.register_voice_command("quit", self.quit_application)
        voice_command.register_voice_command("exit", self.quit_application)
        voice_command.register_voice_command("stop", self.quit_application)
    
    def switch_to_ocr(self):
        """Switch to OCR mode"""
        self.current_mode = "ocr"
        tts_engine.speak_text("Switching to text reading mode")
        print("Mode: OCR Text Reading")
    
    def switch_to_objects(self):
        """Switch to object detection mode"""
        self.current_mode = "object"
        tts_engine.speak_text("Switching to object detection mode")
        print("Mode: Object Detection")
    
    def switch_to_colors(self):
        """Switch to color detection mode"""
        self.current_mode = "color"
        tts_engine.speak_text("Switching to color detection mode")
        print("Mode: Color Detection")
    
    def announce_help(self):
        """Announce available commands"""
        help_text = ("Available voice commands: read text, detect objects, detect colors, help, quit. "
                    "Press Q to quit, or use number keys 1 through 4 to switch modes.")
        tts_engine.speak_text(help_text)
        print("Help announced")
    
    def quit_application(self):
        """Quit the application"""
        tts_engine.speak_text("Goodbye!")
        self.is_running = False
        print("Quitting application...")
    
    def start(self):
        """Start the main application"""
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            error_msg = "Error: Could not access webcam."
            print(error_msg)
            tts_engine.speak_text(error_msg)
            return
        
        # Start voice command listening
        if voice_command.start_voice_listening():
            tts_engine.speak_text("Voice commands activated. Say 'viso' followed by a command.")
        
        self.is_running = True
        tts_engine.speak_text("Viso-Sonic Assistant started. Currently in text reading mode.")
        
        print("Controls:")
        print("- Press '1' for OCR mode")
        print("- Press '2' for Object Detection mode") 
        print("- Press '3' for Color Detection mode")
        print("- Press '4' for Voice Command mode")
        print("- Press 'q' to quit")
        print("- Use voice commands: 'viso [command]'")
        
        # Capture background frame for motion detection
        ret, self.background_frame = self.cap.read()
        
        # Main processing loop
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to grab frame")
                break
            
            # Process frame based on current mode
            self.process_frame(frame)
            
            # Display frame with annotations
            display_frame = self.annotate_frame(frame.copy())
            cv2.imshow("Viso-Sonic Assistant", display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                self.switch_to_ocr()
            elif key == ord('2'):
                self.switch_to_objects()
            elif key == ord('3'):
                self.switch_to_colors()
            elif key == ord('4'):
                tts_engine.speak_text("Voice command mode active. Say 'viso' followed by your command.")
                print("Listening for voice commands...")
        
        self.cleanup()
    
    def process_frame(self, frame):
        """Process frame based on current mode"""
        current_time = time.time()
        
        # Limit announcements to prevent spam
        if current_time - self.last_announcement < self.announcement_interval:
            return
        
        try:
            if self.current_mode == "ocr":
                self.process_ocr(frame)
            elif self.current_mode == "object":
                self.process_objects(frame)
            elif self.current_mode == "color":
                self.process_colors(frame)
        except Exception as e:
            print(f"Processing error: {e}")
    
    def process_ocr(self, frame):
        """Process OCR on frame"""
        text = ocr_reader.read_text(frame)
        if text and text.strip() and not text.startswith("OCR Error"):
            # Clean up the text
            cleaned_text = ' '.join(text.split())
            if len(cleaned_text) > 10:  # Only announce substantial text
                print(f"OCR: {cleaned_text}")
                tts_engine.speak_text(f"Text detected: {cleaned_text}")
                self.last_announcement = time.time()
    
    def process_objects(self, frame):
        """Process object detection on frame"""
        shapes = object_detector.detect_shapes_in_frame(frame)
        if shapes:
            shape_descriptions = []
            for shape in shapes[:3]:  # Limit to 3 shapes to avoid spam
                shape_descriptions.append(f"{shape['name']}")
            
            if shape_descriptions:
                description = f"Detected shapes: {', '.join(shape_descriptions)}"
                print(description)
                tts_engine.speak_text(description)
                self.last_announcement = time.time()
    
    def process_colors(self, frame):
        """Process color detection on frame"""
        colors = object_detector.detect_colors_in_frame(frame)
        if colors:
            # Announce top 2 colors
            color_descriptions = []
            for color in colors[:2]:
                if color['percentage'] > 10:  # Only significant colors
                    color_descriptions.append(f"{color['name']} ({color['percentage']:.1f}%)")
            
            if color_descriptions:
                description = f"Dominant colors: {', '.join(color_descriptions)}"
                print(description)
                tts_engine.speak_text(description)
                self.last_announcement = time.time()
    
    def annotate_frame(self, frame):
        """Add annotations to frame"""
        # Add mode indicator
        cv2.putText(frame, f"Mode: {self.current_mode.upper()}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add instructions
        instructions = [
            "1: OCR  2: Objects  3: Colors  4: Voice  Q: Quit",
            f"Say: 'viso [command]' for voice control"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, frame.shape[0] - 50 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        voice_command.stop_voice_listening()
        tts_engine.stop_speaking()
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        print("Viso-Sonic Assistant stopped.")

def main():
    """Main function"""
    try:
        assistant = VisoSonicAssistant()
        assistant.start()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")

if __name__ == "__main__":
    main()