import cv2
import time
import googlemaps
import pyttsx3
import requests
import math
import speech_recognition as sr
from ultralytics import YOLO
import threading
import re

# ============================================================
# CONFIGURATION
# ============================================================
GOOGLE_MAPS_API_KEY = "AIzaSyAp784TQGkfvo7HyIRSS9a5em5SmASGlps"  # ✅ Replace with your API key
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

engine = pyttsx3.init()
engine.setProperty('rate', 165)

# Load YOLOv8 for obstacle detection
model = YOLO("yolov8n.pt")


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def speak(text):
    """Speak out text using TTS and print it."""
    print(f"[TTS]: {text}")
    engine.say(text)
    engine.runAndWait()


def get_voice_command():
    """Capture user's destination via voice input."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("[Navigation] 🎙 Listening for destination...")
        audio = recognizer.listen(source, phrase_time_limit=5)
    try:
        query = recognizer.recognize_google(audio)
        print(f"[Voice] Heard: {query}")
        return query
    except Exception as e:
        print(f"[Navigation] Voice capture failed: {e}")
        return None


def clean_html(raw_html):
    """Strip HTML tags from Google Maps instructions."""
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, '', raw_html)


def calculate_distance(p1, p2):
    """Haversine formula to calculate distance between two lat/lng pairs."""
    R = 6371  # Earth radius (km)
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ============================================================
# GOOGLE MAPS INTEGRATION
# ============================================================
def get_coordinates(place):
    """Fetch coordinates for a given place using Google Maps Geocoding API."""
    try:
        if not place or len(place.strip()) < 3:
            speak("That destination name seems too short.")
            return None

        print(f"[Navigation] Searching for location: {place}")
        result = gmaps.geocode(place)

        # Retry with "near me" for better results
        if not result:
            print("[Navigation] No results found, retrying with 'near me'...")
            result = gmaps.geocode(f"{place} near me")

        if not result:
            speak("Couldn't find that location on Google Maps.")
            return None

        location = result[0]['geometry']['location']
        print(f"[Navigation] Found coordinates: {location['lat']}, {location['lng']}")
        return (location['lat'], location['lng'])
    except Exception as e:
        print(f"[Navigation] Error getting coordinates: {e}")
        return None


def get_route_directions(origin, destination):
    """Retrieve walking directions from Google Maps."""
    try:
        print(f"[DEBUG] Getting route from {origin} to {destination}")
        directions = gmaps.directions(origin, destination, mode="walking")

        if not directions:
            speak("No route found to that destination.")
            return None

        steps = directions[0]['legs'][0]['steps']
        return [clean_html(step['html_instructions']) for step in steps]
    except Exception as e:
        print(f"[Navigation] Route error: {e}")
        return None


def get_current_location():
    """Get approximate current location using IP-based lookup."""
    try:
        loc = requests.get("https://ipinfo.io/").json()
        lat, lng = map(float, loc["loc"].split(","))
        print(f"[Navigation] Current origin (approx): {lat},{lng}")
        return (lat, lng)
    except Exception as e:
        print(f"[Navigation] Error fetching location: {e}")
        speak("Unable to determine current location.")
        return None


# ============================================================
# OBSTACLE DETECTION LOGIC
# ============================================================
def analyze_obstacles(frame):
    """Use YOLOv8 to detect nearby obstacles and return feedback."""
    results = model(frame)
    detections = results[0].boxes
    objects_detected = []

    for box in detections:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        objects_detected.append(label)

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if not objects_detected:
        return "Clear path ahead."

    feedback = []
    if 'person' in objects_detected:
        feedback.append("Person detected ahead.")
    if 'car' in objects_detected or 'bus' in objects_detected or 'bicycle' in objects_detected:
        feedback.append("Vehicle nearby.")
    if 'chair' in objects_detected or 'bench' in objects_detected:
        feedback.append("Obstacle detected on your path.")

    return " ".join(feedback) if feedback else "Clear path ahead."


# ============================================================
# NAVIGATION LOGIC
# ============================================================
def navigation_thread_function(route, dest_name):
    """Thread for continuous navigation and obstacle monitoring."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Unable to access the camera.")
        return

    speak(f"Starting navigation to {dest_name}. Press Q to stop.")

    for idx, step in enumerate(route):
        speak(f"Step {idx + 1}: {step}")

        for _ in range(20):  # roughly 10 seconds per step
            ret, frame = cap.read()
            if not ret:
                continue

            feedback = analyze_obstacles(frame)
            cv2.imshow("Navigation Mode", frame)

            # Speak obstacle feedback every few frames
            if feedback != "Clear path ahead.":
                speak(feedback)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                speak("Navigation stopped by user.")
                cap.release()
                cv2.destroyAllWindows()
                return

        time.sleep(1)

    speak("You have reached your destination.")
    cap.release()
    cv2.destroyAllWindows()


def start_navigation():
    """Activate full navigation workflow."""
    speak("Navigation mode activated. Where would you like to go?")
    destination = get_voice_command()

    if not destination:
        destination = input("Enter destination for navigation (fallback): ")

    dest_coords = get_coordinates(destination)
    if not dest_coords:
        speak("Could not find that place.")
        return

    origin_coords = get_current_location()
    if not origin_coords:
        return

    route = get_route_directions(origin_coords, dest_coords)
    if not route:
        speak("Could not fetch directions.")
        return

    distance_km = calculate_distance(origin_coords, dest_coords)
    speak(f"Distance to {destination} is approximately {round(distance_km, 2)} kilometers.")
    time.sleep(1)

    # Start navigation thread
    thread = threading.Thread(target=navigation_thread_function, args=(route, destination))
    thread.daemon = True
    thread.start()

    # Keep thread running
    while thread.is_alive():
        time.sleep(1)


# ============================================================
# TEST RUN (only if run directly)
# ============================================================
if __name__ == "__main__":
    start_navigation()
