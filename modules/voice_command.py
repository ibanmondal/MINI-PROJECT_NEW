# Add these imports at the top (below existing ones)
from modules.tts_engine import speak_text
import speech_recognition as sr
from modules.navigation.navigation_mode import start_navigation

commands = {}

def register_voice_command(phrase, callback): commands[phrase.lower()] = callback
_listening = False

def start_voice_listening():
    global _listening
    try:
        import speech_recognition as sr
        import threading
        r = sr.Recognizer()
        mic = sr.Microphone()
        _listening = True
        print("Adjusting for ambient noise... Please wait.")
        def loop():
            while _listening:
                try:
                    with mic as source:
                        r.adjust_for_ambient_noise(source, duration=0.8)
                        audio = r.listen(source, timeout=5, phrase_time_limit=5)
                    text = r.recognize_google(audio).lower().strip()
                    print(f"[Voice] Heard: {text}")
                    if text.startswith("viso"):
                        cmd = text.replace("viso", "", 1).strip()
                        cb = commands.get(cmd)
                        if cb: cb()
                except Exception:
                    continue
        threading.Thread(target=loop, daemon=True).start()
        print("Voice command system ready!")
        return True
    except Exception as e:
        print(f"Voice command unavailable: {e}")
        _listening = False
        return False

def stop_voice_listening():
    global _listening
    _listening = False

def activate_navigation_mode():
    """Activates Navigation Mode through voice input."""
    try:
        speak_text("Navigation mode activated. Where would you like to go?")
        r = sr.Recognizer()
        mic = sr.Microphone()

        with mic as source:
            r.adjust_for_ambient_noise(source, duration=0.8)
            audio = r.listen(source, timeout=8, phrase_time_limit=6)

        destination = r.recognize_google(audio)
        speak_text(f"Navigating to {destination}")
        start_navigation(destination)

    except sr.UnknownValueError:
        speak_text("Sorry, I didn't catch that. Please repeat the destination.")
    except Exception as e:
        speak_text("Navigation mode encountered an error.")
        print(f"[Navigation Error] {e}")

