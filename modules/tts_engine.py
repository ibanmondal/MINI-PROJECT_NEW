import pyttsx3
import threading

_engine = None
_rate = 150

def _get_engine():
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
        _engine.setProperty('rate', _rate)
    return _engine

def set_rate(rate: int):
    global _rate
    _rate = int(rate)
    try:
        _get_engine().setProperty('rate', _rate)
    except Exception:
        pass

def speak_text(text: str, async_mode: bool = True):
    if not text:
        return
    def run():
        try:
            eng = _get_engine()
            eng.say(text)
            eng.runAndWait()
        except Exception:
            pass
    if async_mode:
        threading.Thread(target=run, daemon=True).start()
    else:
        run()

def stop_speaking():
    try:
        _get_engine().stop()
    except Exception:
        pass
