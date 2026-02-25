"""
tts_engine.py - Offline Text-to-Speech engine for Vani-Vision.

Uses the native OS speech synthesizer (e.g., SAPI5 on Windows) via pyttsx3.
"""

from loguru import logger
import threading
import multiprocessing

try:
    import pyttsx3
except ImportError:
    logger.error("pyttsx3 is not installed. TTS won't work.")
    pyttsx3 = None

try:
    import pythoncom
except ImportError:
    pythoncom = None

class TTSEngine:
    def __init__(self):
        self._muted = False
        self._current_engine = None
        self._speech_thread = None

    def start(self):
        pass

    def stop(self):
        """Attempts to stop the currently running speech synthesis."""
        if self._current_engine is not None:
            try:
                self._current_engine.stop()
            except Exception as e:
                logger.debug(f"Could not cleanly stop TTS: {e}")
        
    def set_muted(self, muted: bool):
        """Enable or disable voice output."""
        self._muted = muted

    def say(self, text: str):
        """Speaks the text asynchronously by spawning a dedicated daemon thread."""
        if pyttsx3 is None or self._muted or not text:
            return
        
        clean_text = text.replace('*', '').replace('#', '').strip()
        if not clean_text:
            return

        # Stop any currently playing audio
        self.stop()
        
        self._speech_thread = threading.Thread(
            target=self._speak_worker, 
            args=(clean_text,), 
            daemon=True
        )
        self._speech_thread.start()

    def _speak_worker(self, text: str):
        try:
            if pythoncom is not None:
                pythoncom.CoInitialize()
            
            engine = pyttsx3.init()
            self._current_engine = engine
            
            voices = engine.getProperty('voices')
            for voice in voices:
                if "zira" in voice.name.lower() or "female" in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
                    
            rate = engine.getProperty('rate')
            engine.setProperty('rate', max(100, rate - 25))
            
            logger.info("Speaking aloud...")
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            logger.error(f"TTS error while speaking: {e}")
        finally:
            self._current_engine = None
            if pythoncom is not None:
                pythoncom.CoUninitialize()

# Provide a global instance for the app to use
engine = TTSEngine()


