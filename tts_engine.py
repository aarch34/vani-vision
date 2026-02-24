"""
tts_engine.py - Offline Text-to-Speech engine for Vani-Vision.

Uses the native OS speech synthesizer (e.g., SAPI5 on Windows) via pyttsx3.
Now runs synchronously to ensure absolute stability on Windows, bypassing COM threading issues.
"""

from loguru import logger

try:
    import pyttsx3
except ImportError:
    logger.error("pyttsx3 is not installed. TTS won't work.")
    pyttsx3 = None

class TTSEngine:
    def __init__(self):
        self._muted = False

    def start(self):
        """No-op for compatibility with previous threaded version."""
        pass

    def stop(self):
        """No-op for compatibility with previous threaded version."""
        pass
        
    def set_muted(self, muted: bool):
        """Enable or disable voice output."""
        self._muted = muted

    def say(self, text: str):
        """Speaks the text synchronously if not muted."""
        if pyttsx3 is None or self._muted or not text:
            return

        try:
            # Simple cleanup: remove markdown asterisks for cleaner speech
            clean_text = text.replace('*', '').replace('#', '').strip()
            
            engine = pyttsx3.init()
            
            # Select a decent female voice (usually Zira on Windows)
            voices = engine.getProperty('voices')
            for voice in voices:
                if "zira" in voice.name.lower() or "female" in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
            
            # Slightly slower, calmer speaking rate
            rate = engine.getProperty('rate')
            engine.setProperty('rate', max(100, rate - 25))
            
            logger.info("Speaking aloud...")
            engine.say(clean_text)
            engine.runAndWait()
            
        except Exception as e:
            logger.error(f"TTS error while speaking: {e}")

# Provide a global instance for the app to use
engine = TTSEngine()
