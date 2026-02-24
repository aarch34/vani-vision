"""
tts_engine.py - Offline Text-to-Speech engine for Vani-Vision.

Runs a background thread with a queue to process text and speak it aloud
using the native OS speech synthesizer (e.g. SAPI5 on Windows) via pyttsx3,
ensuring the Streamlit UI does not block while speaking.
"""

import threading
import queue
import time
from loguru import logger

try:
    import pyttsx3
except ImportError:
    logger.error("pyttsx3 is not installed. TTS won't work.")
    pyttsx3 = None

class TTSEngine:
    def __init__(self):
        self._queue = queue.Queue()
        self._running = False
        self._thread = None
        self._muted = False
        self._engine = None

    def start(self):
        """Starts the background TTS thread."""
        if pyttsx3 is None:
            logger.warning("pyttsx3 missing. TTSEngine not starting.")
            return

        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        logger.info("TTSEngine background thread started.")

    def stop(self):
        """Stops the background TTS thread."""
        self._running = False
        # Push an empty string to unblock the queue get
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("TTSEngine background thread stopped.")
        
    def set_muted(self, muted: bool):
        """Enable or disable voice output."""
        self._muted = muted
        if muted:
             # Try to clear pending speech
             while not self._queue.empty():
                 try:
                     self._queue.get_nowait()
                 except queue.Empty:
                     break

    def say(self, text: str):
        """Queues text to be spoken if not muted."""
        if not self._muted and text:
             # Simple cleanup: remove markdown asterisks for cleaner speech
             clean_text = text.replace('*', '').replace('#', '').strip()
             self._queue.put(clean_text)

    def _process_loop(self):
        """Continuously waits for text and speaks it."""
        try:
            # Initialize inside the thread as pyttsx3 can be thread-sensitive
            self._engine = pyttsx3.init()
            # Try to select a decent female voice (usually Zira on Windows)
            voices = self._engine.getProperty('voices')
            for voice in voices:
                if "zira" in voice.name.lower() or "female" in voice.name.lower():
                    self._engine.setProperty('voice', voice.id)
                    break
            
            # Slightly slower, calmer speaking rate
            rate = self._engine.getProperty('rate')
            self._engine.setProperty('rate', max(100, rate - 25))
            
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3 engine: {e}")
            self._running = False
            return

        while self._running:
            try:
                text = self._queue.get(timeout=1.0)
                if text is None:
                    continue  # Stop signal
                    
                if not self._muted:
                    self._engine.say(text)
                    self._engine.runAndWait()
                    
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"TTS error while speaking: {e}")
                time.sleep(1.0) # Prevent tight spin on error

# Provide a global instance for the app to use
engine = TTSEngine()
