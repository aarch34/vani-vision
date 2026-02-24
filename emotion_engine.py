"""
emotion_engine.py - Continuous offline facial expression recognition for Vani-Vision.

Runs a background thread that captures frames from the webcam and uses
DeepFace to classify the dominant emotion, storing it to be read by the UI and Core logic.
"""

import threading
import time
import cv2
from loguru import logger
import config
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver

try:
    from deepface import DeepFace
except ImportError:
    logger.error("DeepFace is not installed. Continuous emotion detection won't work.")
    DeepFace = None


class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/video_feed':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while engine._running:
                    frame = engine.get_annotated_frame_bgr()
                    if frame is not None:
                        ret, jpeg = cv2.imencode('.jpg', frame)
                        if ret:
                            self.wfile.write(b'--FRAME\r\n')
                            self.send_header('Content-Type', 'image/jpeg')
                            self.send_header('Content-Length', len(jpeg.tobytes()))
                            self.end_headers()
                            self.wfile.write(jpeg.tobytes())
                            self.wfile.write(b'\r\n')
                    time.sleep(0.03)
            except Exception as e:
                pass
        else:
            self.send_error(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass # Suppress HTTP logs


class StreamingServer(socketserver.ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class EmotionEngine:
    def __init__(self, camera_index=config.WEBCAM_INDEX):
        self.camera_index = camera_index
        self.cap = None
        self._running = False
        self._capture_thread = None
        self._analyze_thread = None
        self._server_thread = None
        self._server = None
        self._latest_emotion = "neutral"
        self._latest_frame_bgr = None
        self._latest_frame_rgb = None
        self._lock = threading.Lock()

    def start(self):
        """Starts the background processing threads and MJPEG server."""
        if DeepFace is None:
            logger.warning("DeepFace missing. EmotionEngine not starting.")
            return

        if self._running:
            return

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logger.error("Could not open webcam for EmotionEngine.")
            return

        self._running = True
        
        # 1. Start Capture Thread (~30 FPS)
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        
        # 2. Start Analyze Thread (1 FPS)
        self._analyze_thread = threading.Thread(target=self._analyze_loop, daemon=True)
        self._analyze_thread.start()
        
        # 3. Start MJPEG Server (Port 5050)
        try:
            self._server = StreamingServer(('', 5050), StreamingHandler)
            self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._server_thread.start()
            logger.info("EmotionEngine MJPEG Server started on port 5050.")
        except Exception as e:
            logger.error(f"Could not start MJPEG server: {e}")
            
        logger.info("EmotionEngine background threads started.")

    def stop(self):
        """Stops the background threads and releases the webcam."""
        self._running = False
        
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        if self._analyze_thread:
            self._analyze_thread.join(timeout=2.0)
        if self._server_thread:
            self._server_thread.join(timeout=2.0)
            
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("EmotionEngine background threads stopped.")

    def _capture_loop(self):
        """Continuously capture frames at hardware speed (~30fps)."""
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            with self._lock:
                self._latest_frame_bgr = frame.copy()
                # Also store RGB for Streamlit UI just in case
                self._latest_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            time.sleep(0.01)

    def _analyze_loop(self):
        """Poll the latest frame once a second and run heavy DeepFace evaluation."""
        while self._running:
            frame = None
            with self._lock:
                if self._latest_frame_bgr is not None:
                    frame = self._latest_frame_bgr.copy()
            
            if frame is None:
                time.sleep(0.1)
                continue

            try:
                # Use enforce_detection=True to catch when the user looks away
                res = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=True,
                    silent=True
                )
                
                if isinstance(res, list):
                    res = res[0]
                    
                emotion = res.get('dominant_emotion', 'neutral')
                
                with self._lock:
                    self._latest_emotion = emotion

            except ValueError as ve:
                if "could not be detected" in str(ve).lower():
                    with self._lock:
                        self._latest_emotion = "no_face"
                else:
                    logger.debug(f"EmotionEngine evaluate ValueError: {ve}")
            except Exception as e:
                logger.debug(f"EmotionEngine evaluate error: {e}")

            # Sleep to preserve CPU
            time.sleep(1.0)
            
    def get_latest_emotion(self) -> str:
        """Returns the last detected dominant emotion."""
        with self._lock:
            return self._latest_emotion

    def get_latest_frame(self):
        """Returns the last captured RGB frame (for legacy stream if needed)."""
        with self._lock:
            if self._latest_frame_rgb is not None:
                return self._latest_frame_rgb.copy()
            return None
            
    def get_annotated_frame_bgr(self):
        """Returns a BGR frame with the emotion overlay drawn on it for MJPEG streaming."""
        frame = None
        emotion = "neutral"
        with self._lock:
            if self._latest_frame_bgr is not None:
                frame = self._latest_frame_bgr.copy()
            emotion = self._latest_emotion
            
        if frame is None:
            return None
            
        EMOTION_EMOJIS = {
            "happy": "HAPPY", "sad": "SAD", "angry": "ANGRY", 
            "fear": "FEAR", "surprise": "SURPRISE", "disgust": "DISGUST", "neutral": "NEUTRAL",
            "no_face": "NO FACE"
        }
        
        text = EMOTION_EMOJIS.get(emotion, emotion.upper())
        
        if emotion == "no_face":
            # Draw a dark red overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 150), -1) # BGR
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            # Add large warning text
            cv2.putText(
                frame, 
                "PLEASE FOCUS ON YOUR STUDIES", 
                (20, frame.shape[0] // 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (0, 0, 255), # Red BGR
                2
            )
        else:
            color = (0, 255, 0) if emotion in ["happy", "surprise", "neutral"] else (0, 0, 255) # Green vs Red in BGR
            cv2.putText(
                frame, 
                text, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                color, 
                2
            )
        return frame

# Provide a global instance for the app to use
engine = EmotionEngine()
