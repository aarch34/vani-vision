"""
ocr_engine.py – Webcam capture + EasyOCR text extraction for Vāṇī-Vision.

Flow:
  1. Capture a frame from webcam (or accept an uploaded image).
  2. Pre-process the image (grayscale, denoise, threshold).
  3. Run EasyOCR to extract text lines with confidence scores.
  4. Return cleaned text string ready for the LLM pipeline.
"""

import cv2
import numpy as np
from PIL import Image
from loguru import logger

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not installed. OCR will be simulated.")

import config

# Lazy-initialised singleton so the model loads only once
_ocr_instance = None


def _get_ocr():
    global _ocr_instance
    global EASYOCR_AVAILABLE
    if _ocr_instance is None:
        if not EASYOCR_AVAILABLE:
            logger.warning("OCR engine unavailable, skipping init.")
            return None
            
        logger.info("Initialising EasyOCR parameters (CPU mode)…")
        try:
            # Download models on first run, use CPU
            _ocr_instance = easyocr.Reader(['en'], gpu=False)
            logger.info("EasyOCR Engine ready.")
        except Exception as e:
            logger.error(f"EasyOCR Pipeline crashed upstream: {e}")
            EASYOCR_AVAILABLE = False
            _ocr_instance = None
    return _ocr_instance


# ─── Image Pre-processing ────────────────────────────────────────────────────

def preprocess_image(img_array: np.ndarray) -> np.ndarray:
    """Apply standard pre-processing to improve OCR accuracy."""
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    # Adaptive threshold works well for handwriting on white paper
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    # Convert back to 3-channel for EasyOCR
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


# ─── Webcam Capture ───────────────────────────────────────────────────────────

def capture_frame(camera_index: int = config.WEBCAM_INDEX) -> np.ndarray | None:
    """
    Open the webcam, grab a single frame, and release the camera.
    Returns an RGB numpy array or None if the camera is unavailable.
    """
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.WEBCAM_FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_FRAME_H)

    if not cap.isOpened():
        logger.error(f"Cannot open webcam at index {camera_index}")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        logger.error("Failed to read frame from webcam.")
        return None

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ─── PIL / Uploaded Image Conversion ─────────────────────────────────────────

def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL image to a BGR numpy array for OpenCV."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ─── Core OCR Function ────────────────────────────────────────────────────────

def extract_text(image: np.ndarray | Image.Image, preprocess: bool = True) -> str:
    """
    Extract text from an image (numpy array or PIL Image).

    Args:
        image:      BGR numpy array OR PIL Image.
        preprocess: Whether to apply adaptive threshold pre-processing.

    Returns:
        Concatenated text string from all detected text boxes.
    """
    if isinstance(image, Image.Image):
        img_bgr = pil_to_cv2(image)
    else:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if preprocess:
        img_bgr = preprocess_image(img_bgr)

    if not EASYOCR_AVAILABLE:
        logger.warning("Simulating OCR – returning placeholder text.")
        return "[OCR Placeholder] If a body has mass m=5 kg and acceleration a=2 m/s², what is the force?"

    ocr = _get_ocr()
    if ocr is None:
        return "[OCR Placeholder] If a body has mass m=5 kg and acceleration a=2 m/s², what is the force?"

    logger.debug("Running EasyOCR prediction…")
    try:
        # EasyOCR returns a list of tuples: (bounding_box, text, confidence)
        results = ocr.readtext(img_bgr)
        lines = []
        for bbox, text, conf in results:
            if conf > 0.4:  # filter low-confidence noise
                lines.append(text.strip())
    
        extracted = " ".join(lines).strip()
        logger.info(f"OCR extracted ({len(extracted)} chars): {extracted[:120]}…")
        return extracted if extracted else ""
    except Exception as e:
        logger.error(f"Upstream OCR crash: {e}")
        return "[OCR Placeholder] If a body has mass m=5 kg and acceleration a=2 m/s², what is the force?"


# ─── Utility: Draw Bounding Boxes ─────────────────────────────────────────────

def draw_bounding_boxes(image: np.ndarray) -> np.ndarray:
    """
    Run OCR and return a copy of the image with detected text regions
    highlighted — useful for the Streamlit preview panel.
    """
    if isinstance(image, Image.Image):
        img_bgr = pil_to_cv2(image)
    else:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if not EASYOCR_AVAILABLE:
        return image if isinstance(image, np.ndarray) else np.array(image)

    ocr = _get_ocr()
    if ocr is None:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
    annotated = img_bgr.copy()
    
    try:
        results = ocr.readtext(img_bgr)
        for bbox, text, conf in results:
            if conf > 0.4:
                # EasyOCR bbox is a list of 4 points: [tl, tr, br, bl]
                box = np.array(bbox, dtype=np.int32)
                cv2.polylines(annotated, [box], True, (0, 255, 128), 2)
                
                # Use top-left point for text placement
                tl = tuple(box[0])
                cv2.putText(
                    annotated, f"{conf:.0%}",
                    tl, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 128), 1
                )
    except Exception as e:
        logger.error(f"Upstream OCR bounding-box crash: {e}")

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
