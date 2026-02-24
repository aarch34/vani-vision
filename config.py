"""
config.py – Central configuration for Vāṇī-Vision.
All tuneable constants live here so that no other file contains magic values.
"""

# ─── App Meta ────────────────────────────────────────────────────────────────
APP_NAME    = "Vāṇī-Vision"
APP_VERSION = "1.0.0"
TAGLINE     = "Empowering Every Learner with an Offline AI Socratic Tutor in Their Own Language"

# ─── Language Support ────────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "English": "en",
    "Hindi (हिन्दी)": "hi",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Tamil (தமிழ்)": "ta",
    "Telugu (తెలుగు)": "te",
}
DEFAULT_LANGUAGE = "English"

# ─── LLM Settings ────────────────────────────────────────────────────────────
# Download Phi-3 Mini GGUF from HuggingFace and place in models/
LLM_MODEL_PATH   = "models/phi-3-mini-4k-instruct-q4.gguf"
LLM_CONTEXT_LEN  = 4096
LLM_MAX_TOKENS   = 512
LLM_TEMPERATURE  = 0.7
LLM_N_THREADS    = 4   # tune to CPU core count
LLM_N_GPU_LAYERS = 0   # 0 = pure CPU

# ─── OCR Settings ─────────────────────────────────────────────────────────────
OCR_LANGUAGE = "en"        # PaddleOCR language code
OCR_USE_GPU  = False       # Must remain False for offline CPU-only target
OCR_ANGLE_CLASSIFICATION = True

# ─── Webcam ───────────────────────────────────────────────────────────────────
WEBCAM_INDEX      = 0
WEBCAM_FRAME_W    = 1280
WEBCAM_FRAME_H    = 720
CAPTURE_COUNTDOWN = 3      # seconds countdown before snap

# ─── Understanding Meter ─────────────────────────────────────────────────────
METER_INITIAL_SCORE    = 30   # starting comprehension %
METER_STEP_CORRECT     = 15   # added when student answers correctly
METER_STEP_PARTIAL     = 7    # added for partial understanding
METER_STEP_INCORRECT   = -5   # deducted for wrong answers

# ─── Socratic Engine ─────────────────────────────────────────────────────────
MAX_SOCRATIC_TURNS = 8        # max back-and-forth turns before a hint
HINT_THRESHOLD     = 2        # wrong answers before giving a direct hint

# ─── UI Theming ──────────────────────────────────────────────────────────────
THEME_PRIMARY   = "#6C63FF"   # indigo-violet
THEME_SECONDARY = "#F5A623"   # warm amber
THEME_BG        = "#0D0F1A"   # near-black navy
THEME_SURFACE   = "#161929"
THEME_TEXT      = "#E8ECF4"
THEME_SUCCESS   = "#2ECC71"
THEME_WARNING   = "#F39C12"
THEME_DANGER    = "#E74C3C"
