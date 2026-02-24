"""
multilingual.py - Language selection and translation utilities for Vani-Vision.

Supports: English, Hindi, Kannada, Tamil, Telugu.

Strategy:
  1. The LLM is prompted to reply directly in the chosen language when possible
     (Phi-3 Mini has decent multilingual capability).
  2. As a fallback, deep-translator (GoogleTranslator - works offline via cache
     or with minimal bandwidth) is used to post-translate English responses.
  3. A curated phrase dictionary for common UI labels ensures the interface
     itself is fully localised without any network call.
"""

import config
from loguru import logger

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False
    logger.warning("deep-translator not installed. Translation fallback disabled.")


# ─── UI Phrase Dictionary ─────────────────────────────────────────────────────
# Maps (phrase_key, lang_code) -> localised string
#
# lang codes: en, hi, kn, ta, te

UI_PHRASES = {
    # ── Buttons ──────────────────────────────────────────────────────────────
    ("capture_btn",    "en"): "Capture from Camera",
    ("capture_btn",    "hi"): "कैमरे से कैप्चर करें",
    ("capture_btn",    "kn"): "ಕ್ಯಾಮರಾದಿಂದ ಸೆರೆಹಿಡಿಯಿರಿ",
    ("capture_btn",    "ta"): "கேமராவில் இருந்து பிடிக்கவும்",
    ("capture_btn",    "te"): "కెమేరా నుండి క్యాప్చర్ చేయండి",

    ("submit_btn",     "en"): "Submit Answer",
    ("submit_btn",     "hi"): "उत्तर जमा करें",
    ("submit_btn",     "kn"): "ಉತ್ತರ ಸಲ್ಲಿಸಿ",
    ("submit_btn",     "ta"): "பதிலை சமர்ப்பிக்கவும்",
    ("submit_btn",     "te"): "సమాధానం సమర్పించండి",

    ("new_session",    "en"): "New Session",
    ("new_session",    "hi"): "नया सत्र",
    ("new_session",    "kn"): "ಹೊಸ ಅಧಿವೇಶನ",
    ("new_session",    "ta"): "புதிய அமர்வு",
    ("new_session",    "te"): "కొత్త సెషన్",

    # ── Labels ───────────────────────────────────────────────────────────────
    ("extracted_text", "en"): "Extracted Question",
    ("extracted_text", "hi"): "निकाला गया प्रश्न",
    ("extracted_text", "kn"): "ಹೊರತೆಗೆದ ಪ್ರಶ್ನೆ",
    ("extracted_text", "ta"): "பிரித்தெடுக்கப்பட்ட கேள்வி",
    ("extracted_text", "te"): "సంగ్రహించిన ప్రశ్న",

    ("understanding",  "en"): "Comprehension Meter",
    ("understanding",  "hi"): "समझ मीटर",
    ("understanding",  "kn"): "ಅರ್ಥಗ್ರಹಣ ಮೀಟರ್",
    ("understanding",  "ta"): "புரிதல் மீட்டர்",
    ("understanding",  "te"): "అర్థం మీటర్",

    ("your_answer",    "en"): "Your Answer",
    ("your_answer",    "hi"): "आपका उत्तर",
    ("your_answer",    "kn"): "ನಿಮ್ಮ ಉತ್ತರ",
    ("your_answer",    "ta"): "உங்கள் பதில்",
    ("your_answer",    "te"): "మీ సమాధానం",

    ("tutor_says",     "en"): "Vani says",
    ("tutor_says",     "hi"): "वाणी कहती है",
    ("tutor_says",     "kn"): "ವಾಣಿ ಹೇಳುತ್ತಾಳೆ",
    ("tutor_says",     "ta"): "வாணி சொல்கிறார்",
    ("tutor_says",     "te"): "వాణి చెప్తోంది",

    ("upload_image",   "en"): "Upload an Image",
    ("upload_image",   "hi"): "एक छवि अपलोड करें",
    ("upload_image",   "kn"): "ಚಿತ್ರವನ್ನು ಅಪ್ಲೋಡ್ ಮಾಡಿ",
    ("upload_image",   "ta"): "படத்தை பதிவேற்றவும்",
    ("upload_image",   "te"): "చిత్రాన్ని అప్లోడ్ చేయండి",
}


def get_phrase(key: str, lang_code: str) -> str:
    """
    Return a localised UI phrase. Falls back to English if the key is missing
    for the requested language.
    """
    return UI_PHRASES.get((key, lang_code)) or UI_PHRASES.get((key, "en"), key)


# ─── Language Code Lookup ─────────────────────────────────────────────────────

def get_lang_code(lang_display_name: str) -> str:
    """Map a display name (from config.SUPPORTED_LANGUAGES keys) to ISO code."""
    return config.SUPPORTED_LANGUAGES.get(lang_display_name, "en")


# ─── Translation Fallback ─────────────────────────────────────────────────────

def translate_to(text: str, target_lang_code: str) -> str:
    """
    Translate text to the target language.

    Uses deep-translator's GoogleTranslator. When offline, it relies on
    the OS/browser-level DNS cache; for truly air-gapped deployments,
    this function is a no-op and returns the original text unchanged.
    """
    if target_lang_code == "en":
        return text

    if not TRANSLATOR_AVAILABLE:
        logger.debug("Translation skipped (deep-translator not installed).")
        return text

    try:
        translated = GoogleTranslator(
            source="auto", target=target_lang_code
        ).translate(text)
        return translated or text
    except Exception as exc:
        logger.warning(f"Translation failed ({exc}). Returning original text.")
        return text


# ─── Language Prompt Suffix ───────────────────────────────────────────────────

LANGUAGE_PROMPT_SUFFIXES = {
    "en": "Please respond in English.",
    "hi": "Kripaya Hindi mein jawab dein. (Please respond in Hindi.)",
    "kn": "Dayavittu Kannada nalli uttara nidi. (Please respond in Kannada.)",
    "ta": "Thayavu seithu Tamil-il padhil kodunga. (Please respond in Tamil.)",
    "te": "Dayachesi Telugu lo spaandhinchandi. (Please respond in Telugu.)",
}


def get_language_suffix(lang_code: str) -> str:
    """Return the language reminder string to append to Socratic prompts."""
    return LANGUAGE_PROMPT_SUFFIXES.get(lang_code, LANGUAGE_PROMPT_SUFFIXES["en"])
