"""
setup_check.py - Vani-Vision environment verification script.

Run this before starting the app to check all dependencies and the model file.
Usage: python setup_check.py
"""

import sys
import os

RESET  = "\033[0m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"

def ok(msg):   print(f"  {GREEN}[OK]{RESET}  {msg}")
def fail(msg): print(f"  {RED}[FAIL]{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}[WARN]{RESET} {msg}")
def info(msg): print(f"  {CYAN}[INFO]{RESET} {msg}")


def check_python():
    v = sys.version_info
    if v.major == 3 and v.minor >= 10:
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        fail(f"Python 3.10+ required (found {v.major}.{v.minor}.{v.micro})")

def check_package(pkg_import, pip_name, critical=True):
    try:
        __import__(pkg_import)
        ok(f"{pip_name}")
    except ImportError:
        msg = f"{pip_name} not installed  -->  pip install {pip_name}"
        if critical:
            fail(msg)
        else:
            warn(msg + " (optional)")

def check_model():
    import config
    path = config.LLM_MODEL_PATH
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        ok(f"Model found: {path}  ({size_mb:.0f} MB)")
    else:
        warn(
            f"Model NOT found at: {path}\n"
            "         The app will run in demo mode (canned responses).\n"
            "         Download from: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf"
        )

def check_webcam():
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ok("Webcam detected (index 0)")
            cap.release()
        else:
            warn("No webcam at index 0 — upload mode will still work.")
    except Exception as e:
        warn(f"Webcam check failed: {e}")


if __name__ == "__main__":
    print(f"\n{BOLD}{CYAN}=== Vani-Vision Setup Check ==={RESET}\n")

    print(f"{BOLD}Python:{RESET}")
    check_python()

    print(f"\n{BOLD}Core dependencies:{RESET}")
    check_package("streamlit",          "streamlit")
    check_package("cv2",                "opencv-python")
    check_package("PIL",                "Pillow")
    check_package("numpy",              "numpy")
    check_package("loguru",             "loguru")

    print(f"\n{BOLD}OCR engine:{RESET}")
    check_package("paddleocr",          "paddleocr",       critical=False)
    check_package("paddle",             "paddlepaddle",    critical=False)

    print(f"\n{BOLD}LLM inference:{RESET}")
    check_package("llama_cpp",          "llama-cpp-python", critical=False)

    print(f"\n{BOLD}Translation:{RESET}")
    check_package("deep_translator",    "deep-translator", critical=False)
    check_package("langdetect",         "langdetect",      critical=False)

    print(f"\n{BOLD}Model file:{RESET}")
    check_model()

    print(f"\n{BOLD}Webcam:{RESET}")
    check_webcam()

    print(f"\n{BOLD}To launch Vani-Vision:{RESET}")
    info("streamlit run app.py")
    print()
