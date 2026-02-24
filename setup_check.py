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
    import requests
    
    ollama_url = "http://localhost:11434/api/tags"
    model_name = getattr(config, "OLLAMA_MODEL", "phi3")
    
    try:
        response = requests.get(ollama_url, timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_tags = [m.get("name") for m in models]
            
            # Simple check: 'phi3' or 'phi3:latest'
            found = False
            for tag in model_tags:
                if tag.startswith(model_name):
                    found = True
                    break
                    
            if found:
                ok(f"Ollama running & model '{model_name}' found.")
            else:
                warn(
                    f"Ollama running, but model '{model_name}' NOT found.\n"
                    f"         Run: ollama pull {model_name}"
                )
        else:
            warn(f"Ollama returned HTTP {response.status_code}")
    except requests.exceptions.RequestException:
        warn(
            f"Ollama is NOT running or not installed at localhost:11434.\n"
            "         The app will run in demo mode (canned responses).\n"
            "         Install from: https://ollama.com/"
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
    check_package("easyocr",            "easyocr",         critical=False)

    print(f"\n{BOLD}LLM inference (Ollama):{RESET}")
    check_package("requests",           "requests",        critical=True)

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
