"""
llm_engine.py - Ollama inference engine for Vani-Vision.

Uses the local Ollama service for 100% offline inference.
Make sure Ollama is installed and running, and the 'phi3' model is pulled:
    ollama pull phi3
"""

import os
import requests
from loguru import logger

import config

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = getattr(config, "OLLAMA_MODEL", "phi3")

def _check_ollama():
    try:
        response = requests.get("http://localhost:11434/", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def build_messages(system_message: str, history: list, user_message: str) -> list:
    """
    Format the conversation history into Ollama's expected message structure.
    history: list of {"role": "user"|"assistant", "content": "..."} dicts.
    """
    messages = [{"role": "system", "content": system_message}]
    
    # Add history
    for turn in history:
        messages.append({
            "role": turn.get("role", "user"),
            "content": turn.get("content", "")
        })
        
    # Add current user message
    messages.append({"role": "user", "content": user_message})
    
    return messages


def generate(system_message: str, history: list, user_message: str) -> str:
    """
    Generate an LLM response via Ollama. Falls back to canned demo responses 
    if Ollama is not running.

    Args:
        system_message: The Socratic persona / language instructions.
        history:        List of previous {"role", "content"} turns.
        user_message:   The latest student input or OCR-extracted question.

    Returns:
        The assistant reply as a plain string.
    """
    if not _check_ollama():
        logger.warning("Ollama is not responding at localhost:11434. Falling back to demo mode.")
        return _demo_response(user_message)

    messages = build_messages(system_message, history, user_message)

    logger.debug(f"Sending prompt to Ollama ({OLLAMA_MODEL})…")
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": config.LLM_TEMPERATURE,
            "num_predict": config.LLM_MAX_TOKENS,
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        reply = response.json().get("message", {}).get("content", "").strip()
        
        logger.debug(f"Ollama reply ({len(reply)} chars): {reply[:100]}...")
        return reply
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API error: {e}")
        return _demo_response(user_message)


# ─── Demo-Mode Stub Responses ─────────────────────────────────────────────────

_DEMO_RESPONSES = [
    "That is a great question! Before I give you the answer, let me ask: "
    "What does Newton's Second Law of Motion tell us about the relationship "
    "between force, mass, and acceleration?",

    "Good thinking! Now, if F = m x a, and we know the mass of the object, "
    "what value would you substitute for 'm' in the formula?",

    "Excellent! You have substituted the mass correctly. "
    "Now look at the acceleration value given in the problem — "
    "can you plug it in and calculate the result?",

    "You are almost there! Check your arithmetic once more. "
    "What unit does force have, and does your answer carry that unit?",

    "Well done! You solved it. Can you now explain in your own words "
    "why a heavier object needs more force to achieve the same acceleration?",
]

_demo_index = 0


def _demo_response(user_message: str) -> str:
    """Cycle through canned Socratic responses for demo / no-model mode."""
    global _demo_index
    response = _DEMO_RESPONSES[_demo_index % len(_DEMO_RESPONSES)]
    _demo_index += 1
    return response


def reset_demo():
    """Reset the demo response counter (call at start of each new session)."""
    global _demo_index
    _demo_index = 0
