"""
llm_engine.py - Phi-3 Mini (quantized GGUF) inference engine for Vani-Vision.

Uses llama-cpp-python for 100% offline CPU-based inference.
Place the model at:  models/phi-3-mini-4k-instruct-q4.gguf
Download from: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
"""

import os
from loguru import logger

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    logger.warning("llama-cpp-python not installed. Running in demo-stub mode.")

import config

_llm_instance = None

# ChatML tokens built via concatenation to avoid XML parser conflicts
_SYS   = "<|" + "system|>"
_USER  = "<|" + "user|>"
_ASST  = "<|" + "assistant|>"
_END   = "<|" + "end|>"


def _get_llm():
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    if not LLAMA_AVAILABLE:
        raise RuntimeError(
            "llama-cpp-python is not installed.\n"
            "Run: pip install llama-cpp-python"
        )

    model_path = config.LLM_MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            "Download Phi-3 Mini Q4 GGUF from:\n"
            "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf\n"
            f"and place it at: {os.path.abspath(model_path)}"
        )

    logger.info(f"Loading LLM: {model_path}  (threads={config.LLM_N_THREADS})")
    _llm_instance = Llama(
        model_path=model_path,
        n_ctx=config.LLM_CONTEXT_LEN,
        n_threads=config.LLM_N_THREADS,
        n_gpu_layers=config.LLM_N_GPU_LAYERS,
        verbose=False,
    )
    logger.info("LLM ready.")
    return _llm_instance


def build_prompt(system_message: str, history: list, user_message: str) -> str:
    """
    Build a Phi-3 ChatML prompt string from system prompt + conversation history.
    history: list of {"role": "user"|"assistant", "content": "..."} dicts.
    """
    role_map = {"user": _USER, "assistant": _ASST}
    parts = [f"{_SYS}\n{system_message}{_END}"]
    for turn in history:
        tag = role_map.get(turn["role"], _USER)
        parts.append(f"{tag}\n{turn['content']}{_END}")
    parts.append(f"{_USER}\n{user_message}{_END}")
    parts.append(_ASST)          # model continues from here
    return "\n".join(parts)


def generate(system_message: str, history: list, user_message: str) -> str:
    """
    Generate an LLM response. Falls back to canned demo responses when the
    real model is unavailable (for demo / CI purposes).

    Args:
        system_message: The Socratic persona / language instructions.
        history:        List of previous {"role", "content"} turns.
        user_message:   The latest student input or OCR-extracted question.

    Returns:
        The assistant reply as a plain string.
    """
    if not LLAMA_AVAILABLE or not os.path.exists(config.LLM_MODEL_PATH):
        return _demo_response(user_message)

    prompt = build_prompt(system_message, history, user_message)
    llm    = _get_llm()

    logger.debug("Sending prompt to LLM…")
    output = llm(
        prompt,
        max_tokens=config.LLM_MAX_TOKENS,
        temperature=config.LLM_TEMPERATURE,
        stop=[_END, _USER, _SYS],
        echo=False,
    )
    reply = output["choices"][0]["text"].strip()
    logger.debug(f"LLM reply ({len(reply)} chars): {reply[:100]}")
    return reply


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
