"""
socratic_prompt.py - Socratic Method prompt engineering for Vani-Vision.

Generates the system prompt that shapes the AI into a Socratic tutor and
manages the escalation logic (from guiding questions -> hints -> near-answer).
"""

import config

# ─── Subject Detection Keywords ───────────────────────────────────────────────

SUBJECT_KEYWORDS = {
    "mathematics": [
        "equation", "solve", "calculate", "integral", "derivative",
        "algebra", "geometry", "trigonometry", "area", "volume",
        "sum", "product", "fraction", "percentage", "ratio",
    ],
    "physics": [
        "force", "velocity", "acceleration", "mass", "gravity", "newton",
        "energy", "work", "power", "momentum", "wave", "frequency",
        "current", "voltage", "resistance", "ohm",
    ],
    "chemistry": [
        "atom", "molecule", "bond", "reaction", "element", "compound",
        "acid", "base", "mole", "periodic", "electron", "proton",
    ],
    "biology": [
        "cell", "organism", "dna", "gene", "photosynthesis", "respiration",
        "ecosystem", "evolution", "mitosis", "meiosis",
    ],
}


def detect_subject(text: str) -> str:
    """Detect the likely academic subject from extracted question text."""
    text_lower = text.lower()
    scores = {subject: 0 for subject in SUBJECT_KEYWORDS}
    for subject, keywords in SUBJECT_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[subject] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# ─── System Prompt Generator ──────────────────────────────────────────────────

def build_system_prompt(
    language: str,
    subject: str,
    understanding_score: int,
    wrong_answer_count: int,
    emotion: str = "neutral",
) -> str:
    """
    Build the LLM system prompt dynamically based on:
    - language: target response language
    - subject: detected academic domain
    - understanding_score: current comprehension meter value (0-100)
    - wrong_answer_count: consecutive incorrect answers (triggers hint escalation)
    - emotion: detected facial expression of the student
    """
    lang_name = language  # e.g. "Hindi", "English"

    # Hint escalation logic
    if wrong_answer_count >= config.HINT_THRESHOLD or emotion in ["sad", "angry", "fear"]:
        mode = "hint"
    elif understanding_score < 30:
        mode = "scaffolded"    # very basic, lots of encouragement
    elif understanding_score < 60:
        mode = "socratic"      # guiding questions
    else:
        mode = "deep"          # deeper probing / extension questions

    mode_instructions = {
        "scaffolded": (
            "The student is a beginner and is struggling. "
            "IMPORTANT: If they are confused, first provide a brief, simple explanation "
            "of the core concept they are stuck on. Break the concept into the simplest possible steps. "
            "After explaining, ask ONE very simple follow-up question. Use warm, highly encouraging language."
        ),
        "socratic": (
            "Use the Socratic Method strictly: NEVER give direct answers. "
            "Ask one guiding question that nudges the student toward the next logical step. "
            "Reference what they already know before asking the next question."
        ),
        "deep": (
            "The student shows good understanding. Challenge them with probing questions "
            "that test deeper reasoning, edge cases, or real-world applications. "
            "Still do NOT reveal final answers directly."
        ),
        "hint": (
            "The student has struggled or looks frustrated. "
            "IMPORTANT: Provide a brief, clear explanation to gently clarify their confusion, "
            "but do not solve the entire problem. Once they understand that small piece, "
            "ask a simpler follow-up question to ensure they got it."
        ),
    }

    prompt = f"""You are Vani, an empathetic offline AI tutor specializing in {subject}.
Your sole purpose is to guide students to understand concepts.

LANGUAGE RULE: Always respond ONLY in {lang_name}. 
If the student writes in any other language, still reply in {lang_name}.

BEHAVIOR RULES:
1. If the student is struggling or confused (in hint/scaffolded mode), provide a brief explanation FIRST, then ask a simple follow-up question.
2. In regular Socratic mode, never give the final answer directly.
3. CRITICAL: Ask EXACTLY ONE single guiding/follow-up question per response. DO NOT ask multiple questions in a row.
4. Acknowledge what the student said before providing explanations or asking your question.
5. Use simple vocabulary appropriate for a school student.
6. CRITICAL: Keep your responses extremely short, UNDER 50 words. Do not write long paragraphs.
7. End every response with a single question mark (?).

CURRENT TEACHING MODE: {mode.upper()}
{mode_instructions[mode]}

SUBJECT: {subject.capitalize()}
STUDENT COMPREHENSION: {understanding_score}%
STUDENT EMOTION: {emotion} (Adjust your tone accordingly)
"""
    return prompt


# ─── First-Turn Prompt (Problem Introduction) ─────────────────────────────────

def build_intro_message(ocr_text: str, language: str) -> str:
    """
    Generate the initial user message that presents the captured question to the AI.
    """
    return (
        f"I have this problem from my textbook or homework:\n\n"
        f'"{ocr_text}"\n\n'
        f"Please help me understand it step by step in {language}."
    )


# ─── Understanding Score Evaluator Prompt ─────────────────────────────────────

def build_evaluation_prompt(student_reply: str, expected_concept: str) -> str:
    """
    Build a short evaluation prompt that asks the LLM to score student understanding
    on a scale of 0-100.  Used internally by understanding_meter.py.
    """
    return (
        f"Evaluate the student's response below for conceptual correctness "
        f"regarding '{expected_concept}'. "
        f"Reply with ONLY a JSON object: {{\"score\": <0-100>, \"feedback\": \"<one sentence>\"}}.\n\n"
        f"Student response: \"{student_reply}\""
    )
