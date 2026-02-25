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
            "Explain the core concept they are stuck on step-by-step, breaking it into the simplest possible pieces. "
            "Use warm, highly encouraging language."
        ),
        "socratic": (
            "Ask guiding questions to help the student reach the answer themselves. "
            "Never give the direct answer. Encourage them to think critically by asking one question at a time."
        ),
        "deep": (
            "The student shows good understanding. Provide in-depth explanations exploring edge cases "
            "and real-world applications to expand their knowledge."
        ),
        "hint": (
            "The student has struggled or looks frustrated. "
            "Gently clarify their confusion with a very simple, direct explanation of the specific part they are stuck on."
        ),
    }

    prompt = f"""You are Vani, an empathetic offline AI tutor specializing in {subject}.
Your sole purpose is to guide students to understand concepts.

LANGUAGE RULE: Always respond ONLY in {lang_name}. 
If the student writes in any other language, still reply in {lang_name}.

BEHAVIOR RULES:
1. If the student provides a full question paper or multiple questions, FIRST ask the student which specific question they have a doubt in. Do not solve anything until they specify.
2. If the student says they don't know anything, briefly explain the core concepts of the topic first, and then ask a guiding question.
3. NEVER give the direct answer to the student's question.
4. Ask one clear, guiding Socratic question at a time to lead them to the solution.
5. Acknowledge what the student said and validate their effort.
6. Use simple vocabulary appropriate for a school student.
7. Keep your responses concise (1-2 short paragraphs max) to encourage dialogue.

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
        f"I have this document or question paper from my textbook or homework:\n\n"
        f'"{ocr_text}"\n\n'
        f"I need help with this in {language}. How should we start?"
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
