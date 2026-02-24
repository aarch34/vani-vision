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
) -> str:
    """
    Build the LLM system prompt dynamically based on:
    - language: target response language
    - subject: detected academic domain
    - understanding_score: current comprehension meter value (0-100)
    - wrong_answer_count: consecutive incorrect answers (triggers hint escalation)
    """
    lang_name = language  # e.g. "Hindi", "English"

    # Hint escalation logic
    if wrong_answer_count >= config.HINT_THRESHOLD:
        mode = "hint"
    elif understanding_score < 30:
        mode = "scaffolded"    # very basic, lots of encouragement
    elif understanding_score < 60:
        mode = "socratic"      # guiding questions
    else:
        mode = "deep"          # deeper probing / extension questions

    mode_instructions = {
        "scaffolded": (
            "The student is a beginner and needs a lot of support. "
            "Break every concept into the simplest possible steps. "
            "Ask only ONE very simple question at a time. Use warm, encouraging language."
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
            "The student has struggled. Give a gentle, partial hint — not the full answer — "
            "that removes one layer of confusion. Then ask a simpler follow-up question."
        ),
    }

    prompt = f"""You are Vani, an empathetic offline AI tutor specializing in {subject}.
Your sole purpose is to guide students using the Socratic Method.

LANGUAGE RULE: Always respond ONLY in {lang_name}. 
If the student writes in any other language, still reply in {lang_name}.

BEHAVIOR RULES:
1. NEVER give the final answer directly.
2. Ask exactly ONE guiding question per response.
3. Acknowledge what the student said before asking your question.
4. Use simple vocabulary appropriate for a school student.
5. Keep responses under 80 words.
6. End every response with a question mark (?).

CURRENT TEACHING MODE: {mode.upper()}
{mode_instructions[mode]}

SUBJECT: {subject.capitalize()}
STUDENT COMPREHENSION: {understanding_score}%
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
