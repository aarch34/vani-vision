"""
understanding_meter.py - Student comprehension scoring engine for Vani-Vision.

Tracks comprehension score (0-100) across a tutoring session using:
  1. Rule-based heuristics for fast, offline scoring.
  2. Optional LLM-based evaluation for nuanced answers (uses llm_engine).

The score drives the Socratic escalation mode in socratic_prompt.py.
"""

import re
import json
from loguru import logger
import config


# ─── Session State ────────────────────────────────────────────────────────────

class UnderstandingMeter:
    """Stateful comprehension tracker for one tutoring session."""

    def __init__(self):
        self.score: int         = config.METER_INITIAL_SCORE
        self.turn_count: int    = 0
        self.wrong_streak: int  = 0
        self.history: list      = []   # [{"turn": n, "score": s, "verdict": v}]

    # ── Heuristic Scoring ────────────────────────────────────────────────────

    POSITIVE_SIGNALS = [
        r"\b(yes|correct|exactly|right|understand|got it|i see|makes sense)\b",
        r"\b(the formula|newton|f\s*=\s*ma|mass|acceleration|force)\b",
        r"\b(substitute|plug in|calculate|equals|result)\b",
        r"\d+\s*(m/s|kg|n|j|w|pa|k)",   # numeric answer with unit
    ]
    NEGATIVE_SIGNALS = [
        r"\b(don'?t know|not sure|confused|no idea|don'?t understand|i give up)\b",
        r"\b(what|why|how)\?$",
        r"^\s*\?+\s*$",
    ]

    def _heuristic_score(self, student_reply: str) -> tuple[str, int]:
        """
        Returns (verdict, delta) where:
          verdict: "correct" | "partial" | "incorrect"
          delta:   score change value
        """
        text = student_reply.lower()

        pos_hits = sum(
            1 for pattern in self.POSITIVE_SIGNALS
            if re.search(pattern, text)
        )
        neg_hits = sum(
            1 for pattern in self.NEGATIVE_SIGNALS
            if re.search(pattern, text)
        )

        if pos_hits >= 2 and neg_hits == 0:
            return "correct", config.METER_STEP_CORRECT
        elif pos_hits >= 1 or neg_hits == 0:
            return "partial", config.METER_STEP_PARTIAL
        else:
            return "incorrect", config.METER_STEP_INCORRECT

    # ── LLM-based Scoring (optional, async-safe) ─────────────────────────────

    def _llm_score(self, student_reply: str, concept: str) -> tuple[str, int]:
        """
        Ask the LLM to evaluate the student's understanding.
        Returns (verdict, delta). Falls back to heuristic on failure.
        """
        try:
            import llm_engine
            import socratic_prompt

            eval_prompt = socratic_prompt.build_evaluation_prompt(
                student_reply, concept
            )
            raw = llm_engine.generate(
                system_message=(
                    "You are a strict but fair exam evaluator. "
                    "Respond ONLY with valid JSON in the format: "
                    '{"score": <0-100>, "feedback": "<one sentence>"}'
                ),
                history=[],
                user_message=eval_prompt,
            )
            data = json.loads(raw.strip())
            score_val = int(data.get("score", 50))

            if score_val >= 70:
                return "correct", config.METER_STEP_CORRECT
            elif score_val >= 40:
                return "partial", config.METER_STEP_PARTIAL
            else:
                return "incorrect", config.METER_STEP_INCORRECT

        except Exception as exc:
            logger.warning(f"LLM eval failed ({exc}), using heuristic.")
            return self._heuristic_score(student_reply)

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        student_reply: str,
        concept: str = "the topic",
        use_llm: bool = False,
    ) -> dict:
        """
        Update the understanding score based on the student's reply.

        Args:
            student_reply: Raw text of the student's answer.
            concept:       Topic / concept being evaluated (for LLM mode).
            use_llm:       If True, use the LLM for nuanced scoring.

        Returns:
            {"score": int, "delta": int, "verdict": str, "badge": str}
        """
        self.turn_count += 1

        if use_llm:
            verdict, delta = self._llm_score(student_reply, concept)
        else:
            verdict, delta = self._heuristic_score(student_reply)

        # Update streak
        if verdict == "incorrect":
            self.wrong_streak += 1
        else:
            self.wrong_streak = 0

        # Clamp score to [0, 100]
        self.score = max(0, min(100, self.score + delta))

        badge = self._badge(self.score)
        record = {
            "turn":    self.turn_count,
            "score":   self.score,
            "delta":   delta,
            "verdict": verdict,
            "badge":   badge,
        }
        self.history.append(record)
        logger.info(f"Meter update: {verdict}  delta={delta:+d}  score={self.score}  badge={badge}")
        return record

    def reset(self):
        """Reset for a new tutoring session."""
        self.score       = config.METER_INITIAL_SCORE
        self.turn_count  = 0
        self.wrong_streak = 0
        self.history     = []

    @staticmethod
    def _badge(score: int) -> str:
        if score >= 85:
            return "Expert"
        elif score >= 65:
            return "Proficient"
        elif score >= 45:
            return "Developing"
        elif score >= 25:
            return "Beginner"
        else:
            return "Needs Help"

    @property
    def color(self) -> str:
        """Return a hex color matching the current score band for the UI gauge."""
        if self.score >= 75:
            return config.THEME_SUCCESS
        elif self.score >= 45:
            return config.THEME_WARNING
        else:
            return config.THEME_DANGER

    @property
    def emoji(self) -> str:
        if self.score >= 85:
            return "🌟"
        elif self.score >= 65:
            return "✅"
        elif self.score >= 45:
            return "📚"
        elif self.score >= 25:
            return "🌱"
        else:
            return "🆘"
