"""Fast structural pass: primary emotion, cognitive distortion, core internal conflict."""
import logging
import re

from app.core.ollama_client import OllamaClient
from app.schemas.emotional_state import EmotionalState

logger = logging.getLogger(__name__)

# Deterministic, model-free safety net. Runs on every message, costs microseconds, and can't be
# talked out of firing by a small local model.
_CRISIS_PATTERNS = [
    r"\bkill (myself|me)\b",
    r"\bend (my|it all|my own) life\b",
    r"\bsuicid",
    r"\bwant(ed)? to die\b",
    r"\bwish i (was|were) dead\b",
    r"\bdon'?t want to (live|be alive|exist)\b",
    r"\bno reason to (live|go on)\b",
    r"\b(hurt|harm|cut) myself\b",
    r"\bself[- ]?harm",
    r"\bbetter off without me\b",
    r"\btake my own life\b",
]
_CRISIS_RE = re.compile("|".join(_CRISIS_PATTERNS), re.IGNORECASE)


def detect_crisis_keywords(text: str) -> bool:
    return bool(_CRISIS_RE.search(text))


_SYSTEM = """You are a careful annotator for a counseling app. Read ONE message from a person \
describing a conflict (inner or with someone else) and fill in the JSON fields.

- primary_emotion: the dominant feeling in one or two words (e.g. anxiety, guilt, shame, resentment, sadness, anger, overwhelm).
- cognitive_distortion: the single best-fitting CBT distortion, or "none" if the thinking looks balanced. Do not force one.
- core_conflict: the underlying tension in a few words, e.g. "autonomy vs. security", "honesty vs. keeping the peace". Empty string if unclear.
- intensity: 1 (calm) to 5 (overwhelming).
- crisis_risk: true only if the person expresses a wish or intent to harm themselves or end their life.

Return only the JSON object."""


class EmotionAnalyzer:
    def __init__(self, llm: OllamaClient):
        self.llm = llm
        self._schema = EmotionalState.model_json_schema()

    async def analyze(self, message: str) -> EmotionalState:
        """Never raises: on any failure it degrades to a neutral state (plus the keyword check)."""
        crisis = detect_crisis_keywords(message)
        try:
            raw = await self.llm.complete(
                [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": message}],
                json_schema=self._schema,
                temperature=0.0,
                num_predict=160,
            )
            state = EmotionalState.model_validate_json(raw)
        except Exception as exc:
            logger.warning("Emotion analysis unavailable: %s", exc)
            state = EmotionalState()
        if crisis:
            state = state.model_copy(update={"crisis_risk": True})
        return state
