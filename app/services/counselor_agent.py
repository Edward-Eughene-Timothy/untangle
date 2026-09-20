"""Offline counselor engine: prompt synthesis + live SSE token streaming from local Ollama."""
import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator

from app.core.config import Settings
from app.core.ollama_client import OllamaClient, OllamaError
from app.schemas.emotional_state import EmotionalState
from app.services.emotion_analyzer import EmotionAnalyzer, detect_crisis_keywords
from app.services.memory_manager import MemoryManager
from app.services.rag_engine import RagEngine, RetrievedChunk

logger = logging.getLogger(__name__)

PERSONA = """You are a warm, grounded counseling companion running privately on the user's own device. \
You help people work through inner conflicts and conflicts with others. You are an AI language model, \
not a therapist: never claim human experiences or feelings of your own, and if asked how you work, say \
plainly that you are an AI running locally on their device and that any delay comes from that hardware.

Every reply has three parts, in plain conversational language (no headings):
1. Reflect: one or two sentences naming what they feel and, if you can, the need underneath it.
2. Contribute something useful: a reframe, an observation about a pattern (offered tentatively, in varied wording), \
a short piece of psychology that normalizes what they feel, or a practical idea. Never reply with \
questions only.
3. Ask at most one question, and only if it moves things forward.
Vary your wording from reply to reply. Don't lean on stock openers such as "It sounds like", "I hear that" \
or "I wonder".

Ground yourself in CBT (examine thoughts and the evidence for them), Nonviolent Communication \
(observations, feelings, needs, requests) and IFS (the different "parts" of a person, met with \
curiosity). Decisions are theirs: offer options, not orders. Do not diagnose. If they only greet you or \
are unclear, respond briefly and invite them to share; never assume a problem. If someone may be in \
danger, encourage them to contact local emergency services, a crisis line, or someone they trust.
Use the reference material below only if it fits, and never mention it."""

GUIDANCE_MODE = """MODE: PRACTICAL GUIDANCE. The person is asking for concrete help, so do not answer with \
questions. Start with one warm sentence, then give 3 to 5 short, specific, doable steps as a numbered \
list, tailored to what they have told you (drawn from CBT, NVC or IFS where relevant, for example: write \
the thought down and list the evidence for and against it; name the need behind the feeling; a grounding \
exercise; one small next action). Keep each step to one or two sentences. If you don't know enough to be \
specific, give steps that fit what you do know instead of asking. End with one brief invitation to try a \
step or to say which one feels hardest."""

TOO_MANY_QUESTIONS_NOTE = """NOTE: you have already asked several questions in this conversation. This time \
lead with something useful: a reframe, an insight about what they are describing, or one small practical \
suggestion, and ask at most one short question."""

CRISIS_GUIDANCE = """SAFETY PRIORITY: the person may be at risk of harming themselves. Respond with calm, \
direct warmth. Acknowledge their pain, say you are glad they told you, ask whether they are safe right \
now, and encourage them to contact local emergency services or a crisis line and to reach out to a \
trusted person. Do not give advice about methods, and do not try to problem-solve the conflict in this \
reply."""


_GREETING_RE = re.compile(
    r"^\s*(h+i+|h+e+l+o+|h+e+y+|hiya|yo+|hola|namaste|good\s+(morning|afternoon|evening))(\s+there)?[\s!.,?]*$",
    re.IGNORECASE,
)
GREETING_REPLY = "Hi, I'm glad you're here. What's on your mind today?"


def is_greeting(text: str) -> bool:
    return bool(_GREETING_RE.match(text))


_GUIDANCE_RE = re.compile(
    r"\b(guide|steps?|tips?|advice|advise|suggest\w*|strateg(y|ies)|techniques?|exercises?|plan|solutions?|"
    r"how to|how (do|can|should|would) (i|we)|what (should|can|do) i|"
    r"help me (to )?(overcome|cope|deal|stop|handle|fix|get over|manage)|"
    r"overcome|cope with|get over|deal with)\b",
    re.IGNORECASE,
)


def wants_guidance(text: str) -> bool:
    """True when the person is explicitly asking for practical help rather than just talking."""
    return bool(_GUIDANCE_RE.search(text))


def summarize_llm_stats(stats: dict) -> dict:
    """Human-friendly view of Ollama's timing fields (all durations arrive in nanoseconds)."""
    ns = 1e9
    gen_s = (stats.get("eval_duration") or 0) / ns
    gen_tokens = stats.get("eval_count")
    return {
        "prompt_tokens": stats.get("prompt_eval_count"),
        "prompt_s": round((stats.get("prompt_eval_duration") or 0) / ns, 1),
        "load_s": round((stats.get("load_duration") or 0) / ns, 1),
        "gen_tokens": gen_tokens,
        "gen_tok_per_s": round(gen_tokens / gen_s, 1) if gen_tokens and gen_s else None,
    }


_STOCK_PHRASES = ("i wonder", "it sounds like", "i hear that", "i hear you", "thank you for sharing")


def style_note(history: list[dict]) -> str | None:
    """If the model's last two replies leaned on the same stock phrases, tell it to avoid them now.
    Small models repeat themselves; a concrete list works far better than a general instruction."""
    replies = [m["content"].lower() for m in history if m["role"] == "assistant"][-2:]
    used = [p for p in _STOCK_PHRASES if any(p in r for r in replies)]
    if not used:
        return None
    listed = ", ".join(f'"{p.capitalize()}"' for p in used)
    return (
        f"STYLE: your recent replies already used {listed}. Do not use those phrases in this reply; "
        "begin in a different way."
    )


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def build_system_prompt(
    profile: str,
    emotion: EmotionalState | None,
    chunks: list[RetrievedChunk],
    crisis: bool,
    extra: str | None = None,
) -> str:
    sections = [PERSONA]
    if crisis:
        sections.append(CRISIS_GUIDANCE)
    elif extra:
        sections.append(extra)
    if profile.strip():
        sections.append(
            "## Background notes from earlier conversations (private). Do NOT bring these up unless "
            "the person's current message clearly relates to them, and never assume they are still "
            "dealing with these issues today.\n" + profile.strip()
        )
    if emotion and (emotion.primary_emotion != "neutral" or emotion.core_conflict):
        lines = [f"- Main feeling: {emotion.primary_emotion} (intensity {emotion.intensity}/5)"]
        if emotion.cognitive_distortion != "none":
            lines.append(
                f"- Possible thinking pattern: {emotion.cognitive_distortion.replace('_', ' ')} "
                "(only raise it if it fits, as a curious question)"
            )
        if emotion.core_conflict:
            lines.append(f"- Underlying tension: {emotion.core_conflict}")
        sections.append("## Automated read of their latest message (tentative, hold lightly)\n" + "\n".join(lines))
    if chunks:
        sections.append(
            "## Reference material on counseling frameworks\n"
            + "\n".join(f"- [{c.topic}] {c.text}" for c in chunks)
        )
    return "\n\n".join(sections)


class CounselorAgent:
    def __init__(
        self,
        settings: Settings,
        llm: OllamaClient,
        rag: RagEngine,
        memory: MemoryManager,
        analyzer: EmotionAnalyzer,
    ):
        self.settings = settings
        self.llm = llm
        self.rag = rag
        self.memory = memory
        self.analyzer = analyzer
        self._tasks: set[asyncio.Task] = set()

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _analyze_and_store(self, message_id: int, message: str) -> EmotionalState:
        state = await self.analyzer.analyze(message)
        try:
            await self.memory.set_message_emotion(message_id, state)
        except Exception as exc:
            logger.warning("Could not store emotion for message %s: %s", message_id, exc)
        return state

    async def _retrieve(self, message: str) -> list[RetrievedChunk]:
        try:
            return await self.rag.search(message)
        except Exception as exc:  # retrieval is an enhancement; never block the reply on it
            logger.warning("Retrieval failed: %s", exc)
            return []

    async def stream(self, session_id: str, message: str) -> AsyncIterator[str]:
        """Yield Server-Sent Events: meta, [safety], token*, done | error."""
        t0 = time.perf_counter()
        yield sse("meta", {"session_id": session_id})  # instant feedback, before any model work

        user_msg_id = await self.memory.add_message(session_id, "user", message)

        if is_greeting(message):
            # A bare greeting needs no model call: reply instantly, skipping retrieval and the
            # long-term profile (which small models tend to over-apply to a plain "hello").
            yield sse("token", {"text": GREETING_REPLY})
            await self.memory.add_message(session_id, "assistant", GREETING_REPLY)
            ms = int((time.perf_counter() - t0) * 1000)
            yield sse(
                "done",
                {"session_id": session_id, "ttft_ms": ms, "total_ms": ms,
                 "emotion": EmotionalState().model_dump()},
            )
            return

        prior_emotion = await self.memory.latest_emotion(session_id, before_id=user_msg_id)

        # Emotion analysis and retrieval run concurrently. If analysis is slower than the
        # timeout we don't make the person wait: we use the previous turn's read and let the
        # analysis finish (shielded) in the background so it is still stored.
        rag_task = self._spawn(self._retrieve(message))
        if self.settings.EMOTION_ANALYSIS_ENABLED:
            emotion_task = self._spawn(self._analyze_and_store(user_msg_id, message))
            try:
                emotion = await asyncio.wait_for(
                    asyncio.shield(emotion_task), timeout=self.settings.EMOTION_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                emotion = prior_emotion or EmotionalState()
        else:
            emotion = EmotionalState()  # crisis keywords are still checked below
        chunks = await rag_task

        crisis = emotion.crisis_risk or detect_crisis_keywords(message)
        if crisis:
            yield sse("safety", {"message": self.settings.CRISIS_RESOURCES_TEXT})

        profile = await self.memory.get_profile()
        history = await self.memory.recent_messages(
            session_id, self.settings.RECENT_TURNS, before_id=user_msg_id
        )
        extra = None
        if wants_guidance(message):
            extra = GUIDANCE_MODE
        elif sum(1 for m in history if m["role"] == "assistant") >= 2:
            extra = TOO_MANY_QUESTIONS_NOTE
        extra = "\n\n".join(n for n in (extra, style_note(history)) if n) or None
        messages = [
            {"role": "system", "content": build_system_prompt(profile, emotion, chunks, crisis, extra)},
            *history,
            {"role": "user", "content": message},
        ]

        parts: list[str] = []
        ttft_ms: int | None = None
        error: str | None = None
        stats: dict = {}
        prep_ms = int((time.perf_counter() - t0) * 1000)  # everything before the model is called
        try:
            async for token in self.llm.stream_chat(messages, stats=stats):
                if ttft_ms is None:
                    ttft_ms = int((time.perf_counter() - t0) * 1000)
                parts.append(token)
                yield sse("token", {"text": token})
        except OllamaError as exc:
            error = str(exc)
        finally:
            # Also runs if the client disconnects mid-stream: keep whatever was generated.
            reply = "".join(parts).strip()
            if reply:
                await asyncio.shield(self.memory.add_message(session_id, "assistant", reply))

        if error:
            yield sse("error", {"message": error})
            return

        await self.memory.after_turn(session_id)
        yield sse(
            "done",
            {
                "session_id": session_id,
                "ttft_ms": ttft_ms,
                "prep_ms": prep_ms,
                "total_ms": int((time.perf_counter() - t0) * 1000),
                "emotion": emotion.model_dump(),
                "llm": summarize_llm_stats(stats),
            },
        )
