"""Offline counselor engine: prompt synthesis + live SSE token streaming from local Ollama."""
import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator

from app.core.config import Settings
from app.core.ollama_client import OllamaClient, OllamaError
from app.schemas.emotional_state import EmotionalState
from app.services.emotion_analyzer import EmotionAnalyzer, detect_crisis_keywords
from app.services.memory_manager import MemoryManager
from app.services.rag_engine import RagEngine, RetrievedChunk

logger = logging.getLogger(__name__)

PERSONA = """You are a warm, steady counseling companion running privately on the user's own \
device. You help people think through inner conflicts and conflicts with others.

How you work:
- Reflect and validate the feeling first. Use Nonviolent Communication's frame: what was observed, \
what is felt, what need sits underneath, and what request might follow.
- Use CBT-style Socratic questions to examine thoughts. If a thinking pattern seems present, offer it \
as a gentle hypothesis ("I wonder if..."), never as a diagnosis or a verdict.
- Where it fits, use IFS-informed language: notice the different "parts" of the person (for example the \
part that wants safety and the part that wants freedom) with curiosity and without judgment.
- Be non-prescriptive. Do not tell the person what to do. Offer perspectives, options and questions, \
and respect that the decision is theirs.
- Keep replies short (about 3 to 6 sentences), in plain conversational language. No headings, no bullet \
lists. Ask at most one or two questions.
- You are not a therapist and cannot diagnose or treat. If someone may be in danger, encourage them to \
contact local emergency services, a crisis line, or someone they trust.
- Use the reference material below only when it genuinely fits. Never mention or quote it as a source."""

CRISIS_GUIDANCE = """SAFETY PRIORITY: the person may be at risk of harming themselves. Respond with calm, \
direct warmth. Acknowledge their pain, say you are glad they told you, ask whether they are safe right \
now, and encourage them to contact local emergency services or a crisis line and to reach out to a \
trusted person. Do not give advice about methods, and do not try to problem-solve the conflict in this \
reply."""


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def build_system_prompt(
    profile: str,
    emotion: EmotionalState | None,
    chunks: list[RetrievedChunk],
    crisis: bool,
) -> str:
    sections = [PERSONA]
    if crisis:
        sections.append(CRISIS_GUIDANCE)
    if profile.strip():
        sections.append(
            "## What you already know about this person (private long-term notes; use gently, "
            "don't recite)\n" + profile.strip()
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
        prior_emotion = await self.memory.latest_emotion(session_id, before_id=user_msg_id)

        # Emotion analysis and retrieval run concurrently. If analysis is slower than the
        # timeout we don't make the person wait: we use the previous turn's read and let the
        # analysis finish (shielded) in the background so it is still stored.
        emotion_task = self._spawn(self._analyze_and_store(user_msg_id, message))
        rag_task = self._spawn(self._retrieve(message))
        try:
            emotion = await asyncio.wait_for(
                asyncio.shield(emotion_task), timeout=self.settings.EMOTION_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            emotion = prior_emotion or EmotionalState()
        chunks = await rag_task

        crisis = emotion.crisis_risk or detect_crisis_keywords(message)
        if crisis:
            yield sse("safety", {"message": self.settings.CRISIS_RESOURCES_TEXT})

        profile = await self.memory.get_profile()
        history = await self.memory.recent_messages(
            session_id, self.settings.RECENT_TURNS, before_id=user_msg_id
        )
        messages = [
            {"role": "system", "content": build_system_prompt(profile, emotion, chunks, crisis)},
            *history,
            {"role": "user", "content": message},
        ]

        parts: list[str] = []
        ttft_ms: int | None = None
        error: str | None = None
        try:
            async for token in self.llm.stream_chat(messages):
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
                "total_ms": int((time.perf_counter() - t0) * 1000),
                "emotion": emotion.model_dump(),
            },
        )
