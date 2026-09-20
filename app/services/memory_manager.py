"""Persistent conversation memory (SQLite) and the long-term User Journey & Profile Summary."""
import asyncio
import logging
import uuid
from datetime import datetime, timezone

from app.core.memory_db import MemoryDB
from app.core.ollama_client import OllamaClient
from app.schemas.chat import MessageOut, SessionDetail, SessionOut
from app.schemas.emotional_state import EmotionalState

logger = logging.getLogger(__name__)

PROFILE_KEY = "user_profile_summary"
COUNTER_KEY = "turns_since_summary"

_SUMMARY_SYSTEM = """You maintain a short, private "User Journey & Profile Summary" for a counseling \
companion that runs entirely on the user's own device.

Update the existing summary using the new conversation excerpt. Keep only what helps future \
conversations: recurring emotions and thinking patterns, core inner conflicts, values, goals, \
important life context, and coping strategies that seemed to help. Write in third person, neutral \
and non-judgmental. Do not diagnose. Do not include contact details, ID numbers or other \
identifying data. Stay under 200 words. Output only the updated summary text."""


class SessionNotFound(LookupError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MemoryManager:
    def __init__(self, db: MemoryDB, llm: OllamaClient, *, summary_every: int = 6):
        self.db = db
        self.llm = llm
        self.summary_every = summary_every
        self._tasks: set[asyncio.Task] = set()
        self._summarizing = False

    # ------------------------------------------------------------------ sessions

    async def create_session(self) -> str:
        sid = uuid.uuid4().hex
        now = _now()
        async with self.db.connect() as db:
            await db.execute(
                "INSERT INTO sessions(id, title, created_at, updated_at) VALUES(?, NULL, ?, ?)",
                (sid, now, now),
            )
            await db.commit()
        return sid

    async def resolve_session(self, session_id: str | None) -> str:
        """Return an existing session id, or create one when `session_id` is None."""
        if session_id is None:
            return await self.create_session()
        async with self.db.connect() as db:
            cur = await db.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,))
            if await cur.fetchone() is None:
                raise SessionNotFound(session_id)
        return session_id

    async def list_sessions(self) -> list[SessionOut]:
        async with self.db.connect() as db:
            cur = await db.execute(
                "SELECT s.*, (SELECT COUNT(*) FROM chat_messages m WHERE m.session_id = s.id) AS n "
                "FROM sessions s ORDER BY s.updated_at DESC, s.created_at DESC"
            )
            rows = await cur.fetchall()
        return [
            SessionOut(
                id=r["id"], title=r["title"], message_count=r["n"],
                created_at=r["created_at"], updated_at=r["updated_at"],
            )
            for r in rows
        ]

    async def get_session(self, session_id: str) -> SessionDetail:
        async with self.db.connect() as db:
            cur = await db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            s = await cur.fetchone()
            if s is None:
                raise SessionNotFound(session_id)
            cur = await db.execute(
                "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY id", (session_id,)
            )
            rows = await cur.fetchall()
        messages = [
            MessageOut(
                id=r["id"], role=r["role"], content=r["content"], created_at=r["created_at"],
                emotion=EmotionalState.model_validate_json(r["emotion_json"]) if r["emotion_json"] else None,
            )
            for r in rows
        ]
        return SessionDetail(
            id=s["id"], title=s["title"], message_count=len(messages),
            created_at=s["created_at"], updated_at=s["updated_at"], messages=messages,
        )

    async def delete_session(self, session_id: str) -> None:
        async with self.db.connect() as db:
            cur = await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            await db.commit()
            if cur.rowcount == 0:
                raise SessionNotFound(session_id)

    # ------------------------------------------------------------------ messages

    async def add_message(
        self, session_id: str, role: str, content: str, emotion: EmotionalState | None = None
    ) -> int:
        now = _now()
        async with self.db.connect() as db:
            cur = await db.execute(
                "INSERT INTO chat_messages(session_id, role, content, emotion_json, created_at) "
                "VALUES(?, ?, ?, ?, ?)",
                (session_id, role, content, emotion.model_dump_json() if emotion else None, now),
            )
            message_id = cur.lastrowid
            await db.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
            if role == "user":
                await db.execute(
                    "UPDATE sessions SET title = ? WHERE id = ? AND title IS NULL",
                    (content[:60].strip(), session_id),
                )
            await db.commit()
        return message_id

    async def set_message_emotion(self, message_id: int, emotion: EmotionalState) -> None:
        async with self.db.connect() as db:
            await db.execute(
                "UPDATE chat_messages SET emotion_json = ? WHERE id = ?",
                (emotion.model_dump_json(), message_id),
            )
            await db.commit()

    async def recent_messages(self, session_id: str, limit: int, before_id: int | None = None) -> list[dict]:
        """Last `limit` messages (oldest first), optionally only those older than `before_id`."""
        query = "SELECT role, content FROM chat_messages WHERE session_id = ?"
        params: list = [session_id]
        if before_id is not None:
            query += " AND id < ?"
            params.append(before_id)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        async with self.db.connect() as db:
            cur = await db.execute(query, params)
            rows = await cur.fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    async def latest_emotion(self, session_id: str, before_id: int | None = None) -> EmotionalState | None:
        query = (
            "SELECT emotion_json FROM chat_messages "
            "WHERE session_id = ? AND role = 'user' AND emotion_json IS NOT NULL"
        )
        params: list = [session_id]
        if before_id is not None:
            query += " AND id < ?"
            params.append(before_id)
        query += " ORDER BY id DESC LIMIT 1"
        async with self.db.connect() as db:
            cur = await db.execute(query, params)
            row = await cur.fetchone()
        return EmotionalState.model_validate_json(row["emotion_json"]) if row else None

    # ------------------------------------------------------------------ long-term profile

    async def get_profile(self) -> str:
        return await self.db.get_state(PROFILE_KEY, "") or ""

    async def set_profile(self, text: str) -> None:
        await self.db.set_state(PROFILE_KEY, text.strip())

    async def clear_profile(self) -> None:
        await self.db.delete_state(PROFILE_KEY)
        await self.db.set_state(COUNTER_KEY, "0")

    async def after_turn(self, session_id: str) -> None:
        """Call after each completed exchange; refreshes the profile in the background."""
        n = int(await self.db.get_state(COUNTER_KEY, "0") or 0) + 1
        await self.db.set_state(COUNTER_KEY, str(n))
        if n >= self.summary_every and not self._summarizing:
            task = asyncio.create_task(self._summarize(session_id))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _summarize(self, session_id: str) -> None:
        self._summarizing = True
        try:
            recent = await self.recent_messages(session_id, limit=self.summary_every * 2 + 2)
            if not recent:
                return
            excerpt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)
            current = await self.get_profile()
            summary = await self.llm.complete(
                [
                    {"role": "system", "content": _SUMMARY_SYSTEM},
                    {
                        "role": "user",
                        "content": f"EXISTING SUMMARY:\n{current or '(none yet)'}\n\nNEW CONVERSATION EXCERPT:\n{excerpt}",
                    },
                ],
                temperature=0.2,
                num_predict=400,
            )
            summary = summary.strip()
            if summary:
                await self.set_profile(summary[:4000])
                await self.db.set_state(COUNTER_KEY, "0")
        except Exception as exc:
            logger.warning("Profile summarization failed (will retry next turn): %s", exc)
        finally:
            self._summarizing = False

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
