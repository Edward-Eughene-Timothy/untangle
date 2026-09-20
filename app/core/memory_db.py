"""SQLite (aiosqlite) storage for sessions, chat history and app-level flags."""
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    title      TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role         TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content      TEXT NOT NULL,
    emotion_json TEXT,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, id);
CREATE TABLE IF NOT EXISTS app_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class MemoryDB:
    def __init__(self, path: str):
        self.path = path

    async def init(self) -> None:
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.executescript(SCHEMA)
            await db.commit()

    @asynccontextmanager
    async def connect(self):
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA foreign_keys=ON")
            yield db

    async def get_state(self, key: str, default: str | None = None) -> str | None:
        async with self.connect() as db:
            cur = await db.execute("SELECT value FROM app_state WHERE key = ?", (key,))
            row = await cur.fetchone()
        return row["value"] if row else default

    async def set_state(self, key: str, value: str) -> None:
        async with self.connect() as db:
            await db.execute(
                "INSERT INTO app_state(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            await db.commit()

    async def delete_state(self, key: str) -> None:
        async with self.connect() as db:
            await db.execute("DELETE FROM app_state WHERE key = ?", (key,))
            await db.commit()
