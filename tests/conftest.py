"""Shared fixtures: every external dependency (Ollama, the web, the embedding model) is faked,
so the suite runs offline in seconds while exercising the real FastAPI / SQLite / Qdrant code."""
import json
import math
import re
import zlib

import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.warmup_service import FetchedPage

DIM = 64

PAGE_TEXT = (
    "Socratic questioning helps a person examine automatic thoughts by asking open, curious questions "
    "rather than giving answers. In cognitive behavioral therapy the counselor invites the person to "
    "test whether a belief is supported by evidence. Nonviolent communication separates observations "
    "from evaluations and connects feelings to underlying needs. Internal family systems describes the "
    "mind as made of parts, each with a positive intent, led by a calm core self. "
) * 4


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder (no model download)."""

    async def warm(self) -> None:
        return None

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for w in re.findall(r"[a-z]+", text.lower()):
            v[zlib.crc32(w.encode()) % DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    async def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    async def embed_query(self, text):
        return self._vec(text)


class FakeOllama:
    def __init__(self):
        self.tokens = ["I hear ", "how heavy ", "this feels."]
        self.emotion = {
            "primary_emotion": "anxiety",
            "cognitive_distortion": "catastrophizing",
            "core_conflict": "autonomy vs. security",
            "intensity": 4,
            "crisis_risk": False,
        }
        self.summary = "Person is torn between autonomy and security."
        self.calls: list[dict] = []

    @property
    def stream_calls(self):
        return [c for c in self.calls if c.get("stream")]

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "gemma2:9b"}]})
        if request.url.path == "/api/chat":
            body = json.loads(request.content)
            self.calls.append(body)
            if body.get("stream"):
                lines = [
                    json.dumps({"message": {"role": "assistant", "content": t}, "done": False})
                    for t in self.tokens
                ]
                lines.append(json.dumps({
                    "message": {"role": "assistant", "content": ""}, "done": True,
                    "prompt_eval_count": 123, "prompt_eval_duration": 2_000_000_000,
                    "eval_count": 20, "eval_duration": 1_000_000_000, "load_duration": 0,
                }))
                return httpx.Response(200, content=("\n".join(lines) + "\n").encode())
            content = json.dumps(self.emotion) if body.get("format") else self.summary
            return httpx.Response(200, json={"message": {"role": "assistant", "content": content}})
        return httpx.Response(200, json={})


async def fake_fetcher(url: str) -> FetchedPage:
    if "FAIL" in url:
        raise ValueError("boom")
    return FetchedPage(url=url, title=url.rsplit("/", 1)[-1], text=f"About {url}. " + PAGE_TEXT)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        SQLITE_DB_PATH=str(tmp_path / "memory.db"),
        QDRANT_STORAGE_PATH=str(tmp_path / "qdrant"),
        FASTEMBED_CACHE_PATH=str(tmp_path / "fe"),
        EMBEDDING_DIM=DIM,
        RAG_MIN_SCORE=0.05,
        PRELOAD_ON_STARTUP=False,
        SUMMARY_EVERY_N_TURNS=1,
    )


@pytest.fixture
def ollama() -> FakeOllama:
    return FakeOllama()


@pytest.fixture
def make_client(settings, ollama):
    """Factory so a test can boot the app several times against the same on-disk state."""

    def _make(fetcher=fake_fetcher, transport=None):
        app = create_app(
            settings,
            embedder=FakeEmbedder(),
            page_fetcher=fetcher,
            ollama_transport=transport or httpx.MockTransport(ollama.handler),
        )
        return TestClient(app)

    return _make


@pytest.fixture
def client(make_client):
    with make_client() as c:
        yield c


@pytest.fixture
def warmed(client):
    r = client.post("/api/v1/setup/warmup", json={"topics": ["cbt"]})
    assert r.status_code == 202
    assert client.get("/api/v1/setup/status").json()["state"] == "completed"
    return client


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = block.split("\n")
        event = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        events.append((event, data))
    return events
