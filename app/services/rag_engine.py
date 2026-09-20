"""Offline retrieval: FastEmbed vectors + local Qdrant similarity search."""
import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass
from typing import Protocol

from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...
    async def warm(self) -> None: ...


class FastEmbedder:
    """Runs BAAI/bge-small-en-v1.5 (ONNX, CPU) locally. Model files are downloaded once, during
    warm-up, into `cache_dir`; afterwards no network access is needed."""

    def __init__(self, model_name: str, cache_dir: str):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None
        self._load_lock = threading.Lock()

    def _load(self):
        with self._load_lock:
            if self._model is None:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(model_name=self.model_name, cache_dir=self.cache_dir)
            return self._model

    async def warm(self) -> None:
        await asyncio.to_thread(self._load)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        def run():
            return [v.tolist() for v in self._load().embed(texts)]

        return await asyncio.to_thread(run)

    async def embed_query(self, text: str) -> list[float]:
        def run():
            return next(iter(self._load().query_embed(text))).tolist()

        return await asyncio.to_thread(run)


@dataclass(frozen=True)
class Chunk:
    text: str
    source_url: str
    title: str
    topic: str
    index: int

    @property
    def id(self) -> str:
        # Deterministic: re-running warm-up overwrites instead of duplicating.
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.source_url}#{self.index}"))


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source_url: str
    title: str
    topic: str
    score: float


class RagEngine:
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        embedder: Embedder,
        *,
        top_k: int = 4,
        min_score: float = 0.3,
    ):
        self.client = client
        self.collection = collection
        self.embedder = embedder
        self.top_k = top_k
        self.min_score = min_score
        # Local-mode Qdrant is not built for concurrent callers; serialize access.
        self._lock = asyncio.Lock()

    async def replace_source(self, source_url: str, chunks: list[Chunk]) -> int:
        """Atomically (per source) swap in the chunks for one page. Returns chunks stored."""
        if not chunks:
            return 0
        vectors = await self.embedder.embed_documents([c.text for c in chunks])
        points = [
            models.PointStruct(
                id=c.id,
                vector=v,
                payload={
                    "text": c.text,
                    "source_url": c.source_url,
                    "title": c.title,
                    "topic": c.topic,
                    "index": c.index,
                },
            )
            for c, v in zip(chunks, vectors)
        ]

        def run():
            self.client.delete(
                self.collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source_url", match=models.MatchValue(value=source_url)
                            )
                        ]
                    )
                ),
            )
            self.client.upsert(self.collection, points=points)

        async with self._lock:
            await asyncio.to_thread(run)
        return len(points)

    async def search(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        if await self.count() == 0:
            return []
        vector = await self.embedder.embed_query(query)

        def run():
            return self.client.query_points(
                self.collection, query=vector, limit=k or self.top_k, with_payload=True
            ).points

        async with self._lock:
            hits = await asyncio.to_thread(run)
        return [
            RetrievedChunk(
                text=h.payload["text"],
                source_url=h.payload.get("source_url", ""),
                title=h.payload.get("title", ""),
                topic=h.payload.get("topic", ""),
                score=h.score,
            )
            for h in hits
            if h.score >= self.min_score
        ]

    async def count(self) -> int:
        def run():
            return self.client.count(self.collection, exact=True).count

        async with self._lock:
            return await asyncio.to_thread(run)
