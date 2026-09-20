"""FastAPI entry point: wiring, CORS and lifespan."""
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.api import api_router
from app.core.config import Settings, get_settings
from app.core.database import create_qdrant_client, ensure_collection
from app.core.memory_db import MemoryDB
from app.core.ollama_client import OllamaClient
from app.services.counselor_agent import CounselorAgent
from app.services.emotion_analyzer import EmotionAnalyzer
from app.services.memory_manager import MemoryManager
from app.services.rag_engine import Embedder, FastEmbedder, RagEngine
from app.services.warmup_service import PageFetcher, WarmupService

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(
    settings: Settings | None = None,
    *,
    embedder: Embedder | None = None,
    page_fetcher: PageFetcher | None = None,
    ollama_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """App factory. The keyword arguments exist so tests can inject fakes; production leaves
    them unset."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db = MemoryDB(settings.SQLITE_DB_PATH)
        await db.init()

        qdrant = create_qdrant_client(settings.QDRANT_STORAGE_PATH)
        ensure_collection(qdrant, settings.QDRANT_COLLECTION, settings.EMBEDDING_DIM)

        emb = embedder or FastEmbedder(settings.EMBEDDING_MODEL, settings.FASTEMBED_CACHE_PATH)
        rag = RagEngine(
            qdrant, settings.QDRANT_COLLECTION, emb,
            top_k=settings.RAG_TOP_K, min_score=settings.RAG_MIN_SCORE,
        )
        llm = OllamaClient(
            settings.OLLAMA_BASE_URL, settings.MODEL_NAME,
            keep_alive=settings.OLLAMA_KEEP_ALIVE, temperature=settings.LLM_TEMPERATURE,
            num_ctx=settings.LLM_NUM_CTX, transport=ollama_transport,
        )
        memory = MemoryManager(db, llm, summary_every=settings.SUMMARY_EVERY_N_TURNS)
        analyzer = EmotionAnalyzer(llm)
        warmup = WarmupService(db, rag, settings, fetcher=page_fetcher)
        counselor = CounselorAgent(settings, llm, rag, memory, analyzer)

        app.state.settings = settings
        app.state.db = db
        app.state.rag = rag
        app.state.llm = llm
        app.state.memory = memory
        app.state.warmup = warmup
        app.state.counselor = counselor

        preload_tasks: list[asyncio.Task] = []
        if settings.PRELOAD_ON_STARTUP and await warmup.is_initialized():
            # Load the embedding model and the LLM into RAM now so the first message is fast.
            preload_tasks = [asyncio.create_task(emb.warm()), asyncio.create_task(llm.preload())]

        try:
            yield
        finally:
            for task in preload_tasks:
                task.cancel()
            await asyncio.gather(*preload_tasks, return_exceptions=True)
            await counselor.aclose()
            await memory.aclose()
            await llm.aclose()
            qdrant.close()

    app = FastAPI(
        title="Untangle",
        version="2.0.0",
        description="A private, on-device companion for untangling hard decisions, inner conflicts and difficult conversations.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix="/api/v1")
    # The chat UI. Mounted last so it never shadows /api or /docs.
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
    return app


app = create_app()
