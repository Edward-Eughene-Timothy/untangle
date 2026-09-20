"""System status: local Ollama, vector store, SQLite, warm-up flag."""
from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request):
    state = request.app.state
    ollama = await state.llm.health()

    qdrant = {"ready": True, "chunks": 0}
    try:
        qdrant["chunks"] = await state.rag.count()
    except Exception as exc:
        qdrant = {"ready": False, "error": str(exc)}

    sqlite = {"ok": True}
    initialized = False
    try:
        initialized = await state.warmup.is_initialized()
    except Exception as exc:
        sqlite = {"ok": False, "error": str(exc)}

    healthy = ollama["reachable"] and ollama["model_available"] and qdrant["ready"] and sqlite["ok"]
    return {
        "status": "ok" if healthy else "degraded",
        "initialized": initialized,
        "ollama": ollama,
        "qdrant": qdrant,
        "sqlite": sqlite,
    }
