"""Application settings (Pydantic v2 BaseSettings, environment / .env driven)."""
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CRISIS_TEXT = (
    "It sounds like you may be carrying something really painful right now. If you might act on "
    "thoughts of harming yourself, or you are in immediate danger, please contact your local "
    "emergency number or a local crisis line right now, or reach out to someone you trust and "
    "stay with them. You don't have to go through this alone."
)


class Settings(BaseSettings):
    # Resolve .env next to the project root, so it is found no matter which directory uvicorn is
    # started from (a relative ".env" silently does nothing when launched from e.g. app/).
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Required by the blueprint -------------------------------------------------------
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    MODEL_NAME: str = "gemma2:9b"  # or "qwen2.5:7b"
    QDRANT_STORAGE_PATH: str = "./qdrant_storage"
    SQLITE_DB_PATH: str = "./memory.db"

    # --- Embeddings / vector store -------------------------------------------------------
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIM: int = 384  # must match EMBEDDING_MODEL
    # FastEmbed's default cache lives in the OS temp dir and can be wiped on reboot, which would
    # silently break offline runtime. Keep it next to the app instead.
    FASTEMBED_CACHE_PATH: str = "./fastembed_cache"
    QDRANT_COLLECTION: str = "counseling_frameworks"

    # --- Chunking / retrieval ------------------------------------------------------------
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    RAG_TOP_K: int = 4
    RAG_MIN_SCORE: float = 0.3  # cosine; raise if unrelated chunks leak into prompts

    # --- Conversation / memory -----------------------------------------------------------
    RECENT_TURNS: int = 8  # last N messages (user + assistant) sent to the model
    SUMMARY_EVERY_N_TURNS: int = 6  # refresh the long-term profile after this many exchanges
    EMOTION_TIMEOUT_S: float = 2.0  # max time to wait for emotion analysis before streaming
    # The emotion pass is a second LLM call per message. On slow hardware it competes with the reply
    # for the same CPU/GPU, so turning it off can cut latency noticeably. The keyword-based safety
    # check still runs either way.
    EMOTION_ANALYSIS_ENABLED: bool = True

    # --- LLM -----------------------------------------------------------------------------
    OLLAMA_KEEP_ALIVE: str = "30m"  # keep the model resident so first-token latency stays low
    LLM_TEMPERATURE: float = 0.6
    LLM_NUM_CTX: int = 4096
    PRELOAD_ON_STARTUP: bool = True

    # --- Warm-up crawler -----------------------------------------------------------------
    CRAWL_USER_AGENT: str = "Untangle/2.0 (local warm-up; respects robots.txt)"
    CRAWL_TIMEOUT_S: float = 20.0
    CRAWL_CONCURRENCY: int = 4

    # --- API -----------------------------------------------------------------------------
    REQUIRE_WARMUP: bool = True  # refuse /stream_counsel until warm-up has succeeded once
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    CRISIS_RESOURCES_TEXT: str = DEFAULT_CRISIS_TEXT

    @field_validator("QDRANT_STORAGE_PATH", "SQLITE_DB_PATH", "FASTEMBED_CACHE_PATH")
    @classmethod
    def _anchor_to_project_root(cls, v: str) -> str:
        """Relative data paths mean "inside the project", not "wherever uvicorn was launched from",
        so starting the server from another directory can never create a second, empty database."""
        if v == ":memory:":
            return v
        p = Path(v).expanduser()
        return str(p if p.is_absolute() else (PROJECT_ROOT / p).resolve())


@lru_cache
def get_settings() -> Settings:
    return Settings()
