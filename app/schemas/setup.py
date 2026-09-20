"""Onboarding / warm-up models."""
from enum import Enum

from pydantic import BaseModel, Field, HttpUrl


class TopicInfo(BaseModel):
    id: str
    label: str
    sources: list[str]


class WarmupRequest(BaseModel):
    topics: list[str] = Field(min_length=1, description="Topic ids from GET /setup/topics")
    extra_urls: list[HttpUrl] = Field(
        default_factory=list, max_length=20, description="Optional additional pages to index"
    )


class WarmupState(str, Enum):
    idle = "idle"
    running = "running"
    completed = "completed"
    failed = "failed"


class WarmupStatus(BaseModel):
    state: WarmupState = WarmupState.idle
    initialized: bool = False
    topics: list[str] = Field(default_factory=list)
    pages_total: int = 0
    pages_done: int = 0
    pages_failed: int = 0
    failed_urls: list[str] = Field(default_factory=list)
    chunks_indexed: int = 0
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
