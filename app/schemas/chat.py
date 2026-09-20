"""Session and message models."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.emotional_state import EmotionalState


class CounselRequest(BaseModel):
    session_id: str | None = Field(
        default=None, description="Omit to start a new session; the id is returned in the 'meta' event"
    )
    message: str = Field(max_length=4000)

    @field_validator("message")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be empty")
        return v


class MessageOut(BaseModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    emotion: EmotionalState | None = None
    created_at: str


class SessionOut(BaseModel):
    id: str
    title: str | None = None
    message_count: int = 0
    created_at: str
    updated_at: str


class SessionDetail(SessionOut):
    messages: list[MessageOut] = Field(default_factory=list)


class ProfileOut(BaseModel):
    summary: str = ""


class ProfileUpdate(BaseModel):
    summary: str = Field(max_length=4000)
