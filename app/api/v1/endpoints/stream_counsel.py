"""Live SSE counseling endpoint."""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.schemas.chat import CounselRequest
from app.services.memory_manager import SessionNotFound

router = APIRouter(tags=["counsel"])


@router.post("/stream_counsel")
async def stream_counsel(body: CounselRequest, request: Request):
    """Stream the counselor's reply as Server-Sent Events.

    Events: `meta` (session id, sent immediately), `safety` (only if self-harm language is
    detected), `token` (repeated), then `done` or `error`. POST is used so the message travels in
    the body; consume it with `fetch()` + a stream reader (EventSource only supports GET).
    """
    state = request.app.state
    if state.settings.REQUIRE_WARMUP and not await state.warmup.is_initialized():
        raise HTTPException(
            status_code=409,
            detail="Setup not finished. Run POST /api/v1/setup/warmup first.",
        )
    try:
        session_id = await state.memory.resolve_session(body.session_id)
    except SessionNotFound:
        raise HTTPException(status_code=404, detail="Session not found")

    return StreamingResponse(
        state.counselor.stream(session_id, body.message),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
