"""Conversation history and long-term profile management."""
from fastapi import APIRouter, HTTPException, Request, Response

from app.schemas.chat import ProfileOut, ProfileUpdate, SessionDetail, SessionOut
from app.services.memory_manager import SessionNotFound

router = APIRouter(tags=["sessions"])


@router.post("/sessions", response_model=SessionOut, status_code=201)
async def create_session(request: Request):
    memory = request.app.state.memory
    sid = await memory.create_session()
    detail = await memory.get_session(sid)
    return SessionOut(**detail.model_dump(exclude={"messages"}))


@router.get("/sessions", response_model=list[SessionOut])
async def list_sessions(request: Request):
    return await request.app.state.memory.list_sessions()


@router.get("/sessions/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str, request: Request):
    try:
        return await request.app.state.memory.get_session(session_id)
    except SessionNotFound:
        raise HTTPException(status_code=404, detail="Session not found")


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, request: Request):
    try:
        await request.app.state.memory.delete_session(session_id)
    except SessionNotFound:
        raise HTTPException(status_code=404, detail="Session not found")
    return Response(status_code=204)


@router.get("/profile", response_model=ProfileOut)
async def get_profile(request: Request):
    """The condensed 'User Journey & Profile Summary' the counselor sees in every conversation."""
    return ProfileOut(summary=await request.app.state.memory.get_profile())


@router.put("/profile", response_model=ProfileOut)
async def update_profile(body: ProfileUpdate, request: Request):
    await request.app.state.memory.set_profile(body.summary)
    return ProfileOut(summary=await request.app.state.memory.get_profile())


@router.delete("/profile", status_code=204)
async def clear_profile(request: Request):
    await request.app.state.memory.clear_profile()
    return Response(status_code=204)
