"""First-time warm-up trigger and status check."""
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from app.schemas.setup import TopicInfo, WarmupRequest, WarmupStatus
from app.services.warmup_service import UnknownTopic, WarmupBusy

router = APIRouter(prefix="/setup", tags=["setup"])


@router.get("/topics", response_model=list[TopicInfo])
async def list_topics(request: Request):
    """Topics the user can pick during onboarding."""
    return request.app.state.warmup.catalog()


@router.get("/status", response_model=WarmupStatus)
async def warmup_status(request: Request):
    return await request.app.state.warmup.get_status()


@router.post("/warmup", response_model=WarmupStatus, status_code=202)
async def start_warmup(body: WarmupRequest, background: BackgroundTasks, request: Request):
    """Start crawling + indexing the selected topics. Returns immediately; poll /setup/status."""
    service = request.app.state.warmup
    try:
        job = await service.start(body)
    except UnknownTopic as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except WarmupBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    background.add_task(service.run, job)
    return job.status
