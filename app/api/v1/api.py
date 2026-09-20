"""Central v1 router."""
from fastapi import APIRouter

from app.api.v1.endpoints import health, sessions, setup, stream_counsel

api_router = APIRouter()
api_router.include_router(setup.router)
api_router.include_router(stream_counsel.router)
api_router.include_router(sessions.router)
api_router.include_router(health.router)
