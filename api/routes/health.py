"""
Health-check endpoint.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
@router.get("/health")
@router.get("/api/health")
async def health_check():
    """Return server health status."""
    return {"status": "ok"}
