"""
Mode listing endpoint.
"""

from fastapi import APIRouter

from services.response_formatter import format_modes_response

router = APIRouter(prefix="/api")


@router.get("/modes")
async def list_modes():
    """Return the available legal-assistance modes with descriptions."""
    return format_modes_response()
