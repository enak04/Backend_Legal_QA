"""
Conversation API endpoints.

Thin route handlers that delegate all logic to
:class:`ConversationManager`.  Error handling translates internal
exceptions into appropriate HTTP responses.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from conversation.manager import ConversationManager, ConversationNotFoundError
from database.models import Mode
from legal_qa.client import (
    LegalQABadResponseError,
    LegalQATimeoutError,
    LegalQAUnavailableError,
)
from services.response_formatter import format_conversation_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


# ── Request / response schemas ────────────────────────────────

class CreateConversationRequest(BaseModel):
    mode: str = Field(
        ...,
        description="One of: actionable, informative, readable",
    )


class SendMessageRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        description="The user's message text",
    )


# ── Helper to get the manager from app state ─────────────────

def _manager(request: Request) -> ConversationManager:
    return request.app.state.conversation_manager


# ── Endpoints ─────────────────────────────────────────────────


@router.post("/conversations")
async def create_conversation(
    body: CreateConversationRequest,
    request: Request,
) -> dict[str, Any]:
    """
    Start a new conversation with the selected mode.

    **Request**::

        { "mode": "actionable" }

    **Response**::

        { "conversation_id": "...", "mode": "actionable" }
    """
    # Validate mode
    try:
        mode = Mode(body.mode)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid mode '{body.mode}'. "
                f"Must be one of: actionable, informative, readable."
            ),
        )

    manager = _manager(request)
    record = await manager.create_conversation(mode)

    return {
        "conversation_id": record.conversation_id,
        "mode": record.mode.value,
        "stage": record.stage.value,
        "created_at": record.created_at,
    }


@router.post("/conversations/{conversation_id}/messages")
async def send_message(
    conversation_id: str,
    body: SendMessageRequest,
    request: Request,
) -> dict[str, Any]:
    """
    Send a message in an existing conversation.

    Returns either a follow-up question or a final Legal_QA answer.
    """
    manager = _manager(request)

    try:
        result = await manager.process_message(
            conversation_id=conversation_id,
            user_message=body.message,
        )
    except ConversationNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation not found: {conversation_id}",
        )
    except LegalQAUnavailableError as exc:
        logger.error("Legal_QA unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=(
                "The legal analysis service is currently unavailable. "
                "Please try again later."
            ),
        )
    except LegalQATimeoutError as exc:
        logger.error("Legal_QA timeout: %s", exc)
        raise HTTPException(
            status_code=504,
            detail=(
                "The legal analysis service took too long to respond. "
                "Please try again."
            ),
        )
    except LegalQABadResponseError as exc:
        logger.error("Legal_QA bad response: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Received an unexpected response from the legal analysis service: {exc}",
        )
    except Exception as exc:
        logger.exception("Unexpected error processing message")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again.",
        )

    return result


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    request: Request,
) -> dict[str, Any]:
    """
    Retrieve the full conversation state.

    Returns conversation metadata, message history, collected facts,
    and any Legal_QA results.
    """
    manager = _manager(request)
    record = await manager.get_conversation(conversation_id)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation not found: {conversation_id}",
        )

    return format_conversation_response(record)
