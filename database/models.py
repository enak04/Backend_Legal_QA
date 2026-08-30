"""
Data models for the conversation system.

These are plain data structures (Pydantic models) used across the
application.  They are independent of the storage backend.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────

class Mode(str, Enum):
    """Legal-assistance response modes."""
    ACTIONABLE = "actionable"
    INFORMATIVE = "informative"
    READABLE = "readable"


class ConversationStage(str, Enum):
    """Tracks where a conversation currently sits in its lifecycle."""
    INITIAL = "initial"
    GATHERING_INFO = "gathering_info"
    READY_FOR_QA = "ready_for_qa"
    ANSWERED = "answered"


class MessageRole(str, Enum):
    """Who sent the message."""
    USER = "user"
    ASSISTANT = "assistant"


# ── Message ───────────────────────────────────────────────────

class Message(BaseModel):
    """A single message in a conversation."""
    role: MessageRole
    content: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── Legal_QA response models ────────────────────────────────

class RetrievedCase(BaseModel):
    """A past case returned by Legal_QA."""
    question: str
    answer: str


class LegalQAResult(BaseModel):
    """The full response from the Legal_QA /predict endpoint."""
    question: str
    answer: str
    reasoning_chain: list[str]
    retrieved_cases: list[RetrievedCase]


# ── Conversation record ──────────────────────────────────────

class ConversationRecord(BaseModel):
    """Full conversation state persisted in the database."""
    conversation_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:12]
    )
    mode: Mode
    stage: ConversationStage = ConversationStage.INITIAL
    facts: dict[str, Any] = Field(default_factory=dict)
    messages: list[Message] = Field(default_factory=list)
    last_assistant_question: str | None = None
    legal_qa_results: list[LegalQAResult] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
