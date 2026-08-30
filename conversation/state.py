"""
Conversation state helpers.

Thin convenience layer that operates on :class:`ConversationRecord`
objects.  Keeps mutation logic out of the manager and follow-up modules.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from database.models import (
    ConversationRecord,
    ConversationStage,
    LegalQAResult,
    Message,
    MessageRole,
)


def add_user_message(state: ConversationRecord, content: str) -> None:
    """Append a user message and move stage forward if still INITIAL."""
    state.messages.append(
        Message(
            role=MessageRole.USER,
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    )
    if state.stage == ConversationStage.INITIAL:
        state.stage = ConversationStage.GATHERING_INFO


def add_assistant_message(state: ConversationRecord, content: str) -> None:
    """Append an assistant (follow-up or answer) message."""
    state.messages.append(
        Message(
            role=MessageRole.ASSISTANT,
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    )
    state.last_assistant_question = content


def update_facts(state: ConversationRecord, new_facts: dict[str, Any]) -> None:
    """Merge newly extracted facts into the conversation record."""
    state.facts.update(new_facts)


def store_legal_qa_result(
    state: ConversationRecord, result: LegalQAResult
) -> None:
    """Persist a Legal_QA response and advance the stage."""
    state.legal_qa_results.append(result)
    state.stage = ConversationStage.ANSWERED


def get_original_problem(state: ConversationRecord) -> str:
    """Return the first user message (the original problem statement)."""
    for msg in state.messages:
        if msg.role == MessageRole.USER:
            return msg.content
    return ""


def get_user_messages(state: ConversationRecord) -> list[str]:
    """Return all user messages in order."""
    return [m.content for m in state.messages if m.role == MessageRole.USER]
