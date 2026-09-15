"""
Conversation manager — the central orchestrator.

Implements the core message-processing flow:

    User message
        → load state
        → add message to history
        → extract & update facts
        → check if follow-up is needed
        → YES: return follow-up question
        → NO:  build query → call Legal_QA → return answer
        → persist state

All conversation logic passes through this single class so that
API routes remain thin.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from conversation.followup import FollowUpEngine
from conversation.state import (
    add_assistant_message,
    add_user_message,
    store_legal_qa_result,
    update_facts,
)
from database.models import (
    ConversationRecord,
    ConversationStage,
    LegalQAResult,
    Mode,
)
from database.repository import ConversationRepository
from legal_qa.client import LegalQAClient
from legal_qa.query_builder import QueryBuilder
from services.intake_service import OpenAIIntakeService

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Stateless orchestrator that coordinates a single message turn.

    Dependencies are injected so the manager is easy to test with
    mocks/fakes.
    """

    def __init__(
        self,
        repository: ConversationRepository,
        legal_qa_client: LegalQAClient,
        followup_engine: FollowUpEngine | None = None,
        query_builder: QueryBuilder | None = None,
        intake_service: OpenAIIntakeService | None = None,
    ) -> None:
        self._repo = repository
        self._legal_qa = legal_qa_client
        self._followup = followup_engine or FollowUpEngine()
        self._query_builder = query_builder or QueryBuilder()
        self._intake = intake_service or OpenAIIntakeService(
            fallback_engine=self._followup,
            query_builder=self._query_builder,
        )

    # ── Public API ────────────────────────────────────────────

    async def create_conversation(self, mode: Mode) -> ConversationRecord:
        """Start a new conversation with the given mode."""
        record = ConversationRecord(mode=mode)
        return await self._repo.create(record)

    async def get_conversation(
        self, conversation_id: str
    ) -> ConversationRecord | None:
        """Load a conversation by ID."""
        return await self._repo.get(conversation_id)

    async def process_message(
        self, conversation_id: str, user_message: str
    ) -> dict[str, Any]:
        """
        Process one user message and return either a follow-up
        question or a final Legal_QA answer.

        Returns a dict suitable for JSON serialisation:

        Follow-up::

            {
                "type": "follow_up",
                "conversation_id": "...",
                "mode": "actionable",
                "message": "Which state ...?"
            }

        Final answer::

            {
                "type": "final_answer",
                "conversation_id": "...",
                "mode": "actionable",
                "answer": "...",
                "reasoning_chain": [...],
                "sources": [...]
            }
        """
        # 1. Load conversation state
        state = await self._repo.get(conversation_id)
        if state is None:
            raise ConversationNotFoundError(conversation_id)

        # 2. Add user message to history
        add_user_message(state, user_message)

        # 3. Analyze turn via intake service (OpenAI or rule-based fallback)
        intake_result = await self._intake.analyze_turn(state, user_message)

        # 4. Update extracted facts
        if intake_result.extracted_facts:
            update_facts(state, intake_result.extracted_facts)
            logger.info(
                "Extracted facts for %s: %s",
                conversation_id,
                intake_result.extracted_facts,
            )
        if intake_result.case_state:
            state.facts["case_state"] = intake_result.case_state

        # 5. Check if more information is needed
        if not intake_result.is_ready_for_qa:
            followup_question = intake_result.followup_question or (
                "Could you provide more details regarding your legal issue?"
            )
            add_assistant_message(state, followup_question)
            state.stage = ConversationStage.GATHERING_INFO
            await self._repo.update(state)

            return {
                "type": "follow_up",
                "conversation_id": state.conversation_id,
                "mode": state.mode.value,
                "message": followup_question,
                "case_state": state.facts.get("case_state"),
                "timestamp": state.messages[-1].timestamp,
            }

        # 6. Enough info — get synthesized query and call Legal_QA
        constructed_question = (
            intake_result.synthesized_query
            or self._query_builder.build(state)
        )
        logger.info(
            "Calling Legal_QA for %s: %s",
            conversation_id,
            constructed_question[:120],
        )

        state.stage = ConversationStage.READY_FOR_QA
        qa_response = await self._legal_qa.predict(
            question=constructed_question,
            mode=state.mode.value,
        )

        # 7. Store the result, format direct second-person address, and mark answered
        result = LegalQAResult(**qa_response)
        direct_answer = self._format_direct_answer(result.answer)
        result.answer = direct_answer
        store_legal_qa_result(state, result)
        add_assistant_message(state, direct_answer)
        state.last_assistant_question = None
        state.stage = ConversationStage.ANSWERED
        await self._repo.update(state)

        return {
            "type": "final_answer",
            "conversation_id": state.conversation_id,
            "mode": state.mode.value,
            "answer": direct_answer,
            "case_state": state.facts.get("case_state"),
            "reasoning_chain": result.reasoning_chain,
            "sources": [
                {"question": c.question, "answer": c.answer}
                for c in result.retrieved_cases
            ],
            "timestamp": state.messages[-1].timestamp,
        }

    @staticmethod
    def _format_direct_answer(raw_answer: str) -> str:
        """
        Transform third-person references ('the user', 'the user can') into
        direct, respectful second-person legal advice ('you', 'you can').
        """
        if not raw_answer:
            return raw_answer

        replacements = [
            (r"\bThe user can\b", "You can"),
            (r"\bthe user can\b", "you can"),
            (r"\bThe user should\b", "You should"),
            (r"\bthe user should\b", "you should"),
            (r"\bThe user is\b", "You are"),
            (r"\bthe user is\b", "you are"),
            (r"\bThe user was\b", "You were"),
            (r"\bthe user was\b", "you were"),
            (r"\bThe user has\b", "You have"),
            (r"\bthe user has\b", "you have"),
            (r"\bThe user must\b", "You must"),
            (r"\bthe user must\b", "you must"),
            (r"\bThe user cannot\b", "You cannot"),
            (r"\bthe user cannot\b", "you cannot"),
            (r"\bThe user may\b", "You may"),
            (r"\bthe user may\b", "you may"),
            (r"\bThe user's\b", "Your"),
            (r"\bthe user's\b", "your"),
            (r"\bthe user\b", "you"),
            (r"\bThe user\b", "You"),
        ]

        text = raw_answer
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)
        return text


# ── Exceptions ────────────────────────────────────────────────

class ConversationNotFoundError(Exception):
    """Raised when a conversation ID does not exist."""

    def __init__(self, conversation_id: str) -> None:
        self.conversation_id = conversation_id
        super().__init__(f"Conversation not found: {conversation_id}")
