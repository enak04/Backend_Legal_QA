"""
Conversation manager — the universal orchestrator.

Coordinates the complete legal intake, triage, research, and action-planning pipeline:

    User message
        → load state
        → check & resolve user corrections
        → add message to history
        → extract facts & spot legal issues
        → detect risk & urgency
        → determine legally important missing facts
        → check if sufficient information exists
            → NO: ask single highest-value question (with 'Why I'm asking this')
            → YES:
                → perform statutory research & precedent retrieval
                → analyze applicability & evidence gaps
                → generate source-grounded 8-part action plan
                → persist structured CaseState and LegalAssessment
"""

from __future__ import annotations

import logging
import re
from typing import Any

from conversation.cases.models import UniversalCaseState
from conversation.followup import FollowUpEngine
from conversation.state import (
    add_assistant_message,
    add_user_message,
    get_universal_case_state,
    resolve_user_correction,
    set_legal_assessment,
    set_universal_case_state,
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
from legal_qa.client import (
    LegalQAClient,
    LegalQATimeoutError,
    LegalQAUnavailableError,
)
from legal_qa.grounded_generator import (
    GroundedLegalAnswerGenerator,
    grounded_answer_generator,
)
from legal_qa.query_builder import QueryBuilder
from legal_qa.research import LegalResearchLayer, legal_research_layer
from services.intake_service import OpenAIIntakeService

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Universal orchestrator that coordinates a single legal consultation turn.
    """

    def __init__(
        self,
        repository: ConversationRepository,
        legal_qa_client: LegalQAClient,
        followup_engine: FollowUpEngine | None = None,
        query_builder: QueryBuilder | None = None,
        intake_service: OpenAIIntakeService | None = None,
        research_layer: LegalResearchLayer | None = None,
        answer_generator: GroundedLegalAnswerGenerator | None = None,
    ) -> None:
        self._repo = repository
        self._legal_qa = legal_qa_client
        self._followup = followup_engine or FollowUpEngine()
        self._query_builder = query_builder or QueryBuilder()
        self._intake = intake_service or OpenAIIntakeService(
            fallback_engine=self._followup,
            query_builder=self._query_builder,
        )
        self._research = research_layer or legal_research_layer
        self._answer_generator = answer_generator or grounded_answer_generator

    # ── Public API ────────────────────────────────────────────

    async def create_conversation(self, mode: Mode) -> ConversationRecord:
        """Start a new conversation with the given mode."""
        record = ConversationRecord(mode=mode)
        # Initialize universal case state
        initial_state = UniversalCaseState()
        set_universal_case_state(record, initial_state)
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
        Process one user message through the general-purpose legal intake,
        triage, research, and action-planning pipeline.
        """
        # 1. Load conversation state
        state = await self._repo.get(conversation_id)
        if state is None:
            raise ConversationNotFoundError(conversation_id)

        # 2. Check and resolve any user corrections / contradictions
        resolve_user_correction(state, user_message)

        # 3. Add user message to history
        add_user_message(state, user_message)

        # 4. Analyze turn via intake service
        intake_result = await self._intake.analyze_turn(state, user_message)

        # 5. Update extracted facts and structured case state
        if intake_result.extracted_facts:
            update_facts(state, intake_result.extracted_facts)
            logger.info(
                "Extracted facts for %s: %s",
                conversation_id,
                intake_result.extracted_facts,
            )
        if intake_result.case_state:
            state.facts["case_state"] = intake_result.case_state

        case_state = get_universal_case_state(state)

        # 6. Check if more information is needed
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

        # 7. Sufficient info gathered → Legal Research & Retrieval
        constructed_question = (
            intake_result.synthesized_query
            or self._query_builder.build(state)
        )
        logger.info(
            "Synthesized legal question for %s: %s",
            conversation_id,
            constructed_question[:120],
        )

        state.stage = ConversationStage.READY_FOR_QA

        # 7a. Retrieve relevant Indian statutory authorities
        statutory_authorities = self._research.research_authorities(case_state)

        # 7b. Query Legal_QA ML inference model for precedents and reasoning chain
        qa_response: dict[str, Any] = {
            "question": constructed_question,
            "answer": "",
            "reasoning_chain": [],
            "retrieved_cases": [],
        }

        try:
            qa_response = await self._legal_qa.predict(
                question=constructed_question,
                mode=state.mode.value,
            )
        except (LegalQATimeoutError, LegalQAUnavailableError):
            raise
        except Exception as exc:
            logger.warning("Legal_QA service call skipped or failed (%s); using grounded statutory synthesis.", exc)

        # 7c. Process precedents (context only, never user facts!)
        precedent_authorities = self._research.process_retrieved_cases(
            qa_response.get("retrieved_cases", []),
            case_state,
        )
        all_authorities = statutory_authorities + precedent_authorities

        # 8. Use Legal_QA backend answer (or generate grounded fallback if empty)
        legal_assessment = self._answer_generator.build_assessment(
            case_state=case_state,
            authorities=all_authorities,
        )
        if qa_response.get("answer"):
            direct_answer = self._format_direct_answer(qa_response["answer"])
        else:
            direct_answer, gen_assessment = await self._answer_generator.generate_answer(
                case_state=case_state,
                authorities=all_authorities,
                base_qa_answer=None,
            )
            if gen_assessment:
                legal_assessment = gen_assessment
            direct_answer = self._format_direct_answer(direct_answer)

        # 9. Persist result and structured assessment
        result = LegalQAResult(
            question=constructed_question,
            answer=direct_answer,
            reasoning_chain=qa_response.get("reasoning_chain", []),
            retrieved_cases=[
                {"question": c.get("question", ""), "answer": c.get("answer", "")}
                for c in qa_response.get("retrieved_cases", [])
            ],
        )
        store_legal_qa_result(state, result)
        set_legal_assessment(state, legal_assessment)
        add_assistant_message(state, direct_answer)
        state.last_assistant_question = None
        state.stage = ConversationStage.ANSWERED
        await self._repo.update(state)

        # Format sources: if retrieved cases exist, return them in sources for legacy API compatibility
        retrieved_cases = qa_response.get("retrieved_cases", [])
        if retrieved_cases:
            sources = [
                {"question": c.get("question", ""), "answer": c.get("answer", "")}
                for c in retrieved_cases
            ]
        else:
            sources = [
                {
                    "source": auth.source,
                    "provision": auth.provision,
                    "authority_type": auth.authority_type,
                    "jurisdiction": auth.jurisdiction,
                    "status": auth.status,
                    "excerpt": auth.key_excerpt,
                }
                for auth in statutory_authorities
            ]

        return {
            "type": "final_answer",
            "conversation_id": state.conversation_id,
            "mode": state.mode.value,
            "answer": direct_answer,
            "case_state": state.facts.get("case_state"),
            "legal_assessment": state.facts.get("legal_assessment"),
            "reasoning_chain": result.reasoning_chain,
            "sources": sources,
            "authorities": [auth.model_dump() for auth in all_authorities],
            "timestamp": state.messages[-1].timestamp,
        }

    @staticmethod
    def _format_direct_answer(raw_answer: str) -> str:
        """
        Transform third-person references ('the client', 'the user') into
        direct, respectful second-person legal advice ('you', 'your').
        """
        if not raw_answer:
            return raw_answer

        replacements = [
            (r"\bThe client can\b", "You can"),
            (r"\bthe client can\b", "you can"),
            (r"\bThe client should\b", "You should"),
            (r"\bthe client should\b", "you should"),
            (r"\bThe client is\b", "You are"),
            (r"\bthe client is\b", "you are"),
            (r"\bThe client was\b", "You were"),
            (r"\bthe client was\b", "you were"),
            (r"\bThe client has\b", "You have"),
            (r"\bthe client has\b", "you have"),
            (r"\bThe client must\b", "You must"),
            (r"\bthe client must\b", "you must"),
            (r"\bThe client cannot\b", "You cannot"),
            (r"\bthe client cannot\b", "you cannot"),
            (r"\bThe client may\b", "You may"),
            (r"\bthe client may\b", "you may"),
            (r"\bThe client's\b", "Your"),
            (r"\bthe client's\b", "your"),
            (r"\bto the client\b", "to you"),
            (r"\bfor the client\b", "for you"),
            (r"\bThe client\b", "You"),
            (r"\bthe client\b", "you"),
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
            (r"\bto the user\b", "to you"),
            (r"\bfor the user\b", "for you"),
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
