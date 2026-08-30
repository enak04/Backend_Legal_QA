"""
Intake service — OpenAI-powered dynamic conversational fact extraction.

Extracts legal facts across multi-turn conversations, determines if enough
details have been collected based on mode, and generates empathetic follow-up
questions or a complete synthesized query for Legal_QA.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from conversation.followup import FollowUpEngine, extract_facts
from database.models import ConversationRecord, MessageRole, Mode
from legal_qa.query_builder import QueryBuilder

logger = logging.getLogger(__name__)


@dataclass
class IntakeAnalysisResult:
    """Result of analyzing a conversation turn during intake."""

    extracted_facts: dict[str, Any] = field(default_factory=dict)
    is_ready_for_qa: bool = False
    followup_question: str | None = None
    synthesized_query: str | None = None


INTAKE_SYSTEM_PROMPT = """You are an expert Legal Intake Assistant for an Indian Legal Assistance system.
Your job is to analyze the conversation between a user and the assistant, conversationally extract all relevant legal facts, determine whether sufficient details have been gathered to formulate an accurate legal inquiry, and either generate an empathetic, context-aware follow-up question or synthesize a complete query for the downstream Legal QA inference service.

### Guidelines by Conversation Mode:

1. **ACTIONABLE** (Guide user toward actionable legal remedies & procedures):
   - A good legal query needs key factual context:
     - Legal domain/category (e.g. Employment, Property/Tenancy, Criminal, Matrimonial/Family, Consumer, Cybercrime, Contract).
     - Jurisdiction/State/Location in India (essential because state laws/procedures vary).
     - Parties involved and their relationship (e.g., employee vs private employer, landlord vs tenant, buyer vs seller).
     - Core dispute details & timeline/duration (e.g., 6 months unpaid salary, illegal eviction notice, cyber fraud amount).
     - Documentation or formal steps taken (e.g., employment contract, rent agreement, FIR, legal notice, consumer complaint).
     - Specific outcome or remedy the user is seeking (e.g., recover salary, stay eviction, get refund).
   - If critical information (especially domain, state/location, key party details, or core facts) is still missing and not yet clarified:
     - Set `is_ready_for_qa` to `false`.
     - Provide a natural, empathetic, and clear `followup_question` asking for 1 or 2 missing details. Acknowledge what the user has said so far rather than acting like a rigid form.
   - If the user has provided enough basic context (or has answered follow-ups, or indicates they don't know further details):
     - Set `is_ready_for_qa` to `true`.
     - Set `followup_question` to `null`.
     - Produce a comprehensive `synthesized_query` that clearly summarizes the user's situation, all facts collected, and asks what legal remedies, relevant laws/acts, and procedural steps apply.

2. **INFORMATIVE** (Help user understand a legal concept or general law):
   - If the user's question is reasonably clear (even without exhaustive case specifics):
     - Set `is_ready_for_qa` to `true` and synthesize the query.
   - Only set `is_ready_for_qa` to `false` if the question is completely vague or incomprehensible (e.g., single-word input like "help" or "law").

3. **READABLE** (Simple, accessible legal explanation):
   - Set `is_ready_for_qa` to `true` immediately and synthesize the query clearly and simply without asking follow-up questions.

### Output JSON Format:
Respond ONLY with a valid JSON object with the following keys:
{
  "extracted_facts": {
    "detected_domain": "...",
    "state": "...",
    "parties": "...",
    "employment_type": "...",
    "property_type": "...",
    "amount": "...",
    "duration_or_dates": "...",
    "documents": "...",
    "prior_actions": "...",
    "key_events": "...",
    "desired_relief": "..."
  },
  "is_ready_for_qa": boolean,
  "followup_question": "string or null",
  "synthesized_query": "string or null"
}
"""


class OpenAIIntakeService:
    """
    Intelligent conversational intake engine powered by OpenAI with
    deterministic rule-based fallback.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        fallback_engine: FollowUpEngine | None = None,
        query_builder: QueryBuilder | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._fallback_engine = fallback_engine or FollowUpEngine()
        self._query_builder = query_builder or QueryBuilder()
        self._client: AsyncOpenAI | None = None

        if self._api_key:
            self._client = AsyncOpenAI(api_key=self._api_key)

    @property
    def is_configured(self) -> bool:
        """Return True if OpenAI API client is configured."""
        return self._client is not None

    async def analyze_turn(
        self,
        state: ConversationRecord,
        latest_user_message: str,
    ) -> IntakeAnalysisResult:
        """
        Analyze the conversation turn to extract facts and decide next action.

        If OpenAI is enabled and reachable, uses LLM intake.
        Otherwise falls back to the rule-based engine.
        """
        if self.is_configured:
            try:
                return await self._analyze_with_openai(state, latest_user_message)
            except Exception as exc:
                logger.warning(
                    "OpenAI intake analysis failed (%s); falling back to rule engine.",
                    exc,
                )

        return self._analyze_with_fallback(state, latest_user_message)

    async def _analyze_with_openai(
        self,
        state: ConversationRecord,
        latest_user_message: str,
    ) -> IntakeAnalysisResult:
        """Call OpenAI to extract facts and determine intake completion."""
        assert self._client is not None

        # Build message history payload
        conversation_history = []
        for msg in state.messages:
            role = "user" if msg.role == MessageRole.USER else "assistant"
            conversation_history.append({"role": role, "content": msg.content})

        user_prompt_content = {
            "mode": state.mode.value,
            "existing_facts": state.facts,
            "conversation_history": conversation_history,
            "latest_user_message": latest_user_message,
        }

        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": INTAKE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Analyze this conversation state and provide JSON output:\n"
                        f"{json.dumps(user_prompt_content, indent=2)}"
                    ),
                },
            ],
        )

        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)

        raw_facts = parsed.get("extracted_facts") or {}
        # Clean out null or empty string values from extracted_facts
        extracted_facts = {
            k: v for k, v in raw_facts.items() if v is not None and v != ""
        }

        is_ready = bool(parsed.get("is_ready_for_qa", False))
        followup = parsed.get("followup_question")
        synthesized_query = parsed.get("synthesized_query")

        # Mode safety enforcement
        if state.mode == Mode.READABLE:
            is_ready = True
            followup = None

        if is_ready:
            followup = None
            if not synthesized_query:
                # Merge existing and newly extracted facts for building query
                temp_state = state.model_copy(deep=True)
                temp_state.facts.update(extracted_facts)
                synthesized_query = self._query_builder.build(temp_state)
        else:
            if not followup:
                followup = (
                    "Could you share a few more details about your situation "
                    "so I can provide the most accurate legal guidance?"
                )

        return IntakeAnalysisResult(
            extracted_facts=extracted_facts,
            is_ready_for_qa=is_ready,
            followup_question=followup,
            synthesized_query=synthesized_query,
        )

    def _analyze_with_fallback(
        self,
        state: ConversationRecord,
        latest_user_message: str,
    ) -> IntakeAnalysisResult:
        """Deterministic rule-based fallback when OpenAI is not available."""
        last_key = self._fallback_engine.get_last_question_key(state)
        new_facts = extract_facts(
            latest_user_message, last_question_key=last_key
        )

        # Create temporary updated state to test follow-up requirement
        temp_state = state.model_copy(deep=True)
        temp_state.facts.update(new_facts)

        followup = self._fallback_engine.needs_followup(temp_state)

        if followup is not None:
            return IntakeAnalysisResult(
                extracted_facts=new_facts,
                is_ready_for_qa=False,
                followup_question=followup,
                synthesized_query=None,
            )

        # Ready for Legal_QA
        query = self._query_builder.build(temp_state)
        return IntakeAnalysisResult(
            extracted_facts=new_facts,
            is_ready_for_qa=True,
            followup_question=None,
            synthesized_query=query,
        )
