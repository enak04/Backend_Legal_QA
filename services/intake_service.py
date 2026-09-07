"""
Intake service — OpenAI-powered modular conversational legal case intake.

Implements a dynamic, hierarchical case intake pipeline:
  1. Identifies candidate legal case types from the central registry
  2. Dynamically injects only relevant case module(s) into prompt context
  3. Maintains structured case state across multi-turn conversations
  4. Selects the single best next question, answers user interruptions,
     and provides conversational summaries once sufficient info is gathered.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from conversation.cases.models import CaseModule, StructuredCaseState
from conversation.cases.registry import CaseRegistry, case_registry
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
    case_state: dict[str, Any] | None = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Base Conversational Intake Instructions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTAKE_BASE_SYSTEM_PROMPT = """You are an intelligent, empathetic, and highly skilled Legal Intake Assistant for an Indian Legal Assistance system.
Your job is to have a natural conversation with the user to understand their legal problem, maintain structured case state, identify relevant legal case types, and ask ONLY the single most useful next question when more information is needed.

### CORE CONVERSATIONAL PRINCIPLES (STRICTLY ENFORCED):
1. **Never Behave Like a Form, Questionnaire, or Interrogator**:
   - NEVER output a checklist, numbered list of questions, or multi-field questionnaire.
   - Let the user explain their situation naturally in their own words.
   - Do NOT interrogate the user or demand all details at once.
   - Accept approximate information initially. Do not demand exact dates or documents immediately.
   - Avoid repetitive stock phrases like "I need more information" or repeated legal disclaimers.

2. **Single Best Next Question & NEVER Repeat Questions**:
   - Ask strictly ONE important question at a time.
   - You may ask two questions together ONLY if they are strongly related and doing so feels completely natural in conversation.
   - The question must build naturally on what the user just said.
   - NEVER ask for information the user has already provided.
   - NEVER ask the same question twice or rephrase an already-asked question.

3. **Gracefully Handle "I Don't Have It", "I Don't Know", or Negative/Unavailable Information (CRITICAL)**:
   - If the user says they don't have a document, don't have proof, don't know a date/detail, or simply cannot provide information:
     - **NEVER repeat or rephrase the question.**
     - **NEVER push, badger, or try to force the information out of the user.**
     - **Reassure the user warmly**: E.g., *"That is completely fine and very common—many agreements happen orally or on trust. In India, oral agreements are legally valid under the Indian Contract Act, and digital records like UPI transfers and WhatsApp chats can serve as evidence."*
     - **Record the information as resolved/unavailable**: E.g. set `written_agreement: "none"`, `exact_date: "unknown to user"`, and add a known fact (e.g., "No written contract exists; arrangement was oral").
     - **Remove the item permanently from `missing_information`**: It is not missing—it simply does not exist.
     - **MOVE FORWARD IMMEDIATELY**: Either ask about an entirely different topic (such as what the other person is saying now, or how much was transferred), or proceed directly to next steps if the main picture is clear.

4. **Handle Off-Topic, Tangential, or Emotional Statements Naturally**:
   - If the user goes off on a tangent, complains about someone, or answers something different from what you asked:
     - Acknowledge and engage empathetically with what they *actually* said.
     - Do NOT stubbornly pull them back or say "You didn't answer my question."
     - Flow with the conversation and gently address the situation as a human lawyer would.

5. **Acknowledge Information and Answer User Interruptions**:
   - If the user interrupts with a question (e.g., "What does limitation period mean?", "Can they arrest me?", "Can WhatsApp chats be used as evidence?"):
     - ANSWER their question first in a clear, accessible, and reassuring manner.
     - Only then, if needed, naturally transition back to the single most relevant next intake question.
   - Briefly acknowledge important information the user provides before asking the next question.

6. **Fact Extraction, Contradictions, and Corrections**:
   - Intelligently extract new facts, parties, timeline events, evidence, and financial amounts.
   - Distinguish where possible between: user personal knowledge, user suspicion/belief, third-party hearsay, and documented evidence.
   - If the user CORDS or CORRECTS earlier statements (e.g., "Actually it was ₹3 lakh, not ₹2 lakh", or "It happened in Delhi, not Mumbai"):
     - Immediately UPDATE the state with the corrected fact.
     - Do NOT keep contradictory information as equally valid.

5. **Dynamic Hierarchical Classification & Multiple Case Types**:
   - Maintain:
     - Primary Category (Civil, Criminal, Family, Other)
     - Subcategory (e.g., Property, Contract, Money Recovery, Theft, Fraud, Divorce, etc.)
     - Specific Case Type
     - Classification Confidence ("High", "Medium", "Low")
     - Possible Alternative Case Types
     - Related Case Types (e.g., Primary: Civil -> Property -> Ownership Dispute; Related: Civil -> Contract; Possible: Criminal -> Fraud).
   - Classification is revisable. Do NOT prematurely lock down the case type when facts are still unclear.

6. **Urgency Detection & Prioritization**:
   - Detect urgency indicators (e.g., received a court summons/notice with an imminent hearing date, imminent threat of illegal eviction or arrest, urgent cyber fraud reported within golden hour, or statutory deadline expiring).
   - If urgent, mark urgency as "urgent", acknowledge the immediate deadline/threat, and prioritize information needed to safeguard the user's rights.

7. **Sufficient Information Detection & Summary**:
   - The intake process must NOT go on forever.
   - When you have collected a reasonably clear understanding of:
     1. The central problem
     2. Main parties involved
     3. Key events and timeline
     4. Relevant agreement or dispute terms
     5. Major available evidence
     6. User's desired outcome
   - STOP asking further questions.
   - Provide a concise conversational summary to the user:
     "Let me make sure I have understood this correctly: [concise 2-3 sentence summary]. Is that an accurate summary?"
   - Once the user confirms the summary (or if all critical details are already completely clear):
     - Set `is_ready_for_qa` to `true`.
     - Set `followup_question` to `null`.
     - Produce a comprehensive `synthesized_query` detailing all facts, parties, jurisdiction, timeline, and relief sought for the downstream legal reasoning engine.

### GUIDELINES BY CONVERSATION MODE:
1. **ACTIONABLE** (Guide user toward actionable legal remedies & procedures):
   - Guide the intake conversation naturally using the principles above.
   - Use the active Case Module(s) below to know what information is important, but decide conversationally how and when to ask.
2. **INFORMATIVE** (Explain legal concepts/sections/general law):
   - If the user's query is reasonably clear, set `is_ready_for_qa` to `true` immediately and synthesize the query.
   - Only ask a follow-up if the user's message is completely vague or single-word.
3. **READABLE** (Simplified explanation):
   - Set `is_ready_for_qa` to `true` immediately without asking follow-up questions.
"""


class OpenAIIntakeService:
    """
    Intelligent conversational intake engine powered by OpenAI with
    modular hierarchical case routing and deterministic rule-based fallback.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        fallback_engine: FollowUpEngine | None = None,
        query_builder: QueryBuilder | None = None,
        registry: CaseRegistry | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._fallback_engine = fallback_engine or FollowUpEngine()
        self._query_builder = query_builder or QueryBuilder()
        self._registry = registry or case_registry
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
        Analyze the conversation turn to extract facts, route to modular case
        types, maintain structured state, and decide next action.
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
        """Call OpenAI with dynamically loaded case module instructions."""
        assert self._client is not None

        # 1. Identify Candidate Case Modules from conversation text & state
        all_user_messages = [
            m.content for m in state.messages if m.role == MessageRole.USER
        ]
        all_user_messages.append(latest_user_message)
        combined_text = " ".join(all_user_messages)

        existing_case_state = state.facts.get("case_state") or {}
        candidate_modules = self._registry.find_candidate_modules(
            text=combined_text,
            current_state=existing_case_state,
            limit=3,
        )

        # 2. Build Dynamic Prompt
        system_prompt = self._build_dynamic_prompt(
            candidate_modules=candidate_modules,
            mode=state.mode,
        )

        conversation_history = []
        for msg in state.messages:
            role = "user" if msg.role == MessageRole.USER else "assistant"
            conversation_history.append({"role": role, "content": msg.content})

        user_prompt_content = {
            "mode": state.mode.value,
            "existing_facts": {
                k: v for k, v in state.facts.items() if k != "case_state"
            },
            "current_case_state": existing_case_state,
            "conversation_history": conversation_history,
            "latest_user_message": latest_user_message,
            "turn_instructions": (
                "CRITICAL: Check conversation_history. If the latest_user_message indicates that the user "
                "does not have a document, does not know, or gave a negative or tangential response, "
                "DO NOT repeat or rephrase the previous question. Reassure the user, mark that fact as "
                "unavailable/none, remove it from missing_information, and move forward."
            ),
        }

        # 3. Call OpenAI Chat Completions with JSON response format
        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Analyze this conversation turn and output valid JSON:\n"
                        f"{json.dumps(user_prompt_content, indent=2)}"
                    ),
                },
            ],
        )

        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)

        return self._parse_openai_response(parsed, state)

    def _build_dynamic_prompt(
        self,
        candidate_modules: list[CaseModule],
        mode: Mode,
    ) -> str:
        """Assemble base instructions + only relevant case modules + schema."""
        prompt_parts = [INTAKE_BASE_SYSTEM_PROMPT]

        # Ingest only the relevant case module definitions
        if candidate_modules:
            prompt_parts.append("\n### DYNAMICALLY LOADED RELEVANT CASE MODULE(S):")
            for mod in candidate_modules:
                mod_lines = [
                    f"\n#### Case Type: {mod.category} -> {mod.subcategory} -> {mod.case_type}",
                    f"Description: {mod.description}",
                ]
                if mod.priority_info:
                    mod_lines.append(
                        f"Priority Information to Clarify: {', '.join(mod.priority_info)}"
                    )
                if mod.relevant_facts:
                    mod_lines.append(
                        f"Relevant Facts to Understand: {', '.join(mod.relevant_facts)}"
                    )
                if mod.potential_evidence:
                    mod_lines.append(
                        f"Potential Evidence: {', '.join(mod.potential_evidence)}"
                    )
                if mod.urgency_indicators:
                    mod_lines.append(
                        f"Urgency Indicators: {', '.join(mod.urgency_indicators)}"
                    )
                if mod.related_modules:
                    mod_lines.append(
                        f"Related Case Modules: {', '.join(mod.related_modules)}"
                    )
                prompt_parts.append("\n".join(mod_lines))

        prompt_parts.append(
            """
### Output JSON Schema:
Respond ONLY with a valid JSON object matching this structure:
{
  "case_classification": {
    "primary_category": "Civil | Criminal | Family | Other",
    "subcategory": "string",
    "specific_case_type": "string",
    "confidence": "High | Medium | Low",
    "possible_alternatives": ["string"],
    "related_case_types": ["string"]
  },
  "urgency": "normal | potentially_urgent | urgent",
  "parties": {
    "user": "string or null",
    "opposing_party": "string or null",
    "other_parties": "string or null"
  },
  "extracted_facts": {
    "detected_domain": "string (e.g. employment_wage, property_land, money_recovery, criminal, etc.)",
    "state": "string or null",
    "core_issue": "string or null",
    "amount": "string or null",
    "duration_or_dates": "string or null",
    "evidence_mentioned": "string or null"
  },
  "known_facts": [
    {
      "fact": "string",
      "source": "user_statement | documented | hearsay",
      "confidence": "high | medium | low",
      "confirmed": boolean,
      "topic": "string"
    }
  ],
  "timeline": [
    {
      "event": "string",
      "date": "string or null",
      "source": "string"
    }
  ],
  "evidence": [
    {
      "type": "string",
      "description": "string",
      "availability": "available | mentioned | pending",
      "supports": "string"
    }
  ],
  "financial_info": {
    "amount": "string or null",
    "currency": "INR",
    "loss": "string or null"
  },
  "communication": [
    {
      "details": "string",
      "admissions": "string or null"
    }
  ],
  "previous_actions": [
    {
      "action": "string",
      "details": "string"
    }
  ],
  "user_goal": "string or null",
  "missing_information": [
    "string"
  ],
  "is_ready_for_qa": boolean,
  "followup_question": "string or null (the single best question, answering interruptions if any, or summary for confirmation)",
  "synthesized_query": "string or null"
}
"""
        )
        return "\n".join(prompt_parts)

    def _parse_openai_response(
        self,
        parsed: dict[str, Any],
        state: ConversationRecord,
    ) -> IntakeAnalysisResult:
        """Extract structured case state and flat facts from OpenAI output."""
        raw_facts = parsed.get("extracted_facts") or {}
        # Clean null / empty strings
        extracted_facts = {
            k: v for k, v in raw_facts.items() if v is not None and v != ""
        }

        # Build structured case state
        classification = parsed.get("case_classification") or {}
        case_state = {
            "primary_category": classification.get("primary_category"),
            "subcategory": classification.get("subcategory"),
            "case_type": classification.get("specific_case_type"),
            "confidence": classification.get("confidence", "Medium"),
            "possible_alternatives": classification.get(
                "possible_alternatives", []
            ),
            "related_case_types": classification.get("related_case_types", []),
            "urgency": parsed.get("urgency", "normal"),
            "parties": parsed.get("parties", {}),
            "known_facts": parsed.get("known_facts", []),
            "timeline": parsed.get("timeline", []),
            "evidence": parsed.get("evidence", []),
            "financial_info": parsed.get("financial_info", {}),
            "communication": parsed.get("communication", []),
            "previous_actions": parsed.get("previous_actions", []),
            "user_goal": parsed.get("user_goal"),
            "missing_information": parsed.get("missing_information", []),
        }

        # Ensure legacy detected_domain key exists in extracted_facts if not present
        if "detected_domain" not in extracted_facts:
            cat = classification.get("primary_category", "").lower()
            subcat = classification.get("subcategory", "").lower()
            case_type = classification.get("specific_case_type", "").lower()

            if "employment" in subcat or "salary" in case_type:
                extracted_facts["detected_domain"] = "employment_wage"
            elif "property" in subcat or "land" in subcat or "tenant" in case_type:
                extracted_facts["detected_domain"] = "property_land"
            elif "money" in subcat or "money" in case_type:
                extracted_facts["detected_domain"] = "money_recovery"
            elif "criminal" in cat or "theft" in case_type or "fraud" in case_type:
                extracted_facts["detected_domain"] = "criminal"
            elif "family" in cat or "divorce" in case_type:
                extracted_facts["detected_domain"] = "family_matrimonial"
            elif "consumer" in subcat:
                extracted_facts["detected_domain"] = "consumer"
            else:
                extracted_facts["detected_domain"] = "general"

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
                temp_state = state.model_copy(deep=True)
                temp_state.facts.update(extracted_facts)
                temp_state.facts["case_state"] = case_state
                synthesized_query = self._query_builder.build(temp_state)
        else:
            if not followup:
                followup = (
                    "Could you share a few more details regarding what happened "
                    "so I can provide the most accurate legal guidance?"
                )

        return IntakeAnalysisResult(
            extracted_facts=extracted_facts,
            is_ready_for_qa=is_ready,
            followup_question=followup,
            synthesized_query=synthesized_query,
            case_state=case_state,
        )

    def _analyze_with_fallback(
        self,
        state: ConversationRecord,
        latest_user_message: str,
    ) -> IntakeAnalysisResult:
        """Deterministic rule-based fallback using CaseRegistry."""
        last_key = self._fallback_engine.get_last_question_key(state)
        new_facts = extract_facts(
            latest_user_message, last_question_key=last_key
        )

        all_user_messages = [
            m.content for m in state.messages if m.role == MessageRole.USER
        ]
        all_user_messages.append(latest_user_message)
        combined_text = " ".join(all_user_messages)

        existing_case_state = state.facts.get("case_state") or {}
        candidate_modules = self._registry.find_candidate_modules(
            text=combined_text,
            current_state=existing_case_state,
            limit=1,
        )

        # Build fallback structured state
        active_mod = candidate_modules[0] if candidate_modules else None
        urgency = "normal"
        lower_msg = latest_user_message.lower()
        if any(
            w in lower_msg
            for w in [
                "court notice",
                "hearing is next week",
                "hearing next week",
                "hearing tomorrow",
                "arrest",
                "summons",
            ]
        ):
            urgency = "urgent"

        case_state = {
            "primary_category": active_mod.category if active_mod else "Civil",
            "subcategory": active_mod.subcategory if active_mod else "General",
            "case_type": (
                active_mod.case_type
                if active_mod
                else "Unknown / Needs Further Classification"
            ),
            "confidence": "Medium",
            "urgency": urgency,
            "possible_alternatives": [],
            "related_case_types": (
                active_mod.related_modules if active_mod else []
            ),
        }

        # Create temporary updated state to test follow-up requirement
        temp_state = state.model_copy(deep=True)
        temp_state.facts.update(new_facts)
        temp_state.facts["case_state"] = case_state

        followup = self._fallback_engine.needs_followup(temp_state)

        if followup is not None:
            return IntakeAnalysisResult(
                extracted_facts=new_facts,
                is_ready_for_qa=False,
                followup_question=followup,
                synthesized_query=None,
                case_state=case_state,
            )

        # Ready for Legal_QA
        query = self._query_builder.build(temp_state)
        return IntakeAnalysisResult(
            extracted_facts=new_facts,
            is_ready_for_qa=True,
            followup_question=None,
            synthesized_query=query,
            case_state=case_state,
        )
