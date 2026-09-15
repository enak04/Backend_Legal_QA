"""
Intake service — OpenAI-powered universal conversational legal case intake.

Implements the general-purpose legal intake, triage, and action-planning pipeline:
  User message
  ↓
  Fact extraction & distinction (facts vs hypotheses)
  ↓
  Universal Case State
  ↓
  Issue identification (multiple simultaneous issues)
  ↓
  Risk/urgency detection
  ↓
  Determine legally important missing facts
  ↓
  Ask the single highest-value question with 'Why I'm asking this'
  ↓
  Update Case State & repeat only when necessary
  ↓
  Synthesize contextual query for legal research & action planning
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from conversation.cases.domains import domain_registry
from conversation.cases.models import (
    ActionTaken,
    CaseModule,
    ConfidenceScores,
    Dates,
    EvidenceItem,
    FactItem,
    Financial,
    Jurisdiction,
    LegalAssessment,
    LegalIssue,
    MissingFact,
    Party,
    Risk,
    StructuredCaseState,
    UniversalCaseState,
)
from conversation.cases.registry import CaseRegistry, case_registry
from conversation.followup import FollowUpEngine, extract_facts
from database.models import ConversationRecord, MessageRole, Mode
from legal_qa.query_builder import QueryBuilder
from services.risk_engine import risk_engine

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
# Universal Legal Intake System Prompt
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTAKE_BASE_SYSTEM_PROMPT = """You are an intelligent, empathetic, and comprehensive Legal AI Advisor for an Indian Legal Assistance system.
You handle ANY legal matter across the ENTIRE spectrum of Indian law (civil, criminal, constitutional, family, commercial, consumer, labor, tenancy, cyber, taxation, and administrative law).

### 1. CORE ARCHITECTURAL PRINCIPLE:
- You are NOT a rigid questionnaire or a fixed checklist.
- You must dynamically determine which questions matter for the specific case based on LEGAL INFORMATION VALUE:
  * Does a missing fact materially change:
    1. Applicable law or statutory regime? (e.g. private company vs government; commercial contract vs consumer dispute)
    2. Jurisdiction or forum? (State/City determining Rent Controller vs Civil Court vs High Court)
    3. Available remedies? (Injunction vs damages vs criminal FIR)
    4. Limitation period or deadline? (e.g. 30 days under NI Act; 2 years under Consumer Protection Act; 6 months under Specific Relief Act)
    5. Urgency or emergency relief? (Ongoing fraud, physical danger, lockout)
    6. Evidence requirements? (Written agreement, electronic logs, notice)
- Avoid asking questions whose answers would not materially change the legal analysis.

### 2. FACTS VS LEGAL CONCLUSIONS & HYPOTHESES (CRITICAL):
- Never treat an inferred legal conclusion as a user-provided fact!
  * If user says: "My company fired me without reason."
    - User-provided fact: "user was terminated without reasons given" (source: user, is_explicit: true)
    - Legal issue / hypothesis: "possible wrongful termination" (status: hypothesis, confidence: 0.7)
  * Clearly distinguish:
    - User-provided facts (explicit statements)
    - Extracted facts (dates, amounts, jurisdiction)
    - Legal issues / hypotheses (subject to verification of documents)
    - Actions already taken (e.g. filed FIR, contacted bank)
    - Evidence items

### 3. DYNAMIC ISSUE SPOTTER (MULTIPLE SIMULTANEOUS ISSUES):
- Spot multiple simultaneous issues from a single situation:
  * "My employer fired me and hasn't paid me for three months." → [unpaid wages, termination, possible wrongful termination]
  * "My landlord changed the locks and kept my deposit." → [possession / eviction dispute, security deposit recovery]
  * "Someone transferred ₹80,000 from my bank account without permission." → [unauthorized financial transaction, cyber fraud, banking dispute]

### 4. RISK & URGENCY DETECTION:
- Identify high-risk situations:
  * Imminent court/statutory deadline, arrest/detention, physical danger, eviction/lockout, ongoing financial fraud, destruction of evidence.
  * If an urgent risk is detected, acknowledge it warmly and explain the immediate emergency step before lengthy questioning.

### 5. CONVERSATION EFFICIENCY & QUESTION FORMULATION:
- Ask ONE focused question at a time.
- Include a concise "Why this matters" explanation when appropriate so the client understands the legal purpose.
- NEVER ask about information already established in previous turns or facts!
- Once sufficient core facts are known to provide a useful legal answer, STOP asking questions and set `is_ready_for_qa = true`.
- Always address the client directly in second person ("You", "Your employer", "Your rights").
"""


class OpenAIIntakeService:
    """
    Universal conversational legal intake engine powered by OpenAI with
    deterministic rule-based fallback.
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
        """True if an OpenAI API key is available."""
        return self._client is not None

    async def analyze_turn(
        self,
        state: ConversationRecord,
        latest_user_message: str,
    ) -> IntakeAnalysisResult:
        """Analyze a conversation turn using UniversalCaseState and dynamic ranking."""
        if not self.is_configured:
            return self._fallback_analyze(state, latest_user_message)

        try:
            return await self._analyze_with_openai(state, latest_user_message)
        except Exception as exc:
            logger.warning(
                "OpenAI intake analysis failed (%s); falling back to rule engine.",
                exc,
            )
            return self._fallback_analyze(state, latest_user_message)

    def _fallback_analyze(
        self,
        state: ConversationRecord,
        latest_user_message: str,
    ) -> IntakeAnalysisResult:
        """Deterministic fallback analysis using FollowUpEngine."""
        case_state_raw = state.facts.get("case_state")
        if isinstance(case_state_raw, UniversalCaseState):
            case_state = case_state_raw
        elif isinstance(case_state_raw, dict):
            case_state = UniversalCaseState(**case_state_raw)
        else:
            case_state = UniversalCaseState()

        # Update universal case state
        user_turn_count = sum(1 for m in state.messages if m.role == MessageRole.USER)
        self._fallback_engine.update_case_state_from_text(case_state, latest_user_message, user_turn_count)

        extracted = extract_facts(latest_user_message)
        if case_state.jurisdiction.state:
            extracted["state"] = case_state.jurisdiction.state
        if case_state.case_domain:
            extracted["detected_domain"] = case_state.case_domain

        is_ready = self._fallback_engine.is_sufficient_information(state, case_state)
        followup = None
        synthesized_query = None

        if is_ready:
            temp_state = state.model_copy(deep=True)
            temp_state.facts.update(extracted)
            temp_state.facts["case_state"] = case_state.model_dump()
            synthesized_query = self._query_builder.build(temp_state)
        else:
            followup = self._fallback_engine.select_highest_value_question(case_state)

        return IntakeAnalysisResult(
            extracted_facts=extracted,
            is_ready_for_qa=is_ready,
            followup_question=followup,
            synthesized_query=synthesized_query,
            case_state=case_state.model_dump(),
        )

    async def _analyze_with_openai(
        self,
        state: ConversationRecord,
        latest_user_message: str,
    ) -> IntakeAnalysisResult:
        """Perform OpenAI-powered universal fact extraction, issue spotting, and question ranking."""
        existing_case_state = state.facts.get("case_state") or {}

        # Retrieve relevant candidate taxonomy modules for context
        candidate_modules = self._registry.find_candidate_modules(
            text=latest_user_message,
            current_state=existing_case_state,
            limit=2,
        )

        system_prompt = self._build_dynamic_prompt(candidate_modules, state.mode)

        conversation_history = []
        for msg in state.messages:
            role = "user" if msg.role == MessageRole.USER else "assistant"
            conversation_history.append({"role": role, "content": msg.content})

        previous_assistant_questions = [
            m.content for m in state.messages if m.role == MessageRole.ASSISTANT
        ]
        user_turn_count = sum(1 for m in state.messages if m.role == MessageRole.USER)

        user_prompt_content = {
            "mode": state.mode.value,
            "user_turn_number": user_turn_count,
            "existing_facts": {
                k: v for k, v in state.facts.items() if k != "case_state"
            },
            "current_case_state": existing_case_state,
            "previous_assistant_questions": previous_assistant_questions,
            "conversation_history": conversation_history,
            "latest_user_message": latest_user_message,
            "guidelines": (
                "1. Distinguish facts vs legal hypotheses.\n"
                "2. Spot all legal issues (multiple simultaneous issues).\n"
                "3. Rank missing facts by legal information value.\n"
                "4. If Turn >= 3 or sufficient facts are established to advise on legal options, set is_ready_for_qa = true.\n"
                "5. Never ask about information already established."
            ),
        }

        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Analyze this conversation turn and output valid JSON according to schema:\n"
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
        """Assemble universal instructions and schema."""
        prompt_parts = [INTAKE_BASE_SYSTEM_PROMPT]

        if candidate_modules:
            prompt_parts.append("\n### REFERENCE DOMAIN PERSPECTIVE (For Context Only):")
            for mod in candidate_modules:
                prompt_parts.append(
                    f"- **{mod.case_type}** ({mod.subcategory}): {mod.description}\n"
                    f"  Key Facts: {', '.join(mod.relevant_facts[:4])}"
                )

        prompt_parts.append(
            """
### Output JSON Schema:
Respond ONLY with a valid JSON object matching this structure:
{
  "jurisdiction": {
    "country": "India",
    "state": "string or null",
    "district": "string or null",
    "city": "string or null"
  },
  "case": {
    "domain": "string (employment | consumer | property | criminal | cybercrime | family | contract | government | general)",
    "issues": [
      {
        "issue": "string",
        "domain": "string",
        "status": "hypothesis | confirmed",
        "confidence": float,
        "applicable_laws": ["string"]
      }
    ],
    "summary": "string"
  },
  "parties": [
    {
      "role": "string",
      "name_or_description": "string",
      "entity_type": "string or null"
    }
  ],
  "financial": {
    "amount": float or null,
    "amount_raw": "string or null",
    "currency": "INR",
    "loss": "string or null"
  },
  "dates": {
    "incident_date": "string or null",
    "notice_date": "string or null",
    "deadline": "string or null",
    "filing_date": "string or null"
  },
  "evidence": [
    {
      "type": "string",
      "description": "string",
      "source": "user_mentioned | uploaded | available | unavailable",
      "relevance": "string",
      "available": boolean
    }
  ],
  "actions_already_taken": [
    {
      "action": "string",
      "details": "string"
    }
  ],
  "user_goal": "string or null",
  "risk": {
    "level": "low | normal | potentially_urgent | urgent | emergency",
    "flags": ["string"],
    "reason": "string or null",
    "recommended_emergency_action": "string or null"
  },
  "missing_facts": [
    {
      "fact_key": "string",
      "description": "string",
      "legal_importance": "HIGH | MEDIUM | LOW",
      "reason": "string",
      "sample_question": "string"
    }
  ],
  "confidence": {
    "facts": float,
    "issue_classification": float,
    "jurisdiction": float,
    "legal_applicability": float,
    "urgency": float
  },
  "extracted_facts": {
    "detected_domain": "string",
    "state": "string or null",
    "core_issue": "string or null",
    "amount": "string or null",
    "duration_or_dates": "string or null"
  },
  "known_facts": [
    {
      "fact": "string",
      "source": "user_statement | extracted | inferred",
      "confidence": float,
      "is_explicit": boolean
    }
  ],
  "is_ready_for_qa": boolean,
  "followup_question": "string or null (If asking a question, provide warm legal validation first, then ask ONE question with 'Why this matters')",
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
        """Parse structured output into UniversalCaseState."""
        raw_facts = parsed.get("extracted_facts") or {}
        extracted_facts = {k: v for k, v in raw_facts.items() if v is not None and v != ""}

        # Build UniversalCaseState
        case_info = parsed.get("case") or {}
        domain = case_info.get("domain") or extracted_facts.get("detected_domain") or "general"
        extracted_facts["detected_domain"] = domain

        jur_data = parsed.get("jurisdiction") or {}
        if jur_data.get("state") and "state" not in extracted_facts:
            extracted_facts["state"] = jur_data["state"]

        raw_issues = case_info.get("issues", [])
        issues_list = []
        for i in raw_issues:
            if isinstance(i, dict):
                issues_list.append(
                    LegalIssue(
                        issue=i.get("issue", "Legal Dispute"),
                        domain=i.get("domain", domain),
                        status=i.get("status", "hypothesis"),
                        confidence=float(i.get("confidence", 0.7)),
                        applicable_laws=i.get("applicable_laws", []),
                    )
                )
            elif isinstance(i, str):
                issues_list.append(LegalIssue(issue=i, domain=domain))

        risk_data = parsed.get("risk") or {}
        urgency_val = parsed.get("urgency") or risk_data.get("level") or "normal"
        risk_obj = Risk(
            level=urgency_val,
            flags=risk_data.get("flags", []),
            reason=risk_data.get("reason"),
            recommended_emergency_action=risk_data.get("recommended_emergency_action"),
        )

        # Check case_classification for legacy & test compatibility
        cc = parsed.get("case_classification") or {}
        primary_category = cc.get("primary_category") or parsed.get("primary_category") or domain.title()
        subcategory = cc.get("subcategory") or parsed.get("subcategory")
        case_type = cc.get("specific_case_type") or cc.get("case_type") or parsed.get("case_type") or (issues_list[0].issue if issues_list else "Legal Inquiry")

        # Check parties format (dict vs list)
        raw_parties = parsed.get("parties", [])
        if isinstance(raw_parties, list):
            parties_val = [Party(**p) if isinstance(p, dict) and "name" in p else p for p in raw_parties]
        else:
            parties_val = raw_parties

        # Financial info
        raw_fin_info = parsed.get("financial_info") or {}
        fin_obj = Financial(**(parsed.get("financial") or {}))
        if isinstance(raw_fin_info, dict) and raw_fin_info.get("amount"):
            if not fin_obj.amount_raw:
                fin_obj.amount_raw = raw_fin_info["amount"]
        elif fin_obj.amount_raw:
            raw_fin_info["amount"] = fin_obj.amount_raw

        # Confidence (can be str or dict)
        confidence_val = cc.get("confidence") or parsed.get("confidence")
        if not confidence_val:
            confidence_val = ConfidenceScores()
        elif isinstance(confidence_val, dict):
            try:
                confidence_val = ConfidenceScores(**confidence_val)
            except Exception:
                pass

        # Build clean UniversalCaseState
        universal_state = UniversalCaseState(
            jurisdiction=Jurisdiction(
                country=jur_data.get("country", "India"),
                state=jur_data.get("state") or extracted_facts.get("state"),
                district=jur_data.get("district"),
                city=jur_data.get("city"),
            ),
            case_domain=domain,
            issues=issues_list,
            summary=case_info.get("summary"),
            financial=fin_obj,
            financial_info=raw_fin_info,
            dates=Dates(**(parsed.get("dates") or {})),
            parties=parties_val,
            evidence=[EvidenceItem(**e) for e in parsed.get("evidence", []) if isinstance(e, dict)],
            actions_already_taken=[ActionTaken(**a) for a in parsed.get("actions_already_taken", []) if isinstance(a, dict)],
            user_goal=parsed.get("user_goal"),
            risk=risk_obj,
            missing_facts=[MissingFact(**m) for m in parsed.get("missing_facts", []) if isinstance(m, dict)],
            confidence=confidence_val,
            known_facts=parsed.get("known_facts", []),
            primary_category=primary_category,
            subcategory=subcategory,
            case_type=case_type,
            urgency=urgency_val,
            possible_alternatives=cc.get("possible_alternatives", parsed.get("possible_alternatives", [])),
            related_case_types=cc.get("related_case_types", parsed.get("related_case_types", [])),
            missing_information=parsed.get("missing_information", []),
        )

        is_ready = bool(parsed.get("is_ready_for_qa", False))
        followup = parsed.get("followup_question")
        synthesized_query = parsed.get("synthesized_query")

        # Mode & Turn Safety Enforcement
        if state.mode in (Mode.READABLE, Mode.INFORMATIVE):
            is_ready = True
            followup = None
        elif state.mode == Mode.ACTIONABLE:
            user_msg_count = sum(1 for m in state.messages if m.role == MessageRole.USER)
            user_messages = [m.content.lower() for m in state.messages if m.role == MessageRole.USER]
            latest_msg = user_messages[-1] if user_messages else ""

            # Check if user demanded immediate advice
            user_demanded_advice = any(
                phrase in latest_msg
                for phrase in [
                    "skip question", "tell me what to do", "give me advice now",
                    "no more question", "just advise", "remedy now", "legal advice now",
                    "what should i do", "what are my options"
                ]
            )

            # Deduplication Guard
            previous_questions_lower = [
                m.content.strip().lower()
                for m in state.messages
                if m.role == MessageRole.ASSISTANT
            ]

            is_duplicate = False
            if followup:
                norm_followup = followup.strip().lower()
                q_clause = norm_followup
                if "?" in norm_followup:
                    q_clause = norm_followup.split("?")[-2].split(".")[-1].strip()

                stop_words = {
                    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
                    "have", "has", "had", "does", "would", "should", "could", "your", "you",
                    "about", "please", "case", "state", "share", "tell"
                }
                q_words = {
                    re.sub(r"[^\w]", "", w)
                    for w in q_clause.split()
                    if len(w) > 3 and re.sub(r"[^\w]", "", w) not in stop_words
                }

                for prev in previous_questions_lower:
                    if norm_followup in prev or prev in norm_followup:
                        is_duplicate = True
                        break
                    if "?" in prev:
                        prev_q_clause = prev.split("?")[-2].split(".")[-1].strip()
                        prev_words = {
                            re.sub(r"[^\w]", "", w)
                            for w in prev_q_clause.split()
                            if len(w) > 3 and re.sub(r"[^\w]", "", w) not in stop_words
                        }
                        shared = q_words & prev_words
                        if len(shared) >= 3 or (q_words and len(shared) / len(q_words) >= 0.5):
                            is_duplicate = True
                            break

            if is_duplicate:
                logger.warning("Duplicate question detected: '%s'. Overriding.", followup)
                if user_msg_count >= 2:
                    is_ready = True
                    followup = None
                else:
                    followup = self._fallback_engine.select_highest_value_question(universal_state)
                    is_ready = False

            elif user_demanded_advice or user_msg_count >= 3:
                # Turn cap / user request: deliver actionable legal guidance!
                is_ready = True
                followup = None

        if is_ready:
            followup = None
            if not synthesized_query:
                temp_state = state.model_copy(deep=True)
                temp_state.facts.update(extracted_facts)
                temp_state.facts["case_state"] = universal_state.model_dump()
                synthesized_query = self._query_builder.build(temp_state)

        return IntakeAnalysisResult(
            extracted_facts=extracted_facts,
            is_ready_for_qa=is_ready,
            followup_question=followup,
            synthesized_query=synthesized_query,
            case_state=universal_state.model_dump(),
        )

    # Backward compatibility alias
    _analyze_with_fallback = _fallback_analyze
