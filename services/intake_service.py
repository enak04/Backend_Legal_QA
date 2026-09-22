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

INTAKE_BASE_SYSTEM_PROMPT = """\
You are an expert, compassionate senior Indian Legal Intake Advocate conducting an initial client consultation in chambers.
You advise across all domains of Indian law (labor/employment, consumer, property/tenancy, cybercrime, contracts/money recovery, family, criminal, and constitutional/administrative law).

### 1. NATURAL, EMPATHETIC CONVERSATIONAL TONE (HUMAN ADVOCATE, NOT A ROBOT):
- Speak with the poise, warmth, and active listening of a seasoned advocate in private practice.
- NEVER sound like an automated questionnaire, bot, or government web form.
- STRICTLY FORBIDDEN: Do NOT use robotic meta-commentary such as:
  * "This information is crucial for determining your legal options."
  * "Please provide the following details to help me assist you."
  * "Why this matters is that..."
  * "As an AI legal assistant..."
- Ask ONE focused, natural question per turn. Never bombard the user with multiple disparate questions at once.
- Always acknowledge what the client just shared with genuine human understanding and empathy before asking your follow-up.

### 2. CORE FACTUAL PILLARS (WHAT SOUND LEGAL ADVICE REQUIRES ACROSS ALL DOMAINS):
Under Indian law, actionable legal remedies depend on 4 concrete factual pillars across any legal domain:
1. **Jurisdiction (Where)**: State and/or City. Territorial jurisdiction is MANDATORY because court benches, service tribunals (CAT benches, State Administrative Tribunals, High Court writ jurisdiction), Labour Courts, Rent Control Courts, and Consumer District Commissions are strictly territorial. NEVER conclude intake without establishing the State or City.
2. **Legal Relationship & Parties**: Government civil servant/PSU vs Private corporate employee vs workman under IDA; Tenant vs Licensee; Consumer vs Commercial entity; Creditor vs Debtor; Accused vs Complainant.
3. **Specific Breach / Actionable Grievance (What actually happened)**:
   - Do NOT stop at superficial confirmation that an event or document exists. Probe the substantive facts:
   - *Employment*: Stated ground for termination (misconduct, performance, redundancy, or no reason), whether an official written termination order or show-cause notice was served, or if termination was verbal/informal.
   - *Consumer*: The specific defect/failure in goods or deficiency in service, and what the merchant/service center refused.
   - *Property / Tenancy*: Type of eviction (lockout, utility cutoff, verbal threat) and lease agreement status.
   - *Contract / Debt*: Breach terms, default date, and whether demand notice was issued.
   - *Cybercrime*: Unauthorized debit mechanism, time elapsed, and whether 1930 / bank dispute was logged.
4. **Quantum & Stakes (How much / Evidence)**: Approximate monetary dues/loss, months of unpaid salary, security deposit amount, product price, or fraud quantum.

### 3. FACT-DRIVEN READINESS STANDARD (THINK LIKE A SENIOR ADVOCATE IN CHAMBERS):
- A real advocate analyzes DOCUMENTARY EVIDENCE deeply:
  * Primary documents required to prove the case: Appointment letter/service order, written termination/dismissal order, last 3-6 months pay slips, bank statements, demand notice/communications.
  * What if primary documents are MISSING, NOT PROVIDED, or WITHHELD by the employer or counterparty?
    - If employer/government withheld the termination order or issued verbal dismissal: Immediately serve a written representation/protest via Registered Post AD placing on record that the client reported to work and was turned away.
    - For Government / PSU employees: File an RTI application under Section 6 of the Right to Information Act, 2005 to obtain certified copies of the termination order, inquiry report, and Last Pay Certificate (LPC).
    - Secondary proof of employment and salary: Bank account statements showing regular salary credits and the sudden stoppage, EPFO UAN passbook, Form 16, and ESIC records under Section 114 Indian Evidence Act / Section 119 BSA 2023.
- If the client asks whether State matters (e.g. "does it not matter which state I am in?"):
  * Respond directly and authoritatively: Territorial jurisdiction determines the exact court bench, service tribunal (CAT/SAT), High Court, or Labour Court having authority over the employer's establishment.
  * Ask for their State/City and do NOT conclude intake until established.
- Set `is_ready_for_qa = true` ONLY when:
  * ALL core pillars (Jurisdiction State/City + Specific Grievance + Financial Quantum/Dues + Documentary status & Evidentiary fallback) are established.
  * OR the client explicitly demands immediate advice (e.g. "tell me what to do now", "give me advice now", "what are my options").
  * OR the client asks a conceptual legal question (e.g. "What is Section 138 NI Act?").
  * OR the client explicitly states they do not possess further details or documents.
- If high-priority legal questions remain in `unresolved_high_priority_legal_questions` (especially State/City, financial quantum, or written order status):
  * You MUST set `is_ready_for_qa = false`.
  * Formulate `followup_question` with advocate warmth: acknowledge their previous answer (e.g., "Understood, thank you for clarifying that you were employed in Mumbai."), then naturally ask the next essential question.
- Once ready:
  * Set `is_ready_for_qa = true` and `followup_question = null`.
  * Formulate a rich, professional legal brief in `synthesized_query` covering: Jurisdiction, Parties, Factual Chronology, Quantum/Dues, Primary Documents available, Documents missing/withheld + Evidentiary Fallback Strategy, and Specific Relief Sought.

### 4. FACTS VS LEGAL HYPOTHESES:
- Record client statements as known facts.
- Spot legal claims and statutory hypotheses with confidence ratings.
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
        previous_assistant_questions = []
        user_texts = []
        user_turn_count = 0
        for msg in state.messages:
            if isinstance(msg, dict):
                r_val = msg.get("role")
                role = "user" if r_val in (MessageRole.USER, "user") else "assistant"
                content = str(msg.get("content", ""))
            elif hasattr(msg, "role"):
                role = "user" if msg.role == MessageRole.USER else "assistant"
                content = str(msg.content)
            else:
                role = "user"
                content = str(msg)

            conversation_history.append({"role": role, "content": content})
            if role == "assistant":
                previous_assistant_questions.append(content)
            else:
                user_turn_count += 1
                user_texts.append(content)

        if latest_user_message and (not user_texts or latest_user_message != user_texts[-1]):
            user_texts.append(latest_user_message)
            user_turn_count += 1
        all_user_text = " ".join(user_texts)

        from conversation.followup import extract_facts
        deterministic_facts = extract_facts(all_user_text)

        # Inspect unresolved high-priority facts to guide intake
        if isinstance(existing_case_state, dict):
            try:
                curr_ucs = UniversalCaseState(**existing_case_state)
            except Exception:
                curr_ucs = UniversalCaseState()
        else:
            curr_ucs = existing_case_state or UniversalCaseState()

        # Update curr_ucs with deterministic facts
        if deterministic_facts.get("state") and not curr_ucs.jurisdiction.state:
            curr_ucs.jurisdiction.state = deterministic_facts["state"]
        if deterministic_facts.get("city") and not curr_ucs.jurisdiction.city:
            curr_ucs.jurisdiction.city = deterministic_facts["city"]
        if deterministic_facts.get("employment_type"):
            curr_ucs.domain_extensions.setdefault("employment", {})["employment_type"] = deterministic_facts["employment_type"]
            curr_ucs.known_facts.append({"fact": f"employment_type: {deterministic_facts['employment_type']}", "source": "user_statement"})
        if deterministic_facts.get("amount") and not curr_ucs.financial.amount:
            curr_ucs.financial.amount = deterministic_facts["amount"]
        if deterministic_facts.get("amount_raw") and not curr_ucs.financial.amount_raw:
            curr_ucs.financial.amount_raw = deterministic_facts["amount_raw"]
        if deterministic_facts.get("dues_period") and not curr_ucs.financial.dues_period:
            curr_ucs.financial.dues_period = deterministic_facts["dues_period"]
        if deterministic_facts.get("written_contract"):
            curr_ucs.domain_extensions.setdefault("employment", {})["written_contract"] = deterministic_facts["written_contract"]
            curr_ucs.known_facts.append({"fact": f"written_contract: {deterministic_facts['written_contract']}", "source": "user_statement"})

        missing_facts = self._fallback_engine.evaluate_missing_facts(curr_ucs, user_messages_text=all_user_text)
        high_missing = [
            {"fact_key": m.fact_key, "question": m.sample_question, "reason": m.reason}
            for m in missing_facts if m.legal_importance == "HIGH"
        ]

        user_prompt_content = {
            "mode": state.mode.value,
            "user_turn_number": user_turn_count,
            "existing_facts": {
                k: v for k, v in state.facts.items() if k != "case_state"
            },
            "current_case_state": curr_ucs.model_dump(),
            "unresolved_high_priority_legal_questions": high_missing,
            "previous_assistant_questions": previous_assistant_questions,
            "conversation_history": conversation_history,
            "latest_user_message": latest_user_message,
            "guidelines": (
                "1. Communicate with warmth, empathy, and professional poise like an experienced senior Indian advocate in a one-on-one chambers consultation.\n"
                "2. Ask ONE focused, natural question per turn. Never bombard the client with multiple disparate questions, questionnaires, or meta-explanations like 'Why this matters is...'\n"
                "3. Assess readiness substantively across ALL legal domains: A case is ready for advice (is_ready_for_qa = true) ONLY when you have established: (a) Jurisdiction (State/City), (b) Specific factual grievance and relationship context (e.g. stated reason for firing/notice pay; defect and merchant refusal; lockout/deposit terms), AND (c) Approximate quantum/duration of dues/loss or key documentary terms (unpaid salary, notice pay, deposit, purchase amount, contract clause). If 'unresolved_high_priority_legal_questions' contains critical unprobed questions (such as financial dues, unpaid salary, or agreement status), DO NOT set is_ready_for_qa = true! Instead, acknowledge what the client just shared with empathy, and ask ONE natural advocate's follow-up question addressing the most critical unresolved dimension.\n"
                "4. If the client's initial message is comprehensive with all necessary facts, resolve in Turn 1 immediately. If the client explicitly demands immediate advice or asks a conceptual legal question, set is_ready_for_qa = true immediately.\n"
                "5. When ready, formulate a rich, detailed legal brief in synthesized_query summarizing Jurisdiction, Parties, Factual Chronology, Quantum/Dues, and Specific Relief sought."
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
  "followup_question": "string or null (If asking a follow-up, provide empathetic legal acknowledgment first, then ask ONE natural, focused question probing the specific facts)",
  "synthesized_query": "string or null"
}
"""
        )
        return "\n".join(prompt_parts)

    @staticmethod
    def _is_question_duplicate(q_text: str, prev_questions: list[str]) -> bool:
        if not q_text or not prev_questions:
            return False
        norm_q = q_text.strip().lower()
        if "*(why this matters:" in norm_q:
            norm_q = norm_q.split("*(why this matters:")[0].strip()
        if "?" in norm_q:
            norm_q = norm_q.split("?")[-2].split(".")[-1].strip()

        stop_words = {
            "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
            "have", "has", "had", "does", "would", "should", "could", "your", "you",
            "about", "please", "case", "state", "share", "tell", "this", "that", "there",
            "determine", "applicable", "framework", "regime", "claim", "under", "statutory",
            "city", "legal", "court", "authority", "forum", "details"
        }
        q_words = {
            re.sub(r"[^\w]", "", w)
            for w in norm_q.split()
            if len(w) > 3 and re.sub(r"[^\w]", "", w) not in stop_words
        }

        for prev in prev_questions:
            norm_prev = prev.strip().lower()
            if "*(why this matters:" in norm_prev:
                norm_prev = norm_prev.split("*(why this matters:")[0].strip()
            if norm_q in norm_prev or norm_prev in norm_q:
                return True
            if "?" in norm_prev:
                norm_prev = norm_prev.split("?")[-2].split(".")[-1].strip()
            prev_words = {
                re.sub(r"[^\w]", "", w)
                for w in norm_prev.split()
                if len(w) > 3 and re.sub(r"[^\w]", "", w) not in stop_words
            }
            shared = q_words & prev_words
            if len(shared) >= 3 or (q_words and len(shared) / len(q_words) >= 0.6):
                return True
        return False

    def _is_fact_already_asked(self, fact: MissingFact, prev_questions: list[str]) -> bool:
        """Check if a missing fact dimension has already been queried in conversation."""
        if not prev_questions:
            return False
        if self._is_question_duplicate(fact.sample_question, prev_questions):
            return True
        key = fact.fact_key.lower()
        for prev in prev_questions:
            p = prev.lower()
            if key in ("employment_type", "employer_type") and any(w in p for w in ["private", "government", "psu", "organization"]):
                return True
            if "jurisdiction" in key and any(w in p for w in ["state", "city", "located", "stationed"]):
                return True
            if ("amount" in key or "dues" in key or "financial" in key) and any(w in p for w in ["dues", "salary", "unpaid", "amount", "gratuity"]):
                return True
            if "contract" in key and any(w in p for w in ["contract", "offer letter", "appointment letter", "agreement"]):
                return True
            if "possession" in key and any(w in p for w in ["locked out", "possession", "vacate"]):
                return True
        return False

    def _parse_openai_response(
        self,
        parsed: dict[str, Any],
        state: ConversationRecord,
    ) -> IntakeAnalysisResult:
        """Parse structured output into UniversalCaseState."""
        raw_facts = parsed.get("extracted_facts") or {}
        extracted_facts = {k: v for k, v in raw_facts.items() if v is not None and v != ""}

        # Collect all user texts across turns
        user_msg_count = 0
        user_messages = []
        previous_questions_lower = []
        for m in state.messages:
            m_role = getattr(m, "role", None)
            if isinstance(m, dict):
                m_role = m.get("role")
            m_content = str(getattr(m, "content", "")) if hasattr(m, "content") else (str(m.get("content", "")) if isinstance(m, dict) else str(m))
            if m_role in (MessageRole.USER, "user"):
                user_msg_count += 1
                user_messages.append(m_content)
            elif m_role in (MessageRole.ASSISTANT, "assistant"):
                previous_questions_lower.append(m_content.strip().lower())

        all_user_text = " ".join(user_messages)
        from conversation.followup import extract_facts
        deterministic_facts = extract_facts(all_user_text)
        for k, v in deterministic_facts.items():
            if k not in extracted_facts or not extracted_facts[k]:
                extracted_facts[k] = v

        # Build UniversalCaseState
        case_info = parsed.get("case") or {}
        existing_domain = state.facts.get("detected_domain") or (
            state.facts.get("case_state", {}).get("case_domain")
            if isinstance(state.facts.get("case_state"), dict)
            else None
        )
        domain = case_info.get("domain") or extracted_facts.get("detected_domain") or existing_domain or "general"
        extracted_facts["detected_domain"] = domain

        jur_data = parsed.get("jurisdiction") or {}
        if extracted_facts.get("state") and not jur_data.get("state"):
            jur_data["state"] = extracted_facts["state"]
        if extracted_facts.get("city") and not jur_data.get("city"):
            jur_data["city"] = extracted_facts["city"]
        elif jur_data.get("state") and "state" not in extracted_facts:
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

        # Merge known deterministic facts into universal_state
        if extracted_facts.get("employment_type"):
            universal_state.domain_extensions.setdefault("employment", {})["employment_type"] = extracted_facts["employment_type"]
            universal_state.known_facts.append({"fact": f"employment_type: {extracted_facts['employment_type']}", "source": "user_statement"})
        if extracted_facts.get("written_contract"):
            universal_state.domain_extensions.setdefault("employment", {})["written_contract"] = extracted_facts["written_contract"]
            universal_state.known_facts.append({"fact": f"written_contract: {extracted_facts['written_contract']}", "source": "user_statement"})
        if extracted_facts.get("amount"):
            universal_state.financial.amount = extracted_facts["amount"]
        if extracted_facts.get("amount_raw"):
            universal_state.financial.amount_raw = extracted_facts["amount_raw"]
        if extracted_facts.get("dues_period"):
            universal_state.financial.dues_period = extracted_facts["dues_period"]
            universal_state.known_facts.append({"fact": f"dues_period: {extracted_facts['dues_period']}", "source": "user_statement"})

        is_ready = bool(parsed.get("is_ready_for_qa", False))
        followup = parsed.get("followup_question")
        synthesized_query = parsed.get("synthesized_query")

        # Mode & Turn Safety Enforcement
        if state.mode in (Mode.READABLE, Mode.INFORMATIVE):
            is_ready = True
        elif state.mode == Mode.ACTIONABLE:
            latest_msg = user_messages[-1].lower() if user_messages else ""

            user_demanded_advice = any(
                phrase in latest_msg
                for phrase in [
                    "skip question", "tell me what to do", "give me advice now",
                    "no more question", "just advise", "remedy now", "legal advice now",
                    "what should i do", "what are my options"
                ]
            )

            # Detect if the client asked whether their state/location matters
            user_asked_state_relevance = bool(
                re.search(
                    r"\b(does\s+it\s+not\s*matter|does\s+(the\s+)?state\s+matter|matter\s+which\s+state|why\s+(does\s+)?state\s+matter|does\s+it\s+matter\s+where|which\s+state\s+i\s+am\s+in)\b",
                    latest_msg,
                    re.I,
                )
            )

            has_jurisdiction = bool(
                universal_state.jurisdiction.state
                or universal_state.jurisdiction.city
                or extracted_facts.get("state")
                or extracted_facts.get("city")
            )

            has_employment_type = bool(
                extracted_facts.get("employment_type")
                or (isinstance(universal_state.domain_extensions, dict) and universal_state.domain_extensions.get("employment", {}).get("employment_type"))
                or any(
                    kw in all_user_text.lower()
                    for kw in [
                        "private company", "private firm", "private tech", "startup", "mnc", "corporate",
                        "tech firm", "it company", "government", "govt", "psu", "civil servant", "public sector"
                    ]
                )
            )

            # Re-evaluate missing facts with updated universal_state and user messages
            missing_after = self._fallback_engine.evaluate_missing_facts(universal_state, user_messages_text=all_user_text)
            high_priority_missing = [m for m in missing_after if m.legal_importance == "HIGH"]

            is_duplicate = False
            if followup:
                is_duplicate = self._is_question_duplicate(followup, previous_questions_lower)

            # Build unasked list: high priority facts for this domain that have not yet been queried
            unasked: list[MissingFact] = [
                m for m in high_priority_missing
                if not self._is_fact_already_asked(m, previous_questions_lower)
            ]

            # PROGRAMMATIC READINESS & ADVOCATE JURISDICTION GATE:
            if user_asked_state_relevance:
                is_ready = False
                followup = (
                    "Yes, territorial jurisdiction matters critically under Indian law. The exact legal forum, court bench, and statutory authority where your petition or claim must be filed—such as the jurisdictional bench of the Central Administrative Tribunal (CAT), State Administrative Tribunal, High Court under Article 226, or the local Labour Court/Authority—depends strictly on the State and City where you were posted or employed. Which State and City were you employed or stationed in?"
                )
                synthesized_query = None
            elif user_demanded_advice or state.mode in (Mode.INFORMATIVE, Mode.READABLE):
                is_ready = True
                followup = None
            else:
                if followup and not is_duplicate and not parsed.get("is_ready_for_qa"):
                    # LLM asked a valid, non-duplicate conversational follow-up question — respect it!
                    is_ready = False
                elif not has_jurisdiction:
                    # Strict Advocate Invariant: Cannot provide final actionable remedies without territorial jurisdiction
                    is_ready = False
                    domain_def = domain_registry.get(universal_state.case_domain)
                    followup = domain_def.get_jurisdiction_question() if domain_def else "To determine the proper legal forum and applicable state laws, could you please confirm which State or City you are located in?"
                    synthesized_query = None
                elif unasked and not user_demanded_advice:
                    # Universal Schema Gate: If any HIGH pillar from the domain manifest is missing, ask it!
                    is_ready = False
                    top_missing = unasked[0]
                    followup = top_missing.sample_question
                    synthesized_query = None
                elif parsed.get("is_ready_for_qa") and parsed.get("synthesized_query"):
                    is_ready = True
                    followup = None
                else:
                    is_ready = True
                    followup = None

        if is_ready:
            followup = None
            jurisdiction_str = universal_state.jurisdiction.state or extracted_facts.get("state") or "India"
            city_str = universal_state.jurisdiction.city or extracted_facts.get("city") or ""
            loc_full = f"{city_str}, {jurisdiction_str}".strip(", ")
            core_issue = universal_state.case_type or extracted_facts.get("core_issue") or (universal_state.issues[0].issue if universal_state.issues else "legal dispute")
            raw_amt = universal_state.financial.amount
            formatted_amt = f"₹{raw_amt:,.0f}" if isinstance(raw_amt, (int, float)) else str(raw_amt or "")
            dues_str = universal_state.financial.amount_raw or (formatted_amt if formatted_amt else universal_state.financial.dues_period) or ""
            initial_statement = user_messages[0] if user_messages else ""

            domain_def = domain_registry.get(universal_state.case_domain)

            # Check if government employee
            is_govt_emp = False
            ext_emp = universal_state.domain_extensions.get("employment", {}) if isinstance(universal_state.domain_extensions, dict) else {}
            if isinstance(ext_emp, dict) and ext_emp.get("employment_type") == "government":
                is_govt_emp = True
            elif any("government" in str(f).lower() for f in universal_state.known_facts):
                is_govt_emp = True

            fact_bullets = [
                f"  - Jurisdiction: {loc_full}",
                f"  - Core Issue: {core_issue}",
            ]
            if dues_str:
                fact_bullets.append(f"  - Monetary Quantum / Dues: {dues_str}")
            if is_govt_emp:
                fact_bullets.append("  - Employment Sector: Government Civil Servant / Public Post")
                fact_bullets.append("  - Required Documentation & Strategy: Appointment order, written dismissal/inquiry order, and pay slips. If withheld, enforce requisition under Section 6 RTI Act 2005 and secondary proof via Bank salary credits under Section 114 Evidence Act / Section 119 BSA 2023.")
                fact_bullets.append("  - Forum Jurisdiction: Central Administrative Tribunal (CAT) / State Administrative Tribunal under Administrative Tribunals Act 1985 Section 19 or High Court under Article 226.")

            for k, v in extracted_facts.items():
                if k not in {"state", "city", "core_issue", "detected_domain", "amount", "amount_raw", "dues_period", "employment_type"}:
                    fact_bullets.append(f"  - {k.replace('_', ' ').title()}: {v}")

            facts_block = "\n".join(fact_bullets)
            if not synthesized_query or len(synthesized_query.strip()) < 120 or (is_govt_emp and "Tribunal" not in synthesized_query):
                if is_govt_emp:
                    statute_focus = "Administrative Tribunals Act 1985 (Section 19), Article 311 & Article 226 of the Constitution of India, and Payment of Wages / Gratuity provisions"
                elif domain_def and domain_def.primary_statutes:
                    statute_focus = ", ".join(domain_def.primary_statutes[:3])
                else:
                    statute_focus = "relevant Indian statutory provisions and judicial forums"

                synthesized_query = (
                    f"Client's legal concern: {initial_statement}\n\n"
                    f"Relevant details established:\n"
                    f"{facts_block}\n\n"
                    f"Spotted legal issues:\n"
                    f"  - {core_issue}\n\n"
                    f"Address the recipient directly as 'you' in second person. What direct actionable "
                    f"legal remedies, filing procedures, evidentiary fallback strategies, and {statute_focus} apply?"
                )

        return IntakeAnalysisResult(
            extracted_facts=extracted_facts,
            is_ready_for_qa=is_ready,
            followup_question=followup,
            synthesized_query=synthesized_query,
            case_state=universal_state.model_dump(),
        )

    # Backward compatibility alias
    _analyze_with_fallback = _fallback_analyze
