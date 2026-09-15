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
import re
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

INTAKE_BASE_SYSTEM_PROMPT = """You are an intelligent, empathetic, and comprehensive Legal AI Assistant for an Indian Legal Assistance system.
You are capable of handling ANY legal question or scenario across the ENTIRE spectrum of Indian law (civil, criminal, constitutional, family, commercial, consumer, labor, tenancy, cyber, taxation, and general legal inquiries).

### 1. BROAD SPECTRUM VERSATILITY (CRITICAL):
- **Not a Narrow Case-by-Case Form**: Do NOT treat conversations as rigid, narrow case silos where you must check off a fixed list of questions.
- **Universal Scope**: Users may ask broad questions, conceptual legal questions, procedural questions, or share complex overlapping situations.
- **Provide Legal Value in Every Response**:
  - When the user asks a question or shares their situation, DO NOT just ask a counter-question.
  - Provide immediate, substantive legal clarity first—briefly explain the applicable law, rights, or legal position under Indian statutes (e.g., Industrial Disputes Act, Indian Contract Act, Negotiable Instruments Act, Consumer Protection Act, etc.).
  - Then, if more specifics are needed for tailored action, ask ONE natural, relevant follow-up question.
- **Adapt to Broad vs. Specific**:
  - If the user asks a general question (e.g., "What are my rights if my company fires me?", "Can police arrest without warrant?"): Answer the legal question clearly and comprehensively!
  - If the user describes a dispute: Acknowledge their rights, outline the legal framework, and ask about their specific situation naturally.

### 2. CRITICAL IDENTITY & STRICTLY PROHIBITED RESPONSES:
- **YOU ARE THE LEGAL COUNSEL/ASSISTANCE PLATFORM**:
  - The user is here SPECIFICALLY to receive legal guidance, remedies, and action plans from this platform.
  - **ABSOLUTELY FORBIDDEN**: NEVER tell the user to "seek legal advice", "consult an attorney", "contact a lawyer", or "seek professional advice". They are ALREADY here consulting this service!
  - **ABSOLUTELY FORBIDDEN**: NEVER ask naive, passive, or patronizing questions like:
    - ❌ "Have you considered reaching out to your employer for clarification?"
    - ❌ "Have you tried talking to the other party to work it out?"
    - ❌ "Have you considered seeking legal advice?"
  - If someone was terminated, cheated, or faced default, they are here for concrete legal recourse. Address the legal aspects directly.

### 3. CONVERSATIONAL FLOW & NEVER FORCE INFORMATION:
- **Never Interrogate or Badger**:
  - If the user says "I don't know", "I don't have it", "No proof", "He just said you are fired nothing else", "No clauses", or gives a short/curt reply:
    - **NEVER repeat, rephrase, or probe the same topic again.**
    - Accept it as a definitive, established fact (e.g., "No verbal agreements or discussions occurred; termination was unilateral and oral", "No termination clauses exist in contract").
    - Remove the item permanently from `missing_information`—it is not missing, it simply does not exist.
    - Reassure the user warmly and acknowledge the legal implications under Indian law.
    - Move forward immediately to a new aspect or proceed to remedies.
- **Single Best Next Question & No Duplicates**:
  - Ask at most ONE natural question at a time.
  - **STRICTLY PROHIBITED**: NEVER ask any question that has already been asked, rephrase an asked question, or re-open an already answered topic.
  - If the user pivots, shares an emotional reaction, or reports serious violations (e.g., racial discrimination, harassment, sudden eviction, or withholding of wages):
    - Acknowledge and validate the legal significance of that violation under Indian law (e.g. Karnataka Shops & Establishments Act, 1961 Section 39; Payment of Wages Act; Constitutional protections).
    - Do NOT ignore it or jump back to an irrelevant standard question.

### 4. CONVERSATION MODES & RESOLUTION:
- **ACTIONABLE MODE (Conversational Legal Intake & Remedies)**:
  - In this mode, the user chose to have a CONVERSATION.
  - Your primary goal is interactive conversational discovery: understanding the scenario, explaining rights/statutes, clarifying missing critical facts, and helping them formulate an action plan.
  - In initial turns (Turns 1-3), `is_ready_for_qa` MUST BE `false` as long as key facts remain unclarified.
  - In `followup_question`, ALWAYS provide immediate substantive legal context first (reassurance, rights under Indian law), and then ask ONE natural, relevant next question.
  - **TURN CAP & RESOLUTION (CRITICAL)**: 
    - Do NOT keep asking questions indefinitely! Once the user has answered 3 to 4 turns, or once the core facts (who, what, state/jurisdiction, amount/unpaid wages, and termination circumstance) are established:
    - STOP asking further questions.
    - Set `is_ready_for_qa = true`.
    - Synthesize a comprehensive query detailing all facts, timeline, statutory violations, and desired relief to deliver the complete actionable legal remedy.
  - Also set `is_ready_for_qa = true` whenever the user explicitly states they have provided all info or asks for the final advice/remedy.
- **INFORMATIVE MODE**:
  - Explain legal concepts or sections. Set `is_ready_for_qa = true` if the question is reasonably clear.
- **READABLE MODE**:
  - Simplified plain-language legal explanation. Always set `is_ready_for_qa = true`.

### 5. LAWYER-GRADE TONE & DIRECT SECOND-PERSON ADDRESS (CRITICAL):
- **Address the Client Directly**:
  - ALWAYS use direct second-person address ("You", "Your employer", "Your rights", "Your contract").
  - NEVER speak in the third person (NEVER say "the user", "the client", or "the employee").
- **Structure of Every Response in `followup_question`**:
  - Speak with the empathy, gravitas, and strategic clarity of an expert legal advocate.
  - In each turn:
    1. **Empathetic Acknowledgment & Reassurance**: Validate their situation warmly.
    2. **Substantive Legal Framework & Rights**: Explicitly explain the relevant Indian statutes, protections, or legal positions (e.g. Karnataka Shops & Establishments Act, Payment of Wages Act, NI Act Section 138, BNSS/CrPC, Consumer Protection Act 2019). Explain *why* certain details matter.
    3. **One Focused Question**: Ask ONE clear, natural question to clarify crucial missing facts, evidence (contracts, payslips, bank statements, notices), or their desired remedy.

### 6. HUMAN-CENTRIC & IMPERFECT INPUT UNDERSTANDING (CRITICAL):
- **Understand that it is a Human Talking and Messages May Not Be Perfect**:
  - The person consulting you is a human experiencing real-world stress or legal trouble, not a lawyer or a machine.
  - Their messages may contain typos, colloquialisms, emotional venting, or incomplete fragments (e.g., "Karnataka and 3 months", "he just said fired nothing else", "no clauses written", "they scammed me", "blocked my number", "paid rent by upi want him out of my life").
  - **NEVER penalize, critique, or get stuck on short, informal, emotional, or imperfect replies.**
  - **A Real Lawyer Connects the Dots**:
    - If a client says *"He just said you are fired nothing else"*: Recognize immediately that no written termination notice, no show-cause letter, and no verbal discussions took place. That makes the termination arbitrary and without due process under Indian labor laws!
    - If a client says *"No clauses written"*: Recognize that the contract is silent on notice periods and termination procedures, meaning statutory rules and default protections govern their rights.
    - If a client says *"Karnataka and 3 months"*: Immediately extract that the matter is under Karnataka jurisdiction and arrears amount to 3 months.
    - If a client says *"Paid rent on time via UPI every month"*: Immediately extract that they have clear digital proof of tenancy and absence of default.
    - If a client says *"Blocked my number"*: Recognize deliberate evasion or bad-faith refusal to communicate.
  - **Humans Do Not Speak Like Checklists**:
    - Clients frequently answer only one part of a question, or pivot to what is burning in their mind (e.g. sharing proof of payment or expressing their urgent fear).
    - **DO NOT repeat the question just because the client didn't answer it directly.**
    - Acknowledge what they *did* share, extract whatever legal fact is implied, and move the case forward.
  - **Never Demand Legal Jargon**: If they describe being cheated or having a bounced cheque, classify the offense under the relevant law (Section 138 of NI Act, BNS / IPC for cheating) without asking them to name the legal statute.
  - **Client Stating Their Goal or Asking for Relief = Ready for Solutions**:
    - When a client says *"I want to stop him from evicting me"*, *"I want to recover my salary"*, *"What legal action can I take?"*, or *"Outline the steps I should take"*, they are explicitly asking for legal remedies.
    - Transition immediately: set `is_ready_for_qa = true` to deliver full, actionable legal remedies.
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

        previous_assistant_questions = [
            m.content for m in state.messages if m.role == MessageRole.ASSISTANT
        ]
        user_turn_count = sum(1 for m in state.messages if m.role == MessageRole.USER)

        if state.mode == Mode.ACTIONABLE:
            mode_instruction = (
                f"5. ACTIVE MODE IS ACTIONABLE (CONVERSATIONAL INTAKE, User Turn #{user_turn_count}):\n"
                "- REMEMBER: THE CLIENT IS A HUMAN TALKING, NOT A MACHINE OR LAWYER. Their messages may not be perfect—they may use fragments, typos, or emotional expressions (e.g., 'paid rent by UPI every month, want him to stop', 'he just said fired nothing else', 'karnataka and 3 months').\n"
                "  - Connect the dots from what they said: infer legal implications without asking them to clarify minor details or restate themselves.\n"
                "  - If the client did not answer a specific previous question and instead spoke about something else (e.g. payment history or desired remedy), NEVER repeat the previous question. Pivot to their stated point.\n"
                "- ABSOLUTELY FORBIDDEN: You must NEVER repeat, rephrase, or ask about any topic already covered in 'previous_assistant_questions': "
                f"{json.dumps(previous_assistant_questions)}.\n"
                "- CLIENT GOAL / DESIRED REMEDY = READY FOR RESOLUTION: If the client stated their goal (e.g. 'I want to stop him', 'I want my deposit protected', 'recover my salary', 'tell me what to do'), or if User Turn >= 3 and core facts are established, "
                "STOP asking questions! Set is_ready_for_qa = true and synthesize the full legal query for complete actionable remedies.\n"
                "- Only if User Turn < 3, no goal was just stated, and critical facts are genuinely missing, ask ONE single new question on an unasked topic."
            )
        else:
            mode_instruction = f"5. ACTIVE MODE IS {state.mode.value.upper()}."

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
            "turn_instructions": (
                "CRITICAL INSTRUCTIONS:\n"
                "1. BROAD SPECTRUM OF QUESTIONS: This system handles ANY legal question across the entire spectrum of Indian law (general legal questions, concepts, rights, procedures, or dispute cases). "
                "ALWAYS provide immediate substantive legal clarity, rights, and relevant provisions first before asking any question.\n"
                "2. YOU ARE THE LEGAL PLATFORM. NEVER tell the user to 'seek legal advice', 'consult an attorney', or 'hire a lawyer'.\n"
                "3. NEVER ask naive, patronizing questions like 'Have you considered asking your employer/other party for clarification?'.\n"
                "4. If the user indicates they don't have a document, don't know a detail, or replied with an off-topic/negative response, "
                "DO NOT repeat or rephrase the previous question. Reassure the user, record it as unavailable, and move forward.\n"
                f"{mode_instruction}"
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

        # Ingest only the relevant case module definitions as background reference
        if candidate_modules:
            prompt_parts.append(
                "\n### RELEVANT LEGAL DOMAIN REFERENCE CONTEXT (NOT a questionnaire or checklist):\n"
                "(Use these reference domains to understand legal elements and rights under Indian law. "
                "Do NOT quiz or interrogate the user with these fields. Answer the user's questions first and converse naturally.)"
            )
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
  "followup_question": "string or null (A complete, professional lawyer consultation response addressed directly to the client ('You'). First provide empathetic validation and substantive legal context citing relevant Indian statutory provisions and rights, then ask ONE natural, focused follow-up question to clarify documentation, evidence, or relief sought)",
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
        elif state.mode == Mode.ACTIONABLE:
            # Actionable mode is a conversational intake flow.
            user_msg_count = sum(
                1 for m in state.messages if m.role == MessageRole.USER
            )
            has_missing = bool(case_state.get("missing_information"))

            # Deduplication Guard: Check if followup question repeats any previous question
            previous_questions_lower = [
                m.content.strip().lower()
                for m in state.messages
                if m.role == MessageRole.ASSISTANT
            ]

            is_duplicate = False
            if followup:
                norm_followup = followup.strip().lower()

                # Extract the main question clause (the part around or before '?')
                q_clause = norm_followup
                if "?" in norm_followup:
                    q_clause = norm_followup.split("?")[-2].split(".")[-1].strip()

                # Extract significant content words from the question clause
                stop_words = {
                    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
                    "have", "has", "had", "having", "does", "done", "doing", "would", "should", "could",
                    "your", "yours", "yourself", "yourselves", "you", "about", "above", "across",
                    "after", "again", "against", "along", "already", "also", "although", "among",
                    "around", "because", "before", "behind", "below", "beside", "between",
                    "both", "during", "each", "either", "else", "enough", "even", "ever", "every",
                    "from", "further", "here", "into", "just", "like", "more", "most", "much",
                    "must", "near", "never", "only", "other", "our", "ours", "please", "regarding",
                    "same", "some", "such", "than", "that", "their", "theirs", "them", "then",
                    "there", "these", "they", "this", "those", "through", "under", "until",
                    "very", "well", "were", "what", "with", "within", "without", "can", "may",
                    "confirm", "considered", "strengthen", "case", "position"
                }
                q_words = {
                    re.sub(r"[^\w]", "", w)
                    for w in q_clause.split()
                    if len(w) > 3 and re.sub(r"[^\w]", "", w) not in stop_words
                }

                for prev in previous_questions_lower:
                    # 1. Full text overlap
                    if norm_followup in prev or prev in norm_followup:
                        is_duplicate = True
                        break

                    # 2. Main question clause overlap
                    if len(q_clause) > 15 and q_clause in prev:
                        is_duplicate = True
                        break

                    if "?" in prev:
                        prev_q_clause = prev.split("?")[-2].split(".")[-1].strip()
                        if len(prev_q_clause) > 15 and (prev_q_clause in norm_followup or q_clause in prev_q_clause):
                            is_duplicate = True
                            break

                        prev_words = {
                            re.sub(r"[^\w]", "", w)
                            for w in prev_q_clause.split()
                            if len(w) > 3 and re.sub(r"[^\w]", "", w) not in stop_words
                        }
                        shared_words = q_words & prev_words
                        if len(shared_words) >= 3 or (q_words and len(shared_words) / len(q_words) >= 0.5):
                            is_duplicate = True
                            break

                    # 3. Key phrase overlap checks
                    for kw in [
                        "verbal agreement",
                        "verbal discussions",
                        "written contract",
                        "appointment letter",
                        "which state",
                        "unpaid salary",
                        "messages or emails",
                        "documented any",
                        "documenting any",
                        "threats",
                    ]:
                        if kw in norm_followup and kw in prev:
                            is_duplicate = True
                            break
                    if is_duplicate:
                        break

            if is_duplicate:
                logger.warning(
                    "Detected duplicate follow-up question: '%s'. Overriding duplicate.",
                    followup,
                )
                if user_msg_count >= 2:
                    # User has answered across 2+ turns; finish intake and provide remedies!
                    is_ready = True
                    followup = None
                else:
                    followup = (
                        "Understood. What specific relief or outcome are you looking to achieve "
                        "(e.g., recovering your unpaid salary, seeking severance compensation, or sending a formal legal notice)?"
                    )
                    is_ready = False
            elif user_msg_count >= 4:
                # Turn cap: after 4 user messages, conclude intake to prevent interrogation loops
                is_ready = True
                followup = None
            elif followup:
                is_ready = False
            elif user_msg_count < 3 and has_missing:
                is_ready = False
                followup = (
                    "Could you share a few more details regarding what happened "
                    "so I can provide the most accurate legal guidance?"
                )

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
