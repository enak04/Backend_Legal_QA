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
    set_language,
    set_legal_assessment,
    set_universal_case_state,
    store_legal_qa_result,
    update_facts,
)
from database.models import (
    ConversationRecord,
    ConversationStage,
    LegalQAResult,
    MessageRole,
    Mode,
)
from conversation.intent_router import TurnIntent, classify_turn_intent
from database.repository import ConversationRepository
from legal_qa.client import (
    LegalQAClient,
    LegalQAError,
    LegalQABadResponseError,
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
from services.language import (
    detect_language,
    is_indic_language,
    localize_system_question,
    normalize_language_code,
    resolve_turn_language,
)

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

    async def create_conversation(self, mode: Mode, language: str = "en") -> ConversationRecord:
        """Start a new conversation with the given mode and language."""
        norm_lang = normalize_language_code(language, default="en")
        record = ConversationRecord(mode=mode, language=norm_lang)
        # Initialize universal case state
        initial_state = UniversalCaseState(language=norm_lang)
        set_universal_case_state(record, initial_state)
        return await self._repo.create(record)

    async def get_conversation(
        self, conversation_id: str
    ) -> ConversationRecord | None:
        """Load a conversation by ID."""
        return await self._repo.get(conversation_id)

    async def list_conversations(
        self,
        limit: int = 50,
        offset: int = 0,
        search: str | None = None,
        stage: str | None = None,
        mode: str | None = None,
    ) -> tuple[list[ConversationRecord], int]:
        """List conversations with optional filtering, search, and pagination."""
        return await self._repo.list_all(
            limit=limit,
            offset=offset,
            search=search,
            stage=stage,
            mode=mode,
        )

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation by ID."""
        return await self._repo.delete(conversation_id)

    async def get_conversation_stats(self) -> dict[str, Any]:
        """Aggregate statistics across all conversations till date."""
        records, total_count = await self._repo.list_all(limit=10000, offset=0)
        
        total_messages = 0
        user_messages = 0
        assistant_messages = 0
        by_stage: dict[str, int] = {}
        by_mode: dict[str, int] = {}
        languages: dict[str, int] = {}
        file_count = 0

        for r in records:
            # Stage
            stage_val = r.stage.value if hasattr(r.stage, "value") else str(r.stage)
            by_stage[stage_val] = by_stage.get(stage_val, 0) + 1

            # Mode
            mode_val = r.mode.value if hasattr(r.mode, "value") else str(r.mode)
            by_mode[mode_val] = by_mode.get(mode_val, 0) + 1

            # Language
            lang = r.language or "en"
            languages[lang] = languages.get(lang, 0) + 1

            # Messages
            total_messages += len(r.messages)
            for m in r.messages:
                role = m.role.value if hasattr(m.role, "value") else str(m.role)
                if role == "user":
                    user_messages += 1
                else:
                    assistant_messages += 1

            # Files
            file_count += len(r.files)

        return {
            "total_conversations": total_count,
            "total_messages": total_messages,
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "total_files": file_count,
            "by_stage": by_stage,
            "by_mode": by_mode,
            "languages": languages,
        }

    async def process_message(
        self, conversation_id: str, user_message: str, language: str | None = None
    ) -> dict[str, Any]:
        """
        Process one user message through the general-purpose legal intake,
        triage, research, and action-planning pipeline.
        """
        # 1. Load conversation state
        state = await self._repo.get(conversation_id)
        if state is None:
            raise ConversationNotFoundError(conversation_id)

        # 1b. Update or detect language dynamically (supports switching languages mid-chat)
        if language and language.lower() not in ("auto", "none"):
            active_lang = normalize_language_code(language)
            set_language(state, active_lang)
        else:
            resolved_lang = resolve_turn_language(user_message, current_lang=state.language or "en")
            if resolved_lang != state.language:
                set_language(state, resolved_lang)

        # 2. Check and resolve any user corrections / contradictions
        resolve_user_correction(state, user_message)

        # 3. Add user message to history
        add_user_message(state, user_message)

        # 4. Analyze turn via intake service
        intake_result = await self._intake.analyze_turn(state, user_message)

        is_new_issue = getattr(intake_result, "is_new_legal_issue", False)
        turn_intent = classify_turn_intent(user_message, state, parsed_is_new_issue=is_new_issue)

        # 4a. Handle Conceptual Questions (educational / definition inquiries)
        if turn_intent == TurnIntent.CONCEPTUAL_QUESTION:
            direct_answer = (
                intake_result.direct_legal_answer
                or self._generate_conceptual_fallback(user_message)
            )
            add_assistant_message(state, direct_answer, original_content=intake_result.direct_legal_answer or direct_answer)
            state.last_assistant_question = None
            state.stage = ConversationStage.ANSWERED
            await self._repo.update(state)

            return {
                "type": "final_answer",
                "conversation_id": state.conversation_id,
                "mode": state.mode.value,
                "language": state.language,
                "answer": direct_answer,
                "original_answer": intake_result.direct_legal_answer or direct_answer,
                "message": direct_answer,
                "original_message": intake_result.direct_legal_answer or direct_answer,
                "case_state": state.facts.get("case_state"),
                "legal_assessment": state.facts.get("legal_assessment"),
                "reasoning_chain": [],
                "sources": [],
                "authorities": [],
                "timestamp": state.messages[-1].timestamp,
            }

        # 5. Update extracted facts and structured case state
        is_new_dispute = (turn_intent == TurnIntent.NEW_DISPUTE) or is_new_issue
        if is_new_dispute:
            # Set active case anchor index to this turn so subsequent queries are scoped strictly to this dispute
            user_turn_count = sum(1 for m in state.messages if m.role == MessageRole.USER)
            state.facts["active_case_anchor_idx"] = max(0, user_turn_count - 1)

            # Cleanly re-initialize dispute-specific facts so previous matter doesn't contaminate new dispute
            for k in [
                "legal_assessment", "core_issue", "employment_type", "salary",
                "written_contract", "dues_period", "amount", "amount_raw",
                "detected_domain", "city", "state", "summary"
            ]:
                state.facts.pop(k, None)

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
        if not intake_result.is_ready_for_qa and intake_result.followup_question:
            original_followup = intake_result.followup_question
            followup_question = original_followup
            active_lang = state.language or "en"
            if is_indic_language(active_lang):
                followup_question = localize_system_question(followup_question, active_lang)

            add_assistant_message(state, followup_question, original_content=original_followup)
            state.stage = ConversationStage.GATHERING_INFO
            await self._repo.update(state)

            return {
                "type": "follow_up",
                "conversation_id": state.conversation_id,
                "mode": state.mode.value,
                "language": state.language,
                "message": followup_question,
                "original_message": original_followup,
                "answer": followup_question,
                "case_state": state.facts.get("case_state"),
                "timestamp": state.messages[-1].timestamp,
            }

        # 7. Check if this is an ongoing conversational follow-up turn after initial advice
        has_prior_answers = bool(state.legal_qa_results or state.stage == ConversationStage.ANSWERED)
        conversational_reply = intake_result.direct_legal_answer or intake_result.followup_question
        if (turn_intent == TurnIntent.FOLLOWUP_ON_ADVICE or has_prior_answers) and not is_new_dispute:
            statutory_authorities = self._research.research_authorities(case_state)
            if conversational_reply:
                direct_answer = conversational_reply
            else:
                direct_answer = (
                    f"Regarding your question about '{user_message}': "
                    f"In connection with your {case_state.case_domain or 'legal'} dispute, you should adhere to the prescribed statutory timelines and preserve all documentation. "
                    "If you are drafting a legal notice or calculating dues, provide 15 to 30 days for compliance before initiating formal proceedings before the jurisdictional forum."
                )
            add_assistant_message(state, direct_answer, original_content=conversational_reply or direct_answer)
            state.last_assistant_question = None
            state.stage = ConversationStage.ANSWERED
            await self._repo.update(state)

            return {
                "type": "final_answer",
                "conversation_id": state.conversation_id,
                "mode": state.mode.value,
                "language": state.language,
                "answer": direct_answer,
                "original_answer": conversational_reply or direct_answer,
                "message": direct_answer,
                "case_state": state.facts.get("case_state"),
                "legal_assessment": state.facts.get("legal_assessment"),
                "reasoning_chain": [],
                "sources": [],
                "authorities": [auth.model_dump() for auth in statutory_authorities],
                "timestamp": state.messages[-1].timestamp,
            }

        # 8. Initial Advice Turn: Research & Precedent Retrieval
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
        except (LegalQAError, Exception) as exc:
            if state.mode == Mode.ACTIONABLE and (self._answer_generator.is_configured or intake_result.direct_legal_answer):
                logger.warning("Legal_QA service call failed or unavailable (%s); falling back to grounded statutory answer generator.", exc)
                qa_response = {
                    "question": constructed_question,
                    "answer": "",
                    "reasoning_chain": [],
                    "retrieved_cases": [],
                }
            else:
                raise

        # 7c. Process precedents (context only, never user facts!)
        precedent_authorities = self._research.process_retrieved_cases(
            qa_response.get("retrieved_cases", []),
            case_state,
        )
        all_authorities = statutory_authorities + precedent_authorities

        # 8. Determine final legal consultation response and preserve original English reply
        legal_assessment = self._answer_generator.build_assessment(
            case_state=case_state,
            authorities=all_authorities,
            mode=state.mode,
        )
        active_lang = state.language or "en"
        original_answer: str | None = qa_response.get("answer") or intake_result.direct_legal_answer

        if is_indic_language(active_lang) and self._answer_generator.is_configured:
            # Regional languages require full LLM synthesis via GroundedLegalAnswerGenerator
            # to guarantee the 6-part structure, bilingual statutory citations, and proper native vernacular.
            raw_direct, gen_assessment = await self._answer_generator.generate_answer(
                case_state=case_state,
                authorities=all_authorities,
                base_qa_answer=qa_response.get("answer") or intake_result.direct_legal_answer,
                language=active_lang,
                mode=state.mode,
            )
            if gen_assessment:
                legal_assessment = gen_assessment
            direct_answer = raw_direct
            if not original_answer:
                english_direct, _ = await self._answer_generator.generate_answer(
                    case_state=case_state,
                    authorities=all_authorities,
                    base_qa_answer=None,
                    language="en",
                    mode=state.mode,
                )
                original_answer = english_direct
        elif self._answer_generator.is_configured and not qa_response.get("answer"):
            raw_direct, gen_assessment = await self._answer_generator.generate_answer(
                case_state=case_state,
                authorities=all_authorities,
                base_qa_answer=intake_result.direct_legal_answer,
                language=active_lang,
                mode=state.mode,
            )
            if gen_assessment:
                legal_assessment = gen_assessment
            direct_answer = raw_direct
            original_answer = raw_direct
        elif qa_response.get("answer"):
            direct_answer = qa_response["answer"]
            original_answer = qa_response["answer"]
        elif intake_result.direct_legal_answer:
            direct_answer = intake_result.direct_legal_answer
            original_answer = intake_result.direct_legal_answer
        else:
            raw_direct, gen_assessment = await self._answer_generator.generate_answer(
                case_state=case_state,
                authorities=all_authorities,
                base_qa_answer=None,
                language=active_lang,
                mode=state.mode,
            )
            if gen_assessment:
                legal_assessment = gen_assessment
            direct_answer = raw_direct
            original_answer = raw_direct

        if not original_answer:
            original_answer = direct_answer

        # 9. Persist result and structured assessment
        result = LegalQAResult(
            question=constructed_question,
            answer=direct_answer,
            original_answer=original_answer,
            reasoning_chain=qa_response.get("reasoning_chain", []),
            retrieved_cases=[
                {"question": c.get("question", ""), "answer": c.get("answer", "")}
                for c in qa_response.get("retrieved_cases", [])
            ],
        )
        store_legal_qa_result(state, result)
        set_legal_assessment(state, legal_assessment)
        add_assistant_message(state, direct_answer, original_content=original_answer)
        state.facts["original_answer"] = original_answer
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
            "language": state.language,
            "answer": direct_answer,
            "original_answer": original_answer,
            "message": direct_answer,
            "original_message": original_answer,
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

    @classmethod
    def _enrich_answer_with_advocate_evidence(
        cls,
        answer: str,
        case_state: UniversalCaseState,
        evidence_assessment: dict[str, Any] | None,
    ) -> str:
        text = cls._format_direct_answer(answer)
        if not text:
            return text

        # Check government employment
        is_govt = False
        if isinstance(case_state.domain_extensions, dict):
            ext_emp = case_state.domain_extensions.get("employment", {})
            if isinstance(ext_emp, dict) and ext_emp.get("employment_type") == "government":
                is_govt = True
        if not is_govt:
            for f in case_state.known_facts:
                if "government" in str(f).lower() or "civil servant" in str(f).lower():
                    is_govt = True
                    break

        if is_govt:
            # Cleanse erroneous references to Industrial Disputes Act or Labour Courts for civil servants
            text = re.sub(
                r"\b(?:under\s+the\s+)?Industrial\s+Disputes\s+Act(?:,\s*1947)?",
                "under the Administrative Tribunals Act, 1985 (Section 19) and Article 311 of the Constitution of India",
                text,
                flags=re.I,
            )
            text = re.sub(
                r"\b(?:appropriate\s+)?labor\s+court\s+or\s+tribunal\b",
                "Central Administrative Tribunal (CAT) / State Administrative Tribunal or High Court under Article 226",
                text,
                flags=re.I,
            )
            text = re.sub(
                r"\blabor\s+department\b",
                "Appellate Authority under Service Rules or Central Administrative Tribunal",
                text,
                flags=re.I,
            )
            text = re.sub(
                r"\bConsumer\s+Protection\s+Act(?:,\s*2019)?\b",
                "Administrative Tribunals Act, 1985",
                text,
                flags=re.I,
            )

        # Append deep documentary evidence checklist & fallback strategy if not already present
        if evidence_assessment and ("fallback" not in text.lower() and "checklist" not in text.lower()):
            req_docs = evidence_assessment.get("required_documents_checklist") or []
            fallback = evidence_assessment.get("evidentiary_fallback_strategy") or []
            if req_docs or fallback:
                parts = [text.strip(), "\n\n### Evidentiary Strategy & Required Documents"]
                if req_docs:
                    parts.append("**Primary Documents to Gather**:")
                    for doc in req_docs[:4]:
                        parts.append(f"- {doc}")
                if fallback:
                    parts.append("\n**If Documents are Withheld or Missing (Evidentiary Fallback)**:")
                    for fb in fallback[:3]:
                        parts.append(f"- {fb}")
                text = "\n".join(parts)

        return text

    @staticmethod
    def _generate_conceptual_fallback(query: str) -> str:
        """Provide a clear, educational legal explanation for general legal queries."""
        clean = query.strip().lower()
        if "court" in clean:
            return (
                "In India, a court is a judicial institution established by law to resolve disputes, "
                "administer justice, and uphold statutory and constitutional rights. "
                "The judicial system is hierarchical: Civil Courts handle property, contract, and civil grievances; "
                "Criminal Courts (Magistrate and Sessions Courts) try penal offences; High Courts exercise writ and "
                "appellate jurisdiction in their respective States; and the Supreme Court of India serves as the apex constitutional forum."
            )
        if "severance" in clean or "notice pay" in clean:
            return (
                "Severance pay or notice pay in India refers to monetary compensation payable to an employee "
                "upon involuntary termination without sufficient statutory notice. Under Section 25F of the Industrial Disputes Act, 1947, "
                "and state Shops and Establishments Acts, an employer is generally required to provide 30 to 90 days' written notice "
                "or wages in lieu of notice, plus retrenchment compensation equivalent to 15 days' average pay for each completed year of service."
            )
        if "gratuity" in clean:
            return (
                "Under the Payment of Gratuity Act, 1972, gratuity is a statutory retirement benefit payable to employees "
                "who have rendered continuous service of at least 5 years in an establishment with 10 or more employees. "
                "It is calculated at 15 days' wages for every completed year of service, subject to the statutory ceiling."
            )
        if "fir" in clean:
            return (
                "A First Information Report (FIR) is a formal written document prepared by police upon receiving information "
                "about the commission of a cognizable offence (under Section 154 CrPC / Section 173 BNSS). "
                "It sets the criminal justice and investigative machinery into motion."
            )
        return (
            f"Regarding your legal inquiry '{query}': In Indian law, statutory rights and procedures are governed by specific enactments "
            "and procedural codes. Depending on the nature of the matter, remedies may lie through statutory representations, dispute resolution boards, "
            "or competent jurisdictional courts and tribunals."
        )


# ── Exceptions ────────────────────────────────────────────────

class ConversationNotFoundError(Exception):
    """Raised when a conversation ID does not exist."""

    def __init__(self, conversation_id: str) -> None:
        self.conversation_id = conversation_id
        super().__init__(f"Conversation not found: {conversation_id}")
