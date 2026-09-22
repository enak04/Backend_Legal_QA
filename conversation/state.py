"""
Conversation state helpers.

Manages updates to :class:`ConversationRecord` and :class:`UniversalCaseState`.
Supports fact updates, user corrections & contradiction resolution, evidence tracking,
and maintains clear separation between raw messages, structured state, and legal assessments.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any

from conversation.cases.models import (
    ActionTaken,
    EvidenceItem,
    FactItem,
    LegalAssessment,
    UniversalCaseState,
)
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


def get_universal_case_state(record: ConversationRecord) -> UniversalCaseState:
    """Retrieve typed UniversalCaseState from record.facts."""
    raw = record.facts.get("case_state")
    if isinstance(raw, UniversalCaseState):
        return raw
    if isinstance(raw, dict):
        try:
            return UniversalCaseState(**raw)
        except Exception:
            domain = raw.get("case_domain") or record.facts.get("detected_domain") or "general"
            jur_raw = raw.get("jurisdiction", {}) if isinstance(raw.get("jurisdiction"), dict) else {}
            return UniversalCaseState(
                case_domain=domain,
                jurisdiction=Jurisdiction(
                    country=jur_raw.get("country", "India"),
                    state=jur_raw.get("state") or record.facts.get("state"),
                    city=jur_raw.get("city") or record.facts.get("city"),
                ),
                primary_category=raw.get("primary_category") or domain.title(),
                case_type=raw.get("case_type") or record.facts.get("core_issue"),
            )
    domain = record.facts.get("detected_domain") or "general"
    return UniversalCaseState(
        case_domain=domain,
        jurisdiction=Jurisdiction(
            country="India",
            state=record.facts.get("state"),
            city=record.facts.get("city"),
        ),
        primary_category=domain.title(),
    )


def set_universal_case_state(record: ConversationRecord, case_state: UniversalCaseState) -> None:
    """Store typed UniversalCaseState inside record.facts."""
    record.facts["case_state"] = case_state.model_dump()


def get_legal_assessment(record: ConversationRecord) -> LegalAssessment | None:
    """Retrieve typed LegalAssessment if generated."""
    raw = record.facts.get("legal_assessment")
    if isinstance(raw, LegalAssessment):
        return raw
    if isinstance(raw, dict):
        try:
            return LegalAssessment(**raw)
        except Exception:
            pass
    return None


def set_legal_assessment(record: ConversationRecord, assessment: LegalAssessment) -> None:
    """Store LegalAssessment inside record.facts."""
    record.facts["legal_assessment"] = assessment.model_dump()


def resolve_user_correction(
    record: ConversationRecord,
    correction_text: str,
) -> bool:
    """
    Handle user corrections and contradiction resolution (e.g. 'No, I resigned, not fired').
    Marks conflicting facts as superseded, updates fact state, and re-evaluates dependent issues.
    """
    corr_lower = correction_text.lower()
    case_state = get_universal_case_state(record)
    modified = False

    # Check for resignation vs termination correction
    if re.search(r"\b(not fired|didn't fire|i resigned|resignation|willingly left)\b", corr_lower):
        # Supersede termination facts
        for f in case_state.known_facts:
            fact_str = f.get("fact", "") if isinstance(f, dict) else str(f)
            if "fired" in fact_str.lower() or "terminated" in fact_str.lower():
                if isinstance(f, dict):
                    f["superseded"] = True
                modified = True

        # Re-evaluate issues: rule out wrongful termination, add resignation dues recovery
        for issue in case_state.issues:
            if "termination" in issue.issue.lower():
                issue.status = "ruled_out"

        case_state.known_facts.append(
            FactItem(
                id=uuid.uuid4().hex[:8],
                fact=f"Correction: User tendered resignation / was not fired ({correction_text})",
                category="employment",
                source="user_correction",
                confidence=1.0,
                is_explicit=True,
                turn=len(record.messages),
            ).model_dump()
        )
        record.facts["employment_status"] = "resigned"
        modified = True

    # Check for date / financial corrections
    amt_match = re.search(r"\b(actually|not (\d+),? but|it is|amount is)\s*(?:rs\.?|₹)?\s*([\d,]+)", corr_lower)
    if amt_match:
        new_amt_str = amt_match.group(3).replace(",", "")
        try:
            new_amt = float(new_amt_str)
            case_state.financial.amount = new_amt
            case_state.financial.amount_raw = f"₹{new_amt:,.0f}"
            record.facts["amount"] = new_amt
            modified = True
        except ValueError:
            pass

    if modified:
        set_universal_case_state(record, case_state)

    return modified
