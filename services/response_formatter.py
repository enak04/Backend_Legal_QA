"""
Response formatting utilities.

Transforms internal data structures into the JSON shapes expected
by the frontend. Exposes compact structured case state for UI display
without exposing internal raw chain-of-thought.
"""

from __future__ import annotations

from typing import Any

from database.models import ConversationRecord


def format_conversation_response(
    record: ConversationRecord,
) -> dict[str, Any]:
    """
    Format a full conversation record for ``GET /api/conversations/{id}``.
    Returns raw messages, structured CaseState, and LegalAssessment as separate objects.
    """
    case_state = record.facts.get("case_state")
    compact_state = None

    if isinstance(case_state, dict):
        raw_issues = case_state.get("issues", [])
        active_issues = []
        for i in raw_issues:
            if isinstance(i, dict) and i.get("status") != "ruled_out":
                active_issues.append(i.get("issue"))
            elif isinstance(i, str):
                active_issues.append(i)

        if not active_issues and case_state.get("case_type"):
            active_issues = [case_state["case_type"]]

        financial_info = case_state.get("financial") or {}
        amt = financial_info.get("amount_raw") if isinstance(financial_info, dict) else None
        if not amt and isinstance(financial_info, dict) and financial_info.get("amount"):
            amt = f"₹{financial_info['amount']:,.0f}"

        compact_state = {
            "domain": case_state.get("case_domain") or case_state.get("subcategory") or case_state.get("primary_category"),
            "issues": active_issues,
            "jurisdiction": case_state.get("jurisdiction"),
            "urgency": case_state.get("urgency") or (case_state.get("risk", {}).get("level") if isinstance(case_state.get("risk"), dict) else "normal"),
            "amount": amt,
            "evidence_count": len(case_state.get("evidence", [])),
            "user_goal": case_state.get("user_goal"),
        }

    return {
        "conversation_id": record.conversation_id,
        "mode": record.mode.value,
        "stage": record.stage.value,
        "case_state": case_state,
        "compact_case_state": compact_state,
        "legal_assessment": record.facts.get("legal_assessment"),
        "facts": record.facts,
        "messages": [
            {
                "role": msg.role.value,
                "content": msg.content,
                "timestamp": msg.timestamp,
            }
            for msg in record.messages
        ],
        "legal_qa_results": [
            {
                "question": r.question,
                "answer": r.answer,
                "reasoning_chain": r.reasoning_chain,
                "sources": [
                    {"question": c.question, "answer": c.answer}
                    for c in r.retrieved_cases
                ],
            }
            for r in record.legal_qa_results
        ],
        "files": [
            {
                "file_id": f.file_id,
                "filename": f.filename,
                "content_type": f.content_type,
                "size_bytes": f.size_bytes,
                "uploaded_at": f.uploaded_at,
            }
            for f in record.files
        ],
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def format_modes_response() -> list[dict[str, str]]:
    """
    Return the list of available modes with descriptions for
    ``GET /api/modes``.
    """
    return [
        {
            "id": "actionable",
            "name": "Actionable",
            "description": (
                "Guides you toward an appropriate legal remedy or "
                "action. Collects relevant facts through a "
                "conversational intake process before providing "
                "actionable legal guidance."
            ),
        },
        {
            "id": "informative",
            "name": "Informative",
            "description": (
                "Helps you understand a legal issue, law, section, "
                "concept, or legal question. Provides direct, "
                "informative answers without unnecessary follow-up "
                "questions."
            ),
        },
        {
            "id": "readable",
            "name": "Readable",
            "description": (
                "Makes legal information easier for a normal user "
                "to understand. Provides simplified, accessible "
                "explanations of legal concepts and procedures."
            ),
        },
    ]
