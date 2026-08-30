"""
Response formatting utilities.

Transforms internal data structures into the JSON shapes expected
by the frontend.  Keeps formatting logic out of the API routes
and conversation manager.
"""

from __future__ import annotations

from typing import Any

from database.models import ConversationRecord, MessageRole


def format_conversation_response(
    record: ConversationRecord,
) -> dict[str, Any]:
    """
    Format a full conversation record for ``GET /api/conversations/{id}``.

    Returns a dict that can be directly serialised to JSON.
    """
    return {
        "conversation_id": record.conversation_id,
        "mode": record.mode.value,
        "stage": record.stage.value,
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
