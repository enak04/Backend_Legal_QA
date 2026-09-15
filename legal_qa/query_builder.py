"""
Query builder — constructs a coherent Legal_QA question from
accumulated conversation state.

Instead of sending only the latest user message, this module
combines:
  - the original problem statement
  - all collected facts
  - relevant conversation context

into a single, well-formed question that Legal_QA can process
effectively.
"""

from __future__ import annotations

from database.models import ConversationRecord, MessageRole


class QueryBuilder:
    """Builds a coherent legal question from conversation state."""

    def build(self, state: ConversationRecord) -> str:
        """
        Construct the question string to send to Legal_QA.

        Strategy:
        1. If only one user message and no facts → send it directly.
        2. If facts are available → construct a contextualised query
           combining the original problem with collected facts.
        """
        user_messages = [
            m.content for m in state.messages if m.role == MessageRole.USER
        ]

        if not user_messages:
            return ""

        # ── Simple case: single message, no extra facts ──────
        non_meta_facts = {
            k: v
            for k, v in state.facts.items()
            if k not in ("detected_domain", "case_state") and not isinstance(v, (dict, list))
        }

        if len(user_messages) == 1 and not non_meta_facts:
            return user_messages[0]

        # ── Complex case: build contextualised query ─────────
        original_problem = user_messages[0]
        parts: list[str] = []

        # Opening with original problem
        parts.append(f"Client's legal concern: {original_problem}")

        # Add collected facts
        if non_meta_facts:
            fact_lines = []
            for key, value in non_meta_facts.items():
                readable_key = key.replace("_", " ").title()
                fact_lines.append(f"  - {readable_key}: {value}")
            parts.append(
                "Relevant details established:\n" + "\n".join(fact_lines)
            )

        # Add any additional context from later messages
        # (skip the first message since it's already included,
        #  and skip very short answers that are just fact-responses)
        additional_context = []
        for msg in user_messages[1:]:
            if len(msg.split()) > 5:  # only include substantive messages
                additional_context.append(msg)

        if additional_context:
            parts.append(
                "Additional factual context from client: "
                + " ".join(additional_context)
            )

        # Closing: ask for the appropriate legal guidance
        domain = state.facts.get("detected_domain", "general")
        parts.append(
            "Based on the above, what direct actionable legal remedies, procedures, "
            "and relevant Indian statutory provisions apply to the client?"
        )

        return "\n\n".join(parts)
