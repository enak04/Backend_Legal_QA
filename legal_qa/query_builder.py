"""
Query builder — constructs a coherent Legal_QA question from
accumulated Universal Case State and conversation context.

Combines:
  - the original legal problem statement
  - all collected facts and jurisdiction
  - identified legal issues & hypotheses
  - relevant conversation context

into a well-formed query that Legal_QA can process effectively.
"""

from __future__ import annotations

from database.models import ConversationRecord, MessageRole


class QueryBuilder:
    """Builds a coherent legal question from Universal Case State."""

    def build(self, state: ConversationRecord) -> str:
        """
        Construct the question string to send to Legal_QA.
        """
        user_messages = [
            m.content for m in state.messages if m.role == MessageRole.USER
        ]

        if not user_messages:
            return ""

        original_problem = user_messages[0]
        case_state = state.facts.get("case_state") or {}

        # Add facts from state.facts
        non_meta_facts = {
            k: v
            for k, v in state.facts.items()
            if k not in ("detected_domain", "case_state", "legal_assessment") and not isinstance(v, (dict, list))
        }

        # Pull jurisdiction if available
        has_case_issues = isinstance(case_state, dict) and bool(case_state.get("issues"))
        if isinstance(case_state, dict):
            jur = case_state.get("jurisdiction") or {}
            if isinstance(jur, dict) and jur.get("state") and "state" not in non_meta_facts:
                non_meta_facts["state"] = jur["state"]

            fin = case_state.get("financial") or {}
            if isinstance(fin, dict) and fin.get("amount_raw") and "amount" not in non_meta_facts:
                non_meta_facts["amount"] = fin["amount_raw"]

        if len(user_messages) == 1 and not non_meta_facts and not has_case_issues:
            return original_problem

        parts: list[str] = []
        parts.append(f"Client's legal concern: {original_problem}")

        if non_meta_facts:
            fact_lines = []
            for key, value in non_meta_facts.items():
                readable_key = key.replace("_", " ").title()
                fact_lines.append(f"  - {readable_key}: {value}")
            parts.append(
                "Relevant details established:\n" + "\n".join(fact_lines)
            )

        # Add identified issues if present
        if isinstance(case_state, dict) and case_state.get("issues"):
            issue_lines = []
            for iss in case_state["issues"]:
                iss_name = iss.get("issue") if isinstance(iss, dict) else str(iss)
                issue_lines.append(f"  - {iss_name}")
            parts.append("Spotted legal issues:\n" + "\n".join(issue_lines))

        # Add substantive context from later user messages
        additional_context = []
        for msg in user_messages[1:]:
            if len(msg.split()) > 4:
                additional_context.append(msg)

        if additional_context:
            parts.append(
                "Additional factual context from client: "
                + " ".join(additional_context)
            )

        # Closing: ask for actionable guidance in second person
        parts.append(
            "Address the recipient directly as 'you' in second person. "
            "Based on the above, what direct actionable legal remedies, procedures, "
            "and relevant Indian statutory provisions apply?"
        )

        return "\n\n".join(parts)
