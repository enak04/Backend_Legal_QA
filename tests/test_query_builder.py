"""
Tests for the query builder.
"""

from __future__ import annotations

import pytest

from database.models import (
    ConversationRecord,
    ConversationStage,
    Message,
    MessageRole,
    Mode,
)
from legal_qa.query_builder import QueryBuilder


class TestQueryBuilder:
    def _make_state(
        self,
        user_messages: list[str],
        facts: dict | None = None,
    ) -> ConversationRecord:
        messages = [
            Message(role=MessageRole.USER, content=msg)
            for msg in user_messages
        ]
        return ConversationRecord(
            mode=Mode.ACTIONABLE,
            stage=ConversationStage.READY_FOR_QA,
            messages=messages,
            facts=facts or {},
        )

    def test_single_message_no_facts(self):
        builder = QueryBuilder()
        state = self._make_state(["What is Section 302 IPC?"])
        result = builder.build(state)
        assert result == "What is Section 302 IPC?"

    def test_single_message_with_facts(self):
        builder = QueryBuilder()
        state = self._make_state(
            ["My employer hasn't paid me"],
            facts={
                "employment_type": "private",
                "state": "Karnataka",
                "detected_domain": "employment_wage",
            },
        )
        result = builder.build(state)
        assert "employer hasn't paid me" in result
        assert "Private" in result or "private" in result.lower()
        assert "Karnataka" in result

    def test_multi_message_builds_context(self):
        builder = QueryBuilder()
        state = self._make_state(
            [
                "My employer hasn't paid me",
                "Private company",
                "Karnataka",
                "It has been really bad and the company keeps making excuses about payments",
            ],
            facts={
                "employment_type": "private",
                "state": "Karnataka",
                "detected_domain": "employment_wage",
            },
        )
        result = builder.build(state)
        # Should include original problem
        assert "employer" in result.lower()
        # Should include facts
        assert "Karnataka" in result
        # Should include substantive additional context
        assert "excuses" in result.lower() or "remedies" in result.lower()

    def test_empty_messages_returns_empty(self):
        builder = QueryBuilder()
        state = self._make_state([])
        result = builder.build(state)
        assert result == ""

    def test_detected_domain_excluded_from_facts(self):
        builder = QueryBuilder()
        state = self._make_state(
            ["My employer hasn't paid me"],
            facts={"detected_domain": "employment_wage"},
        )
        result = builder.build(state)
        # With only detected_domain (a meta-fact), should be treated
        # as single message with no real facts
        assert result == "My employer hasn't paid me"
