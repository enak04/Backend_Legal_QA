"""
Tests for the follow-up question engine.

Covers domain detection, fact extraction, and mode-aware
follow-up behaviour.
"""

from __future__ import annotations

import pytest

from conversation.followup import (
    FollowUpEngine,
    detect_domain,
    extract_facts,
)
from database.models import (
    ConversationRecord,
    ConversationStage,
    Message,
    MessageRole,
    Mode,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Domain detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDomainDetection:
    def test_employment_domain(self):
        domain = detect_domain("My employer hasn't paid my salary")
        assert domain.name == "employment_wage"

    def test_property_domain(self):
        domain = detect_domain(
            "My landlord is trying to evict me from the flat"
        )
        assert domain.name == "property_land"

    def test_criminal_domain(self):
        domain = detect_domain("I want to file an FIR for theft")
        assert domain.name == "criminal"

    def test_family_domain(self):
        domain = detect_domain("I want to file for divorce")
        assert domain.name == "family_matrimonial"

    def test_consumer_domain(self):
        domain = detect_domain("I bought a defective product")
        assert domain.name == "consumer"

    def test_general_fallback(self):
        domain = detect_domain("Tell me about Article 21")
        assert domain.name == "general"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fact extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestFactExtraction:
    def test_extract_state(self):
        facts = extract_facts("I live in Karnataka")
        assert facts.get("state") == "Karnataka"

    def test_extract_employment_type(self):
        facts = extract_facts("I work in a private company")
        assert facts.get("employment_type") == "private"

    def test_extract_government(self):
        facts = extract_facts("I'm a government employee")
        assert facts.get("employment_type") == "government"

    def test_extract_duration(self):
        facts = extract_facts("It has been 3 months")
        assert facts.get("duration") == "3 months"

    def test_extract_boolean_yes(self):
        facts = extract_facts("Yes", last_question_key="fir_filed")
        assert facts.get("fir_filed") == "yes"

    def test_extract_boolean_no(self):
        facts = extract_facts("No", last_question_key="written_contract")
        assert facts.get("written_contract") == "no"

    def test_short_answer_stored_under_key(self):
        facts = extract_facts(
            "Residential flat", last_question_key="property_type"
        )
        assert facts.get("property_type") == "Residential flat"

    def test_multiple_facts_at_once(self):
        facts = extract_facts(
            "I'm a private company employee in Maharashtra for 6 months"
        )
        assert facts.get("employment_type") == "private"
        assert facts.get("state") == "Maharashtra"
        assert facts.get("duration") == "6 months"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Follow-up engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestFollowUpEngine:
    def _make_state(
        self,
        mode: Mode,
        user_messages: list[str] | None = None,
        facts: dict | None = None,
    ) -> ConversationRecord:
        messages = []
        if user_messages:
            for msg in user_messages:
                messages.append(
                    Message(role=MessageRole.USER, content=msg)
                )
        return ConversationRecord(
            mode=mode,
            stage=ConversationStage.GATHERING_INFO,
            messages=messages,
            facts=facts or {},
        )

    def test_readable_never_asks(self):
        engine = FollowUpEngine()
        state = self._make_state(
            Mode.READABLE,
            user_messages=["pay"],
        )
        assert engine.needs_followup(state) is None

    def test_informative_short_question_asks(self):
        engine = FollowUpEngine()
        state = self._make_state(
            Mode.INFORMATIVE,
            user_messages=["law?"],
        )
        result = engine.needs_followup(state)
        assert result is not None  # should ask for more detail

    def test_informative_clear_question_no_followup(self):
        engine = FollowUpEngine()
        state = self._make_state(
            Mode.INFORMATIVE,
            user_messages=[
                "What are the grounds for divorce under Hindu Marriage Act?"
            ],
        )
        result = engine.needs_followup(state)
        assert result is None

    def test_actionable_asks_followup_for_missing_facts(self):
        engine = FollowUpEngine()
        state = self._make_state(
            Mode.ACTIONABLE,
            user_messages=["My employer hasn't paid me."],
        )
        result = engine.needs_followup(state)
        assert result is not None
        assert isinstance(result, str)

    def test_actionable_no_followup_when_enough_facts(self):
        engine = FollowUpEngine()
        state = self._make_state(
            Mode.ACTIONABLE,
            user_messages=["My employer hasn't paid me."],
            facts={
                "employment_type": "private",
                "state": "Karnataka",
                "detected_domain": "employment_wage",
            },
        )
        result = engine.needs_followup(state)
        assert result is None
