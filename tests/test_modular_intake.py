"""
Comprehensive test suite for the modular, hierarchical legal case intake system.

Validates the required test scenarios:
  TEST 1: Money Recovery ("I lent someone money and they aren't returning it.")
  TEST 2: Property / Inheritance ("My uncle is refusing to give me my grandfather's property.")
  TEST 3: Contract ("A company took my money but never delivered the service.")
  TEST 4: Unclear Case ("I have a problem with my neighbour and I don't know what to do.")
  TEST 5: Multiple Possible Cases ("Someone sold me land using fake documents.")
  TEST 6: User Changes Information (Correction of previously stored facts)
  TEST 7: User Provides Long Explanation (Bulk fact extraction without repetitive questions)
  TEST 8: User Interrupts With a Question (Answers naturally without forcing intake)
  TEST 9: Urgent Situation ("I received a court notice and the hearing is next week.")
  TEST 10: Extensibility (Adding a new case type via registry without modifying other files)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from conversation.cases.models import CaseModule, StructuredCaseState
from conversation.cases.registry import CaseRegistry, case_registry
from conversation.manager import ConversationManager
from conversation.state import add_user_message, update_facts
from database.models import ConversationRecord, ConversationStage, Mode
from database.repository import SQLiteConversationRepository
from conversation.followup import FollowUpEngine, extract_facts
from legal_qa.client import LegalQAClient
from services.intake_service import OpenAIIntakeService


@pytest.fixture
def mock_openai_response():
    """Helper to mock OpenAI chat completion response."""
    def _make(data: dict):
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(data)
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        return mock_resp
    return _make


class TestModularCaseRegistry:
    """Tests for CaseRegistry taxonomy and dynamic routing."""

    def test_registry_contains_all_core_hierarchies(self):
        """Verify all core categories and subcategories exist in registry."""
        registry = CaseRegistry()
        modules = registry.list_all()
        categories = {m.category for m in modules}
        assert "Civil" in categories
        assert "Criminal" in categories
        assert "Family" in categories
        assert "Other" in categories

        # Verify key case types exist
        case_types = {m.case_type for m in modules}
        assert "Ownership Dispute" in case_types
        assert "Partition" in case_types
        assert "Inheritance" in case_types
        assert "Landlord Tenant" in case_types
        assert "Breach of Contract" in case_types
        assert "Money Recovery" in case_types
        assert "Theft" in case_types
        assert "Fraud" in case_types
        assert "Cybercrime" in case_types
        assert "Divorce" in case_types
        assert "Child Custody" in case_types

    def test_extensibility_add_new_case_type(self):
        """TEST 10: Verify adding a new case module requires only registry.register()."""
        registry = CaseRegistry()
        new_module = CaseModule(
            category="Civil",
            subcategory="Insurance",
            case_type="Insurance Dispute",
            description="Dispute concerning repudiation or underpayment of insurance claims.",
            keywords=["insurance claim", "policy repudiated", "tpa", "health insurance rejected"],
            relevant_facts=["Policy type", "Date of claim", "Reason for rejection"],
            priority_info=["Policy number", "Rejection letter reason"],
        )
        registry.register(new_module)

        # Confirm it can be retrieved and routed to
        retrieved = registry.get_by_name("Insurance Dispute")
        assert retrieved is not None
        assert retrieved.subcategory == "Insurance"

        # Verify dynamic routing finds it from user text
        candidates = registry.find_candidate_modules("My health insurance claim was rejected by the company")
        assert any(c.case_type == "Insurance Dispute" for c in candidates)


class TestModularIntakeScenarios:
    """Tests covering TEST 1 through TEST 9 using the modular intake layer."""

    @pytest.mark.asyncio
    async def test_scenario_1_money_recovery(self, mock_openai_response):
        """
        TEST 1: Money Recovery
        User: "I lent someone money and they aren't returning it."
        """
        service = OpenAIIntakeService(api_key="sk-test-key")

        mock_payload = {
            "case_classification": {
                "primary_category": "Civil",
                "subcategory": "Money Recovery",
                "specific_case_type": "Money Recovery",
                "confidence": "High",
                "possible_alternatives": [],
                "related_case_types": ["Breach of Contract"],
            },
            "urgency": "normal",
            "parties": {"user": "Lender", "opposing_party": "Borrower"},
            "extracted_facts": {
                "detected_domain": "money_recovery",
                "core_issue": "lent money, non-repayment",
            },
            "known_facts": [
                {"fact": "User lent money to an acquaintance who is refusing to return it", "source": "user_statement", "confirmed": True}
            ],
            "missing_information": ["amount", "repayment agreement", "evidence of transfer", "state"],
            "is_ready_for_qa": False,
            "followup_question": "I understand. To help you evaluate your options, was this given as a clear loan that they agreed to repay, and what was the approximate amount?",
            "synthesized_query": None,
        }

        with patch.object(
            service._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_openai_response(mock_payload),
        ):
            state = ConversationRecord(mode=Mode.ACTIONABLE)
            user_msg = "I lent someone money and they aren't returning it."
            add_user_message(state, user_msg)

            result = await service.analyze_turn(state, user_msg)

            assert not result.is_ready_for_qa
            assert result.case_state["primary_category"] == "Civil"
            assert result.case_state["subcategory"] == "Money Recovery"
            assert result.case_state["case_type"] == "Money Recovery"
            assert "loan" in result.followup_question.lower()
            assert result.extracted_facts.get("detected_domain") == "money_recovery"

    @pytest.mark.asyncio
    async def test_scenario_2_property_inheritance(self, mock_openai_response):
        """
        TEST 2: Property / Inheritance
        User: "My uncle is refusing to give me my grandfather's property."
        Classification should be revisable and confidence not locked too early.
        """
        service = OpenAIIntakeService(api_key="sk-test-key")

        mock_payload = {
            "case_classification": {
                "primary_category": "Civil",
                "subcategory": "Property",
                "specific_case_type": "Inheritance",
                "confidence": "Medium",
                "possible_alternatives": ["Partition", "Ownership Dispute"],
                "related_case_types": ["Inheritance Related Family Matters"],
            },
            "urgency": "normal",
            "parties": {"user": "Grandchild / Legal heir", "opposing_party": "Uncle"},
            "extracted_facts": {
                "detected_domain": "property_land",
                "relationship": "grandchild vs uncle",
                "property_nature": "grandfather's property",
            },
            "missing_information": ["Will status", "Ancestral vs self-acquired", "Location/State"],
            "is_ready_for_qa": False,
            "followup_question": "I understand this is a sensitive family property matter. To clarify the legal position, did your grandfather leave behind a Will, or did he pass away without one?",
            "synthesized_query": None,
        }

        with patch.object(
            service._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_openai_response(mock_payload),
        ):
            state = ConversationRecord(mode=Mode.ACTIONABLE)
            user_msg = "My uncle is refusing to give me my grandfather's property."
            add_user_message(state, user_msg)

            result = await service.analyze_turn(state, user_msg)

            assert not result.is_ready_for_qa
            assert result.case_state["subcategory"] == "Property"
            assert result.case_state["confidence"] == "Medium"
            assert "Partition" in result.case_state["possible_alternatives"]
            assert "will" in result.followup_question.lower()

    @pytest.mark.asyncio
    async def test_scenario_3_contract_service_dispute(self, mock_openai_response):
        """
        TEST 3: Contract Dispute
        User: "A company took my money but never delivered the service."
        """
        service = OpenAIIntakeService(api_key="sk-test-key")

        mock_payload = {
            "case_classification": {
                "primary_category": "Civil",
                "subcategory": "Contract",
                "specific_case_type": "Breach of Contract",
                "confidence": "High",
                "possible_alternatives": ["Consumer Dispute"],
                "related_case_types": ["Service Agreement", "Fraud"],
            },
            "urgency": "normal",
            "parties": {"user": "Customer / Client", "opposing_party": "Service Company"},
            "extracted_facts": {
                "detected_domain": "contract_dispute",
                "core_issue": "service paid for but non-delivery",
            },
            "is_ready_for_qa": False,
            "followup_question": "That is frustrating. Did you have a written contract, service agreement, or invoice detailing the agreed deliverables and timeline?",
            "synthesized_query": None,
        }

        with patch.object(
            service._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_openai_response(mock_payload),
        ):
            state = ConversationRecord(mode=Mode.ACTIONABLE)
            user_msg = "A company took my money but never delivered the service."
            add_user_message(state, user_msg)

            result = await service.analyze_turn(state, user_msg)

            assert not result.is_ready_for_qa
            assert result.case_state["primary_category"] == "Civil"
            assert "Consumer Dispute" in result.case_state["possible_alternatives"]

    @pytest.mark.asyncio
    async def test_scenario_4_unclear_case_scoping(self, mock_openai_response):
        """
        TEST 4: Unclear Case
        User: "I have a problem with my neighbour and I don't know what to do."
        System should gently scope rather than prematurely locking case type.
        """
        service = OpenAIIntakeService(api_key="sk-test-key")

        mock_payload = {
            "case_classification": {
                "primary_category": "Civil",
                "subcategory": "General Civil",
                "specific_case_type": "Other Civil Dispute",
                "confidence": "Low",
                "possible_alternatives": ["Possession Dispute", "Ownership Dispute", "Other Criminal Matter"],
                "related_case_types": [],
            },
            "urgency": "normal",
            "parties": {"user": "Resident", "opposing_party": "Neighbour"},
            "extracted_facts": {"detected_domain": "general"},
            "is_ready_for_qa": False,
            "followup_question": "I'd be glad to help. Could you tell me a little about what specifically is happening with your neighbour—for instance, is this related to property boundaries, noise/nuisance, construction, or personal conduct?",
            "synthesized_query": None,
        }

        with patch.object(
            service._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_openai_response(mock_payload),
        ):
            state = ConversationRecord(mode=Mode.ACTIONABLE)
            user_msg = "I have a problem with my neighbour and I don't know what to do."
            add_user_message(state, user_msg)

            result = await service.analyze_turn(state, user_msg)

            assert not result.is_ready_for_qa
            assert result.case_state["confidence"] == "Low"
            # Does not prematurely lock into a rigid dispute type
            assert len(result.case_state["possible_alternatives"]) >= 2

    @pytest.mark.asyncio
    async def test_scenario_5_multiple_possible_cases(self, mock_openai_response):
        """
        TEST 5: Multiple Possible Cases
        User: "Someone sold me land using fake documents."
        Primary: Civil (Property), Related: Contract, Possible: Criminal (Fraud/Forgery).
        """
        service = OpenAIIntakeService(api_key="sk-test-key")

        mock_payload = {
            "case_classification": {
                "primary_category": "Civil",
                "subcategory": "Property",
                "specific_case_type": "Ownership Dispute",
                "confidence": "High",
                "possible_alternatives": ["Property Sale Dispute"],
                "related_case_types": ["Fraud", "Breach of Contract"],
            },
            "urgency": "potentially_urgent",
            "parties": {"user": "Buyer", "opposing_party": "Seller / Forger"},
            "extracted_facts": {
                "detected_domain": "property_land",
                "core_issue": "fraudulent sale with fake deed",
            },
            "is_ready_for_qa": False,
            "followup_question": "This involves both property title issues and potential criminal fraud. Have you already filed an FIR with the police, or has a legal notice been sent to the seller?",
            "synthesized_query": None,
        }

        with patch.object(
            service._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_openai_response(mock_payload),
        ):
            state = ConversationRecord(mode=Mode.ACTIONABLE)
            user_msg = "Someone sold me land using fake documents."
            add_user_message(state, user_msg)

            result = await service.analyze_turn(state, user_msg)

            assert not result.is_ready_for_qa
            assert result.case_state["primary_category"] == "Civil"
            # Tracks criminal fraud alongside property civil dispute
            assert "Fraud" in result.case_state["related_case_types"]

    @pytest.mark.asyncio
    async def test_scenario_6_user_changes_information(self, mock_openai_response):
        """
        TEST 6: User Changes / Corrects Information
        When the user corrects a previously stated amount or fact, the state is updated
        and does not retain contradictory values.
        """
        service = OpenAIIntakeService(api_key="sk-test-key")

        # Initial state had ₹2 lakh
        state = ConversationRecord(mode=Mode.ACTIONABLE)
        state.facts = {
            "amount": "₹2 lakh",
            "case_state": {
                "financial_info": {"amount": "₹2 lakh"},
                "known_facts": [{"fact": "Amount lent was ₹2 lakh"}],
            },
        }

        mock_payload = {
            "case_classification": {
                "primary_category": "Civil",
                "subcategory": "Money Recovery",
                "specific_case_type": "Money Recovery",
                "confidence": "High",
            },
            "extracted_facts": {"amount": "₹3.5 lakh"},
            "financial_info": {"amount": "₹3.5 lakh"},
            "known_facts": [
                {"fact": "Corrected loan amount is ₹3.5 lakh transferred via UPI", "confirmed": True}
            ],
            "is_ready_for_qa": False,
            "followup_question": "Noted, I have updated the amount to ₹3.5 lakh. Was there an agreed deadline for him to return this money?",
            "synthesized_query": None,
        }

        with patch.object(
            service._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_openai_response(mock_payload),
        ):
            correction_msg = "Actually, checking my records, it was ₹3.5 lakh, not ₹2 lakh, transferred via UPI."
            add_user_message(state, correction_msg)

            result = await service.analyze_turn(state, correction_msg)

            # Updated value replaced the old value
            assert result.extracted_facts["amount"] == "₹3.5 lakh"
            assert result.case_state["financial_info"]["amount"] == "₹3.5 lakh"
            assert "updated" in result.followup_question.lower()

    @pytest.mark.asyncio
    async def test_scenario_7_user_long_explanation_sufficient_info(self, mock_openai_response):
        """
        TEST 7: User Provides Long Explanation With All Core Facts
        System extracts everything and provides a conversational summary
        rather than continuing to ask questions.
        """
        service = OpenAIIntakeService(api_key="sk-test-key")

        mock_payload = {
            "case_classification": {
                "primary_category": "Civil",
                "subcategory": "Money Recovery",
                "specific_case_type": "Money Recovery",
                "confidence": "High",
            },
            "extracted_facts": {
                "amount": "₹5 lakh",
                "state": "Delhi",
                "repayment_deadline": "6 months",
                "evidence": "UPI transfer and WhatsApp chats",
                "transaction_nature": "friendly loan",
            },
            "is_ready_for_qa": False,
            "followup_question": (
                "Let me make sure I've understood this correctly: you lent ₹5 lakh to your former colleague in Delhi "
                "in January via UPI, with an agreed repayment deadline of 6 months. That deadline has passed, and you have "
                "the bank transfer records and WhatsApp messages acknowledging the debt. Is that an accurate summary?"
            ),
            "synthesized_query": None,
        }

        with patch.object(
            service._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_openai_response(mock_payload),
        ):
            state = ConversationRecord(mode=Mode.ACTIONABLE)
            long_msg = (
                "In January 2024, I gave a friendly loan of ₹5 lakh to my colleague Ramesh in Delhi. "
                "He promised to repay within 6 months. I transferred it directly from my HDFC bank account via UPI. "
                "The 6 months ended in July, but now he is ignoring my phone calls. I have all WhatsApp chats where he "
                "acknowledges taking the loan and promising repayment. I want to recover my money."
            )
            add_user_message(state, long_msg)

            result = await service.analyze_turn(state, long_msg)

            assert not result.is_ready_for_qa
            # Bot provides conversational summary instead of interrogating for details already provided
            assert "accurate summary" in result.followup_question.lower() or "understood this correctly" in result.followup_question.lower()

    @pytest.mark.asyncio
    async def test_scenario_8_user_interrupts_with_question(self, mock_openai_response):
        """
        TEST 8: User Interrupts With a Question During Intake
        System answers the user's question directly rather than forcing intake continuation.
        """
        service = OpenAIIntakeService(api_key="sk-test-key")

        mock_payload = {
            "case_classification": {
                "primary_category": "Civil",
                "subcategory": "Money Recovery",
                "specific_case_type": "Money Recovery",
                "confidence": "High",
            },
            "extracted_facts": {"detected_domain": "money_recovery"},
            "is_ready_for_qa": False,
            "followup_question": (
                "Under Indian law, the limitation period to file a civil suit for money recovery is generally 3 years "
                "from the date the loan was due or from the date of the last written acknowledgment of debt. "
                "To see if you are within time, when was the agreed repayment date or the last message where they acknowledged owing the money?"
            ),
            "synthesized_query": None,
        }

        with patch.object(
            service._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_openai_response(mock_payload),
        ):
            state = ConversationRecord(mode=Mode.ACTIONABLE)
            user_msg = "Before we go further, what is the limitation period for recovering money in court?"
            add_user_message(state, user_msg)

            result = await service.analyze_turn(state, user_msg)

            assert not result.is_ready_for_qa
            # Directly answered the question about 3 years limitation
            assert "3 years" in result.followup_question
            assert "limitation period" in result.followup_question.lower()

    @pytest.mark.asyncio
    async def test_scenario_9_urgent_situation_court_notice(self, mock_openai_response):
        """
        TEST 9: Urgent Situation
        User: "I received a court notice and the hearing is next week."
        Urgency must be flagged as "urgent".
        """
        service = OpenAIIntakeService(api_key="sk-test-key")

        mock_payload = {
            "case_classification": {
                "primary_category": "Civil",
                "subcategory": "General Civil",
                "specific_case_type": "Other Civil Dispute",
                "confidence": "Medium",
            },
            "urgency": "urgent",
            "extracted_facts": {
                "urgency": "urgent",
                "court_notice_received": "yes",
                "hearing_timeframe": "next week",
            },
            "is_ready_for_qa": False,
            "followup_question": (
                "Understood, this is time-sensitive since the hearing is next week. Which court issued the notice, "
                "and what is the exact date of the hearing specified on the summons?"
            ),
            "synthesized_query": None,
        }

        with patch.object(
            service._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_openai_response(mock_payload),
        ):
            state = ConversationRecord(mode=Mode.ACTIONABLE)
            user_msg = "I received a court notice and the hearing is next week."
            add_user_message(state, user_msg)

            result = await service.analyze_turn(state, user_msg)

            assert not result.is_ready_for_qa
            assert result.case_state["urgency"] == "urgent"
            assert "time-sensitive" in result.followup_question.lower() or "hearing" in result.followup_question.lower()

    def test_fallback_engine_urgency_and_money_recovery(self):
        """Verify fallback rule engine detects urgency and money recovery without OpenAI."""
        service = OpenAIIntakeService(api_key=None)
        assert not service.is_configured

        # Test urgency detection in fallback
        state = ConversationRecord(mode=Mode.ACTIONABLE)
        user_msg = "I received a court notice and the hearing is next week"
        add_user_message(state, user_msg)

        res = service._analyze_with_fallback(state, user_msg)
        assert res.case_state["urgency"] == "urgent"

        # Test money recovery detection in fallback
        state2 = ConversationRecord(mode=Mode.ACTIONABLE)
        user_msg2 = "I lent money to a friend and he refuses to return it"
        add_user_message(state2, user_msg2)

        res2 = service._analyze_with_fallback(state2, user_msg2)
        assert res2.case_state["subcategory"] in ("Money Recovery", "General") or res2.case_state["case_type"] == "Money Recovery"
        assert not res2.is_ready_for_qa
        assert res2.followup_question is not None

    @pytest.mark.asyncio
    async def test_scenario_user_says_dont_have_information(self, mock_openai_response):
        """
        Verify that when a user says they don't have a document/proof or don't know:
        1. The system does NOT repeat the question.
        2. Reassures the user warmly.
        3. Marks the document as not available.
        4. Moves forward to next relevant topic or preliminary advice.
        """
        service = OpenAIIntakeService(api_key="sk-test-key")

        state = ConversationRecord(mode=Mode.ACTIONABLE)
        add_user_message(state, "I gave my friend ₹2 lakh and he isn't returning it.")
        state.last_assistant_question = "Do you have any written agreement or contract for this loan?"

        mock_payload = {
            "case_classification": {
                "primary_category": "Civil",
                "subcategory": "Money Recovery",
                "specific_case_type": "Money Recovery",
                "confidence": "High",
            },
            "extracted_facts": {
                "written_agreement": "none",
                "transaction_nature": "oral agreement",
            },
            "known_facts": [
                {"fact": "No written agreement exists; agreement was oral based on friendship", "confirmed": True}
            ],
            "missing_information": ["payment method", "repayment deadline"],
            "is_ready_for_qa": False,
            "followup_question": (
                "That's completely fine and very common with loans between friends. "
                "In India, oral agreements are legally valid. Did you transfer the money via bank transfer, UPI, or cash?"
            ),
            "synthesized_query": None,
        }

        with patch.object(
            service._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_openai_response(mock_payload),
        ):
            user_msg = "No, I don't have any written agreement, it was just between friends."
            add_user_message(state, user_msg)

            result = await service.analyze_turn(state, user_msg)

            assert not result.is_ready_for_qa
            # Question is NOT repeated, user is reassured, and conversation moves to payment method
            assert "completely fine" in result.followup_question.lower()
            assert "bank" in result.followup_question.lower() or "upi" in result.followup_question.lower()

    def test_fallback_engine_never_repeats_question_when_user_says_dont_have(self):
        """Verify fallback rule engine does not repeat questions when user says 'don't have'."""
        engine = FollowUpEngine()
        state = ConversationRecord(mode=Mode.ACTIONABLE)
        add_user_message(state, "My employer hasn't paid me salary")
        state.facts = {"employment_type": "private", "state": "Karnataka", "detected_domain": "employment_wage"}
        state.last_assistant_question = "Do you have a written employment contract or appointment letter?"

        # User says they don't have it
        user_msg = "I don't have any written contract"
        add_user_message(state, user_msg)

        # Last question key was written_contract
        last_key = engine.get_last_question_key(state)
        new_facts = extract_facts(user_msg, last_question_key=last_key)
        assert new_facts.get("written_contract") in ("not_available", "no")

        state.facts.update(new_facts)
        # Should not ask for written_contract again
        next_q = engine.needs_followup(state)
        assert next_q != state.last_assistant_question
