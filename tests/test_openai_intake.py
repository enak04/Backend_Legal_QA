"""
Unit and integration tests for OpenAI-powered conversational legal intake.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conversation.manager import ConversationManager
from conversation.state import add_user_message
from database.models import ConversationRecord, ConversationStage, MessageRole, Mode
from database.repository import SQLiteConversationRepository
from legal_qa.client import LegalQAClient
from services.intake_service import OpenAIIntakeService


@pytest.fixture
def mock_openai_response():
    """Helper to create a mock OpenAI chat completion response."""
    def _make(data: dict):
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(data)
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        return mock_resp
    return _make


class TestOpenAIIntakeService:
    """Tests for OpenAIIntakeService standalone behavior."""

    @pytest.mark.asyncio
    async def test_unconfigured_uses_fallback(self):
        """When no API key is provided, service falls back to rule engine."""
        service = OpenAIIntakeService(api_key=None)
        assert not service.is_configured

        state = ConversationRecord(mode=Mode.ACTIONABLE)
        add_user_message(state, "I have not been paid salary for 3 months in Karnataka")

        result = await service.analyze_turn(state, "I have not been paid salary for 3 months in Karnataka")
        # Rule fallback extracts state and duration
        assert "state" in result.extracted_facts
        assert result.extracted_facts["state"] == "Karnataka"

    @pytest.mark.asyncio
    async def test_openai_followup_turn(self, mock_openai_response):
        """OpenAI intake detects missing facts and returns a conversational follow-up."""
        service = OpenAIIntakeService(api_key="sk-test-key")
        assert service.is_configured

        openai_data = {
            "extracted_facts": {
                "detected_domain": "employment_wage",
                "state": "Maharashtra",
                "core_issue": "unpaid salary for 4 months",
            },
            "is_ready_for_qa": False,
            "followup_question": "Were you working for a private company or a government department?",
            "synthesized_query": None,
        }

        with patch.object(
            service._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_openai_response(openai_data),
        ):
            state = ConversationRecord(mode=Mode.ACTIONABLE)
            add_user_message(state, "My boss in Mumbai hasn't paid me in 4 months")

            result = await service.analyze_turn(
                state, "My boss in Mumbai hasn't paid me in 4 months"
            )

            assert not result.is_ready_for_qa
            assert result.followup_question == "Were you working for a private company or a government department?"
            assert result.extracted_facts["state"] == "Maharashtra"
            assert result.extracted_facts["detected_domain"] == "employment_wage"

    @pytest.mark.asyncio
    async def test_openai_ready_for_qa_turn(self, mock_openai_response):
        """OpenAI intake detects all necessary facts and provides synthesized query."""
        service = OpenAIIntakeService(api_key="sk-test-key")

        openai_data = {
            "extracted_facts": {
                "detected_domain": "property_land",
                "state": "Delhi",
                "property_type": "residential flat",
                "parties": "tenant vs landlord",
                "issue": "illegal eviction without notice",
            },
            "is_ready_for_qa": True,
            "followup_question": None,
            "synthesized_query": "The tenant in Delhi was evicted from a residential flat without notice. What legal remedies apply under Delhi Rent Control Act?",
        }

        with patch.object(
            service._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_openai_response(openai_data),
        ):
            state = ConversationRecord(mode=Mode.ACTIONABLE)
            state.facts = {"state": "Delhi"}
            add_user_message(state, "I am a tenant in a Delhi flat and my landlord changed locks today")

            result = await service.analyze_turn(
                state, "I am a tenant in a Delhi flat and my landlord changed locks today"
            )

            assert result.is_ready_for_qa
            assert result.followup_question is None
            assert "Delhi Rent Control Act" in (result.synthesized_query or "")

    @pytest.mark.asyncio
    async def test_openai_api_error_fallback(self):
        """If OpenAI API raises an exception, service falls back to rule engine."""
        service = OpenAIIntakeService(api_key="sk-test-key")

        with patch.object(
            service._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=Exception("OpenAI API rate limit exceeded"),
        ):
            state = ConversationRecord(mode=Mode.ACTIONABLE)
            add_user_message(state, "I worked at a private company in Karnataka")

            result = await service.analyze_turn(
                state, "I worked at a private company in Karnataka"
            )

            # Successfully fell back to rule engine
            assert "state" in result.extracted_facts
            assert result.extracted_facts["state"] == "Karnataka"


class TestConversationManagerWithOpenAI:
    """Integration tests for ConversationManager with OpenAIIntakeService."""

    @pytest.mark.asyncio
    async def test_full_flow_with_mocked_openai(self, mock_openai_response, tmp_path):
        """Verify multi-turn conversation flow through ConversationManager with OpenAI."""
        repo = SQLiteConversationRepository(str(tmp_path / "intake_test.db"))
        await repo.initialize()
        legal_qa_client = LegalQAClient(base_url="http://mock-legal-qa")
        intake_service = OpenAIIntakeService(api_key="sk-test-key")

        manager = ConversationManager(
            repository=repo,
            legal_qa_client=legal_qa_client,
            intake_service=intake_service,
        )

        conv = await manager.create_conversation(mode=Mode.ACTIONABLE)
        conv_id = conv.conversation_id

        # Turn 1: User gives partial info → OpenAI asks follow-up
        turn1_data = {
            "extracted_facts": {"state": "Karnataka", "detected_domain": "employment_wage"},
            "is_ready_for_qa": False,
            "followup_question": "Is your employer a private company or a government entity?",
            "synthesized_query": None,
        }

        with patch.object(
            intake_service._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_openai_response(turn1_data),
        ):
            resp1 = await manager.process_message(
                conv_id, "My employer in Bangalore has not paid my salary for 3 months"
            )

            assert resp1["type"] == "follow_up"
            assert resp1["message"] == "Is your employer a private company or a government entity?"

            updated_state = await repo.get(conv_id)
            assert updated_state.stage == ConversationStage.GATHERING_INFO
            assert updated_state.facts.get("state") == "Karnataka"

        # Turn 2: User answers follow-up → OpenAI marks ready & synthesizes query for Legal_QA
        turn2_data = {
            "extracted_facts": {"employment_type": "private"},
            "is_ready_for_qa": True,
            "followup_question": None,
            "synthesized_query": "An employee at a private firm in Bangalore, Karnataka is facing 3 months unpaid salary. What legal remedies apply under Indian labour laws?",
        }

        qa_mock_resp = {
            "question": "An employee at a private firm...",
            "answer": "Under the Payment of Wages Act, 1936 and Industrial Disputes Act...",
            "reasoning_chain": ["Step 1: Verify employment status", "Step 2: Issue legal notice"],
            "retrieved_cases": [
                {"question": "Unpaid wages case", "answer": "Remedy under Payment of Wages Act"}
            ],
        }

        with patch.object(
            intake_service._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_openai_response(turn2_data),
        ), patch.object(
            legal_qa_client,
            "predict",
            new_callable=AsyncMock,
            return_value=qa_mock_resp,
        ) as mock_predict:
            resp2 = await manager.process_message(
                conv_id, "It is a private tech startup."
            )

            assert resp2["type"] == "final_answer"
            assert len(resp2["answer"]) > 50  # Meaningful legal answer was generated
            assert resp2["legal_assessment"] is not None  # Structured assessment present
            assert len(resp2["reasoning_chain"]) == 2
            assert len(resp2["sources"]) == 1

            mock_predict.assert_called_once_with(
                question=turn2_data["synthesized_query"],
                mode="actionable",
            )

            final_state = await repo.get(conv_id)
            assert final_state.stage == ConversationStage.ANSWERED
            assert final_state.facts.get("employment_type") == "private"
            assert len(final_state.messages) == 4  # user, assistant, user, assistant

    @pytest.mark.asyncio
    async def test_turn3_brief_location_does_not_prematurely_close_without_financial_dues(
        self, mock_openai_response, tmp_path
    ):
        """
        Verify that a single-word location response on Turn 3 (e.g. 'Mumbei')
        does NOT trigger premature closure when critical facts like financial dues remain unasked.
        """
        service = OpenAIIntakeService(api_key="sk-test-key")
        state = ConversationRecord(mode=Mode.ACTIONABLE)

        add_user_message(state, "I was fired from my job")
        state.messages.append(type("Msg", (), {"role": MessageRole.ASSISTANT, "content": "What reason was given for your termination?"})())
        add_user_message(state, "It says due to performance issues but my manager always appreciated my work")
        state.messages.append(type("Msg", (), {"role": MessageRole.ASSISTANT, "content": "Which state or city were you employed in?"})())
        add_user_message(state, "Mumbei")

        # Mock OpenAI eagerly attempting premature closure on turn 3
        openai_turn3 = {
            "case": {"domain": "employment"},
            "jurisdiction": {"state": "Maharashtra", "city": "Mumbai"},
            "is_ready_for_qa": True,
            "followup_question": None,
            "synthesized_query": "The client was terminated in Mumbai...",
        }

        with patch.object(
            service._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_openai_response(openai_turn3),
        ):
            res = await service.analyze_turn(state, "Mumbei")
            assert not res.is_ready_for_qa, "Should NOT close intake on Turn 3 with unasked financial dues!"
            assert res.followup_question is not None
            assert any(w in res.followup_question.lower() for w in ["dues", "salary", "pay", "gratuity", "amount"])

    def test_statutory_catalog_domain_isolation(self):
        """
        Verify that research_authorities strictly isolates domains so that
        Maharashtra Rent Control Act NEVER contaminates an employment dispute in Mumbai,
        and Maharashtra Shops and Establishments Act 2017 is correctly retrieved.
        """
        from conversation.cases.models import Jurisdiction, LegalIssue, UniversalCaseState
        from legal_qa.research import legal_research_layer

        ucs = UniversalCaseState(
            case_domain="employment",
            jurisdiction=Jurisdiction(country="India", state="Maharashtra", city="Mumbai"),
            issues=[LegalIssue(issue="wrongful termination", domain="employment", status="confirmed")],
        )

        auths = legal_research_layer.research_authorities(ucs)
        sources = [a.source for a in auths]

        # Rent Control must NEVER be retrieved for an employment dispute
        assert not any("Rent Control" in s for s in sources), "Cross-domain pollution: Rent Control in employment!"
        # Maharashtra Shops and Establishments Act must be retrieved for Maharashtra
        assert any("Maharashtra Shops" in s for s in sources), "State-specific employment act not retrieved!"

