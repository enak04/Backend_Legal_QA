"""
Tests for conversation API endpoints.

Covers:
  1. Conversation creation
  2. Mode selection
  3. Mode persistence
  4. Sending messages
  5. Conversation history (GET)
  6. Fact persistence
  7. Follow-up questions
  8. Calling Legal_QA with question + mode
  9. Returning Legal_QA results
  10. Legal_QA timeout
  11. Invalid mode
  12. Invalid conversation
  13. Multiple messages in a conversation
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from legal_qa.client import LegalQATimeoutError, LegalQAUnavailableError

pytestmark = pytest.mark.anyio


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Conversation creation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestConversationCreation:
    async def test_create_conversation(self, test_client):
        resp = await test_client.post(
            "/api/conversations", json={"mode": "actionable"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "conversation_id" in data
        assert data["mode"] == "actionable"

    async def test_create_all_modes(self, test_client):
        for mode in ("actionable", "informative", "readable"):
            resp = await test_client.post(
                "/api/conversations", json={"mode": mode}
            )
            assert resp.status_code == 200
            assert resp.json()["mode"] == mode


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Mode selection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestModeSelection:
    async def test_mode_stored_correctly(self, test_client):
        resp = await test_client.post(
            "/api/conversations", json={"mode": "informative"}
        )
        cid = resp.json()["conversation_id"]

        get_resp = await test_client.get(f"/api/conversations/{cid}")
        assert get_resp.json()["mode"] == "informative"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Mode persistence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestModePersistence:
    async def test_mode_persists_across_messages(self, test_client):
        resp = await test_client.post(
            "/api/conversations", json={"mode": "actionable"}
        )
        cid = resp.json()["conversation_id"]

        # Send a message
        await test_client.post(
            f"/api/conversations/{cid}/messages",
            json={"message": "My employer hasn't paid me."},
        )

        # Check mode is still actionable
        get_resp = await test_client.get(f"/api/conversations/{cid}")
        assert get_resp.json()["mode"] == "actionable"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Sending messages
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSendMessage:
    async def test_send_message_returns_response(self, test_client):
        resp = await test_client.post(
            "/api/conversations", json={"mode": "readable"}
        )
        cid = resp.json()["conversation_id"]

        msg_resp = await test_client.post(
            f"/api/conversations/{cid}/messages",
            json={"message": "What is Section 498A of the IPC?"},
        )
        assert msg_resp.status_code == 200
        data = msg_resp.json()
        assert data["type"] in ("follow_up", "final_answer")
        assert data["conversation_id"] == cid

    async def test_empty_message_rejected(self, test_client):
        resp = await test_client.post(
            "/api/conversations", json={"mode": "actionable"}
        )
        cid = resp.json()["conversation_id"]

        msg_resp = await test_client.post(
            f"/api/conversations/{cid}/messages",
            json={"message": ""},
        )
        assert msg_resp.status_code == 422  # Pydantic validation


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Conversation history
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestConversationHistory:
    async def test_get_conversation_returns_messages(self, test_client):
        resp = await test_client.post(
            "/api/conversations", json={"mode": "informative"}
        )
        cid = resp.json()["conversation_id"]

        await test_client.post(
            f"/api/conversations/{cid}/messages",
            json={"message": "What is habeas corpus?"},
        )

        get_resp = await test_client.get(f"/api/conversations/{cid}")
        data = get_resp.json()
        assert len(data["messages"]) >= 1
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "What is habeas corpus?"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Fact persistence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestFactPersistence:
    async def test_facts_extracted_and_stored(self, test_client):
        resp = await test_client.post(
            "/api/conversations", json={"mode": "actionable"}
        )
        cid = resp.json()["conversation_id"]

        # First message triggers domain detection
        await test_client.post(
            f"/api/conversations/{cid}/messages",
            json={"message": "My employer hasn't paid my salary."},
        )

        # Answer about employer type
        await test_client.post(
            f"/api/conversations/{cid}/messages",
            json={"message": "Private company."},
        )

        get_resp = await test_client.get(f"/api/conversations/{cid}")
        facts = get_resp.json()["facts"]
        assert "employment_type" in facts or "detected_domain" in facts


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Follow-up questions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestFollowUpQuestions:
    async def test_actionable_asks_followup(self, test_client):
        resp = await test_client.post(
            "/api/conversations", json={"mode": "actionable"}
        )
        cid = resp.json()["conversation_id"]

        msg_resp = await test_client.post(
            f"/api/conversations/{cid}/messages",
            json={"message": "My employer hasn't paid me."},
        )
        data = msg_resp.json()
        assert data["type"] == "follow_up"
        assert data["mode"] == "actionable"
        assert "message" in data

    async def test_readable_skips_followup(self, test_client):
        resp = await test_client.post(
            "/api/conversations", json={"mode": "readable"}
        )
        cid = resp.json()["conversation_id"]

        msg_resp = await test_client.post(
            f"/api/conversations/{cid}/messages",
            json={
                "message": "Explain Section 498A of the IPC in simple terms."
            },
        )
        data = msg_resp.json()
        # Readable mode should go directly to Legal_QA
        assert data["type"] == "final_answer"

    async def test_informative_direct_for_clear_question(self, test_client):
        resp = await test_client.post(
            "/api/conversations", json={"mode": "informative"}
        )
        cid = resp.json()["conversation_id"]

        msg_resp = await test_client.post(
            f"/api/conversations/{cid}/messages",
            json={
                "message": (
                    "What are the grounds for divorce under the Hindu "
                    "Marriage Act, 1955?"
                )
            },
        )
        data = msg_resp.json()
        assert data["type"] == "final_answer"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Calling Legal_QA with question + mode
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLegalQACall:
    async def test_legal_qa_called_with_correct_mode(
        self, test_client, mock_legal_qa_client
    ):
        resp = await test_client.post(
            "/api/conversations", json={"mode": "readable"}
        )
        cid = resp.json()["conversation_id"]

        await test_client.post(
            f"/api/conversations/{cid}/messages",
            json={"message": "Explain contempt of court."},
        )

        # Verify predict was called with mode="readable"
        mock_legal_qa_client.predict.assert_called_once()
        call_kwargs = mock_legal_qa_client.predict.call_args
        assert call_kwargs.kwargs["mode"] == "readable"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. Returning Legal_QA results
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLegalQAResults:
    async def test_final_answer_has_required_fields(self, test_client):
        resp = await test_client.post(
            "/api/conversations", json={"mode": "readable"}
        )
        cid = resp.json()["conversation_id"]

        msg_resp = await test_client.post(
            f"/api/conversations/{cid}/messages",
            json={"message": "What is the right to information act?"},
        )
        data = msg_resp.json()
        assert data["type"] == "final_answer"
        assert "answer" in data
        assert "reasoning_chain" in data
        assert "sources" in data
        assert isinstance(data["sources"], list)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. Legal_QA timeout
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLegalQATimeout:
    async def test_timeout_returns_504(
        self, test_client, mock_legal_qa_client
    ):
        mock_legal_qa_client.predict.side_effect = LegalQATimeoutError(
            "timeout"
        )

        resp = await test_client.post(
            "/api/conversations", json={"mode": "readable"}
        )
        cid = resp.json()["conversation_id"]

        msg_resp = await test_client.post(
            f"/api/conversations/{cid}/messages",
            json={"message": "Explain fundamental rights."},
        )
        assert msg_resp.status_code == 504

    async def test_unavailable_returns_503(
        self, test_client, mock_legal_qa_client
    ):
        mock_legal_qa_client.predict.side_effect = LegalQAUnavailableError(
            "down"
        )

        resp = await test_client.post(
            "/api/conversations", json={"mode": "readable"}
        )
        cid = resp.json()["conversation_id"]

        msg_resp = await test_client.post(
            f"/api/conversations/{cid}/messages",
            json={"message": "Explain fundamental rights."},
        )
        assert msg_resp.status_code == 503


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. Invalid mode
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestInvalidMode:
    async def test_invalid_mode_returns_400(self, test_client):
        resp = await test_client.post(
            "/api/conversations", json={"mode": "invalid_mode"}
        )
        assert resp.status_code == 400
        assert "Invalid mode" in resp.json()["detail"]

    async def test_missing_mode_returns_422(self, test_client):
        resp = await test_client.post("/api/conversations", json={})
        assert resp.status_code == 422


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 12. Invalid conversation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestInvalidConversation:
    async def test_nonexistent_conversation_returns_404(self, test_client):
        resp = await test_client.get("/api/conversations/nonexistent123")
        assert resp.status_code == 404

    async def test_message_to_nonexistent_returns_404(self, test_client):
        resp = await test_client.post(
            "/api/conversations/nonexistent123/messages",
            json={"message": "Hello"},
        )
        assert resp.status_code == 404


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 13. Multiple messages in a conversation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMultiTurnConversation:
    async def test_full_actionable_flow(self, test_client):
        """Simulate a complete multi-turn actionable conversation."""
        # Create conversation
        resp = await test_client.post(
            "/api/conversations", json={"mode": "actionable"}
        )
        cid = resp.json()["conversation_id"]

        # Turn 1: initial problem → expect follow-up
        msg1 = await test_client.post(
            f"/api/conversations/{cid}/messages",
            json={"message": "My employer hasn't paid my salary for months."},
        )
        data1 = msg1.json()
        assert data1["type"] == "follow_up"

        # Turn 2: answer follow-up
        msg2 = await test_client.post(
            f"/api/conversations/{cid}/messages",
            json={"message": "Private company in Karnataka."},
        )
        data2 = msg2.json()
        # May be another follow-up or final answer depending on
        # how many facts were extracted
        assert data2["type"] in ("follow_up", "final_answer")

        # Keep answering until we get a final answer
        if data2["type"] == "follow_up":
            msg3 = await test_client.post(
                f"/api/conversations/{cid}/messages",
                json={"message": "3 months unpaid."},
            )
            data3 = msg3.json()
            assert data3["type"] in ("follow_up", "final_answer")

        # Verify conversation history has all messages
        get_resp = await test_client.get(f"/api/conversations/{cid}")
        conv = get_resp.json()
        user_msgs = [
            m for m in conv["messages"] if m["role"] == "user"
        ]
        assert len(user_msgs) >= 2  # at least 2 user messages
