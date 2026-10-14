"""
Unit and integration tests for the Admin Portal and endpoints.

Verifies:
  1. Admin authentication (login, logout, session verification)
  2. Protected routes authorization enforcement
  3. System statistics computation
  4. Conversation listing with user & system message counters
  5. Detailed message inspection for both user and system messages
  6. Search, stage/mode filtering, and pagination
  7. Conversation export in JSON and Markdown formats
  8. Conversation deletion
  9. Admin HTML dashboard serving
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from config import settings
from database.models import (
    ConversationRecord,
    LegalQAResult,
    Message,
    MessageRole,
    Mode,
    RetrievedCase,
)
from services.admin_auth import create_admin_token

pytestmark = pytest.mark.anyio


@pytest.fixture
def admin_headers():
    token = create_admin_token(settings.admin_username)
    return {"Authorization": f"Bearer {token}"}


class TestAdminAuth:
    """Verify login, token issuance, and protected endpoint access control."""

    async def test_admin_login_success(self, test_client: AsyncClient):
        resp = await test_client.post(
            "/api/admin/login",
            json={
                "username": settings.admin_username,
                "password": settings.admin_password,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert data["username"] == settings.admin_username
        assert data["role"] == "admin"
        # Verify cookie was set
        assert "admin_token" in resp.cookies

    async def test_admin_login_invalid_credentials(self, test_client: AsyncClient):
        resp = await test_client.post(
            "/api/admin/login",
            json={"username": "wrong_user", "password": "wrong_password"},
        )
        assert resp.status_code == 401
        assert "Invalid" in resp.json()["detail"]

    async def test_admin_me_authorized(self, test_client: AsyncClient, admin_headers):
        resp = await test_client.get("/api/admin/me", headers=admin_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["authenticated"] is True
        assert data["username"] == settings.admin_username

    async def test_admin_me_unauthorized(self, test_client: AsyncClient):
        resp = await test_client.get("/api/admin/me")
        assert resp.status_code == 401

    async def test_admin_logout(self, test_client: AsyncClient):
        resp = await test_client.post("/api/admin/logout")
        assert resp.status_code == 200
        assert "logged out" in resp.json()["message"].lower()


class TestAdminConversationAccess:
    """Verify listing and detailed inspection of user and system messages."""

    async def test_list_conversations_with_message_counts(
        self, test_client: AsyncClient, manager, admin_headers
    ):
        # Create a conversation with user and system messages
        conv = await manager.create_conversation(Mode.ACTIONABLE)
        conv.messages = [
            Message(role=MessageRole.USER, content="Hello, I need legal help with rent."),
            Message(
                role=MessageRole.ASSISTANT,
                content="I can help with tenancy issues. Where is the property located?",
            ),
            Message(role=MessageRole.USER, content="It is in Bengaluru, Karnataka."),
            Message(
                role=MessageRole.ASSISTANT,
                content="Under Karnataka Rent Act, what type of agreement do you have?",
            ),
        ]
        await manager._repo.update(conv)

        # Admin calls list endpoint
        resp = await test_client.get(
            "/api/admin/conversations", headers=admin_headers
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "conversations" in data
        assert data["pagination"]["total"] >= 1

        target = next(
            (c for c in data["conversations"] if c["conversation_id"] == conv.conversation_id),
            None,
        )
        assert target is not None
        assert target["total_messages"] == 4
        assert target["user_messages_count"] == 2
        assert target["system_messages_count"] == 2
        assert target["mode"] == "actionable"

    async def test_get_conversation_detail_full_transcript(
        self, test_client: AsyncClient, manager, admin_headers
    ):
        conv = await manager.create_conversation(Mode.INFORMATIVE)
        conv.facts = {"employment_duration": "3 years", "salary_unpaid_months": "2"}
        conv.messages = [
            Message(role=MessageRole.USER, content="My salary is delayed."),
            Message(role=MessageRole.ASSISTANT, content="Under Section 15 of Payment of Wages Act..."),
        ]
        conv.legal_qa_results = [
            LegalQAResult(
                question="Remedy for delayed salary?",
                answer="Filing claim before labor commissioner.",
                reasoning_chain=["Labor tribunal jurisdiction"],
                retrieved_cases=[
                    RetrievedCase(question="Past wage delay case", answer="Relief granted.")
                ],
            )
        ]
        await manager._repo.update(conv)

        resp = await test_client.get(
            f"/api/admin/conversations/{conv.conversation_id}",
            headers=admin_headers,
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["conversation_id"] == conv.conversation_id
        assert data["mode"] == "informative"
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "My salary is delayed."
        assert data["messages"][1]["role"] == "assistant"
        assert "Payment of Wages Act" in data["messages"][1]["content"]

        assert data["stats"]["user_messages"] == 1
        assert data["stats"]["system_messages"] == 1
        assert data["facts"]["employment_duration"] == "3 years"
        assert len(data["legal_qa_results"]) == 1

    async def test_admin_stats(self, test_client: AsyncClient, manager, admin_headers):
        c1 = await manager.create_conversation(Mode.ACTIONABLE)
        c1.messages = [
            Message(role=MessageRole.USER, content="Query 1"),
            Message(role=MessageRole.ASSISTANT, content="Response 1"),
        ]
        await manager._repo.update(c1)

        c2 = await manager.create_conversation(Mode.READABLE)
        c2.messages = [
            Message(role=MessageRole.USER, content="Query 2"),
        ]
        await manager._repo.update(c2)

        resp = await test_client.get("/api/admin/stats", headers=admin_headers)
        assert resp.status_code == 200
        stats = resp.json()

        assert stats["total_conversations"] >= 2
        assert stats["total_messages"] >= 3
        assert stats["user_messages"] >= 2
        assert stats["assistant_messages"] >= 1
        assert "actionable" in stats["by_mode"]
        assert "readable" in stats["by_mode"]

    async def test_search_and_filters(
        self, test_client: AsyncClient, manager, admin_headers
    ):
        c1 = await manager.create_conversation(Mode.ACTIONABLE)
        c1.messages = [
            Message(role=MessageRole.USER, content="Tenant copyright dispute intellectual property"),
        ]
        await manager._repo.update(c1)

        # Search matching
        resp = await test_client.get(
            "/api/admin/conversations?search=intellectual+property",
            headers=admin_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert any(c["conversation_id"] == c1.conversation_id for c in data["conversations"])

        # Search non-matching
        resp = await test_client.get(
            "/api/admin/conversations?search=nonexistentxyz123",
            headers=admin_headers,
        )
        assert resp.status_code == 200
        assert len(resp.json()["conversations"]) == 0

    async def test_delete_conversation(
        self, test_client: AsyncClient, manager, admin_headers
    ):
        conv = await manager.create_conversation(Mode.ACTIONABLE)
        cid = conv.conversation_id

        # Delete as admin
        resp = await test_client.delete(
            f"/api/admin/conversations/{cid}", headers=admin_headers
        )
        assert resp.status_code == 200

        # Now get should return 404
        get_resp = await test_client.get(
            f"/api/admin/conversations/{cid}", headers=admin_headers
        )
        assert get_resp.status_code == 404

    async def test_export_conversation(
        self, test_client: AsyncClient, manager, admin_headers
    ):
        conv = await manager.create_conversation(Mode.ACTIONABLE)
        conv.messages = [
            Message(role=MessageRole.USER, content="Hello legal team"),
            Message(role=MessageRole.ASSISTANT, content="Hello, how can I advise?"),
        ]
        await manager._repo.update(conv)

        # JSON export
        resp_json = await test_client.get(
            f"/api/admin/conversations/{conv.conversation_id}/export?format=json",
            headers=admin_headers,
        )
        assert resp_json.status_code == 200
        assert resp_json.headers["content-type"].startswith("application/json")
        data = resp_json.json()
        assert data["conversation_id"] == conv.conversation_id

        # Markdown export
        resp_md = await test_client.get(
            f"/api/admin/conversations/{conv.conversation_id}/export?format=markdown",
            headers=admin_headers,
        )
        assert resp_md.status_code == 200
        assert "Legal Assistance Consultation Transcript" in resp_md.text
        assert "Hello legal team" in resp_md.text

    async def test_admin_dashboard_html_view(self, test_client: AsyncClient):
        resp = await test_client.get("/admin")
        assert resp.status_code == 200
        assert "<!DOCTYPE html>" in resp.text
        assert "Legal Assistance Admin Portal" in resp.text
        assert "Administrator Login" in resp.text
