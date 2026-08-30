"""
Shared pytest fixtures for the test suite.

Provides:
  - An in-memory SQLite repository
  - A mock Legal_QA client
  - A configured ConversationManager
  - A FastAPI TestClient
"""

from __future__ import annotations

import os
import tempfile
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Ensure test config before importing app modules
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_conversations.db")
os.environ.setdefault("LEGAL_QA_BASE_URL", "http://mock-legal-qa:8000")
os.environ.setdefault("CORS_ORIGINS", "*")

from app import app
from conversation.followup import FollowUpEngine
from conversation.manager import ConversationManager
from database.models import Mode
from database.repository import SQLiteConversationRepository
from legal_qa.client import LegalQAClient
from legal_qa.query_builder import QueryBuilder


# ── Fake Legal_QA response ───────────────────────────────────

MOCK_LEGAL_QA_RESPONSE: dict[str, Any] = {
    "question": "test question",
    "answer": (
        "Under the Payment of Wages Act, 1936, your employer is "
        "obligated to pay wages within the prescribed period."
    ),
    "reasoning_chain": [
        "Identified relevant statute: Payment of Wages Act, 1936",
        "Karnataka labor tribunal has jurisdiction",
        "Remedies include filing complaint with labor commissioner",
    ],
    "retrieved_cases": [
        {
            "question": "What remedies exist for unpaid wages?",
            "answer": (
                "Under Section 15 of the Payment of Wages Act, a "
                "worker may file a complaint."
            ),
        }
    ],
}


# ── Fixtures ──────────────────────────────────────────────────

@pytest_asyncio.fixture
async def db_repo(tmp_path):
    """Fresh SQLite repository per test, stored in a temp directory."""
    db_path = str(tmp_path / "test.db")
    repo = SQLiteConversationRepository(db_path)
    await repo.initialize()
    return repo


@pytest.fixture
def mock_legal_qa_client():
    """Mock LegalQAClient whose predict() returns canned data."""
    client = AsyncMock(spec=LegalQAClient)
    client.predict.return_value = MOCK_LEGAL_QA_RESPONSE.copy()
    client.health_check.return_value = True
    return client


@pytest_asyncio.fixture
async def manager(db_repo, mock_legal_qa_client):
    """ConversationManager wired with test doubles."""
    return ConversationManager(
        repository=db_repo,
        legal_qa_client=mock_legal_qa_client,
        followup_engine=FollowUpEngine(),
        query_builder=QueryBuilder(),
    )


@pytest_asyncio.fixture
async def test_client(db_repo, mock_legal_qa_client):
    """
    HTTPX AsyncClient pointed at the FastAPI app.

    Overrides app.state so routes use the test repository and
    mock Legal_QA client.
    """
    mgr = ConversationManager(
        repository=db_repo,
        legal_qa_client=mock_legal_qa_client,
        followup_engine=FollowUpEngine(),
        query_builder=QueryBuilder(),
    )
    app.state.conversation_manager = mgr

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        yield client
