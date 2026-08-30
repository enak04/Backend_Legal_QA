"""
Tests for the Legal_QA HTTP client.

All tests mock the HTTP transport layer so no real Legal_QA
service is needed.
"""

from __future__ import annotations

import pytest
import respx
from httpx import Response

from legal_qa.client import (
    LegalQABadResponseError,
    LegalQAClient,
    LegalQATimeoutError,
    LegalQAUnavailableError,
)

pytestmark = pytest.mark.anyio

BASE_URL = "http://mock-legal-qa:8000"


class TestLegalQAClient:
    @respx.mock
    async def test_successful_predict(self):
        respx.post(f"{BASE_URL}/predict").mock(
            return_value=Response(
                200,
                json={
                    "question": "test",
                    "answer": "test answer",
                    "reasoning_chain": ["step1"],
                    "retrieved_cases": [
                        {"question": "q", "answer": "a"}
                    ],
                },
            )
        )

        client = LegalQAClient(BASE_URL, timeout=10)
        result = await client.predict("test", "actionable")

        assert result["answer"] == "test answer"
        assert result["reasoning_chain"] == ["step1"]

    @respx.mock
    async def test_predict_sends_correct_payload(self):
        route = respx.post(f"{BASE_URL}/predict").mock(
            return_value=Response(
                200,
                json={
                    "question": "q",
                    "answer": "a",
                    "reasoning_chain": [],
                    "retrieved_cases": [],
                },
            )
        )

        client = LegalQAClient(BASE_URL, timeout=10)
        await client.predict("What is bail?", "informative")

        request = route.calls[0].request
        import json
        body = json.loads(request.content)
        assert body["question"] == "What is bail?"
        assert body["mode"] == "informative"

    @respx.mock
    async def test_bad_status_raises(self):
        respx.post(f"{BASE_URL}/predict").mock(
            return_value=Response(500, json={"error": "internal"})
        )

        client = LegalQAClient(BASE_URL, timeout=10)
        with pytest.raises(LegalQABadResponseError):
            await client.predict("test", "actionable")

    @respx.mock
    async def test_missing_field_raises(self):
        respx.post(f"{BASE_URL}/predict").mock(
            return_value=Response(
                200,
                json={"question": "test", "answer": "test"},
                # Missing reasoning_chain and retrieved_cases
            )
        )

        client = LegalQAClient(BASE_URL, timeout=10)
        with pytest.raises(LegalQABadResponseError, match="missing"):
            await client.predict("test", "actionable")

    @respx.mock
    async def test_health_check_success(self):
        respx.get(f"{BASE_URL}/health").mock(
            return_value=Response(200, json={"status": "ok"})
        )

        client = LegalQAClient(BASE_URL)
        assert await client.health_check() is True

    @respx.mock
    async def test_health_check_failure(self):
        respx.get(f"{BASE_URL}/health").mock(
            return_value=Response(500)
        )

        client = LegalQAClient(BASE_URL)
        assert await client.health_check() is False
