"""
Tests for the modes endpoint.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.anyio


class TestModesEndpoint:
    async def test_list_modes(self, test_client):
        resp = await test_client.get("/api/modes")
        assert resp.status_code == 200
        modes = resp.json()
        assert len(modes) == 3

        mode_ids = [m["id"] for m in modes]
        assert "actionable" in mode_ids
        assert "informative" in mode_ids
        assert "readable" in mode_ids

    async def test_modes_have_descriptions(self, test_client):
        resp = await test_client.get("/api/modes")
        modes = resp.json()
        for mode in modes:
            assert "name" in mode
            assert "description" in mode
            assert len(mode["description"]) > 20


class TestHealthEndpoint:
    async def test_health_returns_ok(self, test_client):
        resp = await test_client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
