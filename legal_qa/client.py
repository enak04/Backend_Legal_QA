"""
HTTP client for the Legal_QA inference service.

Communicates with Legal_QA **only** through its ``POST /predict``
endpoint.  Never imports Legal_QA code or loads its models.

All errors are translated into application-specific exceptions so
that the rest of the backend can handle them uniformly.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# ── Custom exceptions ─────────────────────────────────────────

class LegalQAError(Exception):
    """Base class for Legal_QA client errors."""


class LegalQAUnavailableError(LegalQAError):
    """Legal_QA service could not be reached."""


class LegalQATimeoutError(LegalQAError):
    """Legal_QA did not respond within the timeout."""


class LegalQABadResponseError(LegalQAError):
    """Legal_QA returned a malformed or error response."""


# ── Client ────────────────────────────────────────────────────

class LegalQAClient:
    """
    Async HTTP client for the Legal_QA service.

    Parameters
    ----------
    base_url : str
        Root URL of the Legal_QA service (e.g. ``http://localhost:8000``).
    timeout : int
        Request timeout in seconds.  Defaults to 120 because ML
        inference (HKG + PPO + DSSM + GPT-4o) can be slow.
    """

    def __init__(self, base_url: str, timeout: int = 120) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def predict(self, question: str, mode: str) -> dict[str, Any]:
        """
        Call ``POST /predict`` on the Legal_QA service.

        Parameters
        ----------
        question : str
            The constructed legal question.
        mode : str
            One of ``"actionable"``, ``"informative"``, ``"readable"``.

        Returns
        -------
        dict
            The Legal_QA response containing ``question``, ``answer``,
            ``reasoning_chain``, and ``retrieved_cases``.

        Raises
        ------
        LegalQAUnavailableError
            Could not connect to Legal_QA.
        LegalQATimeoutError
            Legal_QA did not respond in time.
        LegalQABadResponseError
            Legal_QA returned an error or malformed JSON.
        """
        payload = {"question": question, "mode": mode}
        url = f"{self._base_url}/predict"

        logger.info("Legal_QA request → %s  mode=%s", url, mode)

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(timeout=self._timeout, connect=1.5)
            ) as client:
                response = await client.post(url, json=payload)
        except httpx.ConnectError as exc:
            logger.error("Legal_QA unreachable: %s", exc)
            raise LegalQAUnavailableError(
                f"Cannot connect to Legal_QA at {self._base_url}"
            ) from exc
        except httpx.TimeoutException as exc:
            logger.error("Legal_QA timeout after %ds", self._timeout)
            raise LegalQATimeoutError(
                f"Legal_QA did not respond within {self._timeout}s"
            ) from exc
        except httpx.HTTPError as exc:
            logger.error("Legal_QA HTTP error: %s", exc)
            raise LegalQAUnavailableError(str(exc)) from exc

        # ── Validate response ────────────────────────────────
        if response.status_code != 200:
            error_detail = response.text[:500]
            logger.error(
                "Legal_QA returned %d: %s",
                response.status_code,
                error_detail,
            )
            raise LegalQABadResponseError(
                f"Legal_QA returned HTTP {response.status_code}: "
                f"{error_detail}"
            )

        try:
            data = response.json()
        except Exception as exc:
            raise LegalQABadResponseError(
                "Legal_QA returned invalid JSON"
            ) from exc

        # Sanity-check required fields
        for field in ("question", "answer", "reasoning_chain", "retrieved_cases"):
            if field not in data:
                raise LegalQABadResponseError(
                    f"Legal_QA response missing required field: {field}"
                )

        return data

    async def health_check(self) -> bool:
        """
        Check if Legal_QA is reachable and healthy.

        Returns ``True`` if the service responds with status ``ok``.
        """
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10)) as client:
                resp = await client.get(f"{self._base_url}/health")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False
