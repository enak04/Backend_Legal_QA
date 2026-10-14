"""
Admin authentication and session management.

Provides token-based authentication for administrative endpoints and the
Admin Dashboard using HMAC-SHA256 signatures (standard JWT-compatible).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from typing import Any

from fastapi import HTTPException, Request, status

from config import settings

logger = logging.getLogger(__name__)


def _b64_encode(data: bytes) -> str:
    """Base64url encode without padding."""
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64_decode(data_str: str) -> bytes:
    """Base64url decode with automatic padding restoration."""
    rem = len(data_str) % 4
    if rem:
        data_str += "=" * (4 - rem)
    return base64.urlsafe_b64decode(data_str.encode("utf-8"))


def create_admin_token(username: str, expiry_hours: int | None = None) -> str:
    """Generate a signed access token for the admin user."""
    hours = expiry_hours or settings.admin_session_expiry_hours
    now = int(time.time())
    exp = now + (hours * 3600)

    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": username,
        "role": "admin",
        "iat": now,
        "exp": exp,
        "jti": secrets.token_hex(8),
    }

    hdr_b64 = _b64_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{hdr_b64}.{payload_b64}".encode("utf-8")

    signature = hmac.new(
        settings.admin_secret_key.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()
    sig_b64 = _b64_encode(signature)

    return f"{hdr_b64}.{payload_b64}.{sig_b64}"


def verify_admin_token(token: str) -> dict[str, Any] | None:
    """
    Verify the token signature and expiration.
    Returns the decoded payload dict if valid, else None.
    """
    if not token or not isinstance(token, str):
        return None

    parts = token.strip().split(".")
    if len(parts) != 3:
        return None

    hdr_b64, payload_b64, sig_b64 = parts
    signing_input = f"{hdr_b64}.{payload_b64}".encode("utf-8")

    expected_sig = hmac.new(
        settings.admin_secret_key.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()

    try:
        actual_sig = _b64_decode(sig_b64)
    except Exception:
        return None

    if not hmac.compare_digest(expected_sig, actual_sig):
        return None

    try:
        payload_bytes = _b64_decode(payload_b64)
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception:
        return None

    # Check expiration
    exp = payload.get("exp")
    if exp is None or time.time() > float(exp):
        return None

    return payload


def verify_admin_credentials(username: str, password: str) -> bool:
    """Timing-safe check of username and password against configured values."""
    user_match = secrets.compare_digest(
        username.strip().encode("utf-8"),
        settings.admin_username.encode("utf-8"),
    )
    pass_match = secrets.compare_digest(
        password.encode("utf-8"),
        settings.admin_password.encode("utf-8"),
    )
    return user_match and pass_match


async def get_current_admin(request: Request) -> dict[str, Any]:
    """
    FastAPI dependency to extract and authenticate the admin user.
    Checks:
      1. Authorization header: Bearer <token>
      2. Cookie: admin_token
      3. Query parameter: token
    """
    token: str | None = None

    # 1. Bearer header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[len("Bearer "):].strip()

    # 2. Cookie
    if not token:
        token = request.cookies.get("admin_token")

    # 3. Query param
    if not token:
        token = request.query_params.get("token")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in as admin.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = verify_admin_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired admin session token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload
