"""
Admin routes and web dashboard for Legal_QA application backend.

Provides:
  - Admin login/logout authentication endpoints
  - Aggregated analytics and stats
  - Full access to all conversations till date
  - Complete transcript of messages sent by user and system/assistant
  - Search, filtering, pagination, and chat export
  - Self-contained, responsive Admin Dashboard Web UI at /admin
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from config import settings
from conversation.manager import ConversationManager
from database.models import ConversationRecord
from services.admin_auth import (
    create_admin_token,
    get_current_admin,
    verify_admin_credentials,
)

router = APIRouter(tags=["Admin"])


# ── Schemas ───────────────────────────────────────────────────

class AdminLoginRequest(BaseModel):
    username: str = Field(..., description="Admin username")
    password: str = Field(..., description="Admin password")


class AdminLoginResponse(BaseModel):
    token: str
    username: str
    role: str = "admin"
    expires_in_hours: int


# ── Helper ────────────────────────────────────────────────────

def _manager(request: Request) -> ConversationManager:
    return request.app.state.conversation_manager


def _format_conversation_summary(record: ConversationRecord) -> dict[str, Any]:
    """Format a conversation record into a lightweight summary for table display."""
    user_count = sum(1 for m in record.messages if (m.role.value if hasattr(m.role, "value") else str(m.role)) == "user")
    assistant_count = len(record.messages) - user_count
    last_msg = record.messages[-1] if record.messages else None

    return {
        "conversation_id": record.conversation_id,
        "mode": record.mode.value if hasattr(record.mode, "value") else str(record.mode),
        "stage": record.stage.value if hasattr(record.stage, "value") else str(record.stage),
        "language": record.language,
        "total_messages": len(record.messages),
        "user_messages_count": user_count,
        "system_messages_count": assistant_count,
        "last_message": {
            "role": last_msg.role.value if hasattr(last_msg.role, "value") else str(last_msg.role),
            "content": last_msg.content[:160] + ("…" if len(last_msg.content) > 160 else ""),
            "timestamp": last_msg.timestamp,
        } if last_msg else None,
        "files_count": len(record.files),
        "has_legal_qa": len(record.legal_qa_results) > 0,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def _format_conversation_detail(record: ConversationRecord) -> dict[str, Any]:
    """Format a complete conversation record with full message history and metadata."""
    messages = []
    user_count = 0
    assistant_count = 0

    for m in record.messages:
        role_str = m.role.value if hasattr(m.role, "value") else str(m.role)
        if role_str == "user":
            user_count += 1
        else:
            assistant_count += 1

        messages.append({
            "role": role_str,
            "sender_label": "User" if role_str == "user" else "System / Assistant",
            "content": m.content,
            "timestamp": m.timestamp,
        })

    return {
        "conversation_id": record.conversation_id,
        "mode": record.mode.value if hasattr(record.mode, "value") else str(record.mode),
        "stage": record.stage.value if hasattr(record.stage, "value") else str(record.stage),
        "language": record.language,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "last_assistant_question": record.last_assistant_question,
        "stats": {
            "total_messages": len(messages),
            "user_messages": user_count,
            "system_messages": assistant_count,
        },
        "messages": messages,
        "facts": record.facts,
        "files": [f.model_dump() for f in record.files],
        "legal_qa_results": [r.model_dump() for r in record.legal_qa_results],
    }


# ── Auth Endpoints ────────────────────────────────────────────

@router.post("/api/admin/login", response_model=AdminLoginResponse)
async def admin_login(body: AdminLoginRequest, response: Response):
    """
    Authenticate admin credentials and return signed access token.
    Sets an HTTP-only cookie for web browser dashboard access.
    """
    if not verify_admin_credentials(body.username, body.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin username or password.",
        )

    token = create_admin_token(body.username)
    max_age = settings.admin_session_expiry_hours * 3600

    # Set cookie for browser session
    response.set_cookie(
        key="admin_token",
        value=token,
        max_age=max_age,
        httponly=True,
        samesite="lax",
        secure=False,  # Set to True when HTTPS is enabled
        path="/",
    )

    return AdminLoginResponse(
        token=token,
        username=body.username,
        role="admin",
        expires_in_hours=settings.admin_session_expiry_hours,
    )


@router.post("/api/admin/logout")
async def admin_logout(response: Response):
    """Log out and clear the admin session cookie."""
    response.delete_cookie(key="admin_token", path="/")
    return {"message": "Admin session logged out successfully."}


@router.get("/api/admin/me")
async def admin_me(admin: dict[str, Any] = Depends(get_current_admin)):
    """Check current admin session validity."""
    return {
        "authenticated": True,
        "username": admin.get("sub", settings.admin_username),
        "role": admin.get("role", "admin"),
        "expires_at": admin.get("exp"),
    }


# ── Analytics & Stats ─────────────────────────────────────────

@router.get("/api/admin/stats")
async def get_admin_stats(
    request: Request,
    admin: dict[str, Any] = Depends(get_current_admin),
) -> dict[str, Any]:
    """Retrieve system overview statistics across all chats till date."""
    manager = _manager(request)
    stats = await manager.get_conversation_stats()
    return stats


# ── Conversation Management Endpoints ─────────────────────────

@router.get("/api/admin/conversations")
async def list_admin_conversations(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    search: str | None = Query(None, description="Search conversation ID, messages, or facts"),
    stage: str | None = Query(None, description="Filter by stage"),
    mode: str | None = Query(None, description="Filter by mode"),
    admin: dict[str, Any] = Depends(get_current_admin),
) -> dict[str, Any]:
    """
    List all conversations till date with filtering, search, and pagination.
    Shows message counts for both user and system.
    """
    manager = _manager(request)
    offset = (page - 1) * page_size
    records, total = await manager.list_conversations(
        limit=page_size,
        offset=offset,
        search=search,
        stage=stage,
        mode=mode,
    )

    total_pages = (total + page_size - 1) // page_size if total > 0 else 1

    return {
        "conversations": [_format_conversation_summary(r) for r in records],
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
        },
    }


@router.get("/api/admin/conversations/{conversation_id}")
async def get_admin_conversation_detail(
    conversation_id: str,
    request: Request,
    admin: dict[str, Any] = Depends(get_current_admin),
) -> dict[str, Any]:
    """
    Get full conversation state, including all messages sent by user and system,
    extracted facts, legal QA research results, and file attachments.
    """
    manager = _manager(request)
    record = await manager.get_conversation(conversation_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation not found: {conversation_id}",
        )

    return _format_conversation_detail(record)


@router.delete("/api/admin/conversations/{conversation_id}")
async def delete_admin_conversation(
    conversation_id: str,
    request: Request,
    admin: dict[str, Any] = Depends(get_current_admin),
) -> dict[str, Any]:
    """Delete a conversation and its attached files."""
    manager = _manager(request)
    deleted = await manager.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation not found: {conversation_id}",
        )
    return {"message": f"Conversation {conversation_id} deleted successfully."}


@router.get("/api/admin/conversations/{conversation_id}/export")
async def export_admin_conversation(
    conversation_id: str,
    request: Request,
    format: str = Query("json", pattern="^(json|markdown|txt)$"),
    admin: dict[str, Any] = Depends(get_current_admin),
):
    """Export conversation transcript in JSON, Markdown, or Plain Text format."""
    manager = _manager(request)
    record = await manager.get_conversation(conversation_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation not found: {conversation_id}",
        )

    detail = _format_conversation_detail(record)

    if format == "json":
        return Response(
            content=json.dumps(detail, indent=2),
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="chat_{conversation_id}.json"'
            },
        )

    # Text / Markdown transcript
    lines = [
        f"# Legal Assistance Consultation Transcript",
        f"- **Conversation ID:** {detail['conversation_id']}",
        f"- **Mode:** {detail['mode']}",
        f"- **Stage:** {detail['stage']}",
        f"- **Language:** {detail['language']}",
        f"- **Created:** {detail['created_at']}",
        f"- **Updated:** {detail['updated_at']}",
        f"- **Total Messages:** {detail['stats']['total_messages']} ({detail['stats']['user_messages']} user, {detail['stats']['system_messages']} system)",
        "",
        "## Messages Transcript",
        "",
    ]

    for idx, msg in enumerate(detail["messages"], start=1):
        role_label = msg["sender_label"].upper()
        lines.append(f"### [{idx}] {role_label} ({msg['timestamp']})")
        lines.append(msg["content"])
        lines.append("")

    if detail.get("facts"):
        lines.append("## Extracted Facts")
        for k, v in detail["facts"].items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")

    if detail.get("legal_qa_results"):
        lines.append("## Legal QA Results")
        for res in detail["legal_qa_results"]:
            lines.append(f"**Synthesized Answer:** {res.get('answer', '')}")
            lines.append("")

    content_str = "\n".join(lines)
    ext = "md" if format == "markdown" else "txt"
    media = "text/markdown" if format == "markdown" else "text/plain"

    return Response(
        content=content_str,
        media_type=media,
        headers={
            "Content-Disposition": f'attachment; filename="chat_{conversation_id}.{ext}"'
        },
    )


# ── Admin Dashboard Single Page Application ───────────────────

_ADMIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Legal QA Admin Portal | Conversation Monitor</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-base: #090d16;
      --bg-surface: #0f172a;
      --bg-card: rgba(30, 41, 59, 0.7);
      --bg-card-hover: rgba(51, 65, 85, 0.8);
      --border-subtle: rgba(148, 163, 184, 0.12);
      --border-active: rgba(99, 102, 241, 0.4);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --primary: #6366f1;
      --primary-hover: #4f46e5;
      --primary-glow: rgba(99, 102, 241, 0.25);
      --accent-cyan: #06b6d4;
      --accent-emerald: #10b981;
      --accent-rose: #f43f5e;
      --accent-amber: #f59e0b;
      --user-bubble: #1e1b4b;
      --user-border: rgba(99, 102, 241, 0.3);
      --system-bubble: #064e3b;
      --system-border: rgba(16, 185, 129, 0.3);
      --shadow-lg: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
      --radius-sm: 6px;
      --radius-md: 10px;
      --radius-lg: 16px;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      background-color: var(--bg-base);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
      line-height: 1.5;
    }

    /* Scrollbars */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg-base); }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }

    /* Login Modal */
    #login-overlay {
      position: fixed;
      inset: 0;
      background: rgba(9, 13, 22, 0.85);
      backdrop-filter: blur(12px);
      z-index: 100;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 1.5rem;
    }

    .login-card {
      width: 100%;
      max-width: 420px;
      background: var(--bg-surface);
      border: 1px solid var(--border-active);
      border-radius: var(--radius-lg);
      padding: 2.5rem;
      box-shadow: 0 25px 50px -12px rgba(99, 102, 241, 0.15);
      text-align: center;
      animation: popIn 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }

    @keyframes popIn {
      from { transform: scale(0.95); opacity: 0; }
      to { transform: scale(1); opacity: 1; }
    }

    .brand-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      background: rgba(99, 102, 241, 0.1);
      border: 1px solid rgba(99, 102, 241, 0.3);
      padding: 0.35rem 0.85rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 700;
      color: var(--primary);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 1.25rem;
    }

    .login-title {
      font-size: 1.6rem;
      font-weight: 800;
      margin-bottom: 0.5rem;
      background: linear-gradient(135deg, #fff 40%, #94a3b8 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .login-desc {
      color: var(--text-muted);
      font-size: 0.875rem;
      margin-bottom: 2rem;
    }

    .form-group {
      text-align: left;
      margin-bottom: 1.25rem;
    }

    .form-label {
      display: block;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 0.4rem;
    }

    .form-input {
      width: 100%;
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 0.75rem 1rem;
      color: var(--text-main);
      font-size: 0.95rem;
      outline: none;
      transition: all 0.2s ease;
    }

    .form-input:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 3px var(--primary-glow);
    }

    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      font-family: inherit;
      font-size: 0.875rem;
      font-weight: 600;
      padding: 0.75rem 1.25rem;
      border-radius: var(--radius-md);
      cursor: pointer;
      transition: all 0.2s ease;
      border: 1px solid transparent;
      text-decoration: none;
    }

    .btn-primary {
      background: var(--primary);
      color: white;
      width: 100%;
      box-shadow: 0 4px 12px var(--primary-glow);
    }

    .btn-primary:hover {
      background: var(--primary-hover);
      transform: translateY(-1px);
    }

    .btn-primary:active {
      transform: translateY(0);
    }

    .btn-secondary {
      background: rgba(255, 255, 255, 0.05);
      border-color: var(--border-subtle);
      color: var(--text-main);
    }

    .btn-secondary:hover {
      background: rgba(255, 255, 255, 0.1);
      border-color: rgba(255, 255, 255, 0.2);
    }

    .btn-danger {
      background: rgba(244, 63, 94, 0.15);
      border-color: rgba(244, 63, 94, 0.3);
      color: var(--accent-rose);
    }

    .btn-danger:hover {
      background: rgba(244, 63, 94, 0.25);
    }

    .btn-sm {
      padding: 0.4rem 0.75rem;
      font-size: 0.8rem;
    }

    /* Main App Layout */
    .app-header {
      background: rgba(15, 23, 42, 0.8);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border-subtle);
      position: sticky;
      top: 0;
      z-index: 50;
      padding: 0.85rem 2rem;
    }

    .header-inner {
      max-width: 1600px;
      margin: 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
    }

    .header-brand {
      display: flex;
      align-items: center;
      gap: 0.85rem;
    }

    .brand-icon {
      width: 36px;
      height: 36px;
      background: linear-gradient(135deg, var(--primary), var(--accent-cyan));
      border-radius: var(--radius-sm);
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-weight: 800;
      font-size: 1.1rem;
      box-shadow: 0 4px 10px rgba(99, 102, 241, 0.3);
    }

    .header-title-group h1 {
      font-size: 1.15rem;
      font-weight: 800;
      letter-spacing: -0.02em;
    }

    .header-title-group p {
      font-size: 0.75rem;
      color: var(--text-muted);
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 1rem;
    }

    .user-pill {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      background: rgba(30, 41, 59, 0.6);
      border: 1px solid var(--border-subtle);
      padding: 0.35rem 0.85rem;
      border-radius: 9999px;
      font-size: 0.8rem;
      color: var(--text-muted);
    }

    .user-pill .dot {
      width: 8px;
      height: 8px;
      background: var(--accent-emerald);
      border-radius: 50%;
      box-shadow: 0 0 8px var(--accent-emerald);
    }

    .app-main {
      flex: 1;
      max-width: 1600px;
      width: 100%;
      margin: 0 auto;
      padding: 2rem;
      display: flex;
      flex-direction: column;
      gap: 2rem;
    }

    /* Stats Grid */
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 1.25rem;
    }

    .stat-card {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: 1.5rem;
      position: relative;
      overflow: hidden;
      transition: border-color 0.2s ease, transform 0.2s ease;
    }

    .stat-card:hover {
      border-color: rgba(99, 102, 241, 0.3);
      transform: translateY(-2px);
    }

    .stat-card::after {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 2px;
      background: linear-gradient(90deg, transparent, var(--stat-color, var(--primary)), transparent);
      opacity: 0.8;
    }

    .stat-label {
      font-size: 0.8rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 0.5rem;
    }

    .stat-value {
      font-size: 2.2rem;
      font-weight: 800;
      color: var(--text-main);
      display: flex;
      align-items: baseline;
      gap: 0.5rem;
    }

    .stat-sub {
      font-size: 0.8rem;
      color: var(--text-dim);
      margin-top: 0.35rem;
    }

    /* Controls Bar */
    .controls-bar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: 1rem 1.25rem;
    }

    .search-box {
      flex: 1;
      min-width: 280px;
      position: relative;
    }

    .search-box input {
      width: 100%;
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 0.65rem 1rem 0.65rem 2.5rem;
      color: var(--text-main);
      font-size: 0.875rem;
      outline: none;
      transition: all 0.2s ease;
    }

    .search-box input:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 3px var(--primary-glow);
    }

    .search-icon {
      position: absolute;
      left: 0.85rem;
      top: 50%;
      transform: translateY(-50%);
      color: var(--text-dim);
      pointer-events: none;
    }

    .filter-group {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    select.filter-select {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 0.65rem 1rem;
      color: var(--text-main);
      font-size: 0.85rem;
      outline: none;
      cursor: pointer;
    }

    /* Content Layout (Split View) */
    .content-layout {
      display: grid;
      grid-template-columns: 1fr 1.35fr;
      gap: 1.5rem;
      align-items: start;
    }

    @media (max-width: 1024px) {
      .content-layout {
        grid-template-columns: 1fr;
      }
    }

    /* Conversation Table / List */
    .list-container {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }

    .list-header {
      padding: 1rem 1.25rem;
      border-bottom: 1px solid var(--border-subtle);
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-weight: 700;
      font-size: 0.95rem;
    }

    .conv-items {
      max-height: 750px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
    }

    .conv-card {
      padding: 1.15rem 1.25rem;
      border-bottom: 1px solid var(--border-subtle);
      cursor: pointer;
      transition: all 0.15s ease;
      background: transparent;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      position: relative;
    }

    .conv-card:hover {
      background: rgba(255, 255, 255, 0.02);
    }

    .conv-card.active {
      background: rgba(99, 102, 241, 0.08);
      border-left: 3px solid var(--primary);
    }

    .conv-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
    }

    .conv-id {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--accent-cyan);
    }

    .conv-date {
      font-size: 0.75rem;
      color: var(--text-dim);
    }

    .conv-badges {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.4rem;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
      padding: 0.2rem 0.55rem;
      border-radius: 9999px;
      font-size: 0.725rem;
      font-weight: 600;
      text-transform: capitalize;
    }

    .badge-mode {
      background: rgba(99, 102, 241, 0.15);
      color: #a5b4fc;
      border: 1px solid rgba(99, 102, 241, 0.25);
    }

    .badge-stage {
      background: rgba(6, 182, 212, 0.15);
      color: #67e8f9;
      border: 1px solid rgba(6, 182, 212, 0.25);
    }

    .badge-stage.answered {
      background: rgba(16, 185, 129, 0.15);
      color: #6ee7b7;
      border-color: rgba(16, 185, 129, 0.25);
    }

    .badge-msg-count {
      background: rgba(248, 250, 252, 0.08);
      color: var(--text-muted);
    }

    .conv-snippet {
      font-size: 0.825rem;
      color: var(--text-muted);
      line-height: 1.4;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }

    /* Inspector / Viewer Panel */
    .inspector-container {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      display: flex;
      flex-direction: column;
      height: 820px;
      overflow: hidden;
    }

    .inspector-header {
      padding: 1.25rem;
      border-bottom: 1px solid var(--border-subtle);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      background: rgba(15, 23, 42, 0.9);
    }

    .inspector-title {
      font-size: 1.05rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }

    .inspector-actions {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .inspector-meta {
      padding: 0.75rem 1.25rem;
      background: rgba(0, 0, 0, 0.2);
      border-bottom: 1px solid var(--border-subtle);
      display: flex;
      flex-wrap: wrap;
      gap: 1.5rem;
      font-size: 0.8rem;
    }

    .meta-item {
      display: flex;
      align-items: center;
      gap: 0.4rem;
      color: var(--text-muted);
    }

    .meta-item strong {
      color: var(--text-main);
    }

    .inspector-tabs {
      display: flex;
      border-bottom: 1px solid var(--border-subtle);
      background: rgba(15, 23, 42, 0.5);
    }

    .tab-btn {
      padding: 0.65rem 1.25rem;
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-muted);
      background: none;
      border: none;
      border-bottom: 2px solid transparent;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .tab-btn:hover {
      color: var(--text-main);
    }

    .tab-btn.active {
      color: var(--primary);
      border-bottom-color: var(--primary);
      background: rgba(99, 102, 241, 0.05);
    }

    .inspector-body {
      flex: 1;
      overflow-y: auto;
      padding: 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
      background: radial-gradient(circle at top right, rgba(99, 102, 241, 0.03), transparent 70%);
    }

    /* Message Bubbles */
    .msg-group {
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      max-width: 88%;
      animation: fadeIn 0.2s ease;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .msg-group.user {
      align-self: flex-start;
    }

    .msg-group.assistant {
      align-self: flex-end;
    }

    .msg-header {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.725rem;
      color: var(--text-dim);
      padding: 0 0.25rem;
    }

    .msg-group.assistant .msg-header {
      flex-direction: row-reverse;
    }

    .role-tag {
      font-weight: 700;
      text-transform: uppercase;
      font-size: 0.65rem;
      letter-spacing: 0.04em;
      padding: 0.15rem 0.45rem;
      border-radius: 4px;
    }

    .role-tag.user {
      background: rgba(99, 102, 241, 0.25);
      color: #c7d2fe;
    }

    .role-tag.system {
      background: rgba(16, 185, 129, 0.25);
      color: #a7f3d0;
    }

    .msg-content {
      padding: 1rem 1.25rem;
      border-radius: var(--radius-md);
      font-size: 0.9rem;
      line-height: 1.6;
      white-space: pre-wrap;
      word-break: break-word;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    .msg-group.user .msg-content {
      background: var(--user-bubble);
      border: 1px solid var(--user-border);
      color: #f1f5f9;
      border-top-left-radius: 2px;
    }

    .msg-group.assistant .msg-content {
      background: var(--system-bubble);
      border: 1px solid var(--system-border);
      color: #ecfdf5;
      border-top-right-radius: 2px;
    }

    /* Structured Panels in Inspector */
    .detail-card {
      background: rgba(15, 23, 42, 0.7);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 1.25rem;
    }

    .detail-title {
      font-size: 0.875rem;
      font-weight: 700;
      color: var(--accent-cyan);
      margin-bottom: 0.75rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    pre.code-block {
      background: #060911;
      border: 1px solid rgba(255, 255, 255, 0.06);
      border-radius: var(--radius-sm);
      padding: 0.85rem;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.8rem;
      color: #cbd5e1;
      overflow-x: auto;
    }

    /* Empty States */
    .empty-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100%;
      min-height: 280px;
      padding: 2rem;
      text-align: center;
      color: var(--text-dim);
    }

    .empty-icon {
      font-size: 2.5rem;
      margin-bottom: 1rem;
      opacity: 0.6;
    }

    /* Pagination */
    .pagination-bar {
      padding: 0.85rem 1.25rem;
      border-top: 1px solid var(--border-subtle);
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 0.85rem;
      color: var(--text-muted);
    }

    /* Toast Notification */
    #toast {
      position: fixed;
      bottom: 2rem;
      right: 2rem;
      background: var(--bg-surface);
      border: 1px solid var(--border-active);
      color: var(--text-main);
      padding: 0.85rem 1.35rem;
      border-radius: var(--radius-md);
      box-shadow: var(--shadow-lg);
      z-index: 200;
      font-size: 0.875rem;
      display: none;
      align-items: center;
      gap: 0.6rem;
      animation: popIn 0.2s ease;
    }
  </style>
</head>
<body>

  <!-- Login Modal -->
  <div id="login-overlay">
    <div class="login-card">
      <div class="brand-badge">Legal QA Portal</div>
      <h2 class="login-title">Administrator Login</h2>
      <p class="login-desc">Sign in to monitor all active and past consultations</p>
      
      <form id="login-form">
        <div class="form-group">
          <label class="form-label" for="username">Username</label>
          <input type="text" id="username" class="form-input" required autocomplete="username" value="admin" placeholder="admin">
        </div>
        <div class="form-group">
          <label class="form-label" for="password">Password</label>
          <input type="password" id="password" class="form-input" required autocomplete="current-password" placeholder="••••••••">
        </div>
        <button type="submit" class="btn btn-primary" id="login-btn" style="margin-top: 0.75rem;">
          Authenticate & Access Chats
        </button>
      </form>
      <div id="login-error" style="color: var(--accent-rose); font-size: 0.825rem; margin-top: 1rem; display: none;"></div>
    </div>
  </div>

  <!-- Header -->
  <header class="app-header">
    <div class="header-inner">
      <div class="header-brand">
        <div class="brand-icon">⚖</div>
        <div class="header-title-group">
          <h1>Legal Assistance Admin Portal</h1>
          <p>Complete historical transcript of user and system messages</p>
        </div>
      </div>
      
      <div class="header-actions">
        <div class="user-pill">
          <span class="dot"></span>
          <span id="current-user-display">Admin</span>
        </div>
        <button class="btn btn-secondary btn-sm" id="refresh-btn" title="Refresh data">
          ⟳ Refresh
        </button>
        <button class="btn btn-danger btn-sm" id="logout-btn">
          Log Out
        </button>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <main class="app-main">
    
    <!-- Stats Cards -->
    <section class="stats-grid">
      <div class="stat-card" style="--stat-color: var(--primary);">
        <div class="stat-label">Total Consultations</div>
        <div class="stat-value" id="stat-total-chats">-</div>
        <div class="stat-sub">Persistent sessions till date</div>
      </div>
      <div class="stat-card" style="--stat-color: var(--accent-cyan);">
        <div class="stat-label">Total Messages</div>
        <div class="stat-value" id="stat-total-msgs">-</div>
        <div class="stat-sub">Across all conversations</div>
      </div>
      <div class="stat-card" style="--stat-color: #818cf8;">
        <div class="stat-label">User Inquiries</div>
        <div class="stat-value" id="stat-user-msgs">-</div>
        <div class="stat-sub">Sent by citizens & clients</div>
      </div>
      <div class="stat-card" style="--stat-color: var(--accent-emerald);">
        <div class="stat-label">System Responses</div>
        <div class="stat-value" id="stat-system-msgs">-</div>
        <div class="stat-sub">Grounded advice & triage</div>
      </div>
    </section>

    <!-- Controls & Filters -->
    <section class="controls-bar">
      <div class="search-box">
        <span class="search-icon">🔍</span>
        <input type="text" id="search-input" placeholder="Search by Conversation ID, client facts, or message content...">
      </div>

      <div class="filter-group">
        <select class="filter-select" id="filter-mode">
          <option value="">All Modes</option>
          <option value="actionable">Actionable</option>
          <option value="informative">Informative</option>
          <option value="readable">Readable</option>
        </select>

        <select class="filter-select" id="filter-stage">
          <option value="">All Stages</option>
          <option value="initial">Initial</option>
          <option value="gathering_info">Gathering Info</option>
          <option value="ready_for_qa">Ready for QA</option>
          <option value="answered">Answered</option>
        </select>

        <select class="filter-select" id="page-size-select">
          <option value="15">15 per page</option>
          <option value="25" selected>25 per page</option>
          <option value="50">50 per page</option>
        </select>
      </div>
    </section>

    <!-- Split View Content -->
    <div class="content-layout">
      
      <!-- Left: Conversation List -->
      <section class="list-container">
        <div class="list-header">
          <span>Conversations History</span>
          <span id="filtered-count" style="font-size: 0.8rem; color: var(--text-muted); font-weight: 500;">-</span>
        </div>
        
        <div class="conv-items" id="conversation-list">
          <div class="empty-state">
            <div class="empty-icon">⏳</div>
            <p>Loading conversations…</p>
          </div>
        </div>

        <div class="pagination-bar">
          <button class="btn btn-secondary btn-sm" id="prev-page-btn" disabled>← Previous</button>
          <span id="page-indicator">Page 1 of 1</span>
          <button class="btn btn-secondary btn-sm" id="next-page-btn" disabled>Next →</button>
        </div>
      </section>

      <!-- Right: Detailed Chat Inspector -->
      <section class="inspector-container">
        <div class="inspector-header">
          <div class="inspector-title">
            <span>💬 Chat Inspector:</span>
            <span id="selected-conv-id" style="font-family: 'JetBrains Mono', monospace; color: var(--accent-cyan); font-size: 0.9rem;">
              Select a chat
            </span>
          </div>
          <div class="inspector-actions" id="inspector-actions" style="display: none;">
            <button class="btn btn-secondary btn-sm" id="export-json-btn">JSON</button>
            <button class="btn btn-secondary btn-sm" id="export-txt-btn">Text</button>
            <button class="btn btn-danger btn-sm" id="delete-conv-btn">Delete</button>
          </div>
        </div>

        <div class="inspector-meta" id="inspector-meta" style="display: none;">
          <div class="meta-item"><span>Mode:</span> <strong id="meta-mode">-</strong></div>
          <div class="meta-item"><span>Stage:</span> <strong id="meta-stage">-</strong></div>
          <div class="meta-item"><span>Language:</span> <strong id="meta-lang">-</strong></div>
          <div class="meta-item"><span>Messages:</span> <strong id="meta-msgs">-</strong></div>
          <div class="meta-item"><span>Updated:</span> <strong id="meta-date">-</strong></div>
        </div>

        <div class="inspector-tabs" id="inspector-tabs" style="display: none;">
          <button class="tab-btn active" data-tab="messages">Messages Timeline</button>
          <button class="tab-btn" data-tab="facts">Extracted Facts</button>
          <button class="tab-btn" data-tab="legal_qa">Legal QA Precedents</button>
        </div>

        <div class="inspector-body" id="inspector-body">
          <div class="empty-state">
            <div class="empty-icon">👈</div>
            <h3 style="margin-bottom: 0.5rem; color: var(--text-main);">No Conversation Selected</h3>
            <p>Select any consultation from the list on the left to inspect every message exchanged between the user and the system.</p>
          </div>
        </div>
      </section>

    </div>

  </main>

  <div id="toast">Notice message</div>

  <script>
    // State management
    let state = {
      token: null,
      currentUser: null,
      currentPage: 1,
      pageSize: 25,
      search: '',
      stage: '',
      mode: '',
      conversations: [],
      selectedConvId: null,
      selectedConvData: null,
      activeTab: 'messages'
    };

    // Helper: Toast
    function showToast(msg) {
      const toast = document.getElementById('toast');
      toast.textContent = msg;
      toast.style.display = 'flex';
      setTimeout(() => { toast.style.display = 'none'; }, 3200);
    }

    // Auth Headers
    function getHeaders() {
      const h = { 'Content-Type': 'application/json' };
      if (state.token) {
        h['Authorization'] = `Bearer ${state.token}`;
      }
      return h;
    }

    // Check login state on boot
    async function checkAuth() {
      try {
        const res = await fetch('/api/admin/me', { headers: getHeaders() });
        if (res.ok) {
          const data = await res.json();
          state.currentUser = data.username;
          document.getElementById('current-user-display').textContent = data.username;
          document.getElementById('login-overlay').style.display = 'none';
          loadDashboard();
        } else {
          document.getElementById('login-overlay').style.display = 'flex';
        }
      } catch (err) {
        document.getElementById('login-overlay').style.display = 'flex';
      }
    }

    // Login action
    document.getElementById('login-form').addEventListener('submit', async (e) => {
      e.preventDefault();
      const user = document.getElementById('username').value.trim();
      const pass = document.getElementById('password').value;
      const errBox = document.getElementById('login-error');
      errBox.style.display = 'none';

      try {
        const res = await fetch('/api/admin/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username: user, password: pass })
        });
        if (!res.ok) {
          const err = await res.json();
          errBox.textContent = err.detail || 'Authentication failed. Please check credentials.';
          errBox.style.display = 'block';
          return;
        }
        const data = await res.json();
        state.token = data.token;
        state.currentUser = data.username;
        document.getElementById('current-user-display').textContent = data.username;
        document.getElementById('login-overlay').style.display = 'none';
        showToast('Logged in successfully!');
        loadDashboard();
      } catch (err) {
        errBox.textContent = 'Server connection error.';
        errBox.style.display = 'block';
      }
    });

    // Logout action
    document.getElementById('logout-btn').addEventListener('click', async () => {
      await fetch('/api/admin/logout', { method: 'POST', headers: getHeaders() });
      state.token = null;
      state.currentUser = null;
      document.getElementById('login-overlay').style.display = 'flex';
      showToast('Logged out.');
    });

    // Load Stats
    async function loadStats() {
      try {
        const res = await fetch('/api/admin/stats', { headers: getHeaders() });
        if (res.ok) {
          const s = await res.json();
          document.getElementById('stat-total-chats').textContent = s.total_conversations.toLocaleString();
          document.getElementById('stat-total-msgs').textContent = s.total_messages.toLocaleString();
          document.getElementById('stat-user-msgs').textContent = s.user_messages.toLocaleString();
          document.getElementById('stat-system-msgs').textContent = s.assistant_messages.toLocaleString();
        }
      } catch (e) {
        console.error('Failed to load stats', e);
      }
    }

    // Load Conversations List
    async function loadConversations() {
      const listEl = document.getElementById('conversation-list');
      listEl.innerHTML = '<div class="empty-state"><div class="empty-icon">⏳</div><p>Loading...</p></div>';

      const params = new URLSearchParams({
        page: state.currentPage,
        page_size: state.pageSize,
      });
      if (state.search) params.append('search', state.search);
      if (state.stage) params.append('stage', state.stage);
      if (state.mode) params.append('mode', state.mode);

      try {
        const res = await fetch(`/api/admin/conversations?${params.toString()}`, { headers: getHeaders() });
        if (!res.ok) {
          if (res.status === 401) checkAuth();
          return;
        }
        const data = await res.json();
        state.conversations = data.conversations;

        // Render Pagination
        const p = data.pagination;
        document.getElementById('filtered-count').textContent = `${p.total} total conversations`;
        document.getElementById('page-indicator').textContent = `Page ${p.page} of ${p.total_pages || 1}`;
        document.getElementById('prev-page-btn').disabled = !p.has_prev;
        document.getElementById('next-page-btn').disabled = !p.has_next;

        if (state.conversations.length === 0) {
          listEl.innerHTML = '<div class="empty-state"><div class="empty-icon">🔍</div><p>No matching conversations found.</p></div>';
          return;
        }

        listEl.innerHTML = '';
        state.conversations.forEach(c => {
          const card = document.createElement('div');
          card.className = `conv-card ${state.selectedConvId === c.conversation_id ? 'active' : ''}`;
          card.onclick = () => selectConversation(c.conversation_id);

          const timeStr = new Date(c.updated_at).toLocaleString();
          const lastText = c.last_message ? `${c.last_message.role === 'user' ? 'Client' : 'System'}: "${c.last_message.content}"` : 'No messages yet';

          card.innerHTML = `
            <div class="conv-top">
              <span class="conv-id">#${c.conversation_id}</span>
              <span class="conv-date">${timeStr}</span>
            </div>
            <div class="conv-badges">
              <span class="badge badge-mode">${c.mode}</span>
              <span class="badge badge-stage ${c.stage}">${c.stage.replace('_', ' ')}</span>
              <span class="badge badge-msg-count">${c.total_messages} msgs (${c.user_messages_count} user, ${c.system_messages_count} sys)</span>
            </div>
            <div class="conv-snippet">${escapeHtml(lastText)}</div>
          `;
          listEl.appendChild(card);
        });

      } catch (err) {
        listEl.innerHTML = '<div class="empty-state"><p>Error loading conversations.</p></div>';
      }
    }

    // Select and inspect conversation
    async function selectConversation(convId) {
      state.selectedConvId = convId;
      document.querySelectorAll('.conv-card').forEach(el => el.classList.remove('active'));
      event?.currentTarget?.classList?.add('active');

      const body = document.getElementById('inspector-body');
      body.innerHTML = '<div class="empty-state"><div class="empty-icon">⏳</div><p>Loading full transcript...</p></div>';

      document.getElementById('selected-conv-id').textContent = `#${convId}`;
      document.getElementById('inspector-actions').style.display = 'flex';
      document.getElementById('inspector-meta').style.display = 'flex';
      document.getElementById('inspector-tabs').style.display = 'flex';

      try {
        const res = await fetch(`/api/admin/conversations/${convId}`, { headers: getHeaders() });
        if (!res.ok) throw new Error('Failed to load chat detail');
        state.selectedConvData = await res.json();
        renderSelectedConversation();
      } catch (e) {
        body.innerHTML = '<div class="empty-state"><p>Failed to load conversation details.</p></div>';
      }
    }

    // Render inspector content based on active tab
    function renderSelectedConversation() {
      const d = state.selectedConvData;
      if (!d) return;

      document.getElementById('meta-mode').textContent = d.mode;
      document.getElementById('meta-stage').textContent = d.stage;
      document.getElementById('meta-lang').textContent = d.language;
      document.getElementById('meta-msgs').textContent = `${d.stats.total_messages} (${d.stats.user_messages} user, ${d.stats.system_messages} system)`;
      document.getElementById('meta-date').textContent = new Date(d.updated_at).toLocaleString();

      const body = document.getElementById('inspector-body');
      body.innerHTML = '';

      if (state.activeTab === 'messages') {
        if (!d.messages || d.messages.length === 0) {
          body.innerHTML = '<div class="empty-state"><p>No messages recorded in this conversation yet.</p></div>';
          return;
        }

        d.messages.forEach(m => {
          const isUser = m.role === 'user';
          const group = document.createElement('div');
          group.className = `msg-group ${isUser ? 'user' : 'assistant'}`;

          const time = new Date(m.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

          group.innerHTML = `
            <div class="msg-header">
              <span class="role-tag ${isUser ? 'user' : 'system'}">${isUser ? 'User / Client' : 'System / Assistant'}</span>
              <span>${time}</span>
            </div>
            <div class="msg-content">${escapeHtml(m.content)}</div>
          `;
          body.appendChild(group);
        });

      } else if (state.activeTab === 'facts') {
        const factsCard = document.createElement('div');
        factsCard.className = 'detail-card';
        factsCard.innerHTML = `
          <div class="detail-title">Extracted Client Facts & Circumstances</div>
          <pre class="code-block">${JSON.stringify(d.facts, null, 2)}</pre>
        `;
        body.appendChild(factsCard);

        if (d.files && d.files.length > 0) {
          const filesCard = document.createElement('div');
          filesCard.className = 'detail-card';
          filesCard.style.marginTop = '1rem';
          filesCard.innerHTML = `
            <div class="detail-title">Attached Files (${d.files.length})</div>
            <pre class="code-block">${JSON.stringify(d.files, null, 2)}</pre>
          `;
          body.appendChild(filesCard);
        }

      } else if (state.activeTab === 'legal_qa') {
        if (!d.legal_qa_results || d.legal_qa_results.length === 0) {
          body.innerHTML = '<div class="empty-state"><p>No Legal QA precedents or answers generated for this conversation yet.</p></div>';
          return;
        }

        d.legal_qa_results.forEach((qa, idx) => {
          const qaCard = document.createElement('div');
          qaCard.className = 'detail-card';
          qaCard.innerHTML = `
            <div class="detail-title">Legal Analysis #${idx + 1}</div>
            <p style="font-weight: 600; margin-bottom: 0.5rem; color: #f1f5f9;">Question Formulated:</p>
            <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1rem;">${escapeHtml(qa.question)}</p>
            <p style="font-weight: 600; margin-bottom: 0.5rem; color: #f1f5f9;">Grounded Answer:</p>
            <p style="font-size: 0.85rem; color: #a7f3d0; margin-bottom: 1rem; white-space: pre-wrap;">${escapeHtml(qa.answer)}</p>
            ${qa.retrieved_cases && qa.retrieved_cases.length ? `
              <p style="font-weight: 600; margin-bottom: 0.5rem; color: #f1f5f9;">Retrieved Precedents (${qa.retrieved_cases.length}):</p>
              <pre class="code-block">${JSON.stringify(qa.retrieved_cases, null, 2)}</pre>
            ` : ''}
          `;
          body.appendChild(qaCard);
        });
      }
    }

    // Tabs handling
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.activeTab = btn.dataset.tab;
        renderSelectedConversation();
      });
    });

    // Exports
    document.getElementById('export-json-btn').addEventListener('click', () => {
      if (state.selectedConvId) {
        window.open(`/api/admin/conversations/${state.selectedConvId}/export?format=json`, '_blank');
      }
    });

    document.getElementById('export-txt-btn').addEventListener('click', () => {
      if (state.selectedConvId) {
        window.open(`/api/admin/conversations/${state.selectedConvId}/export?format=txt`, '_blank');
      }
    });

    // Delete conversation
    document.getElementById('delete-conv-btn').addEventListener('click', async () => {
      if (!state.selectedConvId) return;
      if (!confirm(`Are you sure you want to permanently delete conversation #${state.selectedConvId}?`)) return;

      try {
        const res = await fetch(`/api/admin/conversations/${state.selectedConvId}`, {
          method: 'DELETE',
          headers: getHeaders()
        });
        if (res.ok) {
          showToast(`Deleted conversation #${state.selectedConvId}`);
          state.selectedConvId = null;
          state.selectedConvData = null;
          document.getElementById('inspector-actions').style.display = 'none';
          document.getElementById('inspector-meta').style.display = 'none';
          document.getElementById('inspector-tabs').style.display = 'none';
          document.getElementById('selected-conv-id').textContent = 'Select a chat';
          document.getElementById('inspector-body').innerHTML = '<div class="empty-state"><p>Conversation deleted.</p></div>';
          loadDashboard();
        } else {
          showToast('Failed to delete conversation.');
        }
      } catch (e) {
        showToast('Error deleting conversation.');
      }
    });

    // Search and Filters
    let searchDebounce;
    document.getElementById('search-input').addEventListener('input', (e) => {
      clearTimeout(searchDebounce);
      searchDebounce = setTimeout(() => {
        state.search = e.target.value.trim();
        state.currentPage = 1;
        loadConversations();
      }, 300);
    });

    document.getElementById('filter-mode').addEventListener('change', (e) => {
      state.mode = e.target.value;
      state.currentPage = 1;
      loadConversations();
    });

    document.getElementById('filter-stage').addEventListener('change', (e) => {
      state.stage = e.target.value;
      state.currentPage = 1;
      loadConversations();
    });

    document.getElementById('page-size-select').addEventListener('change', (e) => {
      state.pageSize = parseInt(e.target.value, 10);
      state.currentPage = 1;
      loadConversations();
    });

    document.getElementById('prev-page-btn').addEventListener('click', () => {
      if (state.currentPage > 1) {
        state.currentPage--;
        loadConversations();
      }
    });

    document.getElementById('next-page-btn').addEventListener('click', () => {
      state.currentPage++;
      loadConversations();
    });

    document.getElementById('refresh-btn').addEventListener('click', () => {
      loadDashboard();
      showToast('Dashboard data refreshed.');
    });

    function loadDashboard() {
      loadStats();
      loadConversations();
    }

    function escapeHtml(str) {
      if (!str) return '';
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    }

    // Init
    checkAuth();
  </script>
</body>
</html>
"""


@router.get("/admin", response_class=HTMLResponse)
@router.get("/admin/", response_class=HTMLResponse)
async def serve_admin_dashboard():
    """Serve the modern single-page Admin Dashboard."""
    return HTMLResponse(content=_ADMIN_HTML)
