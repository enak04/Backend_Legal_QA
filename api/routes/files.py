"""
File attachment API endpoints.

Allows uploading document attachments linked to a conversation and retrieving them.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import Response

from conversation.manager import ConversationNotFoundError
from database.models import FileAttachment
from conversation.state import update_facts

router = APIRouter(prefix="/api")


# ── Helper to retrieve repository from app state ─────────────

def _repo(request: Request):
    return request.app.state.conversation_manager._repo


def _manager(request: Request):
    return request.app.state.conversation_manager


# ── Endpoints ─────────────────────────────────────────────────

@router.post("/conversations/{conversation_id}/files")
async def upload_file(
    conversation_id: str,
    request: Request,
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """
    Upload a document attachment for a conversation.
    Stores the binary file and appends metadata to the conversation record.
    """
    manager = _manager(request)
    repo = _repo(request)

    # 1. Load conversation state
    state = await repo.get(conversation_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation not found: {conversation_id}",
        )

    # 2. Read file content
    content = await file.read()
    size_bytes = len(content)

    # 3. Create unique file ID
    file_id = uuid.uuid4().hex[:12]
    filename = file.filename or "uploaded_file"
    content_type = file.content_type or "application/octet-stream"

    # 4. Save binary to DB
    await repo.save_file(
        conversation_id=conversation_id,
        file_id=file_id,
        filename=filename,
        content_type=content_type,
        content=content,
    )

    # 5. Add to files list in ConversationRecord
    attachment = FileAttachment(
        file_id=file_id,
        filename=filename,
        content_type=content_type,
        size_bytes=size_bytes,
        uploaded_at=datetime.now(timezone.utc).isoformat(),
    )
    state.files.append(attachment)

    # 6. Inform facts metadata about the upload so the intake engine is aware
    update_facts(state, {f"uploaded_document_{file_id}": filename})

    # 7. Update database
    await repo.update(state)

    return {
        "file_id": file_id,
        "filename": filename,
        "content_type": content_type,
        "size_bytes": size_bytes,
        "uploaded_at": attachment.uploaded_at,
    }


@router.get("/conversations/{conversation_id}/files/{file_id}")
async def download_file(
    conversation_id: str,
    file_id: str,
    request: Request,
) -> Response:
    """
    Download/retrieve the binary file associated with a conversation.
    """
    repo = _repo(request)

    # Verify conversation exists
    state = await repo.get(conversation_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation not found: {conversation_id}",
        )

    # Load file
    file_data = await repo.get_file(file_id)
    if file_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {file_id}",
        )

    # Return response as attachment
    headers = {
        "Content-Disposition": f'attachment; filename="{file_data["filename"]}"'
    }
    return Response(
        content=file_data["content"],
        media_type=file_data["content_type"],
        headers=headers,
    )
