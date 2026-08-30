"""
Integration tests for file attachment API endpoints (upload/download).
"""

from __future__ import annotations

import io
import pytest
from database.models import Mode


@pytest.mark.asyncio
async def test_file_upload_and_download(test_client):
    """Verify that document files can be uploaded and downloaded via API endpoints."""
    # 1. Create a conversation
    resp = await test_client.post(
        "/api/conversations", json={"mode": "actionable"}
    )
    cid = resp.json()["conversation_id"]

    # 2. Upload a sample text file
    file_content = b"This is a sample employment contract content."
    file_name = "contract.txt"
    files = {
        "file": (file_name, io.BytesIO(file_content), "text/plain")
    }

    upload_resp = await test_client.post(
        f"/api/conversations/{cid}/files",
        files=files
    )
    assert upload_resp.status_code == 200
    upload_data = upload_resp.json()
    assert upload_data["filename"] == file_name
    assert upload_data["content_type"] == "text/plain"
    assert upload_data["size_bytes"] == len(file_content)
    assert "file_id" in upload_data

    file_id = upload_data["file_id"]

    # 3. Check that the file attachment is logged in the conversation facts
    conv_resp = await test_client.get(f"/api/conversations/{cid}")
    assert conv_resp.status_code == 200
    conv_data = conv_resp.json()
    assert len(conv_data["files"]) == 1
    assert conv_data["files"][0]["file_id"] == file_id
    assert conv_data["facts"][f"uploaded_document_{file_id}"] == file_name

    # 4. Download/Retrieve the uploaded file content
    download_resp = await test_client.get(
        f"/api/conversations/{cid}/files/{file_id}"
    )
    assert download_resp.status_code == 200
    assert download_resp.content == file_content
    assert "text/plain" in download_resp.headers["content-type"]
    assert file_name in download_resp.headers["content-disposition"]


@pytest.mark.asyncio
async def test_file_upload_invalid_conversation(test_client):
    """Verify uploading to a non-existent conversation returns 404."""
    files = {
        "file": ("test.txt", io.BytesIO(b"test content"), "text/plain")
    }
    upload_resp = await test_client.post(
        "/api/conversations/nonexistent_id/files",
        files=files
    )
    assert upload_resp.status_code == 404
