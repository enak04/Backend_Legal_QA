"""
Unit tests for MongoDBConversationRepository using mock motor objects.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from database.models import ConversationRecord, Mode
from database.repository import MongoDBConversationRepository


@pytest.fixture
def mock_motor_client():
    """Mock motor AsyncIOMotorClient and database objects."""
    with patch("motor.motor_asyncio.AsyncIOMotorClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_col = MagicMock()
        mock_files_col = MagicMock()

        mock_client_cls.return_value = mock_client
        mock_client.__getitem__.return_value = mock_db
        mock_db.__getitem__.side_effect = lambda name: mock_files_col if name == "files" else mock_col

        yield {
            "client": mock_client,
            "col": mock_col,
            "files_col": mock_files_col,
        }


class TestMongoDBConversationRepository:
    """Verify CRUD and file management calls on MongoDB repository."""

    @pytest.mark.asyncio
    async def test_initialize(self, mock_motor_client):
        repo = MongoDBConversationRepository("mongodb://localhost:27017")
        
        repo._col.create_index = AsyncMock()
        repo._files_col.create_index = AsyncMock()

        await repo.initialize()

        repo._col.create_index.assert_called_once_with("conversation_id", unique=True)
        repo._files_col.create_index.assert_called_once_with("file_id", unique=True)

    @pytest.mark.asyncio
    async def test_create_conversation(self, mock_motor_client):
        repo = MongoDBConversationRepository("mongodb://localhost:27017")
        record = ConversationRecord(mode=Mode.ACTIONABLE)

        repo._col.insert_one = AsyncMock()

        res = await repo.create(record)

        assert res.conversation_id == record.conversation_id
        repo._col.insert_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_conversation(self, mock_motor_client):
        repo = MongoDBConversationRepository("mongodb://localhost:27017")
        cid = "conv123"
        record_doc = {
            "conversation_id": cid,
            "mode": "actionable",
            "stage": "initial",
            "facts": {},
            "messages": [],
            "last_assistant_question": None,
            "legal_qa_results": [],
            "files": [],
            "created_at": "2026-08-30T04:10:00Z",
            "updated_at": "2026-08-30T04:10:00Z",
        }

        repo._col.find_one = AsyncMock(return_value=record_doc)

        res = await repo.get(cid)

        assert res is not None
        assert res.conversation_id == cid
        repo._col.find_one.assert_called_once_with({"conversation_id": cid})

    @pytest.mark.asyncio
    async def test_save_and_get_file(self, mock_motor_client):
        repo = MongoDBConversationRepository("mongodb://localhost:27017")
        
        repo._files_col.insert_one = AsyncMock()
        
        file_content = b"sample content"
        await repo.save_file(
            conversation_id="conv123",
            file_id="file999",
            filename="doc.txt",
            content_type="text/plain",
            content=file_content,
        )

        repo._files_col.insert_one.assert_called_once()

        # Retrieve file
        repo._files_col.find_one = AsyncMock(return_value={
            "file_id": "file999",
            "conversation_id": "conv123",
            "filename": "doc.txt",
            "content_type": "text/plain",
            "content": file_content,
        })

        file_doc = await repo.get_file("file999")
        assert file_doc is not None
        assert file_doc["filename"] == "doc.txt"
        assert file_doc["content"] == file_content
