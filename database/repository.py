"""
Conversation persistence layer.

Provides an abstract ``ConversationRepository`` interface and a concrete
``SQLiteConversationRepository`` implementation.  The abstraction lets us
swap SQLite for PostgreSQL later with zero changes to the business logic.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from database.models import ConversationRecord, Mode, FileAttachment


# ── Abstract interface ────────────────────────────────────────

class ConversationRepository(ABC):
    """Storage contract – implement for any database backend."""

    @abstractmethod
    async def initialize(self) -> None:
        """Create tables / run migrations if needed."""

    @abstractmethod
    async def create(self, record: ConversationRecord) -> ConversationRecord:
        """Persist a brand-new conversation."""

    @abstractmethod
    async def get(self, conversation_id: str) -> ConversationRecord | None:
        """Load a conversation by ID.  Returns ``None`` if not found."""

    @abstractmethod
    async def update(self, record: ConversationRecord) -> ConversationRecord:
        """Overwrite an existing conversation record."""

    @abstractmethod
    async def delete(self, conversation_id: str) -> bool:
        """Delete a conversation.  Returns ``True`` if it existed."""

    @abstractmethod
    async def save_file(
        self,
        conversation_id: str,
        file_id: str,
        filename: str,
        content_type: str,
        content: bytes,
    ) -> None:
        """Store the file content linked to a conversation."""

    @abstractmethod
    async def get_file(self, file_id: str) -> dict[str, Any] | None:
        """Retrieve a file document by ID, containing content bytes."""


# ── SQLite implementation ─────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id  TEXT PRIMARY KEY,
    mode             TEXT NOT NULL,
    stage            TEXT NOT NULL,
    facts            TEXT NOT NULL DEFAULT '{}',
    messages         TEXT NOT NULL DEFAULT '[]',
    last_assistant_question TEXT,
    legal_qa_results TEXT NOT NULL DEFAULT '[]',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
"""

_CREATE_FILES_TABLE = """
CREATE TABLE IF NOT EXISTS files (
    file_id         TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    filename        TEXT NOT NULL,
    content_type    TEXT NOT NULL,
    content         BLOB NOT NULL,
    uploaded_at     TEXT NOT NULL
);
"""


class SQLiteConversationRepository(ConversationRepository):
    """Async SQLite-backed conversation store."""

    def __init__(self, db_path: str = "conversations.db") -> None:
        # Strip the "sqlite:///" prefix produced by typical DATABASE_URL values
        clean = db_path
        if clean.startswith("sqlite:///"):
            clean = clean[len("sqlite:///"):]
        self._db_path = clean

    # ── lifecycle ─────────────────────────────────────────────

    async def initialize(self) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(_CREATE_TABLE)
            await db.execute(_CREATE_FILES_TABLE)
            await db.commit()

    # ── CRUD ──────────────────────────────────────────────────

    async def create(self, record: ConversationRecord) -> ConversationRecord:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO conversations
                    (conversation_id, mode, stage, facts, messages,
                     last_assistant_question, legal_qa_results,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.conversation_id,
                    record.mode.value,
                    record.stage.value,
                    json.dumps(record.facts),
                    json.dumps([m.model_dump() for m in record.messages]),
                    record.last_assistant_question,
                    json.dumps([r.model_dump() for r in record.legal_qa_results]),
                    record.created_at,
                    record.updated_at,
                ),
            )
            await db.commit()
        return record

    async def get(self, conversation_id: str) -> ConversationRecord | None:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            
            # Fetch associated files
            file_cursor = await db.execute(
                "SELECT file_id, filename, content_type, length(content) as size_bytes, uploaded_at FROM files WHERE conversation_id = ?",
                (conversation_id,),
            )
            file_rows = await file_cursor.fetchall()
            files_list = [
                FileAttachment(
                    file_id=fr["file_id"],
                    filename=fr["filename"],
                    content_type=fr["content_type"],
                    size_bytes=fr["size_bytes"],
                    uploaded_at=fr["uploaded_at"],
                )
                for fr in file_rows
            ]

            return self._row_to_record(row, files_list)

    async def update(self, record: ConversationRecord) -> ConversationRecord:
        record.updated_at = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                UPDATE conversations
                SET mode = ?,
                    stage = ?,
                    facts = ?,
                    messages = ?,
                    last_assistant_question = ?,
                    legal_qa_results = ?,
                    updated_at = ?
                WHERE conversation_id = ?
                """,
                (
                    record.mode.value,
                    record.stage.value,
                    json.dumps(record.facts),
                    json.dumps([m.model_dump() for m in record.messages]),
                    record.last_assistant_question,
                    json.dumps([r.model_dump() for r in record.legal_qa_results]),
                    record.updated_at,
                    record.conversation_id,
                ),
            )
            await db.commit()
        return record

    async def delete(self, conversation_id: str) -> bool:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "DELETE FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
            await db.execute(
                "DELETE FROM files WHERE conversation_id = ?",
                (conversation_id,),
            )
            await db.commit()
            return cursor.rowcount > 0

    # ── File-specific Storage ─────────────────────────────────

    async def save_file(
        self,
        conversation_id: str,
        file_id: str,
        filename: str,
        content_type: str,
        content: bytes,
    ) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO files (file_id, conversation_id, filename, content_type, content, uploaded_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    conversation_id,
                    filename,
                    content_type,
                    content,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            await db.commit()

    async def get_file(self, file_id: str) -> dict[str, Any] | None:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT filename, content_type, content FROM files WHERE file_id = ?",
                (file_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return {
                "filename": row["filename"],
                "content_type": row["content_type"],
                "content": row["content"],
            }

    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def _row_to_record(row: aiosqlite.Row, files_list: list[FileAttachment]) -> ConversationRecord:
        return ConversationRecord(
            conversation_id=row["conversation_id"],
            mode=Mode(row["mode"]),
            stage=row["stage"],
            facts=json.loads(row["facts"]),
            messages=json.loads(row["messages"]),
            last_assistant_question=row["last_assistant_question"],
            legal_qa_results=json.loads(row["legal_qa_results"]),
            files=files_list,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


# ── MongoDB implementation ───────────────────────────────────

class MongoDBConversationRepository(ConversationRepository):
    """Async MongoDB-backed conversation store using motor."""

    def __init__(self, connection_uri: str, database_name: str = "legal_qa") -> None:
        from motor.motor_asyncio import AsyncIOMotorClient
        self._client = AsyncIOMotorClient(connection_uri)
        self._db = self._client[database_name]
        self._col = self._db["conversations"]
        self._files_col = self._db["files"]

    async def initialize(self) -> None:
        import asyncio
        for attempt in range(3):
            try:
                await self._col.create_index("conversation_id", unique=True)
                await self._files_col.create_index("file_id", unique=True)
                break
            except Exception:
                if attempt == 2:
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))

    async def create(self, record: ConversationRecord) -> ConversationRecord:
        import asyncio
        doc = record.model_dump()
        for attempt in range(3):
            try:
                await self._col.insert_one(doc)
                return record
            except Exception:
                if attempt == 2:
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))
        return record

    async def get(self, conversation_id: str) -> ConversationRecord | None:
        import asyncio
        for attempt in range(3):
            try:
                doc = await self._col.find_one({"conversation_id": conversation_id})
                if doc is None:
                    return None
                doc.pop("_id", None)
                return ConversationRecord(**doc)
            except Exception:
                if attempt == 2:
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))
        return None

    async def update(self, record: ConversationRecord) -> ConversationRecord:
        import asyncio
        record.updated_at = datetime.now(timezone.utc).isoformat()
        doc = record.model_dump()
        for attempt in range(3):
            try:
                await self._col.replace_one(
                    {"conversation_id": record.conversation_id},
                    doc
                )
                return record
            except Exception:
                if attempt == 2:
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))
        return record

    async def delete(self, conversation_id: str) -> bool:
        res = await self._col.delete_one({"conversation_id": conversation_id})
        await self._files_col.delete_many({"conversation_id": conversation_id})
        return res.deleted_count > 0

    async def save_file(
        self,
        conversation_id: str,
        file_id: str,
        filename: str,
        content_type: str,
        content: bytes,
    ) -> None:
        from bson import Binary
        await self._files_col.insert_one({
            "file_id": file_id,
            "conversation_id": conversation_id,
            "filename": filename,
            "content_type": content_type,
            "content": Binary(content),
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        })

    async def get_file(self, file_id: str) -> dict[str, Any] | None:
        doc = await self._files_col.find_one({"file_id": file_id})
        if doc is None:
            return None
        return {
            "filename": doc["filename"],
            "content_type": doc["content_type"],
            "content": bytes(doc["content"]),
        }

