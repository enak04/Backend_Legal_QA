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

from database.models import ConversationRecord, Mode


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
            return self._row_to_record(row)

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
            await db.commit()
            return cursor.rowcount > 0

    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def _row_to_record(row: aiosqlite.Row) -> ConversationRecord:
        return ConversationRecord(
            conversation_id=row["conversation_id"],
            mode=Mode(row["mode"]),
            stage=row["stage"],
            facts=json.loads(row["facts"]),
            messages=json.loads(row["messages"]),
            last_assistant_question=row["last_assistant_question"],
            legal_qa_results=json.loads(row["legal_qa_results"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
