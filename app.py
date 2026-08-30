"""
FastAPI application factory and entry point.

Wires together all components:
  - Configuration (from environment)
  - Database (SQLite repository)
  - Legal_QA client
  - Conversation manager
  - API routes
  - CORS middleware

Run with::

    uvicorn app:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.conversations import router as conversations_router
from api.routes.health import router as health_router
from api.routes.modes import router as modes_router
from config import settings
from conversation.followup import FollowUpEngine
from conversation.manager import ConversationManager
from database.repository import SQLiteConversationRepository
from legal_qa.client import LegalQAClient
from legal_qa.query_builder import QueryBuilder
from services.intake_service import OpenAIIntakeService

# ── Logging ───────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── Application lifespan ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise resources on startup, clean up on shutdown."""
    # ── Startup ───────────────────────────────────────────────
    logger.info("Starting Application Backend …")

    # Database
    repo = SQLiteConversationRepository(settings.database_url)
    await repo.initialize()
    logger.info("Database initialised: %s", settings.database_url)

    # Legal_QA client
    legal_qa_client = LegalQAClient(
        base_url=settings.legal_qa_base_url,
        timeout=settings.legal_qa_timeout,
    )
    logger.info(
        "Legal_QA client configured → %s (timeout=%ds)",
        settings.legal_qa_base_url,
        settings.legal_qa_timeout,
    )

    # OpenAI Intake Service
    followup_engine = FollowUpEngine()
    query_builder = QueryBuilder()
    intake_service = OpenAIIntakeService(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        fallback_engine=followup_engine,
        query_builder=query_builder,
    )
    if intake_service.is_configured:
        logger.info(
            "OpenAI intake service active (model=%s)", settings.openai_model
        )
    else:
        logger.info(
            "OpenAI API key not configured; using deterministic rule engine."
        )

    # Conversation manager (the orchestrator)
    manager = ConversationManager(
        repository=repo,
        legal_qa_client=legal_qa_client,
        followup_engine=followup_engine,
        query_builder=query_builder,
        intake_service=intake_service,
    )

    # Store on app state so routes can access it
    app.state.conversation_manager = manager

    logger.info("Application Backend ready on :%d", settings.app_port)

    yield

    # ── Shutdown ──────────────────────────────────────────────
    logger.info("Shutting down Application Backend …")


# ── FastAPI app ───────────────────────────────────────────────

app = FastAPI(
    title="Legal Assistance Application Backend",
    description=(
        "Conversational application backend for legal assistance. "
        "Manages multi-turn conversations, collects relevant facts, "
        "and communicates with the Legal_QA inference service."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────

app.include_router(health_router)
app.include_router(modes_router)
app.include_router(conversations_router)


# ── Direct run ────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
    )
