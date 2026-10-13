"""
Audio and voice consultation API endpoints.

Endpoints:
- POST /api/audio/transcribe: Transcribe speech-to-text (IIT Madras Indic ASR & Whisper).
- POST /api/conversations/{conversation_id}/audio: Direct voice message intake turn.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from conversation.manager import ConversationManager, ConversationNotFoundError
from services.voice_service import voice_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


def _manager(request: Request) -> ConversationManager:
    return request.app.state.conversation_manager


@router.post("/audio/transcribe")
async def transcribe_audio_endpoint(
    file: UploadFile = File(..., description="Audio file (.wav, .mp3, .m4a, .webm, .ogg)"),
    language: str | None = Form(None, description="Optional language code (e.g. 'te', 'hi', 'ta', 'auto')"),
    provider: str = Form("iitm", description="Preferred ASR provider: 'iitm' (IIT Madras) or 'whisper'"),
) -> dict[str, Any]:
    """
    Transcribe speech audio into text.

    Supports Indian regional languages (Telugu, Hindi, Tamil, Kannada, Marathi, etc.)
    using IIT Madras Indic ASR with OpenAI Whisper fallback.
    """
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty audio file uploaded.")

        result = await voice_service.transcribe_audio(
            audio_bytes=content,
            filename=file.filename or "audio.wav",
            content_type=file.content_type or "audio/wav",
            language=language,
            preferred_provider=provider,
        )
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to transcribe audio: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Speech transcription failed: {exc}",
        )


@router.post("/conversations/{conversation_id}/audio")
async def send_audio_message(
    conversation_id: str,
    request: Request,
    file: UploadFile = File(..., description="Spoken voice message from user"),
    language: str | None = Form(None, description="Optional language code (e.g. 'te', 'hi', 'auto')"),
    provider: str = Form("iitm", description="Preferred ASR provider: 'iitm' or 'whisper'"),
) -> dict[str, Any]:
    """
    Send a spoken audio message into an ongoing legal conversation.

    Transcribes the audio, runs legal intake / research turn, and returns
    both the transcribed user query and the advocate's response.
    """
    manager = _manager(request)

    # 1. Transcribe the audio
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty audio file uploaded.")

        transcription_result = await voice_service.transcribe_audio(
            audio_bytes=content,
            filename=file.filename or "audio.wav",
            content_type=file.content_type or "audio/wav",
            language=language,
            preferred_provider=provider,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Voice transcription failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to transcribe voice input: {exc}",
        )

    transcribed_text = transcription_result["text"]
    detected_lang = transcription_result.get("language")

    # 2. Process turn with the transcribed text
    try:
        turn_result = await manager.process_message(
            conversation_id=conversation_id,
            user_message=transcribed_text,
            language=detected_lang or language,
        )
    except ConversationNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation not found: {conversation_id}",
        )
    except Exception as exc:
        logger.exception("Failed to process transcribed message")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing consultation turn: {exc}",
        )

    # 3. Augment response with audio transcription details
    turn_result["transcription"] = transcribed_text
    turn_result["transcription_provider"] = transcription_result.get("provider")

    return turn_result
