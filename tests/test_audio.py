"""
Unit tests for voice transcription service and audio endpoints.
"""

from __future__ import annotations

import io
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport

from app import app
from services.voice_service import VoiceService


@pytest.mark.asyncio
class TestVoiceService:
    """Test IIT Madras Indic ASR and Whisper integration."""

    async def test_transcribe_with_iitm_success(self):
        service = VoiceService()
        fake_audio = b"RIFF....WAVEfmt...."

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"body": "నా యజమాని నన్ను తొలగించారు"}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            text = await service.transcribe_with_iitm(fake_audio, "test.wav", language="te")
            assert text == "నా యజమాని నన్ను తొలగించారు"
            assert mock_post.called

    async def test_transcribe_audio_falls_back_to_whisper(self):
        mock_openai = AsyncMock()
        mock_transcription = MagicMock()
        mock_transcription.text = "मेरे नियोक्ता ने मुझे निकाल दिया"
        mock_openai.audio.transcriptions.create.return_value = mock_transcription

        service = VoiceService(openai_client=mock_openai)
        fake_audio = b"RIFF....WAVE"

        # Mock IITM failing
        with patch.object(service, "transcribe_with_iitm", side_effect=Exception("IITM down")):
            res = await service.transcribe_audio(
                fake_audio,
                filename="test.wav",
                language="hi",
                preferred_provider="iitm",
            )
            assert res["text"] == "मेरे नियोक्ता ने मुझे निकाल दिया"
            assert res["language"] == "hi"
            assert res["provider"] == "whisper"

    async def test_transcribe_audio_auto_detection_with_perso_arabic_urdu_hindi(self):
        mock_openai = AsyncMock()
        mock_transcription = MagicMock()
        mock_transcription.text = "مجھے اپری نوکری سے نکال دیا گیا ہے"
        mock_openai.audio.transcriptions.create.return_value = mock_transcription

        service = VoiceService(openai_client=mock_openai)
        fake_audio = b"RIFF....WAVE"

        res = await service.transcribe_audio(
            fake_audio,
            filename="test.wav",
            language="auto",
        )
        assert res["text"] == "مجھے اپری نوکری سے نکال دیا گیا ہے"
        assert res["language"] == "hi"
        assert res["provider"] == "whisper"

        # Verify language was None for Whisper auto-detection
        call_kwargs = mock_openai.audio.transcriptions.create.call_args.kwargs
        assert "language" not in call_kwargs

    async def test_transcribe_audio_auto_detection_with_devanagari_hindi(self):
        mock_openai = AsyncMock()
        mock_transcription = MagicMock()
        mock_transcription.text = "मुझे अपनी नौकरी से निकाल दिया गया है"
        mock_openai.audio.transcriptions.create.return_value = mock_transcription

        service = VoiceService(openai_client=mock_openai)
        fake_audio = b"RIFF....WAVE"

        res = await service.transcribe_audio(
            fake_audio,
            filename="test.wav",
            language=None,
        )
        assert res["text"] == "मुझे अपनी नौकरी से निकाल दिया गया है"
        assert res["language"] == "hi"
        assert res["provider"] == "whisper"

    async def test_transcribe_audio_telugu_whisper_fallback_omits_unsupported_language(self):
        mock_openai = AsyncMock()
        mock_transcription = MagicMock()
        mock_transcription.text = "నా యజమాని నన్ను తొలగించారు"
        mock_openai.audio.transcriptions.create.return_value = mock_transcription

        service = VoiceService(openai_client=mock_openai)
        fake_audio = b"RIFF....WAVE"

        with patch.object(service, "transcribe_with_iitm", side_effect=Exception("IITM 404")):
            res = await service.transcribe_audio(
                fake_audio,
                filename="test.wav",
                language="te",
                preferred_provider="iitm",
            )
            assert res["text"] == "నా యజమాని నన్ను తొలగించారు"
            assert res["language"] == "te"
            assert res["provider"] == "whisper"

            # Verify that language="te" was NOT passed to Whisper API to avoid 400 error
            call_kwargs = mock_openai.audio.transcriptions.create.call_args.kwargs
            assert "language" not in call_kwargs


@pytest.mark.asyncio
class TestAudioEndpoints:
    """Test FastAPI /api/audio/transcribe and /api/conversations/{id}/audio."""

    async def test_transcribe_endpoint(self, test_client):
        fake_audio_file = io.BytesIO(b"fake audio data")

        with patch("services.voice_service.voice_service.transcribe_audio", new_callable=AsyncMock) as mock_transcribe:
            mock_transcribe.return_value = {
                "text": "నాకు 2 నెలల జీతం ఇవ్వలేదు",
                "language": "te",
                "language_name": "Telugu",
                "provider": "iitm",
            }

            response = await test_client.post(
                "/api/audio/transcribe",
                files={"file": ("speech.wav", fake_audio_file, "audio/wav")},
                data={"language": "te", "provider": "iitm"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["text"] == "నాకు 2 నెలల జీతం ఇవ్వలేదు"
            assert data["language"] == "te"
            assert data["provider"] == "iitm"

    async def test_audio_conversation_turn_endpoint(self, test_client):
        # 1. Create a conversation
        create_resp = await test_client.post(
            "/api/conversations",
            json={"mode": "actionable", "language": "te"},
        )
        assert create_resp.status_code == 200
        conv_id = create_resp.json()["conversation_id"]

        fake_audio_file = io.BytesIO(b"fake voice audio")

        # Mock voice_service transcribe_audio
        with patch("services.voice_service.voice_service.transcribe_audio", new_callable=AsyncMock) as mock_transcribe:
            mock_transcribe.return_value = {
                "text": "నా యజమాని నన్ను తొలగించారు",
                "language": "te",
                "language_name": "Telugu",
                "provider": "iitm",
            }

            response = await test_client.post(
                f"/api/conversations/{conv_id}/audio",
                files={"file": ("voice.wav", fake_audio_file, "audio/wav")},
                data={"language": "te"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["transcription"] == "నా యజమాని నన్ను తొలగించారు"
            assert data["language"] == "te"
            assert data["conversation_id"] == conv_id
