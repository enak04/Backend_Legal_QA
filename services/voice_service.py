"""
Voice transcription and speech-to-text service.

Provides transcription for Indian languages using:
1. IIT Madras Indic ASR API (https://asr.iitm.ac.in/asr/v1/recognize/)
   Native support for Telugu, Hindi, Tamil, Kannada, Marathi, Bengali, Gujarati, Malayalam, Odia.
2. OpenAI Whisper API (whisper-1) as versatile fallback.
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any

import httpx
from openai import AsyncOpenAI

from config import settings
from services.language import (
    detect_language,
    get_language_name,
    is_indic_language,
    normalize_language_code,
)

logger = logging.getLogger(__name__)

IITM_ASR_ENDPOINT = "https://asr.iitm.ac.in/asr/v1/recognize/"

# Mapping of ISO codes to IIT Madras ASR language parameter names
IITM_LANGUAGE_MAP: dict[str, str] = {
    "te": "telugu",
    "hi": "hindi",
    "ta": "tamil",
    "kn": "kannada",
    "mr": "marathi",
    "bn": "bengali",
    "gu": "gujarati",
    "ml": "malayalam",
    "or": "odia",
    "pa": "punjabi",
    "en": "english",
}


# OpenAI Whisper official ISO-639-1 supported language codes for the explicit `language` parameter.
# Languages omitted here (e.g. Telugu 'te', Odia 'or') are transcribed via Whisper auto-detection
# instead of an explicit parameter, which avoids OpenAI 400 'Language is not supported' errors.
WHISPER_SUPPORTED_LANGUAGES: set[str] = {
    "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs", "ca",
    "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fo", "fr",
    "gl", "gu", "ha", "haw", "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it",
    "ja", "jw", "ka", "kk", "km", "kn", "ko", "la", "lb", "ln", "lo", "lt", "lv",
    "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "nn", "no",
    "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn",
    "so", "sq", "sr", "su", "sv", "sw", "ta", "th", "tk", "tl", "tr", "tt", "uk",
    "ur", "uz", "vi", "yi", "yo", "zh", "yue",
}


class VoiceService:
    """Manages audio transcription with Indic ASR and Whisper fallback."""

    def __init__(
        self,
        openai_client: AsyncOpenAI | None = None,
        iitm_asr_url: str = IITM_ASR_ENDPOINT,
        timeout: int = 30,
    ) -> None:
        self._openai_client = openai_client
        self._iitm_url = iitm_asr_url
        self._timeout = timeout

        if not self._openai_client and settings.openai_api_key:
            self._openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def transcribe_with_iitm(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: str = "te",
    ) -> str:
        """
        Transcribe audio via IIT Madras Indic ASR API.
        POST https://asr.iitm.ac.in/asr/v1/recognize/
        multipart: audio (file), language (form field)
        """
        lang_param = IITM_LANGUAGE_MAP.get(language.lower(), language.lower())

        files = {
            "audio": (filename or "audio.wav", audio_bytes, "audio/wav"),
        }
        data = {
            "language": lang_param,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(self._iitm_url, data=data, files=files)
            resp.raise_for_status()
            res_json = resp.json()

            if isinstance(res_json, dict):
                # IITM returns {"body": "transcribed text"} or {"status": "success", "text": "..."}
                text = res_json.get("body") or res_json.get("text") or res_json.get("transcription") or ""
                if text and text != "null":
                    return str(text).strip()
            elif isinstance(res_json, str):
                return res_json.strip()

        raise ValueError(f"IIT Madras ASR returned empty or unexpected response: {resp.text[:200]}")

    async def transcribe_with_whisper(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: str | None = None,
    ) -> str:
        """Transcribe audio via OpenAI Whisper (whisper-1)."""
        if not self._openai_client:
            raise RuntimeError("OpenAI client not configured for Whisper ASR.")

        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = filename or "audio.wav"

        # Check if requested language code is formally supported by Whisper's language parameter
        lang_code = language.lower() if language and language != "auto" else None
        whisper_lang = lang_code if (lang_code and lang_code in WHISPER_SUPPORTED_LANGUAGES) else None

        kwargs: dict[str, Any] = {
            "model": "whisper-1",
            "file": audio_file,
        }
        if whisper_lang:
            kwargs["language"] = whisper_lang

        try:
            transcription = await self._openai_client.audio.transcriptions.create(**kwargs)
            return transcription.text.strip()
        except Exception as exc:
            # If OpenAI rejected the language code with 400 unsupported_language, retry with auto-detection
            if "language" in str(exc).lower() and "language" in kwargs:
                logger.warning(
                    "Whisper rejected language parameter '%s' (%s); retrying with auto-detection.",
                    whisper_lang,
                    exc,
                )
                audio_file.seek(0)
                kwargs.pop("language", None)
                transcription = await self._openai_client.audio.transcriptions.create(**kwargs)
                return transcription.text.strip()
            raise

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: str | None = None,
        preferred_provider: str = "iitm",
    ) -> dict[str, Any]:
        """
        Transcribe audio using the best available provider with automatic fallback.
        
        Returns:
            {
                "text": "Transcribed text",
                "language": "te",
                "language_name": "Telugu",
                "provider": "iitm" | "whisper"
            }
        """
        is_auto = not language or language.strip().lower() in ("auto", "none", "")
        errors: list[str] = []

        # When auto-detection is requested, Whisper is the primary provider because
        # it natively detects speech language across 99 languages.
        if is_auto:
            if self._openai_client:
                try:
                    text = await self.transcribe_with_whisper(
                        audio_bytes,
                        filename=filename,
                        language=None,
                    )
                    if text:
                        detected = detect_language(text, default="hi")
                        return {
                            "text": text,
                            "language": detected,
                            "language_name": get_language_name(detected),
                            "provider": "whisper",
                        }
                except Exception as exc:
                    logger.warning("Whisper auto-detection failed (%s); trying fallback.", exc)
                    errors.append(f"Whisper auto-detect: {exc}")

            # Fallback if Whisper is unavailable in auto mode
            try:
                text = await self.transcribe_with_iitm(audio_bytes, filename=filename, language="hi")
                if text:
                    detected = detect_language(text, default="hi")
                    return {
                        "text": text,
                        "language": detected,
                        "language_name": get_language_name(detected),
                        "provider": "iitm",
                    }
            except Exception as exc:
                errors.append(f"IITM auto fallback: {exc}")

            raise RuntimeError(f"All ASR providers failed to transcribe audio in auto mode. Errors: {'; '.join(errors)}")

        target_lang = normalize_language_code(language, default="en")

        # 1. Try preferred provider first
        if preferred_provider == "iitm" or is_indic_language(target_lang):
            try:
                text = await self.transcribe_with_iitm(audio_bytes, filename=filename, language=target_lang)
                if text:
                    detected = detect_language(text, default=target_lang)
                    return {
                        "text": text,
                        "language": detected,
                        "language_name": get_language_name(detected),
                        "provider": "iitm",
                    }
            except Exception as exc:
                logger.warning("IIT Madras ASR attempt failed (%s); trying Whisper fallback.", exc)
                errors.append(f"IITM ASR: {exc}")

        # 2. Try Whisper fallback / primary
        if self._openai_client:
            try:
                text = await self.transcribe_with_whisper(
                    audio_bytes,
                    filename=filename,
                    language=target_lang if target_lang != "en" else None,
                )
                if text:
                    detected = detect_language(text, default=target_lang)
                    return {
                        "text": text,
                        "language": detected,
                        "language_name": get_language_name(detected),
                        "provider": "whisper",
                    }
            except Exception as exc:
                logger.warning("Whisper ASR attempt failed (%s).", exc)
                errors.append(f"Whisper ASR: {exc}")

        # 3. If primary was Whisper and it failed, try IITM as secondary
        if preferred_provider != "iitm" and is_indic_language(target_lang):
            try:
                text = await self.transcribe_with_iitm(audio_bytes, filename=filename, language=target_lang)
                if text:
                    detected = detect_language(text, default=target_lang)
                    return {
                        "text": text,
                        "language": detected,
                        "language_name": get_language_name(detected),
                        "provider": "iitm",
                    }
            except Exception as exc:
                errors.append(f"IITM ASR secondary: {exc}")

        raise RuntimeError(f"All ASR providers failed to transcribe audio. Errors: {'; '.join(errors)}")


voice_service = VoiceService()
