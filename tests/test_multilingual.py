"""
Tests for multilingual backend capabilities across language detection,
intake, conversation manager, and answer generation.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from conversation.cases.models import (
    LegalAssessment,
    RetrievedAuthority,
    UniversalCaseState,
)
from conversation.manager import ConversationManager
from conversation.state import get_language, get_universal_case_state, set_language
from database.models import ConversationRecord, ConversationStage, Mode
from database.repository import SQLiteConversationRepository
from legal_qa.client import LegalQAClient
from legal_qa.grounded_generator import GroundedLegalAnswerGenerator
from services.language import (
    detect_language,
    get_language_name,
    is_indic_language,
    normalize_language_code,
)


class TestLanguageDetection:
    """Test Indic script, transliteration, and English language detection."""

    def test_detect_telugu_script(self):
        telugu_text = "నా యజమాని నన్ను ఎటువంటి నోటీసు లేకుండా ఉద్యోగం నుండి తొలగించారు"
        assert detect_language(telugu_text) == "te"

    def test_detect_teluglish(self):
        teluglish_text = "naa employer nannu fire chesaru, 2 months jeetham ivvaledhu, nenu em cheyali"
        assert detect_language(teluglish_text) == "te"

    def test_detect_hindi_script(self):
        hindi_text = "मेरे नियोक्ता ने मुझे बिना किसी नोटिस के नौकरी से निकाल दिया"
        assert detect_language(hindi_text) == "hi"

    def test_detect_hinglish(self):
        hinglish_text = "mera employer ne mujhe bina notice terminate kar diya aur salary nahi diya"
        assert detect_language(hinglish_text) == "hi"

    def test_detect_tamil_script(self):
        tamil_text = "என் முதலாளி என்னை எந்த அறிவிப்பும் இன்றி வேலையிலிருந்து நீக்கிவிட்டார்"
        assert detect_language(tamil_text) == "ta"

    def test_detect_perso_arabic_script(self):
        perso_arabic_text = "مجھے اپری نوکری سے نکال دیا گیا ہے"
        assert detect_language(perso_arabic_text) == "hi"

    def test_detect_english(self):
        english_text = "My employer wrongfully terminated me without notice pay in Bengaluru"
        assert detect_language(english_text) == "en"

    def test_language_normalization(self):
        assert normalize_language_code("TE") == "te"
        assert normalize_language_code("hi") == "hi"
        assert normalize_language_code("auto") == "en"
        assert normalize_language_code(None) == "en"
        assert normalize_language_code("unknown_xyz") == "en"

    def test_language_helpers(self):
        assert get_language_name("te") == "Telugu"
        assert get_language_name("hi") == "Hindi"
        assert get_language_name("ta") == "Tamil"
        assert get_language_name("en") == "English"
        assert is_indic_language("te") is True
        assert is_indic_language("hi") is True
        assert is_indic_language("en") is False


class TestMultilingualConversationState:
    """Test conversation state tracking with language metadata."""

    def test_set_and_get_language(self):
        record = ConversationRecord(mode=Mode.ACTIONABLE)
        assert get_language(record) == "en"

        set_language(record, "te")
        assert get_language(record) == "te"
        assert record.language == "te"
        assert record.facts["language"] == "te"

        ucs = get_universal_case_state(record)
        assert ucs.language == "te"
        assert ucs.to_compact_dict()["language"] == "te"


@pytest.mark.asyncio
class TestMultilingualManagerFlow:
    """Test end-to-end conversation turn in Telugu."""

    async def test_process_message_auto_detects_telugu(self, tmp_path):
        db_path = str(tmp_path / "test_telugu.db")
        repo = SQLiteConversationRepository(db_path)
        await repo.initialize()

        qa_client = AsyncMock(spec=LegalQAClient)
        qa_client.predict.return_value = {
            "question": "Wrongful termination in Telangana",
            "answer": "Under Telangana Shops Act Section 47, one month notice is required.",
            "reasoning_chain": ["Step 1: Notice requirement"],
            "retrieved_cases": [],
        }

        mock_intake = AsyncMock()
        mock_intake.analyze_turn.return_value = MagicMock(
            extracted_facts={"jurisdiction_state": "Telangana"},
            case_state={"jurisdiction": {"state": "Telangana", "city": "Hyderabad"}},
            is_ready_for_qa=False,
            followup_question="మీ వద్ద అపాయింట్‌మెంట్ లెటర్ లేదా శాలరీ స్లిప్‌లు ఉన్నాయా?",
            synthesized_query=None,
            direct_legal_answer=None,
        )

        mock_generator = AsyncMock(spec=GroundedLegalAnswerGenerator)
        mock_generator.is_configured = True
        mock_generator.build_assessment.return_value = LegalAssessment()

        manager = ConversationManager(
            repository=repo,
            legal_qa_client=qa_client,
            intake_service=mock_intake,
            answer_generator=mock_generator,
        )

        record = await manager.create_conversation(Mode.ACTIONABLE)
        assert record.language == "en"

        # User sends message in Telugu script
        user_telugu = "నా యజమాని నన్ను ఎలాంటి నోటీసు లేకుండా తీసేశారు. నేను హైదరాబాద్‌లో పనిచేస్తున్నాను."
        result = await manager.process_message(record.conversation_id, user_telugu)

        assert result["type"] == "follow_up"
        assert result["language"] == "te"
        assert result["message"] == "మీ వద్ద అపాయింట్‌మెంట్ లెటర్ లేదా శాలరీ స్లిప్‌లు ఉన్నాయా?"

        updated = await repo.get(record.conversation_id)
        assert updated.language == "te"

    async def test_grounded_generator_receives_target_language(self):
        generator = GroundedLegalAnswerGenerator(openai_client=None)
        state = UniversalCaseState(language="te")
        state.jurisdiction.state = "Telangana"
        state.jurisdiction.city = "Hyderabad"

        authorities = [
            RetrievedAuthority(
                source="Telangana Shops and Establishments Act, 1988",
                provision="Section 47",
                key_excerpt="One month notice required",
                relevance="Statutory notice protection",
            )
        ]

        context = generator._build_context_message(state, authorities, language="te")
        assert "TARGET CLIENT LANGUAGE: Telugu (te)" in context
        assert "Synthesize the entire consultation guidance and action plan in clear, natural Telugu" in context

    async def test_dynamic_language_switching_across_turns(self, tmp_path):
        db_path = str(tmp_path / "test_switch.db")
        repo = SQLiteConversationRepository(db_path)
        await repo.initialize()

        qa_client = AsyncMock(spec=LegalQAClient)
        mock_intake = AsyncMock()
        mock_intake.analyze_turn.return_value = MagicMock(
            extracted_facts={},
            case_state={},
            is_ready_for_qa=False,
            followup_question="Need more info",
            synthesized_query=None,
            direct_legal_answer=None,
        )

        mock_generator = AsyncMock(spec=GroundedLegalAnswerGenerator)
        mock_generator.is_configured = True
        mock_generator.build_assessment.return_value = LegalAssessment()

        manager = ConversationManager(
            repository=repo,
            legal_qa_client=qa_client,
            intake_service=mock_intake,
            answer_generator=mock_generator,
        )

        record = await manager.create_conversation(Mode.ACTIONABLE)

        # Turn 1: English
        r1 = await manager.process_message(record.conversation_id, "I was terminated without notice pay in Bengaluru.")
        assert r1["language"] == "en"

        # Turn 2: User switches to Telugu
        r2 = await manager.process_message(record.conversation_id, "నాకు 2 నెలల జీతం కూడా రాలేదు.")
        assert r2["language"] == "te"

        # Turn 3: Short answer in English ("Yes, written contract") -> stays Telugu
        r3 = await manager.process_message(record.conversation_id, "Yes")
        assert r3["language"] == "te"

        # Turn 4: User switches to Hinglish
        r4 = await manager.process_message(record.conversation_id, "Mera employer paise nahi de raha abhi bhi.")
        assert r4["language"] == "hi"

        # Turn 5: User switches to full English sentence
        r5 = await manager.process_message(record.conversation_id, "Please explain my legal remedies in English now.")
        assert r5["language"] == "en"

    async def test_hindi_consultation_does_not_switch_to_english_on_amount(self, manager):
        """
        Verify that an ongoing Hindi conversation never leaks English follow-up questions
        and that 'बैंगलोर' maps strictly to Karnataka (never West Bengal).
        """
        record = await manager.create_conversation(mode=Mode.ACTIONABLE, language="hi")
        cid = record.conversation_id

        # Turn 1: Fired from job
        t1 = await manager.process_message(cid, "मुझे नौकरी से निकाल दिया गया है")
        assert t1["language"] == "hi"
        # Must not contain English words like 'company' or 'organization'
        assert not any(w in t1["answer"].lower() for w in ["government", "company", "psu"])

        # Turn 2: City provided as 'बैंगलोर'
        t2 = await manager.process_message(cid, "बैंगलोर")
        assert t2["language"] == "hi"

        # Check state: must be Karnataka, NOT West Bengal
        conv = await manager.get_conversation(cid)
        facts = conv.facts or {}
        case_state = facts.get("case_state") or {}
        jur = case_state.get("jurisdiction") or {}
        assert jur.get("city") in ("Bengaluru", "Bangalore", "बैंगलोर")
        assert jur.get("state") == "Karnataka"
        assert facts.get("state") == "Karnataka"

        # Turn 3: Pending amount 'मुझे एक 3000 पेंडिंग करता है'
        t3 = await manager.process_message(cid, "मुझे एक 3000 पेंडिंग करता है")
        assert t3["language"] == "hi"
        # Must NOT be in English
        ans3 = t3.get("answer", "") or t3.get("message", "")
        assert not any(w in ans3.lower() for w in ["private company", "government/psu", "organization", "were you working"])
        # Should contain Devanagari characters
        assert any("\u0900" <= ch <= "\u097F" for ch in ans3)


