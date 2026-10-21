"""
Turn Intent Router for Legal Consultation Pipeline.

Classifies incoming user messages into distinct operational intents:
1. CONCEPTUAL_QUESTION: General legal definitions, statutory terms ("What is a court", "What is severance pay").
   -> Handled directly by conversational advocate; never calls Base ML model or mutates case state.
2. NEW_DISPUTE: Introduction of a brand new legal issue or domain shift.
   -> Resets active case facts, anchors query to this turn, and initiates fresh intake.
3. FOLLOWUP_ON_ADVICE: Ongoing consultation on already-rendered legal advice.
   -> Contextual advice on existing matter; never calls Base ML model.
4. INTAKE_FACT_RESPONSE: Providing answers/facts during active case fact gathering.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from database.models import ConversationRecord, ConversationStage, MessageRole


class TurnIntent(str, Enum):
    CONCEPTUAL_QUESTION = "conceptual_question"
    NEW_DISPUTE = "new_dispute"
    FOLLOWUP_ON_ADVICE = "followup_on_advice"
    INTAKE_FACT_RESPONSE = "intake_fact_response"


_CONCEPTUAL_PREFIX_REGEX = re.compile(
    r"^(what\s+is\s+(a\s+|an\s+|the\s+)?|what\s+are\s+|what\s+does\s+|meaning\s+of\s+|define\s+|"
    r"can\s+(an?\s+)?(employer|landlord|police|company|tenant)\s+|"
    r"is\s+it\s+(legal|illegal|valid|allowed)\s+(to\s+)?|"
    r"how\s+is\s+(severance|gratuity|notice|compensation)\s+calculated|"
    r"who\s+is\s+(a\s+|an\s+)?|difference\s+between\s+)",
    re.IGNORECASE,
)

_CONCEPTUAL_EXACT_TERMS = {
    "what is a court",
    "what is court",
    "what is severance pay",
    "what is severance",
    "what is notice pay",
    "what is notice period",
    "what is gratuity",
    "what is an fir",
    "what is fir",
    "what is zero liability",
    "what is bail",
    "what is anticipatory bail",
    "what is a legal notice",
    "what is legal notice",
    "what are court fees",
    "how do courts work",
    "what does civil court mean",
    "what is labour court",
    "what is consumer forum",
}

_NEW_DISPUTE_REGEX = re.compile(
    r"\b(another legal issue|another issue|new legal issue|new issue|different issue|"
    r"separate issue|another problem|different problem|another case|new case|different case|"
    r"another matter|different matter|one more issue|also have an? (issue|problem|case))\b",
    re.IGNORECASE,
)

_PROPERTY_KEYWORDS = re.compile(
    r"\b(rental apartment|rented flat|landlord|tenant|tenancy|evict|eviction|"
    r"security deposit|vacate|lockout|lease agreement|rent agreement)\b",
    re.IGNORECASE,
)

_EMPLOYMENT_KEYWORDS = re.compile(
    r"\b(fired|terminated|salary|unpaid wages|employer|boss|resigned|layoff|gratuity|severance|pf)\b",
    re.IGNORECASE,
)

_CYBER_KEYWORDS = re.compile(
    r"\b(scam|defrauded|hacked|cybercrime|otp fraud|phishing|frozen account|bank fraud)\b",
    re.IGNORECASE,
)


def is_conceptual_legal_question(message: str) -> bool:
    """Return True if message asks for a general legal definition or educational explanation."""
    clean = message.strip().lower().rstrip("?.,!")
    if clean in _CONCEPTUAL_EXACT_TERMS:
        return True
    if _CONCEPTUAL_PREFIX_REGEX.match(clean):
        # Ensure it is not reporting a personal factual incident (e.g. "What is an FIR against me")
        if any(w in clean for w in ["against me", "filed on me", "my case", "happened to me", "yesterday", "last week"]):
            return False
        return True
    return False


def is_new_dispute_signal(message: str, current_domain: str | None = None) -> bool:
    """Return True if message indicates the start of a completely new legal dispute."""
    clean = message.strip()
    if _NEW_DISPUTE_REGEX.search(clean):
        return True

    # Check for dramatic domain transition
    if current_domain == "employment" and _PROPERTY_KEYWORDS.search(clean) and not _EMPLOYMENT_KEYWORDS.search(clean):
        return True
    if current_domain == "property" and _EMPLOYMENT_KEYWORDS.search(clean) and not _PROPERTY_KEYWORDS.search(clean):
        return True
    if current_domain and current_domain not in ("general", "cybercrime") and _CYBER_KEYWORDS.search(clean):
        return True

    return False


def classify_turn_intent(
    message: str,
    state: ConversationRecord,
    parsed_is_new_issue: bool = False,
) -> TurnIntent:
    """
    Classify the user turn into one of the four operational pipeline intents.
    """
    # 1. Pure conceptual inquiry
    if is_conceptual_legal_question(message):
        return TurnIntent.CONCEPTUAL_QUESTION

    # 2. Check if a new legal dispute is being raised
    current_domain = state.facts.get("detected_domain") or (
        state.facts.get("case_state", {}).get("case_domain")
        if isinstance(state.facts.get("case_state"), dict)
        else None
    )
    if parsed_is_new_issue or is_new_dispute_signal(message, current_domain=current_domain):
        return TurnIntent.NEW_DISPUTE

    # 3. Post-advice follow-up on existing active case
    has_prior_answers = bool(state.legal_qa_results or state.stage == ConversationStage.ANSWERED)
    if has_prior_answers:
        return TurnIntent.FOLLOWUP_ON_ADVICE

    # 4. Standard intake fact gathering
    return TurnIntent.INTAKE_FACT_RESPONSE
