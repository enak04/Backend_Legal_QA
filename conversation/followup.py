"""
Follow-up question engine.

Determines whether a conversation has enough information to call Legal_QA
or whether a targeted follow-up question should be asked first.

Design goals:
  - **Mode-aware**: ACTIONABLE asks more, INFORMATIVE/READABLE ask less.
  - **Domain-specific**: Each legal domain has its own fact requirements.
  - **Modular**: New domains and questions can be added without touching
    the manager or API routes.
  - **Extensible**: The rule-based approach can later be replaced by a
    more sophisticated intake system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from database.models import ConversationRecord, Mode


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Domain definitions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class FactRequirement:
    """One fact that should be collected before calling Legal_QA."""
    key: str                       # fact dictionary key, e.g. "state"
    question: str                  # follow-up question to ask
    priority: int = 0              # higher = asked first
    required: bool = True          # if False, nice-to-have


@dataclass
class LegalDomain:
    """A legal domain with its detection keywords and fact requirements."""
    name: str
    keywords: list[str]
    fact_requirements: list[FactRequirement] = field(default_factory=list)
    # Minimum number of required facts before we call Legal_QA.
    # If the user volunteered enough info, we skip remaining questions.
    min_required_facts: int = 2


# ── Domain catalogue ──────────────────────────────────────────
# Add new domains by appending to this list.

DOMAINS: list[LegalDomain] = [
    LegalDomain(
        name="employment_wage",
        keywords=[
            "salary", "wage", "pay", "employer", "employment",
            "fired", "terminated", "resign", "layoff", "retrench",
            "pf", "provident fund", "gratuity", "bonus",
            "workplace", "harassment at work", "labour", "labor",
        ],
        fact_requirements=[
            FactRequirement(
                key="employment_type",
                question=(
                    "Is your employer a private company, government "
                    "organization, or contractor?"
                ),
                priority=10,
            ),
            FactRequirement(
                key="state",
                question="Which state are you employed in?",
                priority=9,
            ),
            FactRequirement(
                key="duration",
                question=(
                    "How long has this issue been ongoing? "
                    "(e.g., 3 months, 1 year)"
                ),
                priority=8,
            ),
            FactRequirement(
                key="written_contract",
                question=(
                    "Do you have a written employment contract or "
                    "appointment letter?"
                ),
                priority=5,
                required=False,
            ),
        ],
        min_required_facts=2,
    ),
    LegalDomain(
        name="property_land",
        keywords=[
            "property", "land", "tenant", "landlord", "rent",
            "eviction", "lease", "registration", "encroachment",
            "title", "deed", "flat", "apartment", "builder",
            "real estate", "possession", "mutation",
        ],
        fact_requirements=[
            FactRequirement(
                key="property_type",
                question=(
                    "What type of property is this about? "
                    "(e.g., residential flat, agricultural land, commercial)"
                ),
                priority=10,
            ),
            FactRequirement(
                key="state",
                question="In which state is the property located?",
                priority=9,
            ),
            FactRequirement(
                key="ownership_status",
                question=(
                    "Are you the owner, tenant, buyer, or someone else?"
                ),
                priority=8,
            ),
            FactRequirement(
                key="dispute_type",
                question=(
                    "What is the main issue? (e.g., eviction, "
                    "non-registration, encroachment, dispute with builder)"
                ),
                priority=7,
                required=False,
            ),
        ],
        min_required_facts=2,
    ),
    LegalDomain(
        name="criminal",
        keywords=[
            "fir", "police", "crime", "theft", "assault", "murder",
            "fraud", "cheating", "forgery", "bail", "arrest",
            "complaint", "chargesheet", "accused", "victim",
            "cybercrime", "stalking", "threat", "extortion",
            "kidnap", "dowry", "domestic violence",
        ],
        fact_requirements=[
            FactRequirement(
                key="incident_type",
                question=(
                    "What type of incident occurred? "
                    "(e.g., theft, fraud, assault, cybercrime)"
                ),
                priority=10,
            ),
            FactRequirement(
                key="fir_filed",
                question="Has an FIR been filed with the police?",
                priority=9,
            ),
            FactRequirement(
                key="state",
                question="In which state did this occur?",
                priority=8,
            ),
        ],
        min_required_facts=2,
    ),
    LegalDomain(
        name="family_matrimonial",
        keywords=[
            "divorce", "marriage", "custody", "alimony",
            "maintenance", "child", "adoption", "guardianship",
            "domestic violence", "dowry", "498a", "dv act",
            "matrimonial", "husband", "wife", "spouse",
            "separation", "mutual consent",
        ],
        fact_requirements=[
            FactRequirement(
                key="relationship",
                question=(
                    "What is your relationship to the other party? "
                    "(e.g., spouse, parent, guardian)"
                ),
                priority=10,
            ),
            FactRequirement(
                key="issue_type",
                question=(
                    "What is the primary issue? (e.g., divorce, "
                    "custody, maintenance, domestic violence)"
                ),
                priority=9,
            ),
            FactRequirement(
                key="state",
                question=(
                    "Which state are you located in?"
                ),
                priority=8,
            ),
        ],
        min_required_facts=2,
    ),
    LegalDomain(
        name="consumer",
        keywords=[
            "consumer", "product", "defective", "refund",
            "warranty", "service", "overcharged", "misleading",
            "advertisement", "e-commerce", "online purchase",
            "delivery", "insurance claim", "bank", "loan",
        ],
        fact_requirements=[
            FactRequirement(
                key="product_or_service",
                question=(
                    "Is this about a product or a service?"
                ),
                priority=10,
            ),
            FactRequirement(
                key="amount",
                question=(
                    "What is the approximate amount involved?"
                ),
                priority=8,
            ),
            FactRequirement(
                key="complaint_filed",
                question=(
                    "Have you already raised a complaint with the "
                    "company or any consumer forum?"
                ),
                priority=7,
                required=False,
            ),
        ],
        min_required_facts=1,
    ),
    # ── Catch-all / general ──────────────────────────────────
    LegalDomain(
        name="general",
        keywords=[],  # matches everything as fallback
        fact_requirements=[
            FactRequirement(
                key="state",
                question="Which state is this legal matter in?",
                priority=10,
            ),
        ],
        min_required_facts=0,  # general domain doesn't require facts
    ),
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fact extraction (rule-based, LLM-free)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Indian states and union territories for automatic extraction.
_INDIAN_STATES = [
    "andhra pradesh", "arunachal pradesh", "assam", "bihar",
    "chhattisgarh", "goa", "gujarat", "haryana", "himachal pradesh",
    "jharkhand", "karnataka", "kerala", "madhya pradesh",
    "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland",
    "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu",
    "telangana", "tripura", "uttar pradesh", "uttarakhand",
    "west bengal", "delhi", "chandigarh", "puducherry",
    "jammu and kashmir", "ladakh", "lakshadweep",
    "andaman and nicobar", "dadra and nagar haveli",
    "daman and diu",
]

_EMPLOYMENT_TYPES = {
    "private": ["private", "pvt", "private company", "private sector",
                "private limited", "startup"],
    "government": ["government", "govt", "public sector", "psu",
                   "central government", "state government"],
    "contractor": ["contractor", "contract", "outsourced",
                   "third party", "contractual"],
}

_DURATION_PATTERN = re.compile(
    r"(\d+)\s*(months?|years?|weeks?|days?)",
    re.IGNORECASE,
)

_BOOLEAN_YES = {"yes", "yeah", "yep", "haan", "ha", "ji"}
_BOOLEAN_NO = {"no", "nahi", "nope", "na"}


def extract_facts(
    text: str,
    last_question_key: str | None = None,
) -> dict[str, Any]:
    """
    Extract structured facts from free-text user input.

    Uses keyword matching and regex patterns.  Returns a dict of
    extracted facts (may be empty).

    Parameters
    ----------
    text : str
        The user's latest message.
    last_question_key : str | None
        The fact key that the last assistant question was about.
        Helps interpret short answers like "yes" or "Karnataka".
    """
    text_lower = text.lower().strip()
    facts: dict[str, Any] = {}

    # ── State extraction ──────────────────────────────────────
    for state_name in _INDIAN_STATES:
        if state_name in text_lower:
            facts["state"] = state_name.title()
            break

    # ── Employment type ───────────────────────────────────────
    for emp_type, keywords in _EMPLOYMENT_TYPES.items():
        if any(kw in text_lower for kw in keywords):
            facts["employment_type"] = emp_type
            break

    # ── Duration ──────────────────────────────────────────────
    match = _DURATION_PATTERN.search(text_lower)
    if match:
        facts["duration"] = f"{match.group(1)} {match.group(2).lower()}"

    # ── Boolean answers to last question ──────────────────────
    if last_question_key:
        if text_lower in _BOOLEAN_YES:
            facts[last_question_key] = "yes"
        elif text_lower in _BOOLEAN_NO:
            facts[last_question_key] = "no"
        # If it's a short answer and we asked a specific question,
        # store the raw answer under that key.
        elif len(text_lower.split()) <= 5 and last_question_key not in facts:
            # Only if we haven't extracted a structured value already
            already_matched = any(k in facts for k in [
                "state", "employment_type", "duration"
            ])
            if not already_matched:
                facts[last_question_key] = text.strip()

    return facts


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Domain detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_domain(text: str) -> LegalDomain:
    """
    Identify the most likely legal domain from the text.

    Scores each domain by counting keyword matches and returns
    the highest-scoring one.  Falls back to ``general``.
    """
    text_lower = text.lower()
    best_domain = DOMAINS[-1]  # "general" fallback
    best_score = 0

    for domain in DOMAINS:
        if not domain.keywords:
            continue  # skip catch-all for scoring
        score = sum(1 for kw in domain.keywords if kw in text_lower)
        if score > best_score:
            best_score = score
            best_domain = domain

    return best_domain


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Follow-up engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class FollowUpEngine:
    """
    Decides whether to ask a follow-up question or proceed to Legal_QA.

    The engine is **mode-aware**:

    * **ACTIONABLE** — uses domain-specific question trees; collects
      facts until ``min_required_facts`` are satisfied.
    * **INFORMATIVE** — only asks if the question is very vague
      (fewer than ~5 meaningful words).
    * **READABLE** — almost never asks follow-ups; sends straight
      to Legal_QA.
    """

    def needs_followup(self, state: ConversationRecord) -> str | None:
        """
        Return a follow-up question string, or ``None`` if enough
        information is available to call Legal_QA.
        """
        mode = state.mode

        # ── READABLE: never ask follow-ups ────────────────────
        if mode == Mode.READABLE:
            return None

        # ── INFORMATIVE: only if very vague ───────────────────
        if mode == Mode.INFORMATIVE:
            return self._informative_followup(state)

        # ── ACTIONABLE: domain-aware intake ───────────────────
        return self._actionable_followup(state)

    # ── INFORMATIVE mode ──────────────────────────────────────

    @staticmethod
    def _informative_followup(state: ConversationRecord) -> str | None:
        """
        For informative mode, only ask a follow-up if the user's
        question is extremely short or vague.
        """
        user_msgs = [
            m.content for m in state.messages if m.role.value == "user"
        ]
        if not user_msgs:
            return None

        all_text = " ".join(user_msgs)
        word_count = len(all_text.split())

        # If the total user input is very short, ask for elaboration
        if word_count < 4:
            return (
                "Could you provide a bit more detail about your legal "
                "question so I can give you accurate information?"
            )

        return None

    # ── ACTIONABLE mode ───────────────────────────────────────

    @staticmethod
    def _actionable_followup(state: ConversationRecord) -> str | None:
        """
        For actionable mode, detect the domain and collect missing
        high-priority facts one at a time.
        """
        # Build a combined text from all user messages for domain detection
        user_text = " ".join(
            m.content for m in state.messages if m.role.value == "user"
        )
        domain = detect_domain(user_text)

        # Store detected domain as a fact so QueryBuilder can use it
        if "detected_domain" not in state.facts:
            state.facts["detected_domain"] = domain.name

        # Count how many *required* facts we already have
        collected_required = 0
        for req in domain.fact_requirements:
            if req.required and req.key in state.facts:
                collected_required += 1

        # If we have enough, proceed to Legal_QA
        if collected_required >= domain.min_required_facts:
            return None

        # Find the highest-priority missing required fact
        missing = [
            req
            for req in domain.fact_requirements
            if req.required and req.key not in state.facts
        ]
        missing.sort(key=lambda r: r.priority, reverse=True)

        if missing:
            return missing[0].question

        return None

    def get_last_question_key(self, state: ConversationRecord) -> str | None:
        """
        Return the fact key that the last assistant question was
        targeting, so ``extract_facts`` can interpret short answers.
        """
        if not state.last_assistant_question:
            return None

        user_text = " ".join(
            m.content for m in state.messages if m.role.value == "user"
        )
        domain = detect_domain(user_text)

        for req in domain.fact_requirements:
            if req.question == state.last_assistant_question:
                return req.key

        return None
