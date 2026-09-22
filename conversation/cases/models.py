"""
Data models for the Universal Case State, dynamic legal issue spotting,
and extensible domain schemas.

Designed to represent ANY legal matter under Indian law without domain bias.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field, field_validator


# ── Jurisdiction ──────────────────────────────────────────────

class Jurisdiction(BaseModel):
    """Geographic and legal jurisdiction parameters."""
    country: str = "India"
    state: str | None = None
    district: str | None = None
    city: str | None = None


# ── Facts vs Hypotheses ───────────────────────────────────────

class FactItem(BaseModel):
    """
    An extracted or user-stated fact.
    Distinguishes objective statements from legal conclusions.
    """
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    fact: str
    category: str | None = None             # e.g., "employment", "financial", "timeline", "property"
    source: str = "user_stated"             # "user_stated", "extracted", "inferred", "document"
    confidence: float = 1.0                 # 0.0 to 1.0
    is_explicit: bool = True               # True = explicitly stated; False = inferred
    turn: int = 1
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    superseded: bool = False                # Marked True if user corrected this fact


class LegalIssue(BaseModel):
    """
    A legal issue or hypothesis spotted in the case.
    Never confused with a user-provided fact.
    """
    issue: str                              # e.g., "unpaid wages", "wrongful termination", "security deposit dispute"
    domain: str | None = None               # e.g., "employment", "consumer", "property", "criminal"
    status: str = "hypothesis"              # "hypothesis", "confirmed", "ruled_out"
    confidence: float = 0.6                 # 0.0 to 1.0
    applicable_laws: list[str] = Field(default_factory=list)


# ── Parties & Events ──────────────────────────────────────────

class Party(BaseModel):
    """A party involved in the dispute or inquiry."""
    role: str                               # e.g. "client", "opposing_party", "employer", "landlord", "bank"
    name_or_description: str
    entity_type: str | None = None          # "individual", "private_company", "government", "bank"


class Event(BaseModel):
    """A chronological event in the matter."""
    description: str
    date: str | None = None
    normalized_date: str | None = None
    significance: str | None = None
    source: str = "user"


class Dates(BaseModel):
    """Key dates and statutory limitation tracking."""
    incident_date: str | None = None
    notice_date: str | None = None
    deadline: str | None = None
    filing_date: str | None = None
    limitation_flags: list[str] = Field(default_factory=list)


# ── Financials ────────────────────────────────────────────────

class Financial(BaseModel):
    """Financial aspect of the claim or dispute."""
    amount: float | str | None = None
    amount_raw: str | None = None
    currency: str = "INR"
    loss: str | None = None
    dues_period: str | None = None

    @field_validator("amount", mode="before")
    @classmethod
    def _parse_amount(cls, v: Any) -> float | str | None:
        if v is None or isinstance(v, (int, float)):
            return float(v) if v is not None else None
        if isinstance(v, str):
            clean = v.strip().lower()
            lakh_match = re.search(r"([\d,]+(?:\.\d+)?)\s*lakh", clean)
            if lakh_match:
                try:
                    num_val = float(lakh_match.group(1).replace(",", ""))
                    return num_val * 100000.0
                except ValueError:
                    pass
            num_match = re.search(r"[\d,]+(?:\.\d+)?", clean)
            if num_match:
                try:
                    return float(num_match.group(0).replace(",", ""))
                except ValueError:
                    pass
        return v


# ── Evidence & Communications ─────────────────────────────────

class EvidenceItem(BaseModel):
    """An item of documentary, physical, or electronic evidence."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    type: str                               # contract, notice, email, whatsapp, bank_statement, receipt, invoice, photo, etc.
    description: str
    date: str | None = None
    source: str = "user_mentioned"          # "user_mentioned", "uploaded", "available", "unavailable"
    relevance: str = ""
    available: bool = True


class Communication(BaseModel):
    """A record of communication between parties."""
    mode: str = "verbal"                    # verbal, whatsapp, email, letter, notice
    summary: str
    admissions: str | None = None
    date: str | None = None


class ActionTaken(BaseModel):
    """An action the user has already taken in the matter."""
    action: str                             # e.g. "contacted_employer", "sent_legal_notice", "filed_police_complaint"
    date: str | None = None
    outcome: str | None = None
    details: str | None = None


# ── Risk & Urgency ────────────────────────────────────────────

class Risk(BaseModel):
    """Risk and urgency classification."""
    level: str = "normal"                   # "low", "normal", "potentially_urgent", "urgent", "emergency"
    flags: list[str] = Field(default_factory=list)
    reason: str | None = None
    recommended_emergency_action: str | None = None


# ── Missing Facts & Value Ranking ─────────────────────────────

class MissingFact(BaseModel):
    """A legally important fact that remains missing."""
    fact_key: str
    description: str
    legal_importance: str = "HIGH"          # "HIGH", "MEDIUM", "LOW"
    reason: str = ""                        # Why this fact matters legally
    sample_question: str = ""


# ── Confidence Tracking ───────────────────────────────────────

class ConfidenceScores(BaseModel):
    """Separate confidence metrics across key dimensions."""
    facts: float = 0.9
    issue_classification: float = 0.8
    jurisdiction: float = 0.5
    legal_applicability: float = 0.5
    urgency: float = 0.7
    retrieval_relevance: float = 0.5


# ── Legal Authorities & Research ──────────────────────────────

class RetrievedAuthority(BaseModel):
    """A statutory provision, court precedent, or regulation."""
    source: str
    authority_type: str = "statute"         # "statute", "judgment", "rule", "notification"
    jurisdiction: str = "India"
    provision: str                          # Section, Order, or Article
    effective_date: str | None = None
    status: str = "current"                 # "current", "amended", "repealed", "unknown"
    relevance: str = ""
    applicability_conditions: list[str] = Field(default_factory=list)
    key_excerpt: str = ""


class LegalAssessment(BaseModel):
    """Structured legal assessment and action plan."""
    summary: str = ""
    primary_domain: str | None = None
    confirmed_issues: list[str] = Field(default_factory=list)
    applicable_authorities: list[RetrievedAuthority] = Field(default_factory=list)
    evidence_assessment: dict[str, Any] = Field(default_factory=dict)
    action_plan: list[str] = Field(default_factory=list)
    limitations_and_risks: list[str] = Field(default_factory=list)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    ready_for_final_remedy: bool = False


# ── Universal Case State ──────────────────────────────────────

class UniversalCaseState(BaseModel):
    """
    Universal representation of any legal matter.
    Can represent civil, criminal, constitutional, commercial, family,
    cyber, tenancy, consumer, and administrative disputes under Indian law.
    """
    jurisdiction: Jurisdiction = Field(default_factory=Jurisdiction)
    case_domain: str | None = None          # e.g., "employment", "consumer", "property", "criminal", "cybercrime"
    issues: list[LegalIssue] = Field(default_factory=list)
    sub_issues: list[str] = Field(default_factory=list)
    case_posture: str | None = None         # "pre_litigation", "notice_served", "court_case_pending", "post_order"
    summary: str | None = None

    parties: Any = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    dates: Dates = Field(default_factory=Dates)
    financial: Financial = Field(default_factory=Financial)
    financial_info: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    communications: list[Communication] = Field(default_factory=list)
    actions_already_taken: list[ActionTaken] = Field(default_factory=list)
    user_goal: str | None = None
    risk: Risk = Field(default_factory=Risk)
    missing_facts: list[MissingFact] = Field(default_factory=list)
    confidence: Any = Field(default_factory=ConfidenceScores)
    legal_assessment: LegalAssessment | None = None
    domain_extensions: dict[str, Any] = Field(default_factory=dict)

    # Legacy & UI convenience fields
    known_facts: list[dict[str, Any]] = Field(default_factory=list)
    primary_category: str | None = None
    subcategory: str | None = None
    case_type: str | None = None
    urgency: str = "normal"
    possible_alternatives: list[str] = Field(default_factory=list)
    related_case_types: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    summary_confirmed: bool = False

    def to_compact_dict(self) -> dict[str, Any]:
        """Convert to a clean, serializable summary for frontend display."""
        active_issues = [i.issue for i in self.issues if i.status != "ruled_out"]
        if not active_issues and self.case_type:
            active_issues = [self.case_type]

        return {
            "domain": self.case_domain or self.subcategory or self.primary_category,
            "issues": active_issues,
            "jurisdiction": {
                "country": self.jurisdiction.country,
                "state": self.jurisdiction.state,
                "district": self.jurisdiction.district,
                "city": self.jurisdiction.city,
            },
            "urgency": self.risk.level or self.urgency,
            "risk_flags": self.risk.flags,
            "financial": {
                "amount": self.financial.amount_raw or (str(self.financial.amount) if self.financial.amount else None),
                "loss": self.financial.loss,
                "currency": self.financial.currency,
            },
            "evidence_count": len(self.evidence),
            "actions_already_taken": [a.action for a in self.actions_already_taken],
            "user_goal": self.user_goal,
            "confidence": self.confidence.model_dump(),
        }


# ── Backward Compatibility Alias ──────────────────────────────

StructuredCaseState = UniversalCaseState


class CaseModule(BaseModel):
    """
    Specification of a legal case type or domain definition.
    Maintained for taxonomy lookup and rule fallback.
    """
    category: str
    subcategory: str
    case_type: str
    description: str
    keywords: list[str] = Field(default_factory=list)
    relevant_facts: list[str] = Field(default_factory=list)
    priority_info: list[str] = Field(default_factory=list)
    potential_evidence: list[str] = Field(default_factory=list)
    timeline_info: list[str] = Field(default_factory=list)
    financial_info: list[str] = Field(default_factory=list)
    parties_involved: list[str] = Field(default_factory=list)
    jurisdiction_requirements: list[str] = Field(default_factory=list)
    urgency_indicators: list[str] = Field(default_factory=list)
    desired_outcomes: list[str] = Field(default_factory=list)
    related_modules: list[str] = Field(default_factory=list)
    min_required_facts: int = 2
