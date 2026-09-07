"""
Data models for the modular case system and structured conversation state.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class CaseModule(BaseModel):
    """
    Specification of a single legal case type.

    Defines information requirements, priorities, and evidence types
    rather than rigid question scripts.
    """
    category: str                               # e.g., "Civil", "Criminal", "Family", "Other"
    subcategory: str                            # e.g., "Property", "Contract", "Money Recovery"
    case_type: str                              # e.g., "Ownership Dispute", "Breach of Contract"
    description: str                            # Overview of the case type
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
    min_required_facts: int = 2                 # For rule-based fallback


class StructuredCaseState(BaseModel):
    """
    Structured representation of case intelligence maintained across turns.
    """
    primary_category: str | None = None
    subcategory: str | None = None
    case_type: str | None = None
    confidence: str = "Low"                     # "High", "Medium", "Low"
    possible_alternatives: list[str] = Field(default_factory=list)
    related_case_types: list[str] = Field(default_factory=list)

    parties: dict[str, Any] = Field(default_factory=dict)
    known_facts: list[dict[str, Any]] = Field(default_factory=list)
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    financial_info: dict[str, Any] = Field(default_factory=dict)
    communication: list[dict[str, Any]] = Field(default_factory=list)
    previous_actions: list[dict[str, Any]] = Field(default_factory=list)
    user_goal: str | None = None
    missing_information: list[str] = Field(default_factory=list)
    urgency: str = "normal"                     # "normal", "potentially_urgent", "urgent"
    jurisdiction: dict[str, str] = Field(default_factory=dict)
    summary_confirmed: bool = False
