"""
Dynamic Follow-Up & Legal Information Value Engine.

Implements dynamic fact extraction, multi-issue identification, and information-value
ranking. Replaces rigid questionnaires with dynamic question selection that asks:
  "Does this missing fact materially change applicable law, jurisdiction, remedy, or deadline?"
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any

from conversation.cases.domains import domain_registry
from conversation.cases.models import (
    ActionTaken,
    Dates,
    EvidenceItem,
    FactItem,
    Financial,
    Jurisdiction,
    LegalIssue,
    MissingFact,
    Party,
    Risk,
    UniversalCaseState,
)
from database.models import ConversationRecord, MessageRole, Mode
from services.risk_engine import risk_engine


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Dynamic Multi-Issue Spotter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DynamicIssueSpotter:
    """
    Spots multiple simultaneous legal issues from natural language statements.
    Labels them as hypotheses with confidence scores.
    """

    _ISSUE_RULES = [
        # Employment
        (
            re.compile(r"\b(salary|wages?|remuneration|dues|(?:not|hasn't|haven't)\s+(?:been\s+)?paid|pay(?!ment)|unpaid)\b", re.I),
            "unpaid wages",
            "employment",
            ["Payment of Wages Act, 1936", "Karnataka Shops & Commercial Establishments Act, 1961"],
        ),
        (
            re.compile(r"\b(fired|terminated|removed from (my )?job|sacked|dismissed)\b", re.I),
            "termination of employment",
            "employment",
            ["Industrial Disputes Act, 1947", "State Shops and Establishments Acts"],
        ),
        (
            re.compile(r"\b(wrongful|without notice|no notice|no reason|unfairly fired|retaliation)\b", re.I),
            "possible wrongful termination",
            "employment",
            ["Section 39 Karnataka Shops Act / Section 25F Industrial Disputes Act"],
        ),
        # Consumer
        (
            re.compile(r"\b(defect(ive)?|stopped working|broken|faulty|malfunction)\b", re.I),
            "defective goods or product liability",
            "consumer",
            ["Consumer Protection Act, 2019 Section 2(7) & Section 84"],
        ),
        (
            re.compile(r"\b(refuses? to replace|refused refund|no replacement|deficiency)\b", re.I),
            "deficiency in service and unfair trade practice",
            "consumer",
            ["Consumer Protection Act, 2019 Section 2(11) & Section 35"],
        ),
        # Property
        (
            re.compile(r"\b(changed (the )?locks|locked (me )?out|threw.*out|evict(ed|ion))\b", re.I),
            "illegal dispossession / unlawful eviction",
            "property",
            ["Specific Relief Act, 1963 Section 6", "State Rent Control Act"],
        ),
        (
            re.compile(r"\b(security deposit|deposit|not returned deposit|withheld deposit)\b", re.I),
            "security deposit recovery dispute",
            "property",
            ["Indian Contract Act, 1872", "State Tenancy Acts"],
        ),
        # Criminal
        (
            re.compile(r"\b(assault(ed)?|beaten|beat|hit me|physical violence|injury|injured)\b", re.I),
            "physical assault and voluntarily causing hurt",
            "criminal",
            ["Bharatiya Nyaya Sanhita, 2023 Section 115", "BNSS Section 173 (FIR)"],
        ),
        (
            re.compile(r"\b(threat(ened)?|threat to kill|extort(ion)?|blackmail)\b", re.I),
            "criminal intimidation and threat",
            "criminal",
            ["Bharatiya Nyaya Sanhita, 2023 Section 351"],
        ),
        # Cybercrime
        (
            re.compile(r"\b(transferred from my bank|unauthorized transaction|without (my )?permission|otp scam|money stolen)\b", re.I),
            "unauthorized financial transaction",
            "cybercrime",
            ["RBI Master Direction on Customer Protection (2017)", "Information Technology Act, 2000 Section 66D"],
        ),
        (
            re.compile(r"\b(cyber|phishing|hacked|online scam|telegram scam|fake website)\b", re.I),
            "cyber fraud and identity theft",
            "cybercrime",
            ["Information Technology Act, 2000 Section 43 & Section 66C"],
        ),
        # Family
        (
            re.compile(r"\b(financial support|maintenance|stopped paying|refuses to maintain)\b", re.I),
            "denial of maintenance and financial neglect",
            "family",
            ["Section 144 BNSS / Section 125 CrPC", "Protection of Women from Domestic Violence Act, 2005"],
        ),
        (
            re.compile(r"\b(custody|child|divorce|separated|matrimonial)\b", re.I),
            "matrimonial separation and custody dispute",
            "family",
            ["Guardians and Wards Act, 1890", "Special Marriage Act / Hindu Marriage Act"],
        ),
        # Contract
        (
            re.compile(r"\b(contractor|never completed|incomplete work|breach|failed to perform)\b", re.I),
            "breach of contract and failure of performance",
            "contract",
            ["Indian Contract Act, 1872 Section 73 & Section 74"],
        ),
        (
            re.compile(r"\b(recover money|unpaid invoice|repay|loan default|promissory|lent money|borrowed money|refuses to return|not returning)\b", re.I),
            "Money Recovery",
            "contract",
            ["Limitation Act, 1963", "Order XXXVII CPC (Summary Suit)"],
        ),
        # Government
        (
            re.compile(r"\b(government notice|govt notice|municipal notice|demolition notice|show cause notice|notice from (a )?government)\b", re.I),
            "statutory notice from public authority",
            "government",
            ["Constitution of India Article 226", "Principles of Natural Justice"],
        ),
        (
            re.compile(r"\b(rti|information denied|public authority)\b", re.I),
            "RTI compliance and administrative grievance",
            "government",
            ["Right to Information Act, 2005 Section 6 & 19"],
        ),
    ]

    def spot_issues(self, text: str) -> list[LegalIssue]:
        text_lower = text.lower()
        spotted: list[LegalIssue] = []
        seen_issues = set()

        for pattern, issue_name, domain, laws in self._ISSUE_RULES:
            if pattern.search(text_lower):
                if issue_name not in seen_issues:
                    seen_issues.add(issue_name)
                    spotted.append(
                        LegalIssue(
                            issue=issue_name,
                            domain=domain,
                            status="hypothesis",
                            confidence=0.75,
                            applicable_laws=laws,
                        )
                    )

        return spotted


class DomainInfo:
    """Lightweight domain representation for legacy and test compatibility."""
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"DomainInfo(name='{self.name}')"


def detect_domain(text: str) -> DomainInfo:
    """
    Detect the primary legal domain for legacy compatibility.
    Maps to domain names expected by tests.
    """
    text_lower = text.lower()
    if re.search(r"\b(salary|wages?|pay|employer|employee|fired|terminated|job|resigned)\b", text_lower):
        return DomainInfo("employment_wage")
    if re.search(r"\b(landlord|tenant|evict|flat|rent|property|deposit|lease)\b", text_lower):
        return DomainInfo("property_land")
    if re.search(r"\b(fir|theft|assault|police|crime|accused|arrest|bailable|bail)\b", text_lower):
        return DomainInfo("criminal")
    if re.search(r"\b(divorce|maintenance|custody|spouse|wife|husband|marriage|matrimonial)\b", text_lower):
        return DomainInfo("family_matrimonial")
    if re.search(r"\b(consumer|defective|product|warranty|refund|seller|replacement)\b", text_lower):
        return DomainInfo("consumer")
    return DomainInfo("general")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Dynamic Fact Extractor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Delhi", "Jammu and Kashmir", "Chandigarh", "Puducherry",
]

MAJOR_INDIAN_CITIES = {
    "Bengaluru": "Karnataka", "Bangalore": "Karnataka", "Mumbai": "Maharashtra",
    "Pune": "Maharashtra", "Delhi": "Delhi", "New Delhi": "Delhi",
    "Hyderabad": "Telangana", "Chennai": "Tamil Nadu", "Kolkata": "West Bengal",
    "Ahmedabad": "Gujarat", "Gurgaon": "Haryana", "Gurugram": "Haryana",
    "Noida": "Uttar Pradesh", "Jaipur": "Rajasthan", "Lucknow": "Uttar Pradesh",
}


def extract_facts(text: str, last_question_key: str | None = None) -> dict[str, Any]:
    """
    Extract structured facts from text without presupposing domain.
    Accepts optional last_question_key to capture short answers or yes/no responses.
    Returns a dictionary of normalized facts.
    """
    facts: dict[str, Any] = {}
    text_clean = text.strip()
    text_lower = text_clean.lower()

    # Handle short / boolean answers for last_question_key
    if last_question_key:
        if text_lower in ("yes", "y", "true", "correct", "yep", "yeah"):
            facts[last_question_key] = "yes"
        elif text_lower in ("no", "n", "false", "incorrect", "nope"):
            facts[last_question_key] = "no"
        elif len(text_clean.split()) <= 4 and not any(k in text_lower for k in ["salary", "employer", "month", "job"]):
            facts[last_question_key] = text_clean

    # 1. Jurisdiction: State and City
    for city, state in MAJOR_INDIAN_CITIES.items():
        if re.search(rf"\b{city.lower()}\b", text_lower):
            facts["city"] = city
            facts["state"] = state
            break

    if "state" not in facts:
        for state in INDIAN_STATES:
            if re.search(rf"\b{state.lower()}\b", text_lower):
                facts["state"] = state
                break

    # 2. Monetary Amounts
    amount_match = re.search(
        r"(?:(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d+)?)\s*(?:lakh|crore|k)?|([\d,]+(?:\.\d+)?)\s*(?:lakhs?|crores?|k)\s*(?:rs\.?|inr|rupees)?)",
        text_lower,
    )
    if amount_match:
        facts["amount_raw"] = amount_match.group(0).strip()

    # Specific ₹80,000 or 2 lakh patterns
    lakh_match = re.search(r"(\d+(?:\.\d+)?)\s*lakh", text_lower)
    if lakh_match:
        try:
            facts["amount"] = float(lakh_match.group(1)) * 100000
            facts["amount_raw"] = f"₹{float(lakh_match.group(1)):g} Lakh"
        except ValueError:
            pass

    num_match = re.search(r"(?:rs\.?|₹)\s*([\d,]+)", text_lower)
    if num_match and "amount" not in facts:
        raw_num = num_match.group(1).replace(",", "")
        try:
            facts["amount"] = float(raw_num)
            facts["amount_raw"] = f"₹{facts['amount']:,.0f}"
        except ValueError:
            pass

    # 3. Durations & Timelines
    dur_match = re.search(r"\b(\d+)\s*(days?|weeks?|months?|years?)\b", text_lower)
    if dur_match:
        facts["duration"] = f"{dur_match.group(1)} {dur_match.group(2)}"

    date_rel_match = re.search(r"\b(yesterday|today|last week|last month|2 weeks ago|3 months ago)\b", text_lower)
    if date_rel_match:
        facts["relative_date"] = date_rel_match.group(1)

    # 4. Employment Type (Check government before general company)
    if re.search(r"\b(government|govt|psu|civil servant|public sector)\b", text_lower):
        facts["employment_type"] = "government"
    elif re.search(r"\b(private|startup|tech firm|company|mnc|corporate)\b", text_lower):
        facts["employment_type"] = "private"

    # 5. Contract / Documentation
    if re.search(r"\b(no contract|verbal only|no clauses|no agreement|oral|don't have|dont have|no written|not have)\b", text_lower):
        facts["written_contract"] = "no"
    elif re.search(r"\b(written contract|offer letter|appointment letter|registered agreement|lease agreement)\b", text_lower):
        facts["written_contract"] = "yes"

    # 6. Eviction / Lockout
    if re.search(r"\b(changed (the )?locks|locked me out|thrown.*out)\b", text_lower):
        facts["eviction_status"] = "forcible lockout"

    # 7. Actions Already Taken
    actions: list[str] = []
    if re.search(r"\b(filed an? fir|reported to police|police complaint|dialed 112)\b", text_lower):
        actions.append("police_complaint_filed")
    if re.search(r"\b(called 1930|reported on cybercrime|reported to bank|informed bank)\b", text_lower):
        actions.append("bank_or_cyber_notified")
    if re.search(r"\b(sent (a )?legal notice|served notice|demand letter)\b", text_lower):
        actions.append("legal_notice_sent")
    if re.search(r"\b(complained to hr|contacted employer|emailed boss)\b", text_lower):
        actions.append("contacted_employer")

    if actions:
        facts["actions_already_taken"] = actions

    return facts


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Dynamic Legal Information-Value Ranker & Engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FollowUpEngine:
    """
    Evaluates Universal Case State to determine:
      1. Has sufficient information been collected to provide a useful legal answer?
      2. If not, what is the single highest-value missing fact that materially
         changes the legal outcome?
    """

    def __init__(self) -> None:
        self._issue_spotter = DynamicIssueSpotter()

    def update_case_state_from_text(
        self,
        state: UniversalCaseState,
        user_message: str,
        turn_number: int = 1,
    ) -> UniversalCaseState:
        """Update UniversalCaseState with facts, issues, and risks extracted from text."""
        # 1. Extract facts
        extracted = extract_facts(user_message)

        if "state" in extracted and not state.jurisdiction.state:
            state.jurisdiction.state = extracted["state"]
        if "city" in extracted and not state.jurisdiction.city:
            state.jurisdiction.city = extracted["city"]

        if "amount" in extracted and not state.financial.amount:
            state.financial.amount = extracted["amount"]
        if "amount_raw" in extracted and not state.financial.amount_raw:
            state.financial.amount_raw = extracted["amount_raw"]

        if "duration" in extracted and not state.dates.incident_date:
            state.dates.incident_date = extracted["duration"]

        # Record explicit FactItem
        fact_id = uuid.uuid4().hex[:8]
        new_fact = FactItem(
            id=fact_id,
            fact=user_message.strip(),
            category=state.case_domain or "general",
            source="user_stated",
            confidence=1.0,
            is_explicit=True,
            turn=turn_number,
        )
        state.known_facts.append(new_fact.model_dump())

        # 2. Spot issues
        spotted_issues = self._issue_spotter.spot_issues(user_message)
        existing_issue_names = {i.issue for i in state.issues}
        for issue in spotted_issues:
            if issue.issue not in existing_issue_names:
                state.issues.append(issue)
                if not state.case_domain:
                    state.case_domain = issue.domain

        if state.issues:
            if not state.case_type:
                state.case_type = state.issues[0].issue
            if not state.subcategory:
                state.subcategory = state.issues[0].issue

        # 3. Detect risk
        state.risk = risk_engine.detect_risk(user_message, state)
        state.urgency = state.risk.level

        # 4. Actions taken
        if "actions_already_taken" in extracted:
            for act in extracted["actions_already_taken"]:
                if not any(a.action == act for a in state.actions_already_taken):
                    state.actions_already_taken.append(ActionTaken(action=act))

        # 5. Populate domain extensions if detected
        detected = domain_registry.detect_domains(user_message)
        if detected and not state.case_domain:
            state.case_domain = detected[0][0]

        return state

    def evaluate_missing_facts(
        self,
        case_state: UniversalCaseState,
        user_messages_text: str = "",
    ) -> list[MissingFact]:
        """
        Identify and rank missing facts by legal information value.
        """
        missing: list[MissingFact] = []
        domain_name = case_state.case_domain or "general"
        domain_def = domain_registry.get(domain_name)

        # 1. Domain-specific high value questions from registry
        if domain_def:
            for q_def in domain_def.high_value_questions:
                key = q_def["fact_key"]
                # Check if already answered in state or user messages
                if self._is_fact_known(case_state, key, user_messages_text=user_messages_text):
                    continue

                missing.append(
                    MissingFact(
                        fact_key=key,
                        description=q_def.get("reason", "Legally material fact"),
                        legal_importance=q_def.get("importance", "HIGH"),
                        reason=q_def.get("reason", ""),
                        sample_question=q_def.get("question", ""),
                    )
                )

        # 2. Universal core requirements: Jurisdiction
        if not case_state.jurisdiction.state and not self._is_fact_known(case_state, "jurisdiction_state", user_messages_text=user_messages_text) and not any(m.fact_key == "jurisdiction_state" for m in missing):
            missing.append(
                MissingFact(
                    fact_key="jurisdiction_state",
                    description="State jurisdiction in India",
                    legal_importance="HIGH",
                    reason="State laws, local rent control, and labor tribunals are state-specific",
                    sample_question="Which state or city are you located in?",
                )
            )

        # Sort by legal importance: HIGH -> MEDIUM -> LOW
        importance_weight = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        missing.sort(key=lambda m: importance_weight.get(m.legal_importance, 0), reverse=True)

        return missing

    def _is_fact_known(
        self,
        state: UniversalCaseState,
        key: str,
        user_messages_text: str = "",
    ) -> bool:
        """Check if a factual dimension is already established."""
        all_fact_parts = []
        for f in state.known_facts:
            if isinstance(f, dict):
                all_fact_parts.extend(str(v) for v in f.values())
            else:
                all_fact_parts.append(str(f))

        # Include domain_extensions values
        if isinstance(state.domain_extensions, dict):
            for ext_val in state.domain_extensions.values():
                if isinstance(ext_val, dict):
                    all_fact_parts.extend(str(v) for v in ext_val.values())
                else:
                    all_fact_parts.append(str(ext_val))

        all_fact_text = " ".join(all_fact_parts).lower()

        # Check party entity types as well
        party_parts = []
        if isinstance(state.parties, list):
            for p in state.parties:
                if hasattr(p, "role"):
                    party_parts.append(f"{p.role} {p.name_or_description} {p.entity_type or ''}")
                elif isinstance(p, dict):
                    party_parts.append(f"{p.get('role', '')} {p.get('name_or_description', '')} {p.get('entity_type', '')}")
                else:
                    party_parts.append(str(p))
        elif isinstance(state.parties, dict):
            for k, v in state.parties.items():
                party_parts.append(f"{k} {v}")
        party_text = " ".join(party_parts).lower()
        combined_text = f"{all_fact_text} {party_text} {state.summary or ''} {user_messages_text}".lower()

        if "jurisdiction" in key or "state" in key:
            return bool(state.jurisdiction.state or state.jurisdiction.city) or bool(
                re.search(
                    r"\b(karnataka|bangalore|bengaluru|delhi|mumbai|maharashtra|tamil nadu|chennai|hyderabad|telangana|west bengal|kolkata|kerala|uttar pradesh|haryana|gurgaon|noida|pune)\b",
                    combined_text,
                )
            )

        if "employment_type" in key:
            if isinstance(state.domain_extensions, dict) and state.domain_extensions.get("employment", {}).get("employment_type"):
                return True
            return any(
                w in combined_text
                for w in [
                    "private", "government", "psu", "contractor", "startup", "tech",
                    "it firm", "it company", "mnc", "corporate", "firm", "factory",
                    "establishment", "workman", "civil servant"
                ]
            )

        if "contract" in key or "agreement" in key:
            return any(
                w in combined_text
                for w in [
                    "contract", "offer letter", "appointment letter", "lease",
                    "verbal only", "no clauses", "written agreement", "registered deed",
                    "service agreement", "invoice", "receipt"
                ]
            )

        if "possession" in key:
            return any(
                w in combined_text
                for w in [
                    "locked out", "in possession", "thrown out", "dispossessed",
                    "evicted", "vacated", "still living", "occupying"
                ]
            )

        if "police" in key or "fir" in key or "portal" in key or "reporting" in key:
            return any(
                a.action in ["police_complaint_filed", "bank_or_cyber_notified"]
                for a in state.actions_already_taken
            ) or any(w in combined_text for w in ["fir", "police complaint", "cybercrime", "1930", "disputed with bank"])

        if "amount" in key or "paid" in key or "loss" in key or "value" in key:
            return bool(state.financial.amount or state.financial.amount_raw) or bool(
                re.search(r"(?:rs\.?|₹|\blakh|\bcrore)\s*[\d,]+", combined_text)
            )

        if "date" in key or "timing" in key or "hour" in key or "duration" in key:
            return bool(state.dates.incident_date or state.dates.notice_date)

        return False

    def is_sufficient_information(
        self,
        record: ConversationRecord,
        case_state: UniversalCaseState,
    ) -> bool:
        """
        Determine if enough information exists to provide a useful, grounded legal answer.
        Does NOT demand all schema fields!
        """
        # In Readable or Informative modes, answer immediately
        if record.mode in (Mode.READABLE, Mode.INFORMATIVE):
            return True

        # Check if user explicitly asked to skip questions or give advice now
        user_messages = [m.content.lower() for m in record.messages if m.role == MessageRole.USER]
        if user_messages:
            latest = user_messages[-1]
            if any(phrase in latest for phrase in [
                "skip question", "tell me what to do", "give me advice now",
                "no more question", "just advise", "remedy now", "legal advice now",
                "what are my options", "what should i do"
            ]):
                return True

        # Turn Cap: if 3 or more user turns have elapsed, stop asking questions!
        user_turns = len(user_messages)
        if user_turns >= 3:
            return True

        # If turn == 1, ALWAYS ask at least 1 high-value question
        if user_turns == 1:
            return False

        # If turn == 2: If core dimensions (jurisdiction + main issue + relationship) are known, we can answer!
        has_jurisdiction = bool(case_state.jurisdiction.state or case_state.jurisdiction.city)
        has_core_facts = len(case_state.known_facts) >= 2 or bool(case_state.financial.amount or case_state.dates.incident_date)

        missing = self.evaluate_missing_facts(case_state)
        high_missing = [m for m in missing if m.legal_importance == "HIGH"]

        if not high_missing and has_jurisdiction:
            return True

        return False

    def select_highest_value_question(
        self,
        case_state: UniversalCaseState,
    ) -> str:
        """
        Select the single highest-value question with a concise 'Why this matters' explanation.
        """
        missing = self.evaluate_missing_facts(case_state)

        if missing:
            top_missing = missing[0]
            question = top_missing.sample_question
            reason = top_missing.reason

            if reason and "why" not in question.lower():
                return f"{question}\n*(Why this matters: {reason})*"
            return question

        return (
            "Could you share any further details or documents you have regarding this dispute, "
            "or let me know if you are ready for your actionable legal options?"
        )

    # ── Legacy FollowUpEngine API compatibility ───────────────────

    def needs_followup(self, state: ConversationRecord) -> str | None:
        """
        Evaluate whether a follow-up question is needed.
        Returns a question string if needed, or None if no follow-up is needed.
        """
        if state.mode == Mode.READABLE:
            return None

        user_messages = [m.content for m in state.messages if m.role == MessageRole.USER]
        last_msg = user_messages[-1] if user_messages else ""

        if state.mode == Mode.INFORMATIVE:
            # If the user's question is too brief / ambiguous, ask for clarification
            if len(last_msg.strip().split()) <= 2:
                return "Could you please provide more details or context about the legal topic you would like to know about?"
            return None

        if state.mode == Mode.ACTIONABLE:
            facts = state.facts or {}
            # If standard legacy facts are already satisfied:
            if facts.get("employment_type") and facts.get("state"):
                return None

            # Check case state
            raw_cs = facts.get("case_state")
            if isinstance(raw_cs, UniversalCaseState):
                case_state = raw_cs
            elif isinstance(raw_cs, dict):
                case_state = UniversalCaseState(**raw_cs)
            else:
                case_state = UniversalCaseState()
                for msg in user_messages:
                    self.update_case_state_from_text(case_state, msg)
                if facts.get("state"):
                    case_state.jurisdiction.state = facts["state"]
                if facts.get("employment_type"):
                    case_state.employment.employment_type = facts["employment_type"]

            if facts.get("employment_type") and (facts.get("state") or case_state.jurisdiction.state):
                return None

            if self.is_sufficient_information(state, case_state):
                return None

            return self.select_highest_value_question(case_state)

        return None

    def should_ask_followup(
        self,
        state: ConversationRecord,
        latest_message: str,
    ) -> bool:
        """Backward-compatible wrapper."""
        case_state = state.facts.get("case_state")
        if not isinstance(case_state, UniversalCaseState):
            if isinstance(case_state, dict):
                case_state = UniversalCaseState(**case_state)
            else:
                case_state = UniversalCaseState()

        self.update_case_state_from_text(case_state, latest_message, len(state.messages))
        state.facts["case_state"] = case_state.model_dump()
        return not self.is_sufficient_information(state, case_state)

    def get_next_question(self, state: ConversationRecord) -> str:
        """Backward-compatible wrapper."""
        case_state = state.facts.get("case_state")
        if not isinstance(case_state, UniversalCaseState):
            if isinstance(case_state, dict):
                case_state = UniversalCaseState(**case_state)
            else:
                case_state = UniversalCaseState()

        return self.select_highest_value_question(case_state)

    def get_last_question_key(self, state: ConversationRecord) -> str | None:
        """Derive the key of the last question asked by the assistant."""
        last_q = state.last_assistant_question or ""
        if not last_q:
            for m in reversed(state.messages):
                if m.role == MessageRole.ASSISTANT:
                    last_q = m.content
                    break
        last_q_lower = last_q.lower()
        if "contract" in last_q_lower or "appointment letter" in last_q_lower or "written agreement" in last_q_lower:
            return "written_contract"
        if "fir" in last_q_lower or "police complaint" in last_q_lower:
            return "fir_filed"
        if "state" in last_q_lower or "which city" in last_q_lower:
            return "state"
        if "private" in last_q_lower or "government" in last_q_lower or "employer" in last_q_lower:
            return "employment_type"
        if "property" in last_q_lower or "flat" in last_q_lower:
            return "property_type"
        if "amount" in last_q_lower or "salary" in last_q_lower or "how much" in last_q_lower:
            return "amount"
        return None


followup_engine = FollowUpEngine()
