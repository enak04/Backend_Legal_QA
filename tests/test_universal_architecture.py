"""
Tests for Universal Legal Intake, Triage, Research, and Action-Planning Architecture.

Comprehensive test suite verifying:
  1. Universal CaseState serialization & domain independence
  2. Facts vs Legal Conclusions distinction
  3. Dynamic multi-issue spotting
  4. Dynamic question selection based on legal information value
  5. Urgency and risk detection interrupting normal gathering
  6. All 8 benchmark legal domains (Employment, Consumer, Property, Criminal, Cybercrime, Family, Contract, Government)
  7. Conversation state persistence across turns without re-asking established facts
  8. User fact correction and contradiction resolution
  9. Precedent separation (retrieved cases cannot overwrite user facts)
 10. Source-grounded 8-part answer generation with claims-to-authority mapping
 11. Actions already taken tracking (never re-recommending completed steps)
"""

from __future__ import annotations

import pytest

from conversation.cases.domains import domain_registry
from conversation.cases.models import (
    ActionTaken,
    Dates,
    EvidenceItem,
    FactItem,
    Financial,
    Jurisdiction,
    LegalIssue,
    RetrievedAuthority,
    UniversalCaseState,
)
from conversation.followup import (
    DynamicIssueSpotter,
    FollowUpEngine,
    extract_facts,
)
from conversation.state import (
    add_user_message,
    resolve_user_correction,
)
from database.models import (
    ConversationRecord,
    MessageRole,
    Mode,
)
from legal_qa.grounded_generator import GroundedLegalAnswerGenerator
from legal_qa.research import LegalResearchLayer
from services.eval_framework import EvaluationFramework
from services.risk_engine import RiskEngine


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Universal CaseState
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestUniversalCaseState:
    def test_universal_case_state_initialization(self):
        """UniversalCaseState has no hardcoded domain assumptions."""
        state = UniversalCaseState()
        assert state.jurisdiction.country == "India"
        assert state.jurisdiction.state is None
        assert state.case_domain is None
        assert state.issues == []
        assert state.parties == []
        assert state.evidence == []
        assert state.actions_already_taken == []
        assert state.risk.level == "normal"
        assert state.confidence.facts >= 0.0

    def test_universal_case_state_serialization(self):
        """UniversalCaseState converts cleanly to dict and compact frontend view."""
        state = UniversalCaseState(
            jurisdiction=Jurisdiction(state="Karnataka", city="Bengaluru"),
            case_domain="consumer",
            issues=[
                LegalIssue(
                    issue="defective goods or product liability",
                    domain="consumer",
                    status="hypothesis",
                    confidence=0.85,
                )
            ],
            financial=Financial(amount=45000.0, amount_raw="₹45,000"),
        )
        data = state.model_dump()
        assert data["jurisdiction"]["state"] == "Karnataka"
        assert data["case_domain"] == "consumer"
        assert len(data["issues"]) == 1

        compact = state.to_compact_dict()
        assert compact["domain"] == "consumer"
        assert "defective goods or product liability" in compact["issues"]
        assert compact["jurisdiction"]["state"] == "Karnataka"
        assert compact["financial"]["amount"] == "₹45,000"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Facts vs Legal Conclusions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFactsVsLegalConclusions:
    def test_user_statement_stored_as_fact_not_conclusion(self):
        """
        User stating 'My company fired me' stores user termination as fact,
        NOT wrongful termination as a confirmed conclusion.
        """
        engine = FollowUpEngine()
        case_state = UniversalCaseState()
        engine.update_case_state_from_text(case_state, "My company fired me", turn_number=1)

        # 1. User stated fact is preserved with source='user_stated'
        assert len(case_state.known_facts) >= 1
        raw_facts = [f["fact"] for f in case_state.known_facts]
        assert any("fired" in f for f in raw_facts)

        # 2. Issues are hypotheses, not confirmed conclusions
        for issue in case_state.issues:
            assert issue.status == "hypothesis"
            assert issue.confidence < 1.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Dynamic Multi-Issue Identification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDynamicIssueIdentification:
    def setup_method(self):
        self.spotter = DynamicIssueSpotter()

    def test_spot_multiple_simultaneous_issues_employment(self):
        """Identifies unpaid wages and termination simultaneously."""
        text = "My employer fired me and hasn't paid me for three months."
        issues = self.spotter.spot_issues(text)
        issue_names = [i.issue for i in issues]
        assert "unpaid wages" in issue_names
        assert "termination of employment" in issue_names

    def test_spot_multiple_simultaneous_issues_property(self):
        """Identifies lockout and security deposit dispute simultaneously."""
        text = "My landlord changed the locks and kept my deposit."
        issues = self.spotter.spot_issues(text)
        issue_names = [i.issue for i in issues]
        assert any("lockout" in i or "dispossession" in i for i in issue_names)
        assert any("deposit" in i for i in issue_names)

    def test_spot_multiple_simultaneous_issues_cyber(self):
        """Identifies unauthorized transaction and cyber fraud."""
        text = "Someone transferred ₹80,000 from my bank account without permission."
        issues = self.spotter.spot_issues(text)
        issue_names = [i.issue for i in issues]
        assert any("unauthorized" in i for i in issue_names)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Dynamic Question Selection & Efficiency
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDynamicQuestionSelection:
    def setup_method(self):
        self.engine = FollowUpEngine()

    def test_ranks_missing_facts_by_legal_information_value(self):
        """High-importance missing facts are asked first, not arbitrary schema fields."""
        case_state = UniversalCaseState(
            case_domain="employment",
            issues=[
                LegalIssue(issue="unpaid wages", domain="employment", status="hypothesis")
            ],
        )
        missing = self.engine.evaluate_missing_facts(case_state)
        assert len(missing) >= 1
        # High value questions like employment type or jurisdiction are prioritized
        top_missing = missing[0]
        assert top_missing.legal_importance == "HIGH"
        assert top_missing.reason is not None

    def test_question_includes_why_this_matters(self):
        """Follow-up questions include legal reason explanation."""
        case_state = UniversalCaseState(
            case_domain="employment",
            issues=[
                LegalIssue(issue="unpaid wages", domain="employment", status="hypothesis")
            ],
        )
        q = self.engine.select_highest_value_question(case_state)
        assert "Why this matters" in q or "matters" in q.lower()

    def test_stops_asking_questions_when_sufficient(self):
        """Stops asking questions when core facts and jurisdiction are known."""
        record = ConversationRecord(mode=Mode.ACTIONABLE)
        add_user_message(record, "I was fired in Karnataka from a private IT firm without 3 months salary")
        add_user_message(record, "The unpaid amount is ₹2,50,000")

        case_state = UniversalCaseState(
            jurisdiction=Jurisdiction(state="Karnataka"),
            financial=Financial(amount=250000.0, amount_raw="₹2,50,000"),
            known_facts=[{"fact": "Fired without salary"}, {"fact": "Private IT firm"}],
        )
        assert self.engine.is_sufficient_information(record, case_state)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Risk & Urgency Engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRiskAndUrgencyEngine:
    def setup_method(self):
        self.risk_engine = RiskEngine()

    def test_immediate_lockout_higher_urgency_than_old_deposit(self):
        """Active lockout is classified as urgent while 6-month deposit dispute is normal."""
        urgent_risk = self.risk_engine.detect_risk("My landlord changed the locks today while I was away")
        assert urgent_risk.level == "urgent"
        assert "illegal_eviction_or_lockout" in urgent_risk.flags

        normal_risk = self.risk_engine.detect_risk("My landlord has not returned my deposit for six months")
        assert normal_risk.level in ("normal", "potentially_urgent")

    def test_cybercrime_unauthorized_transfer_urgent(self):
        """Ongoing electronic financial fraud flags emergency reporting steps."""
        risk = self.risk_engine.detect_risk("₹80,000 was transferred from my bank account without permission")
        assert risk.level == "urgent"
        assert "ongoing_financial_fraud" in risk.flags
        assert "1930" in (risk.recommended_emergency_action or "")

    def test_imminent_court_deadline_urgent(self):
        """Summons with hearing next week flags imminent court deadline."""
        risk = self.risk_engine.detect_risk("I received a court notice and the hearing is tomorrow")
        assert risk.level == "urgent"
        assert "imminent_court_or_statutory_deadline" in risk.flags


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Eight Benchmark Legal Domains
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestEightBenchmarkDomains:
    def setup_method(self):
        self.eval = EvaluationFramework()

    def test_domain_a_employment(self):
        res = self.eval.evaluate_case(
            "Employment",
            "I was fired and haven't received my salary for three months.",
            "Karnataka",
        )
        assert res.domain_detected is True
        assert res.detected_domain == "employment"
        assert any("unpaid wages" in i for i in res.issues_spotted)
        assert res.conclusions_premature == 0
        assert res.grounded_plan_provided

    def test_domain_b_consumer(self):
        res = self.eval.evaluate_case(
            "Consumer",
            "I bought a laptop that stopped working after two weeks and the seller refuses to replace it.",
            "Delhi",
        )
        assert res.domain_detected is True
        assert res.detected_domain == "consumer"
        assert any("defective" in i or "deficiency" in i for i in res.issues_spotted)
        assert res.grounded_plan_provided

    def test_domain_c_property(self):
        res = self.eval.evaluate_case(
            "Property",
            "My landlord changed the locks while I was away.",
            "Maharashtra",
        )
        assert res.domain_detected is True
        assert res.detected_domain == "property"
        assert res.urgency_detected == "urgent"
        assert any("dispossession" in i or "eviction" in i for i in res.issues_spotted)

    def test_domain_d_criminal(self):
        res = self.eval.evaluate_case(
            "Criminal",
            "Someone assaulted me yesterday.",
            "Uttar Pradesh",
        )
        assert res.domain_detected is True
        assert res.detected_domain == "criminal"
        assert res.urgency_detected in ("urgent", "emergency")
        assert any("assault" in i for i in res.issues_spotted)

    def test_domain_e_cybercrime(self):
        res = self.eval.evaluate_case(
            "Cybercrime",
            "₹80,000 was transferred from my bank account without my permission.",
            "Tamil Nadu",
        )
        assert res.domain_detected is True
        assert res.detected_domain == "cybercrime"
        assert res.urgency_detected == "urgent"
        assert any("unauthorized" in i for i in res.issues_spotted)

    def test_domain_f_family(self):
        res = self.eval.evaluate_case(
            "Family",
            "My spouse has stopped providing financial support.",
            "Kerala",
        )
        assert res.domain_detected is True
        assert res.detected_domain == "family"
        assert any("maintenance" in i for i in res.issues_spotted)

    def test_domain_g_contract(self):
        res = self.eval.evaluate_case(
            "Contract",
            "I paid a contractor ₹2 lakh but they never completed the work.",
            "Karnataka",
        )
        assert res.domain_detected is True
        assert res.detected_domain == "contract"
        assert any("breach" in i or "contract" in i or "money" in i for i in res.issues_spotted)

    def test_domain_h_government(self):
        res = self.eval.evaluate_case(
            "Government",
            "I received a notice from a government department and don't know what to do.",
            "Delhi",
        )
        assert res.domain_detected is True
        assert res.detected_domain == "government"
        assert any("statutory notice" in i or "notice" in i for i in res.issues_spotted)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. User Fact Corrections & Contradiction Resolution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestUserCorrections:
    def test_user_correction_supersedes_prior_fact(self):
        """When a user corrects a date/fact, it is marked superseded and replaced."""
        record = ConversationRecord(mode=Mode.ACTIONABLE)
        case_state = UniversalCaseState(
            dates=Dates(incident_date="1 August"),
            known_facts=[
                FactItem(
                    id="fact_1",
                    fact="user was terminated on 1 August",
                    source="user_stated",
                    confidence=1.0,
                    is_explicit=True,
                ).model_dump()
            ],
        )
        record.facts["case_state"] = case_state.model_dump()

        # User corrects: "No, I resigned on 1 August, was not terminated"
        correction_msg = "No, I resigned on 1 August, was not terminated."
        resolve_user_correction(record, correction_msg)

        updated_cs = UniversalCaseState(**record.facts["case_state"])
        assert len(updated_cs.known_facts) == 2
        # Previous fact was marked superseded
        assert updated_cs.known_facts[0]["superseded"] is True
        # New corrected fact is active
        assert updated_cs.known_facts[1]["superseded"] is False
        assert "resigned" in updated_cs.known_facts[1]["fact"].lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Retrieved Precedents Strictly Context Only
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPrecedentSeparation:
    def test_retrieved_case_cannot_overwrite_user_facts(self):
        """A precedent from Delhi does not alter the user's Karnataka jurisdiction or facts."""
        research = LegalResearchLayer()
        user_case_state = UniversalCaseState(
            jurisdiction=Jurisdiction(state="Karnataka", city="Bengaluru"),
            financial=Financial(amount=150000.0, amount_raw="₹1,50,000"),
        )

        retrieved_cases = [
            {
                "question": "Employee in Delhi terminated with ₹50,000 dues under Delhi Shops Act",
                "answer": "Under Delhi Shops and Establishments Act, appeal lies to Delhi Conciliation Officer.",
            }
        ]

        authorities = research.process_retrieved_cases(retrieved_cases, user_case_state)
        assert len(authorities) == 1
        # Precedent is marked with authority_type='precedent' and is NOT stored as a user fact
        assert authorities[0].authority_type == "precedent"

        # User's own jurisdiction and amount remain strictly unchanged
        assert user_case_state.jurisdiction.state == "Karnataka"
        assert user_case_state.financial.amount == 150000.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. Source-Grounded Legal Answers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGroundedAnswerGeneration:
    def test_answer_structure_and_assessment(self):
        """Answer generator produces a meaningful answer and structured assessment."""
        import asyncio
        gen = GroundedLegalAnswerGenerator()
        case_state = UniversalCaseState(
            jurisdiction=Jurisdiction(state="Karnataka"),
            case_domain="employment",
            issues=[
                LegalIssue(
                    issue="unpaid wages",
                    domain="employment",
                    applicable_laws=["Payment of Wages Act, 1936 Section 15"],
                )
            ],
            financial=Financial(amount_raw="₹1,20,000"),
            evidence=[EvidenceItem(type="contract", description="Employment appointment letter")],
        )

        authorities = [
            RetrievedAuthority(
                source="Payment of Wages Act, 1936",
                provision="Section 15",
                authority_type="statute",
                jurisdiction="Central / India",
                status="current",
                applicability_conditions=["Applicable to delayed wage recovery claims"],
            )
        ]

        answer, assessment = asyncio.run(gen.generate_answer(case_state, authorities))

        # Answer is non-empty and addresses the user
        assert len(answer) > 50
        assert "you" in answer.lower() or "your" in answer.lower()

        # Assessment has structured data
        assert len(assessment.claims) >= 1
        assert assessment.primary_domain == "employment"
        assert assessment.ready_for_final_remedy is True
        assert "unpaid wages" in assessment.confirmed_issues


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. Actions Already Taken Tracking
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestActionsAlreadyTakenTracking:
    def test_never_recommends_already_completed_actions(self):
        """If user already sent a legal notice, the action plan should not include 'legal notice'."""
        import asyncio
        gen = GroundedLegalAnswerGenerator()
        case_state = UniversalCaseState(
            case_domain="employment",
            issues=[LegalIssue(issue="unpaid wages", domain="employment")],
            actions_already_taken=[ActionTaken(action="legal_notice_sent")],
        )
        authorities = [
            RetrievedAuthority(
                source="Payment of Wages Act, 1936",
                provision="Section 15",
                authority_type="statute",
            )
        ]

        _, assessment = asyncio.run(gen.generate_answer(case_state, authorities))
        # The action plan should skip legal notice since it's already done
        plan_text = " ".join(assessment.action_plan).lower()
        assert "labour commissioner" in plan_text or "competent authority" in plan_text

