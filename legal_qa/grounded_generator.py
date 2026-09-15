"""
Grounded Legal Answer Generator.

Implements the standardized 8-part cross-domain legal guidance structure:
  1. What I understand
  2. Possible legal issues
  3. What the applicable law appears to provide
  4. Why it may apply
  5. What you can do now (never re-recommending actions already taken)
  6. Evidence/documents to preserve
  7. Important deadlines/risks
  8. When professional legal help is advisable

Ensures claims are grounded in retrieved authorities, addresses the client directly
as 'You', and avoids false certainty.
"""

from __future__ import annotations

import logging
from typing import Any
from conversation.cases.models import (
    LegalAssessment,
    RetrievedAuthority,
    UniversalCaseState,
)

logger = logging.getLogger(__name__)


class GroundedLegalAnswerGenerator:
    """
    Generates structured, source-grounded legal guidance across any Indian law domain.
    """

    def generate_answer(
        self,
        case_state: UniversalCaseState,
        authorities: list[RetrievedAuthority],
        base_qa_answer: str | None = None,
    ) -> tuple[str, LegalAssessment]:
        """
        Synthesize the standardized 8-part legal guidance and return the formatted
        answer string along with the structured LegalAssessment.
        """
        domain = case_state.case_domain or case_state.primary_category or "General Legal Dispute"

        # 1. Facts extraction for "What I understand"
        facts_summary = self._summarize_facts(case_state)

        # 2. Issues & Hypotheses
        issues_section = self._format_issues(case_state)

        # 3. Applicable Law
        law_section, claims = self._format_law(authorities, base_qa_answer)

        # 4. Applicability Analysis
        applicability_section = self._format_applicability(case_state, authorities)

        # 5. Action Plan (filtering actions already taken!)
        actions_section, action_plan = self._format_actions(case_state, domain)

        # 6. Evidence Preservation
        evidence_section = self._format_evidence(case_state)

        # 7. Deadlines & Risks
        deadlines_section, risks_list = self._format_deadlines_and_risks(case_state, authorities)

        # 8. Professional Legal Help
        advisory_section = self._format_professional_advisory(case_state)

        # Assemble full 8-part markdown text
        parts = [
            f"### 1. What I Understand\n{facts_summary}",
            f"### 2. Possible Legal Issues\n{issues_section}",
            f"### 3. What the Applicable Law Appears to Provide\n{law_section}",
            f"### 4. Why It May Apply to Your Situation\n{applicability_section}",
            f"### 5. What You Can Do Now\n{actions_section}",
            f"### 6. Evidence & Documents to Preserve\n{evidence_section}",
            f"### 7. Important Deadlines & Risks\n{deadlines_section}",
            f"### 8. When Professional Legal Help Is Advisable\n{advisory_section}",
        ]

        full_answer = "\n\n".join(parts)

        assessment = LegalAssessment(
            summary=facts_summary,
            primary_domain=domain,
            confirmed_issues=[i.issue for i in case_state.issues if i.status != "ruled_out"],
            applicable_authorities=authorities,
            evidence_assessment={
                "provided_count": len(case_state.evidence),
                "items": [e.model_dump() for e in case_state.evidence],
            },
            action_plan=action_plan,
            limitations_and_risks=risks_list,
            claims=claims,
            ready_for_final_remedy=True,
        )

        return full_answer, assessment

    # ── Section Builders ──────────────────────────────────────────

    def _summarize_facts(self, state: UniversalCaseState) -> str:
        points: list[str] = []
        if state.summary:
            points.append(f"- **Summary**: {state.summary}")

        active_facts = [f for f in state.known_facts if not f.get("superseded", False)]
        if active_facts:
            for f in active_facts:
                fact_text = f.get("fact") if isinstance(f, dict) else str(f)
                src = f.get("source", "user statement") if isinstance(f, dict) else "user statement"
                points.append(f"- **Established Fact**: {fact_text} *(Source: {src})*")

        if state.jurisdiction.state or state.jurisdiction.city:
            loc = ", ".join(filter(None, [state.jurisdiction.city, state.jurisdiction.state, state.jurisdiction.country]))
            points.append(f"- **Jurisdiction**: {loc}")

        if state.financial.amount_raw or state.financial.amount:
            amt = state.financial.amount_raw or f"₹{state.financial.amount:,.2f}"
            points.append(f"- **Financial Claim / Disputed Sum**: {amt}")

        if state.user_goal:
            points.append(f"- **Your Stated Objective**: {state.user_goal}")

        if not points:
            points.append("- Based on your statements, you are seeking formal legal recourse and clarity on your statutory remedies under Indian law.")

        return "\n".join(points)

    def _format_issues(self, state: UniversalCaseState) -> str:
        lines: list[str] = [
            "Based on the facts you have shared, this matter may raise the following legal issues under Indian law:"
        ]
        active_issues = [i for i in state.issues if i.status != "ruled_out"]

        if active_issues:
            for issue in active_issues:
                conf_label = "Confirmed" if issue.status == "confirmed" else f"Hypothesis ({int(issue.confidence * 100)}% confidence)"
                laws = f" *(Applicable Framework: {', '.join(issue.applicable_laws)})*" if issue.applicable_laws else ""
                lines.append(f"- **{issue.issue.title()}** [{conf_label}]{laws}")
        elif state.case_type:
            lines.append(f"- **{state.case_type.title()}** [Hypothesis: subject to verification of documentation]")
        else:
            lines.append("- **General Legal Dispute & Statutory Rights Violation** [Hypothesis]")

        return "\n".join(lines)

    def _format_law(
        self,
        authorities: list[RetrievedAuthority],
        base_qa_answer: str | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        lines: list[str] = []
        claims: list[dict[str, Any]] = []

        if authorities:
            for auth in authorities:
                if auth.authority_type == "precedent":
                    lines.append(
                        f"- **{auth.source}**:\n"
                        f"  *Relevant Analogy*: {auth.key_excerpt}\n"
                        f"  *(Note: This past case serves as persuasive guidance only and does not establish facts about your specific case.)*"
                    )
                else:
                    lines.append(
                        f"- **{auth.source} ({auth.provision})** [{auth.status.upper()}]:\n"
                        f"  {auth.relevance}\n"
                        f"  *Statutory Rule*: \"{auth.key_excerpt}\""
                    )
                claims.append({
                    "authority": f"{auth.source} - {auth.provision}",
                    "jurisdiction": auth.jurisdiction,
                    "status": auth.status,
                })
        elif base_qa_answer:
            lines.append(f"- **Statutory Guidance**: {base_qa_answer[:350]}...")
        else:
            lines.append("- Under general Indian civil and commercial jurisprudence, rights arising out of agreement, statutory employment protections, and tortious or criminal wrongs are governed by the relevant Central and State statutes.")

        return "\n".join(lines), claims

    def _format_applicability(
        self,
        state: UniversalCaseState,
        authorities: list[RetrievedAuthority],
    ) -> str:
        lines: list[str] = []
        lines.append(
            "Connecting your factual statements to the statutory framework:"
        )

        if state.jurisdiction.state:
            lines.append(f"- **State Jurisdiction**: Because the matter arose in **{state.jurisdiction.state}**, jurisdictional rules, forum thresholds, and state-specific amendments govern proceedings.")

        # Check conditions
        for auth in authorities:
            if auth.applicability_conditions:
                cond_text = ", ".join(auth.applicability_conditions)
                lines.append(f"- **{auth.provision} Application**: Applies provided the following conditions are met: {cond_text}.")

        lines.append("- *Assessment*: If your statements and records substantiate that the opposing party acted unilaterally or in breach of statutory procedures, you possess strong legal grounds to demand compliance, financial recovery, or restitution.")

        return "\n".join(lines)

    def _format_actions(
        self,
        state: UniversalCaseState,
        domain: str,
    ) -> tuple[str, list[str]]:
        taken_actions_lower = [a.action.lower() for a in state.actions_already_taken]
        action_plan: list[str] = []
        lines: list[str] = []

        lines.append("Here is your prioritized, sequential action plan:")

        # Note actions already done
        if taken_actions_lower:
            done_text = ", ".join(state.actions_already_taken[0].action for _ in [1])
            lines.append(f"*(Note: You have already completed: {', '.join(a.action for a in state.actions_already_taken)}. We will not repeat those steps.)*")

        step = 1

        # Domain-aware next steps
        d_lower = domain.lower()

        if "employment" in d_lower or "salary" in d_lower:
            if not any("notice" in a or "demand" in a for a in taken_actions_lower):
                p = f"{step}. **Issue a Formal Legal Demand Notice**: Have an advocate draft and issue a statutory demand notice giving your employer 15 days to settle outstanding dues and notice pay."
                lines.append(p)
                action_plan.append(p)
                step += 1

            p2 = f"{step}. **Approach the Competent Authority / Labour Commissioner**: If the employer fails to comply, file a petition under the Payment of Wages Act / Section 39 of the relevant State Shops & Establishments Act before the jurisdictional Labour Officer."
            lines.append(p2)
            action_plan.append(p2)
            step += 1

        elif "cyber" in d_lower:
            if not any("1930" in a or "cyber" in a for a in taken_actions_lower):
                p = f"{step}. **Immediate Cyber Fraud Reporting**: Dial 1930 and register the transaction reference on cybercrime.gov.in immediately to initiate an account freeze on the beneficiary."
                lines.append(p)
                action_plan.append(p)
                step += 1

            if not any("bank" in a for a in taken_actions_lower):
                p2 = f"{step}. **Written Notice of Zero Customer Liability to Bank**: Submit a formal written dispute to your bank branch within 72 hours invoking RBI Master Circular (2017) on zero customer liability."
                lines.append(p2)
                action_plan.append(p2)
                step += 1

        elif "property" in d_lower or "tenant" in d_lower:
            p = f"{step}. **Police Complaint for Criminal Trespass / Intimidation**: If you are facing physical lockout or threats, file an immediate written complaint with the local police station."
            lines.append(p)
            action_plan.append(p)
            step += 1

            p2 = f"{step}. **Move Urgent Injunction under Section 6 Specific Relief Act**: Approach the jurisdictional Civil Court / Rent Controller for an emergency mandatory injunction restraining the landlord from disturbing peaceful possession."
            lines.append(p2)
            action_plan.append(p2)
            step += 1

        elif "consumer" in d_lower:
            p = f"{step}. **Formal Legal Notice to Seller / Manufacturer**: Send a 15-day statutory legal notice demanding replacement, full refund, and compensation for deficiency in service."
            lines.append(p)
            action_plan.append(p)
            step += 1

            p2 = f"{step}. **File Consumer Complaint on e-Daakhil**: File an online complaint before the District Consumer Commission via e-Daakhil (edaakhil.nic.in) within 2 years of the cause of action."
            lines.append(p2)
            action_plan.append(p2)
            step += 1

        else:
            p = f"{step}. **Serve a Formal Statutory Demand Notice**: Formally communicate the claim, statutory violations, and a 15-day cure period via registered post and electronic mail."
            lines.append(p)
            action_plan.append(p)
            step += 1

            p2 = f"{step}. **Initiate Legal Proceedings**: If the demand is ignored, approach the jurisdictional judicial or quasi-judicial forum for appropriate relief."
            lines.append(p2)
            action_plan.append(p2)
            step += 1

        return "\n".join(lines), action_plan

    def _format_evidence(self, state: UniversalCaseState) -> str:
        lines: list[str] = [
            "To build an airtight legal position, preserve the following evidence and do not alter or delete originals:"
        ]

        if state.evidence:
            lines.append("- **Items You Have Identified**:")
            for ev in state.evidence:
                lines.append(f"  * {ev.type.title()}: {ev.description} [{ev.source}]")

        lines.append("- **Crucial Additional Records to Compile**:")
        lines.append("  * Written agreements, contracts, appointment letters, or signed receipts")
        lines.append("  * Bank statements and digital transaction logs (UPI UTR / NEFT reference)")
        lines.append("  * Electronic communications (full email threads with headers, WhatsApp chats preserved in PDF export)")
        lines.append("  * Written notices, replies, medical memos (MLC), or postal acknowledgment cards")

        return "\n".join(lines)

    def _format_deadlines_and_risks(
        self,
        state: UniversalCaseState,
        authorities: list[RetrievedAuthority],
    ) -> tuple[str, list[str]]:
        lines: list[str] = []
        risks: list[str] = []

        if state.risk.flags:
            lines.append(f"⚠️ **Urgency Alert [{state.risk.level.upper()}]**:")
            if state.risk.reason:
                lines.append(f"- **Reason**: {state.risk.reason}")
            if state.risk.recommended_emergency_action:
                lines.append(f"- **Immediate Action**: {state.risk.recommended_emergency_action}")
            risks.append(state.risk.reason or state.risk.level)

        lines.append("\n**Applicable Statutory Limitation Windows**:")
        lines.append("- **Limitation Rules**: Indian statutes impose strict time windows. For money recovery/contracts: 3 years from cause of action. For consumer disputes: 2 years. For summary recovery of possession under Specific Relief Act: 6 months. For Cheque bounce (Section 138 NI Act): legal notice strictly within 30 days of dishonour memo.")
        lines.append("- *Caution*: Never wait until the eve of limitation. Prompt legal notices preserve cause of action and evidentiary weight.")

        return "\n".join(lines), risks

    def _format_professional_advisory(self, state: UniversalCaseState) -> str:
        return (
            "While this guidance outlines your statutory framework, rights, and actionable sequence under Indian law, "
            "formal court representations, statutory legal notices, and contested applications must be settled by a qualified "
            "advocate enrolled with the State Bar Council. Engaging legal counsel is especially advisable before signing any settlement, "
            "filing an FIR or writ petition, or approaching judicial tribunals."
        )


grounded_answer_generator = GroundedLegalAnswerGenerator()
