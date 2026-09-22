"""
Grounded Legal Answer Generator.

Uses an LLM call (OpenAI) to synthesize a natural, concise, lawyer-like
legal answer from structured case data and retrieved authorities.

The structured LegalAssessment is still computed deterministically and
returned as API metadata.  The user-facing answer is LLM-generated so
it reads like advice from an experienced advocate — direct, empathetic,
and concise — rather than a template dump.
"""

from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from conversation.cases.models import (
    LegalAssessment,
    RetrievedAuthority,
    UniversalCaseState,
)

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# System prompt for the answer-synthesis LLM call
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ANSWER_SYSTEM_PROMPT = """\
You are a senior Indian advocate providing clear, actionable, and empathetic legal guidance in an initial client consultation.

### TONE & STYLE
- Address the client directly in second person ("You", "Your employer", "Your landlord").
- Be warm, authoritative, and practical — like a trusted senior advocate in chambers.
- Be CONCISE and focused. Do not dump every statute in Indian law; cite only what directly applies to THIS client's situation.
- Use plain, empowering language. Explain legal terms simply when you first use them.
- Format with clean Markdown headings and bullet points for readability.

### STRUCTURE (use these exact headings)
1. **Understanding Your Situation** — 2-3 sentences summarizing the client's grievance, facts, and jurisdiction in plain English.
2. **Your Legal Rights & Applicable Laws** — 2-3 specific Indian acts/sections protecting them. Explain each provision in one clear sentence.
   - For employment disputes in private/IT firms: Highlight state-specific Shops and Commercial Establishments Acts (e.g., Section 39 of Karnataka Shops & Establishments Act requiring 30 days notice or wages in lieu), Payment of Wages Act, and contract remedies. Clarify that Industrial Disputes Act applies primarily to workmen unless non-managerial.
   - For tenancy: Cite local Rent Control / Tenancy Acts and Indian Contract Act.
   - For consumer/cyber: Cite Consumer Protection Act 2019 / IT Act 2000 and RBI zero-liability rules where applicable.
3. **What You Should Do Now** — Prioritized, concrete action steps (Step 1, Step 2, Step 3). Specify who does what, by when, and via what channel (e.g., formal demand notice via registered post/email, approaching the Labour Officer/Tribunal, or filing on e-Daakhil).
   - NOTE: Do NOT state that internal mediation or HR escalation is a mandatory legal prerequisite before issuing a legal notice or approaching statutory authorities.
4. **Important Timelines & Deadlines** — Real statutory limitation periods relevant to this dispute (e.g., wage claims, consumer complaints, or notice response windows).
5. **Documents to Preserve** — Clear bullet list of crucial records and communications to gather as evidence.
6. **A Word of Caution** — 1-2 practical cautionary sentences noting potential employer/counterparty defenses and advising consultation with a practicing advocate for formal filings.

### RULES
- Do NOT include statutes that are irrelevant to the established facts.
- Do NOT invent case numbers or nonexistent statutory sections.
- Keep the entire consultation guidance under 450 words. Focus on clarity, precision, and practical next steps.
"""


class GroundedLegalAnswerGenerator:
    """
    Generates a natural, LLM-powered legal answer plus a structured
    LegalAssessment from case state and retrieved authorities.
    """

    def __init__(self, openai_client: AsyncOpenAI | None = None, model: str = "gpt-4o-mini") -> None:
        self._client = openai_client
        self._model = model

    @property
    def is_configured(self) -> bool:
        """True if an OpenAI client is available for LLM synthesis."""
        return self._client is not None

    def build_assessment(
        self,
        case_state: UniversalCaseState,
        authorities: list[RetrievedAuthority],
    ) -> LegalAssessment:
        """Build the structured LegalAssessment metadata."""
        return self._build_assessment(case_state, authorities)

    async def generate_answer(
        self,
        case_state: UniversalCaseState,
        authorities: list[RetrievedAuthority],
        base_qa_answer: str | None = None,
    ) -> tuple[str, LegalAssessment]:
        """
        Synthesize a concise, lawyer-like legal answer via LLM and return
        both the user-facing text and the structured LegalAssessment.
        """
        # 1. Build the structured LegalAssessment (deterministic — always computed)
        assessment = self._build_assessment(case_state, authorities)

        # 2. Build the context message for the LLM
        context = self._build_context_message(case_state, authorities, base_qa_answer)

        # 3. Call LLM to generate natural answer
        if self.is_configured:
            try:
                answer = await self._call_llm(context)
            except Exception as exc:
                logger.warning(
                    "LLM answer synthesis failed (%s); falling back to template.",
                    exc,
                )
                answer = self._fallback_template(case_state, authorities, assessment, base_qa_answer)
        else:
            logger.info("No OpenAI client configured for answer generator; using template fallback.")
            answer = self._fallback_template(case_state, authorities, assessment, base_qa_answer)

        return answer, assessment

    # ── LLM Call ──────────────────────────────────────────────────

    async def _call_llm(self, context_message: str) -> str:
        """Make the actual OpenAI chat completion call."""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": context_message},
            ],
            temperature=0.3,
            max_tokens=1200,
        )
        return response.choices[0].message.content.strip()

    # ── Context Builder ───────────────────────────────────────────

    def _build_context_message(
        self,
        state: UniversalCaseState,
        authorities: list[RetrievedAuthority],
        base_qa_answer: str | None = None,
    ) -> str:
        """
        Build a structured context message that gives the LLM everything
        it needs to produce a grounded answer.
        """
        parts: list[str] = []

        # Client facts
        parts.append("## CLIENT FACTS")
        if state.summary:
            parts.append(f"Summary: {state.summary}")

        active_facts = [f for f in state.known_facts if not f.get("superseded", False)]
        if active_facts:
            parts.append("Established facts:")
            for f in active_facts:
                fact_text = f.get("fact") if isinstance(f, dict) else str(f)
                parts.append(f"  - {fact_text}")

        # Jurisdiction
        if state.jurisdiction.state or state.jurisdiction.city:
            loc = ", ".join(filter(None, [
                state.jurisdiction.city,
                state.jurisdiction.state,
                state.jurisdiction.country,
            ]))
            parts.append(f"Jurisdiction: {loc}")
        else:
            parts.append("Jurisdiction: India (state/city NOT specified by client)")

        # Financial
        if state.financial.amount_raw or state.financial.amount:
            amt = state.financial.amount_raw or f"₹{state.financial.amount:,.2f}"
            parts.append(f"Financial claim: {amt}")

        # Dates
        if state.dates.incident_date:
            parts.append(f"Incident date: {state.dates.incident_date}")

        # User goal
        if state.user_goal:
            parts.append(f"Client's stated goal: {state.user_goal}")

        # Issues
        parts.append("\n## IDENTIFIED LEGAL ISSUES")
        active_issues = [i for i in state.issues if i.status != "ruled_out"]
        if active_issues:
            for issue in active_issues:
                status = "confirmed" if issue.status == "confirmed" else "likely"
                laws_str = ", ".join(issue.applicable_laws) if issue.applicable_laws else "to be determined"
                parts.append(f"  - {issue.issue} ({status}) — under: {laws_str}")
        else:
            parts.append(f"  - {state.case_type or 'General legal dispute'}")

        # Applicable statutes (only statutory authorities)
        statutory = [a for a in authorities if a.authority_type != "precedent"]
        if statutory:
            parts.append("\n## APPLICABLE STATUTES (use only what's relevant)")
            for auth in statutory:
                conditions = ", ".join(auth.applicability_conditions) if auth.applicability_conditions else "general"
                parts.append(
                    f"  - {auth.source}, {auth.provision}: {auth.relevance}\n"
                    f"    Key rule: \"{auth.key_excerpt}\"\n"
                    f"    Applies if: {conditions}"
                )

        # Precedents
        precedents = [a for a in authorities if a.authority_type == "precedent"]
        if precedents:
            parts.append("\n## RETRIEVED PRECEDENTS (use only if directly relevant)")
            for p in precedents:
                parts.append(f"  - {p.source} ({p.jurisdiction}): {p.key_excerpt[:200]}")

        # Base QA model answer (if available)
        if base_qa_answer:
            parts.append(f"\n## BASE MODEL GUIDANCE\n{base_qa_answer[:500]}")

        # Actions already taken
        if state.actions_already_taken:
            parts.append("\n## ACTIONS CLIENT HAS ALREADY TAKEN")
            for a in state.actions_already_taken:
                parts.append(f"  - {a.action}")

        # Risk flags
        if state.risk.flags:
            parts.append(f"\n## URGENCY: {state.risk.level.upper()}")
            if state.risk.reason:
                parts.append(f"Reason: {state.risk.reason}")
            if state.risk.recommended_emergency_action:
                parts.append(f"Emergency action: {state.risk.recommended_emergency_action}")

        parts.append(
            "\n## INSTRUCTION\n"
            "Based on the above, provide your legal guidance to the client. "
            "Be concise, direct, and actionable. Only cite statutes that ACTUALLY "
            "apply to the established facts. Skip irrelevant authorities."
        )

        return "\n".join(parts)

    # ── Structured Assessment Builder ─────────────────────────────

    def _build_assessment(
        self,
        state: UniversalCaseState,
        authorities: list[RetrievedAuthority],
    ) -> LegalAssessment:
        """Build the structured LegalAssessment for API metadata."""
        domain = state.case_domain or state.primary_category or "General Legal Dispute"

        # Facts summary
        facts_parts: list[str] = []
        if state.summary:
            facts_parts.append(f"- **Summary**: {state.summary}")
        active_facts = [f for f in state.known_facts if not f.get("superseded", False)]
        for f in active_facts:
            fact_text = f.get("fact") if isinstance(f, dict) else str(f)
            src = f.get("source", "user statement") if isinstance(f, dict) else "user statement"
            facts_parts.append(f"- **Established Fact**: {fact_text} *(Source: {src})*")
        if state.financial.amount_raw or state.financial.amount:
            amt = state.financial.amount_raw or f"₹{state.financial.amount:,.2f}"
            facts_parts.append(f"- **Financial Claim / Disputed Sum**: {amt}")
        facts_summary = "\n".join(facts_parts) if facts_parts else "Based on your statements."

        # Claims
        claims = [
            {
                "authority": f"{auth.source} - {auth.provision}",
                "jurisdiction": auth.jurisdiction,
                "status": auth.status,
            }
            for auth in authorities
        ]

        # Check government employment
        is_govt = False
        if isinstance(state.domain_extensions, dict):
            ext_emp = state.domain_extensions.get("employment", {})
            if isinstance(ext_emp, dict) and ext_emp.get("employment_type") == "government":
                is_govt = True
        if not is_govt:
            for f in state.known_facts:
                f_str = str(f).lower()
                if "government" in f_str or "civil servant" in f_str:
                    is_govt = True
                    break
        if not is_govt and isinstance(state.parties, list):
            for p in state.parties:
                p_str = str(p).lower()
                if "government" in p_str or "public service" in p_str:
                    is_govt = True
                    break

        # Evidence & document assessment
        evidence_assessment = self._build_evidence_assessment(state, domain, is_govt)

        # Action plan
        action_plan = self._build_action_plan(state, domain, is_govt)

        return LegalAssessment(
            summary=facts_summary,
            primary_domain=domain,
            confirmed_issues=[i.issue for i in state.issues if i.status != "ruled_out"],
            applicable_authorities=authorities,
            evidence_assessment=evidence_assessment,
            action_plan=action_plan,
            limitations_and_risks=[],
            claims=claims,
            ready_for_final_remedy=True,
        )

    def _build_evidence_assessment(
        self,
        state: UniversalCaseState,
        domain: str,
        is_govt: bool = False,
    ) -> dict[str, Any]:
        """
        Deep evidentiary analysis:
        1. Confirmed documents provided by client
        2. Required documents checklist for court/tribunal filing
        3. Evidentiary fallback strategy if documents are missing or withheld
        """
        d = domain.lower()
        required_docs: list[str] = []
        fallback_strategy: list[str] = []

        if "employment" in d or "salary" in d:
            if is_govt:
                required_docs = [
                    "Appointment Letter / Service Confirmation Order",
                    "Official Termination / Dismissal Order or Show Cause Notice",
                    "Departmental Inquiry Officer's Report and Findings (if inquiry was conducted)",
                    "Last 3-6 months' Salary Slips / Bank Account Statement showing non-payment of ₹3 Lakh dues",
                    "Copies of statutory representations or appeals filed before the Departmental Appellate Authority"
                ]
                fallback_strategy = [
                    "If official termination order or inquiry findings are withheld: File an urgent application under Section 6 of the Right to Information (RTI) Act, 2005 requesting certified copies of the termination order, inquiry report, and file notings.",
                    "If appointment order is missing: Use Bank Account salary credit statements, GPF/NPS statement, Service Book extract, and Employee ID Card as secondary proof of government employment.",
                    "If unpaid dues calculation is disputed: Submit a requisition under RTI for the LPC (Last Pay Certificate) and due-drawn statement from the DDO (Drawing and Disbursing Officer)."
                ]
            else:
                required_docs = [
                    "Offer Letter / Employment Agreement (stating designation, salary, notice period clause)",
                    "Formal Termination Letter / Email stating grounds for dismissal",
                    "Salary slips for the last 3 months + Bank statements reflecting salary credits and non-payment",
                    "Full & Final (F&F) settlement statement or email correspondence demanding pending dues",
                    "EPFO / UAN passbook showing employment tenure and PF contributions"
                ]
                fallback_strategy = [
                    "If written employment contract is missing: Produce Bank statements showing regular monthly salary credits from the employer, EPFO UAN ledger, ESIC registration, company email signature, and ID card as conclusive secondary proof of employment.",
                    "If termination was verbal without written notice: Immediately send an email/speed post letter placing on record that you reported for duty and were verbally turned away; demand written reasons and formal notice under protest.",
                    "If dues records are withheld by employer: Move an application under Section 33C(2) Industrial Disputes Act / Payment of Wages Act requiring the employer to produce wage registers and muster rolls before the Authority."
                ]
        elif "property" in d or "tenant" in d:
            required_docs = [
                "Registered Lease / Leave and License Agreement",
                "Rent receipts / Bank UPI transaction statements proving regular payment of rent",
                "Security deposit payment receipt / bank debit reference",
                "Formal written notice of termination/eviction (if served)",
                "Photographs / Police Diary (GD) entry of lockout or utility disconnection"
            ]
            fallback_strategy = [
                "If written agreement is oral or expired: Use monthly bank/UPI rent debit statements and electricity/gas bills in client's name to prove settled possession.",
                "If landlord cuts off electricity or water: File an emergency application under Section 29 Maharashtra Rent Control Act / state equivalent before the Rent Controller / Civil Court for immediate restoration with police assistance."
            ]
        elif "consumer" in d:
            required_docs = [
                "Tax Invoice / Retail Bill of Purchase",
                "Warranty Card / Product Guarantee Certificate",
                "Authorized Service Center Inspection Report / Job Sheet acknowledging the defect",
                "Written complaint / email chain / support chat transcripts requesting repair/replacement/refund",
                "Formal 15-day statutory legal notice served on manufacturer and seller"
            ]
            fallback_strategy = [
                "If physical bill is lost: Download electronic invoice from the e-commerce portal, or obtain transaction confirmation from credit card / UPI bank statement under Rule 5 of Consumer Protection (E-Commerce) Rules, 2020.",
                "If service center refused to give job sheet: Send an email capturing the refusal with date and time of visit, and file complaint on the National Consumer Helpline (consumerhelpline.gov.in / 1915) to establish documentary trail."
            ]
        elif "cyber" in d:
            required_docs = [
                "Bank Account Statement showing unauthorized debit transactions with UTR/Transaction IDs",
                "Copy of formal written zero-liability complaint submitted to bank within 72 hours",
                "Acknowledgment receipt from National Cyber Crime Reporting Portal (cybercrime.gov.in / 1930)",
                "Screenshots of fraudulent SMS, phishing links, spoofed emails, or communication channels"
            ]
            fallback_strategy = [
                "If bank refuses to accept zero-liability complaint: Send complaint via registered email to the Nodal Grievance Redressal Officer of the bank, and escalate to the RBI Banking Ombudsman (cms.rbi.org.in).",
                "If SMS/call logs are deleted: Request call detail records (CDR) and SMS logs from the telecom service provider immediately."
            ]
        else:
            required_docs = [
                "Written agreement, purchase order, or correspondence establishing legal relationship",
                "Bank statements / financial receipts establishing monetary transaction / loss",
                "Formal statutory legal notice demanding performance / remedy"
            ]
            fallback_strategy = [
                "If agreement is oral: Compile bank records, WhatsApp admissions, and email trails to substantiate oral contract under Section 10 of the Indian Contract Act, 1872."
            ]

        return {
            "provided_count": len(state.evidence),
            "items": [e.model_dump() for e in state.evidence],
            "required_documents_checklist": required_docs,
            "evidentiary_fallback_strategy": fallback_strategy,
        }

    def _build_action_plan(
        self,
        state: UniversalCaseState,
        domain: str,
        is_govt: bool = False,
    ) -> list[str]:
        """Build a prioritised action plan based on domain and government status."""
        taken_lower = [a.action.lower() for a in state.actions_already_taken]
        plan: list[str] = []
        step = 1
        d = domain.lower()

        if "employment" in d or "salary" in d:
            if is_govt:
                plan.append(f"{step}. Submit a formal departmental representation/appeal to the designated Appellate Authority under Service Rules challenging wrongful dismissal.")
                step += 1
                plan.append(f"{step}. If representation is unanswered or rejected, approach the Central Administrative Tribunal (CAT) or State Administrative Tribunal under Section 19 of the Administrative Tribunals Act, 1985 / High Court under Article 226.")
                step += 1
                plan.append(f"{step}. If service records, inquiry reports, or due-drawn statements are withheld, file an urgent application under Section 6 of the RTI Act, 2005.")
                step += 1
            else:
                if not any("notice" in a or "demand" in a for a in taken_lower):
                    plan.append(f"{step}. Issue a formal legal demand notice to your employer.")
                    step += 1
                plan.append(f"{step}. Approach the Labour Commissioner / competent authority under the Shops & Establishments Act or Industrial Disputes Act.")
                step += 1
        elif "cyber" in d:
            if not any("1930" in a or "cyber" in a for a in taken_lower):
                plan.append(f"{step}. Dial 1930 and register on cybercrime.gov.in immediately.")
                step += 1
            if not any("bank" in a for a in taken_lower):
                plan.append(f"{step}. Submit written zero-liability dispute to your bank within 72 hours.")
                step += 1
        elif "property" in d or "tenant" in d:
            plan.append(f"{step}. File a police complaint if facing physical lockout or threats.")
            step += 1
            plan.append(f"{step}. Move urgent injunction before the Civil Court / Rent Controller.")
            step += 1
        elif "consumer" in d:
            plan.append(f"{step}. Send a 15-day statutory legal notice to the seller.")
            step += 1
            plan.append(f"{step}. File consumer complaint on e-Daakhil (edaakhil.nic.in).")
            step += 1
        else:
            plan.append(f"{step}. Serve a formal statutory demand notice via registered post.")
            step += 1
            plan.append(f"{step}. Initiate legal proceedings if the demand is ignored.")
            step += 1

        return plan

    # ── Fallback Template (when LLM is unavailable) ───────────────

    def _fallback_template(
        self,
        state: UniversalCaseState,
        authorities: list[RetrievedAuthority],
        assessment: LegalAssessment,
        base_qa_answer: str | None = None,
    ) -> str:
        """
        Produce a reasonable text answer without an LLM call.
        Used when OpenAI is not configured or the call fails.
        """
        parts: list[str] = []

        # Understanding
        parts.append("### Understanding Your Situation")
        if state.summary:
            parts.append(state.summary)
        elif base_qa_answer:
            parts.append(base_qa_answer)

        # Legal position
        statutory = [a for a in authorities if a.authority_type != "precedent"]
        if statutory:
            parts.append("\n### Your Legal Position")
            for auth in statutory[:3]:  # Cap at 3 most relevant
                parts.append(
                    f"- **{auth.source} ({auth.provision})**: {auth.relevance}"
                )

        # Actions
        if assessment.action_plan:
            parts.append("\n### What You Should Do Now")
            for step in assessment.action_plan:
                parts.append(step)

        # Evidence
        parts.append("\n### Documents to Keep Safe")
        parts.append("- Employment/appointment letters and contracts")
        parts.append("- Bank statements showing salary history")
        parts.append("- All communications (emails, WhatsApp, letters)")
        parts.append("- Any notices received or sent")

        # Caution
        parts.append(
            "\n### A Word of Caution\n"
            "This guidance outlines your statutory rights under Indian law. "
            "For formal legal notices, court filings, and contested proceedings, "
            "please consult a qualified advocate enrolled with your State Bar Council."
        )

        return "\n".join(parts)


# Module-level instance — will be properly initialized in app.py
grounded_answer_generator = GroundedLegalAnswerGenerator()
