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
You are a senior Indian advocate providing clear, actionable legal guidance.

### TONE & STYLE
- Address the client directly in second person ("You", "Your employer").
- Be warm but professional — like a trusted lawyer in a first consultation.
- Be CONCISE. Do not dump every statute you know. Focus on what matters MOST for THIS specific case.
- Use simple language a non-lawyer can understand. Explain legal terms when you first use them.
- Never manufacture citations or case numbers you are not sure about.

### STRUCTURE (use these exact headings)
1. **Understanding Your Situation** — 2-3 sentences summarising what happened, in your own words.
2. **Your Legal Position** — Which laws protect them, citing only the MOST relevant 2-3 statutes/provisions. Include the specific section and a one-line plain-English explanation. Skip any statute that does NOT directly apply to the facts.
3. **What You Should Do Now** — Numbered, concrete, prioritised action steps (max 4-5 steps). Each step should say WHO does WHAT by WHEN.
4. **Important Deadlines** — Only if there are actual time-sensitive deadlines relevant to this case. Skip this section if none apply.
5. **Documents to Keep Safe** — Brief bullet list of evidence to preserve, specific to their case.
6. **A Word of Caution** — 1-2 sentences noting any risks, gaps in their case, or when to definitely hire a lawyer.

### RULES
- Do NOT include statutes that are irrelevant to the established facts (e.g., don't cite cheque bounce law if no cheque is involved, don't cite gratuity act if tenure is unknown/under 5 years).
- Do NOT assume jurisdiction (state/city) unless the client has explicitly stated it.
- Do NOT repeat the same information in multiple sections.
- Do NOT use phrases like "Hypothesis (70% confidence)" or "persuasive_context_only" — those are internal metadata, not client-facing language.
- If retrieved precedents are relevant, weave their guidance naturally into your advice — do NOT dump raw case excerpts.
- Keep the entire answer under 400 words. Quality over quantity.
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
                answer = self._fallback_template(case_state, authorities, assessment)
        else:
            logger.info("No OpenAI client configured for answer generator; using template fallback.")
            answer = self._fallback_template(case_state, authorities, assessment)

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

        # Action plan
        action_plan = self._build_action_plan(state, domain)

        return LegalAssessment(
            summary=facts_summary,
            primary_domain=domain,
            confirmed_issues=[i.issue for i in state.issues if i.status != "ruled_out"],
            applicable_authorities=authorities,
            evidence_assessment={
                "provided_count": len(state.evidence),
                "items": [e.model_dump() for e in state.evidence],
            },
            action_plan=action_plan,
            limitations_and_risks=[],
            claims=claims,
            ready_for_final_remedy=True,
        )

    def _build_action_plan(self, state: UniversalCaseState, domain: str) -> list[str]:
        """Build a prioritised action plan based on domain."""
        taken_lower = [a.action.lower() for a in state.actions_already_taken]
        plan: list[str] = []
        step = 1
        d = domain.lower()

        if "employment" in d or "salary" in d:
            if not any("notice" in a or "demand" in a for a in taken_lower):
                plan.append(f"{step}. Issue a formal legal demand notice to your employer.")
                step += 1
            plan.append(f"{step}. Approach the Labour Commissioner / competent authority.")
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
