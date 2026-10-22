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

from conversation.cases.domains import domain_registry
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

### MULTILINGUAL GUIDANCE
- If a Target Client Language is specified (e.g., Telugu, Hindi, Tamil, Kannada, Marathi, Bengali, Gujarati, etc.):
  * Formulate the entire 6-part legal consultation guidance and action steps directly in that language.
  * Explain legal rights and procedures simply and empathetically.
  * Maintain official Indian statutory titles bilingually (e.g. native script followed by official English Act name in parentheses, such as "తెలంగాణ దుకాణాలు మరియు సంస్థల చట్టం, 1988 (Telangana Shops and Establishments Act, 1988) - సెక్షన్ 47" or "वेतन भुगतान अधिनियम (Payment of Wages Act)").
  * If the client communicates in English, respond in English.
"""


INFORMATIVE_SYSTEM_PROMPT = """\
You are an expert Indian legal scholar and advocate providing clear, accurate, and comprehensive legal information in response to a user's informative inquiry.

### TONE & STYLE
- Address the user respectfully and directly in second person ("You", "Your rights").
- Provide objective, authoritative, and helpful legal explanation of the law, statutory provisions, rights, and restrictions.
- Be CONCISE and focused (under 450 words).
- Format with clean Markdown headings and bullet points.

### STRUCTURE (use these exact headings for Informative Guidance)
1. **Understanding Your Inquiry** — Brief 1-2 sentence acknowledgment of the legal topic or question asked.
2. **Key Legal Framework & Applicable Laws** — 2-3 specific Indian acts/sections governing this topic (e.g. Mental Healthcare Act, 2017, Rights of Persons with Disabilities Act, 2016).
3. **Rights & Legal Protections** — Clear explanation of statutory rights, non-discrimination provisions, and entitlements.
4. **Restrictions, Safeguards & Limitations** — Any legal restrictions, boundaries, or procedures mandated by law.
5. **Authorities & Redressal Bodies** — Statutory bodies or authorities responsible for oversight and grievance redressal (e.g., State Mental Health Authority, Legal Services Authority).
6. **Next Steps & Practical Guidance** — Empathetic, practical advice (e.g., medical consultation, approaching legal aid or specialized advisors).

### RULES
- Ground your answer strictly in the established legal topic and the provided Base Legal Research Answer / Applicable Statutes.
- Do NOT invent fictional labor/wage claims, salary slips, or employer dispute notices if the inquiry is about a general or civil legal topic like mental health, constitutional rights, or family law.
- Do NOT cite statutes that do not pertain to the established topic.

### MULTILINGUAL GUIDANCE
- If a Target Client Language is specified (e.g., Telugu, Hindi, Tamil, Kannada, Marathi, Bengali, Gujarati, etc.):
  * Formulate the entire informative guidance directly in that language.
  * Translate and adapt the substantive legal principles accurately into clear, natural vernacular.
  * Maintain official Indian statutory titles bilingually (e.g. native script followed by official English Act name in parentheses, such as "మానసిక ఆరోగ్య సంరక్షణ చట్టం, 2017 (Mental Healthcare Act, 2017)").
  * If the client communicates in English, respond in English.
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
        mode: Any = None,
    ) -> LegalAssessment:
        """Build the structured LegalAssessment metadata."""
        return self._build_assessment(case_state, authorities, mode=mode)

    async def generate_answer(
        self,
        case_state: UniversalCaseState,
        authorities: list[RetrievedAuthority],
        base_qa_answer: str | None = None,
        language: str | None = None,
        mode: Any = None,
    ) -> tuple[str, LegalAssessment]:
        """
        Synthesize a concise, lawyer-like legal answer via LLM and return
        both the user-facing text and the structured LegalAssessment.
        """
        # 1. Build the structured LegalAssessment (deterministic — always computed)
        assessment = self._build_assessment(case_state, authorities, mode=mode)

        # 2. Build the context message for the LLM
        context = self._build_context_message(case_state, authorities, base_qa_answer, language=language, mode=mode)

        # 3. Call LLM to generate natural answer
        if self.is_configured:
            try:
                answer = await self._call_llm(context, mode=mode)
            except Exception as exc:
                logger.warning(
                    "LLM answer synthesis failed (%s); falling back to template.",
                    exc,
                )
                answer = self._fallback_template(case_state, authorities, assessment, base_qa_answer, mode=mode)
        else:
            logger.info("No OpenAI client configured for answer generator; using template fallback.")
            answer = self._fallback_template(case_state, authorities, assessment, base_qa_answer, mode=mode)

        return answer, assessment

    # ── LLM Call ──────────────────────────────────────────────────

    async def _call_llm(self, context_message: str, mode: Any = None) -> str:
        """Make the actual OpenAI chat completion call."""
        is_informative = str(mode).lower() in ("informative", "mode.informative")
        sys_prompt = INFORMATIVE_SYSTEM_PROMPT if is_informative else ANSWER_SYSTEM_PROMPT
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": sys_prompt},
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
        language: str | None = None,
        mode: Any = None,
    ) -> str:
        """
        Build a structured context message that gives the LLM everything
        it needs to produce a grounded answer.
        """
        parts: list[str] = []
        is_informative = str(mode).lower() in ("informative", "mode.informative")

        # Target language
        lang = language or getattr(state, "language", None) or "en"
        from services.language import get_language_name, is_indic_language
        lang_name = get_language_name(lang)
        if is_indic_language(lang):
            parts.append(f"## TARGET CLIENT LANGUAGE: {lang_name} ({lang})")
            parts.append(
                f"IMPORTANT: The client communicates in {lang_name}. Synthesize the entire consultation "
                f"guidance and action plan in clear, natural {lang_name}. Keep official statutory and act names "
                f"bilingual (e.g. native script followed by official English Act name in parentheses)."
            )

        if is_informative:
            parts.append(
                "## INFORMATIVE LEGAL CONSULTATION:\n"
                "The user is asking an informative/conceptual question. Explain the applicable Indian law, "
                "statutory rights, non-discrimination protections, restrictions, and authorities objectively and clearly. "
                "Do NOT construct employer demand notices, wage slips, or litigation dispute steps unless explicitly asked."
            )

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

        # Base QA model answer (if available) - ground truth
        if base_qa_answer:
            parts.append(
                f"\n## BASE LEGAL RESEARCH ANSWER (GROUND TRUTH):\n{base_qa_answer}\n"
                f"IMPORTANT: Ensure your legal guidance faithfully conveys the substance of this answer, "
                f"adapting it clearly into the target client language without introducing unrelated statutory claims."
            )

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
            "Be concise, direct, and authoritative. Only cite statutes that ACTUALLY "
            "apply to the established facts. Skip irrelevant authorities."
        )

        return "\n".join(parts)

    # ── Structured Assessment Builder ─────────────────────────────

    def _build_assessment(
        self,
        state: UniversalCaseState,
        authorities: list[RetrievedAuthority],
        mode: Any = None,
    ) -> LegalAssessment:
        """Build the structured LegalAssessment for API metadata."""
        is_informative = str(mode).lower() in ("informative", "mode.informative")
        domain = state.case_domain or state.primary_category or ""
        if not domain or domain.lower() in ("general", "general legal dispute"):
            for iss in state.issues:
                if iss.domain and iss.domain.lower() not in ("general", "general legal dispute"):
                    domain = iss.domain
                    break
        if not domain or domain.lower() in ("general", "general legal dispute"):
            facts_text = " ".join([f.get("fact", "") if isinstance(f, dict) else str(f) for f in state.known_facts]).lower()
            issue_text = " ".join([i.issue for i in state.issues] + [state.case_type or "", state.summary or "", facts_text]).lower()
            if any(w in issue_text for w in ["fired", "terminat", "salary", "wage", "employ"]):
                domain = "employment"
            elif any(w in issue_text for w in ["rent", "evict", "tenant", "landlord", "flat", "property"]):
                domain = "property"
            elif any(w in issue_text for w in ["consumer", "defect", "warranty", "refund"]):
                domain = "consumer"
            elif any(w in issue_text for w in ["mental", "psychiatr", "health"]):
                domain = "Healthcare & Mental Health"
            elif any(w in issue_text for w in ["disabil"]):
                domain = "Disability Rights"
            else:
                domain = "General Legal Dispute"

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
        evidence_assessment = self._build_evidence_assessment(state, domain, is_govt, is_informative=is_informative)

        # Action plan
        action_plan = self._build_action_plan(state, domain, is_govt, is_informative=is_informative)

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
        is_informative: bool = False,
    ) -> dict[str, Any]:
        """
        Deep evidentiary analysis:
        1. Confirmed documents provided by client
        2. Required documents checklist for court/tribunal filing
        3. Evidentiary fallback strategy if documents are missing or withheld
        """
        d = domain.lower()
        domain_def = domain_registry.get(domain)

        if is_informative:
            required_docs = [
                "Relevant identity documents or disability/medical certificates (if applicable)",
                "Correspondence or records of any decision or service denial",
            ]
            fallback_strategy = [
                "Maintain copies of medical evaluations, official correspondence, and written representations.",
            ]
        elif is_govt and ("employment" in d or "salary" in d):
            required_docs = [
                "Appointment Letter / Service Confirmation Order",
                "Official Termination / Dismissal Order or Show Cause Notice",
                "Departmental Inquiry Officer's Report and Findings (if inquiry was conducted)",
                "Last 3-6 months' Salary Slips / Bank Account Statement showing non-payment of pending dues",
                "Copies of statutory representations or appeals filed before the Departmental Appellate Authority",
            ]
            fallback_strategy = [
                "If official termination order or inquiry findings are withheld: File an urgent application under Section 6 of the Right to Information (RTI) Act, 2005 requesting certified copies of the termination order, inquiry report, and file notings.",
                "If appointment order is missing: Use Bank Account salary credit statements, GPF/NPS statement, Service Book extract, and Employee ID Card as secondary proof of government employment.",
                "If unpaid dues calculation is disputed: Submit a requisition under RTI for the LPC (Last Pay Certificate) and due-drawn statement from the DDO (Drawing and Disbursing Officer).",
            ]
        elif "healthcare" in d or "mental" in d or "disabil" in d:
            required_docs = [
                "Medical evaluation / diagnosis documentation by a registered medical practitioner",
                "Official communications or notifications regarding denial of services or accommodation",
            ]
            fallback_strategy = [
                "Request written reasons for any adverse administrative action under the Rights of Persons with Disabilities Act, 2016.",
            ]
        elif domain_def and domain_def.required_documents:
            required_docs = list(domain_def.required_documents)
            fallback_strategy = list(domain_def.evidentiary_fallback_strategy)
        else:
            required_docs = [
                "Written agreement, purchase order, or correspondence establishing legal relationship",
                "Bank statements / financial receipts establishing monetary transaction / loss",
                "Formal statutory legal notice demanding performance / remedy",
            ]
            fallback_strategy = [
                "If agreement is oral: Compile bank records, WhatsApp admissions, and email trails to substantiate oral contract under Section 10 of the Indian Contract Act, 1872.",
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
        is_informative: bool = False,
    ) -> list[str]:
        """Build a prioritised action plan based on domain and government status."""
        if is_informative:
            return [
                "1. Review your statutory rights and protections under the applicable Indian legislation.",
                "2. If rights are infringed, approach the designated statutory authority or regulatory commission.",
                "3. Consult a specialized advocate or legal aid authority for individual case representation.",
            ]

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
        elif "healthcare" in d or "mental" in d or "disabil" in d:
            plan.append(f"{step}. Review rights and non-discrimination guarantees under the Mental Healthcare Act, 2017 / RPwD Act, 2016.")
            step += 1
            plan.append(f"{step}. If facing discrimination or denial of care, approach the State Mental Health Authority or State Commissioner for Persons with Disabilities.")
            step += 1
            plan.append(f"{step}. Seek assistance from the District Legal Services Authority (DLSA) for free legal aid and representation.")
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
        case_state: UniversalCaseState | None = None,
        authorities: list[RetrievedAuthority] | None = None,
        assessment: LegalAssessment | None = None,
        base_qa_answer: str | None = None,
        mode: Any = None,
        *,
        state: UniversalCaseState | None = None,
    ) -> str:
        """
        Produce a reasonable text answer without an LLM call.
        Used when OpenAI is not configured or the call fails.
        """
        cs = case_state or state or UniversalCaseState()
        auths = authorities or []
        assess = assessment or LegalAssessment()

        is_informative = str(mode).lower() in ("informative", "mode.informative")
        if is_informative and base_qa_answer:
            return base_qa_answer

        parts: list[str] = []

        # Understanding
        parts.append("### Understanding Your Situation")
        if cs.summary:
            parts.append(cs.summary)
        elif base_qa_answer:
            parts.append(base_qa_answer)

        # Legal position
        statutory = [a for a in auths if a.authority_type != "precedent"]
        if statutory:
            parts.append("\n### Your Legal Position")
            for auth in statutory[:3]:  # Cap at 3 most relevant
                parts.append(
                    f"- **{auth.source} ({auth.provision})**: {auth.relevance}"
                )

        # Actions
        if assess.action_plan:
            parts.append("\n### What You Should Do Now")
            for step in assess.action_plan:
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
