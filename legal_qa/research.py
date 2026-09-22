"""
Legal Research and Retrieval Layer.

Separates conversation handling from legal authority retrieval and research.
Prioritizes authoritative Indian statutes, current provisions, and relevant rules.
Treats past cases/precedents as persuasive context ONLY, never as user facts.
Enforces strict domain isolation so statutes from different legal domains
(e.g., Rent Control vs Employment) are never cross-polluted.
"""

from __future__ import annotations

import logging
from typing import Any
from conversation.cases.models import RetrievedAuthority, UniversalCaseState

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Authoritative Statutory Catalog for Indian Law
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STATUTORY_CATALOG: list[dict[str, Any]] = [
    # ── Employment & Wages ────────────────────────────────────
    {
        "source": "Payment of Wages Act, 1936",
        "authority_type": "statute",
        "domain": "employment",
        "jurisdiction": "Central / India",
        "provision": "Section 5 & Section 15",
        "status": "current",
        "relevance": "Mandates timely payment of wages and provides a summary procedure before the Authority under the Payment of Wages Act for recovery of delayed or deducted wages with up to 10x compensation.",
        "applicability_conditions": ["employee whose wages are unpaid or delayed", "wages within statutory threshold"],
        "key_excerpt": "Section 5 mandates that the wages of every person employed upon or in any railway, factory or industrial or other establishment shall be paid before the expiry of the seventh or tenth day after the last day of the wage-period.",
    },
    {
        "source": "Industrial Disputes Act, 1947",
        "authority_type": "statute",
        "domain": "employment",
        "jurisdiction": "Central / India",
        "provision": "Section 25F",
        "status": "current",
        "relevance": "Conditions precedent to retrenchment/termination of workmen who have completed one year of continuous service: mandatory 1 month notice or wages in lieu thereof, and 15 days compensation for every completed year of service.",
        "applicability_conditions": ["workman as defined under Section 2(s)", "continuous service of 240 days"],
        "key_excerpt": "No workman employed in any industry who has been in continuous service for not less than one year under an employer shall be retrenched by that employer until the workman has been given one month's notice in writing or has been paid wages in lieu of notice.",
    },
    {
        "source": "Maharashtra Shops and Establishments (Regulation of Employment and Conditions of Service) Act, 2017",
        "authority_type": "statute",
        "domain": "employment",
        "jurisdiction": "Maharashtra",
        "provision": "Section 13 & Section 14",
        "status": "current",
        "relevance": "Governs service conditions, discharge/dismissal, and mandatory notice period or wages in lieu thereof for employees in commercial establishments in Maharashtra.",
        "applicability_conditions": ["employed in Maharashtra", "commercial establishment / IT / private firm", "completed continuous service"],
        "key_excerpt": "No employee who has been in continuous employment shall be discharged or dismissed without at least thirty days' notice in writing or wages in lieu of notice, except for misconduct.",
    },
    {
        "source": "Karnataka Shops and Commercial Establishments Act, 1961",
        "authority_type": "statute",
        "domain": "employment",
        "jurisdiction": "Karnataka",
        "provision": "Section 39",
        "status": "current",
        "relevance": "Strictly prohibits termination of an employee who has been in employment for six months or more without at least one month's written notice or wages in lieu, and without reasonable cause. Gives right of appeal to the jurisdictional Labour Officer.",
        "applicability_conditions": ["employed in Karnataka", "commercial establishment / IT / startup / shop", "completed 6 months service"],
        "key_excerpt": "No employer shall dispense with the services of an employee employed continuously for a period of not less than six months, except for a reasonable cause and without giving such employee at least one month's notice or wages in lieu of such notice.",
    },
    {
        "source": "Delhi Shops and Establishments Act, 1954",
        "authority_type": "statute",
        "domain": "employment",
        "jurisdiction": "Delhi",
        "provision": "Section 30",
        "status": "current",
        "relevance": "Notice of termination of employment: no employer shall dispense with the services of an employee who has been in continuous employment for not less than three months without giving at least one month's notice in writing or wages in lieu thereof.",
        "applicability_conditions": ["employed in Delhi", "commercial establishment / shop / private firm", "completed 3 months service"],
        "key_excerpt": "No employer shall dispense with the services of an employee who has been in his continuous employment for not less than three months, without giving such person at least one month's notice in writing or wages in lieu thereof.",
    },
    {
        "source": "Payment of Gratuity Act, 1972",
        "authority_type": "statute",
        "domain": "employment",
        "jurisdiction": "Central / India",
        "provision": "Section 4",
        "status": "current",
        "relevance": "Mandatory gratuity payment to employees who have rendered continuous service for at least 5 years upon termination, resignation, retirement, or death.",
        "applicability_conditions": ["completed 5 years continuous service", "establishment having 10 or more employees"],
        "key_excerpt": "Gratuity shall be payable to an employee on the termination of his employment after he has rendered continuous service for not less than five years.",
    },
    {
        "source": "Administrative Tribunals Act, 1985",
        "authority_type": "statute",
        "domain": "employment",
        "jurisdiction": "Central / India",
        "provision": "Section 14 & Section 19",
        "status": "current",
        "relevance": "Exclusive jurisdiction of Central Administrative Tribunal (CAT) and State Administrative Tribunals (SAT) over service matters, wrongful dismissal, and unpaid dues of persons appointed to public services and civil posts.",
        "applicability_conditions": ["government or public service employee", "civil post under Union or State", "exhaustion of departmental remedies"],
        "key_excerpt": "The Central Administrative Tribunal shall exercise all the jurisdiction, powers and authority exercisable by all courts in relation to recruitment and conditions of service of persons appointed to civil services.",
    },
    {
        "source": "Constitution of India",
        "authority_type": "statute",
        "domain": "employment",
        "jurisdiction": "Central / India",
        "provision": "Article 311 & Article 226",
        "status": "current",
        "relevance": "Constitutional protection for government servants: no dismissal, removal, or reduction in rank without a formal departmental inquiry giving reasonable opportunity of being heard; enforceable via writ petition before the High Court.",
        "applicability_conditions": ["member of civil service of Union or State", "dismissal without inquiry or violation of natural justice"],
        "key_excerpt": "No person who is a member of a civil service of the Union or an all-India service or a civil service of a State shall be dismissed or removed by an authority subordinate to that by which he was appointed, and only after an inquiry.",
    },

    # ── Consumer Protection ───────────────────────────────────
    {
        "source": "Consumer Protection Act, 2019",
        "authority_type": "statute",
        "domain": "consumer",
        "jurisdiction": "Central / India",
        "provision": "Section 2(7) & Section 35",
        "status": "current",
        "relevance": "Defines 'consumer' and establishes right to file a consumer complaint before the District Consumer Disputes Redressal Commission for defective goods, deficiency in service, or unfair trade practice.",
        "applicability_conditions": ["buyer of goods or services for consideration", "not for resale or commercial purpose"],
        "key_excerpt": "A consumer can file a complaint against a trader, manufacturer, or service provider for unfair contract, unfair trade practice, defect in goods, or deficiency in service.",
    },
    {
        "source": "Consumer Protection Act, 2019",
        "authority_type": "statute",
        "domain": "consumer",
        "jurisdiction": "Central / India",
        "provision": "Section 69",
        "status": "current",
        "relevance": "Statutory limitation period: complaint must be filed within 2 years from the date on which the cause of action has arisen.",
        "applicability_conditions": ["cause of action arose within 2 years", "or sufficient cause shown for delay"],
        "key_excerpt": "The District Commission, the State Commission or the National Commission shall not admit a complaint unless it is filed within two years from the date on which the cause of action has arisen.",
    },
    {
        "source": "Consumer Protection (E-Commerce) Rules, 2020",
        "authority_type": "rule",
        "domain": "consumer",
        "jurisdiction": "Central / India",
        "provision": "Rule 5 & Rule 6",
        "status": "current",
        "relevance": "Obligations of e-commerce marketplace entities and sellers regarding refund timelines, warranty enforcement, and nodal grievance officer appointment.",
        "applicability_conditions": ["transaction on e-commerce portal (Amazon, Flipkart, etc.)"],
        "key_excerpt": "Every marketplace e-commerce entity shall ensure that the grievance officer acknowledges the receipt of any consumer complaint within forty-eight hours and redresses the complaint within one month.",
    },

    # ── Property & Tenancy ────────────────────────────────────
    {
        "source": "Specific Relief Act, 1963",
        "authority_type": "statute",
        "domain": "property",
        "jurisdiction": "Central / India",
        "provision": "Section 6",
        "status": "current",
        "relevance": "Provides a summary suit for immediate recovery of possession by any person dispossessed of immovable property without their consent and without due course of law, regardless of title. Must be brought within 6 months.",
        "applicability_conditions": ["physical dispossession occurred without due process", "filed within 6 months of dispossession"],
        "key_excerpt": "If any person is dispossessed without his consent of immovable property otherwise than in due course of law, he or any person claiming through him may, by suit, recover possession thereof.",
    },
    {
        "source": "Transfer of Property Act, 1882",
        "authority_type": "statute",
        "domain": "property",
        "jurisdiction": "Central / India",
        "provision": "Section 108 & Section 106",
        "status": "current",
        "relevance": "Defines lessor's obligations: quiet enjoyment without unlawful interference, and requirement of formal 15 days notice for terminating monthly tenancy.",
        "applicability_conditions": ["lease of immovable property", "absence of contrary contract"],
        "key_excerpt": "The lessor is bound to disclose latent material defects and is deemed to contract with the lessee that, if the latter pays the rent, he may hold the property during the time specified in the lease without interruption.",
    },
    {
        "source": "Maharashtra Rent Control Act, 1999",
        "authority_type": "statute",
        "domain": "property",
        "jurisdiction": "Maharashtra",
        "provision": "Section 24 & Section 29",
        "status": "current",
        "relevance": "Competent Authority jurisdiction for licensee recovery; Section 29 strictly prohibits cut off or withholding of essential supply or service (electricity/water) without just cause under penalty.",
        "applicability_conditions": ["property situated in Maharashtra", "tenancy or leave and license"],
        "key_excerpt": "No landlord either himself or through any person acting or purporting to act on his behalf shall, without just or sufficient cause, cut off or withhold any essential supply or service enjoyed by the tenant.",
    },

    # ── Criminal Law (BNS & BNSS 2023) ────────────────────────
    {
        "source": "Bharatiya Nyaya Sanhita, 2023",
        "authority_type": "statute",
        "domain": "criminal",
        "jurisdiction": "Central / India",
        "provision": "Section 115 (Voluntarily Causing Hurt) & Section 351 (Criminal Intimidation)",
        "status": "current",
        "relevance": "Substantive penal provisions for physical assault, bodily harm, and threats of injury to person, reputation, or property.",
        "applicability_conditions": ["physical assault or threat of injury"],
        "key_excerpt": "Whoever voluntarily causes hurt shall be punished with imprisonment or fine. Whoever threatens another with injury to his person, reputation or property commits criminal intimidation.",
    },
    {
        "source": "Bharatiya Nagarik Suraksha Sanhita, 2023",
        "authority_type": "statute",
        "domain": "criminal",
        "jurisdiction": "Central / India",
        "provision": "Section 173 (Information in Cognizable Cases - FIR)",
        "status": "current",
        "relevance": "Mandatory registration of First Information Report (FIR) by police upon receiving information disclosing commission of a cognizable offense, including electronic FIR (e-FIR).",
        "applicability_conditions": ["disclosure of cognizable offense to police officer"],
        "key_excerpt": "Every information relating to the commission of a cognizable offence, if given orally to an officer in charge of a police station, shall be reduced to writing and registered as an FIR.",
    },
    {
        "source": "Bharatiya Nagarik Suraksha Sanhita, 2023",
        "authority_type": "statute",
        "domain": "criminal",
        "jurisdiction": "Central / India",
        "provision": "Section 175(3) & Section 223",
        "status": "current",
        "relevance": "Procedure if police refuse to register FIR: complainant can send substance to Superintendent of Police, or file a private complaint before the Judicial Magistrate.",
        "applicability_conditions": ["refusal by local police to register FIR"],
        "key_excerpt": "Any person aggrieved by a refusal on the part of an officer in charge of a police station to record the information may send the substance of such information to the Superintendent of Police or approach the Magistrate.",
    },

    # ── Cybercrime & Financial Fraud ──────────────────────────
    {
        "source": "RBI Master Direction - Customer Protection – Limiting Liability in Unauthorized Electronic Banking Transactions, 2017",
        "authority_type": "rule",
        "domain": "cybercrime",
        "jurisdiction": "Central / India",
        "provision": "Paragraph 6 - Zero Liability of a Customer",
        "status": "current",
        "relevance": "A customer has ZERO liability for an unauthorized transaction where the deficiency lies in the banking system, or third-party breach where customer notifies bank within 3 working days.",
        "applicability_conditions": ["unauthorized electronic debit from bank/card/wallet", "reported within 3 working days"],
        "key_excerpt": "A customer's entitlement to zero liability shall arise where the unauthorized transaction occurs in the event of third party breach where customer notifies the bank within three working days of receiving the communication from the bank.",
    },
    {
        "source": "Information Technology Act, 2000",
        "authority_type": "statute",
        "domain": "cybercrime",
        "jurisdiction": "Central / India",
        "provision": "Section 43 & Section 66D",
        "status": "current",
        "relevance": "Penalties for unauthorized computer access, data extraction, and punishment for cheating by personation by using computer resources (imprisonment up to 3 years and fine).",
        "applicability_conditions": ["unauthorized access or cyber fraud via electronic device"],
        "key_excerpt": "Whoever, by means for any communication device or computer resource cheats by personating, shall be punished with imprisonment of either description for a term which may extend to three years and fine.",
    },

    # ── Family & Maintenance ──────────────────────────────────
    {
        "source": "Bharatiya Nagarik Suraksha Sanhita, 2023",
        "authority_type": "statute",
        "domain": "family",
        "jurisdiction": "Central / India",
        "provision": "Section 144 (formerly Section 125 CrPC)",
        "status": "current",
        "relevance": "Statutory right of wife, minor children, and dependent parents to claim monthly maintenance from a person having sufficient means who neglects or refuses to maintain them. Includes interim maintenance.",
        "applicability_conditions": ["neglect or refusal to maintain dependent spouse or children", "claimant unable to maintain themselves"],
        "key_excerpt": "If any person having sufficient means neglects or refuses to maintain his wife, unable to maintain herself, or his legitimate or illegitimate minor child, a Magistrate of the first class may order monthly maintenance.",
    },
    {
        "source": "Protection of Women from Domestic Violence Act, 2005",
        "authority_type": "statute",
        "domain": "family",
        "jurisdiction": "Central / India",
        "provision": "Section 12, Section 19 (Residence Order), Section 20 (Monetary Relief)",
        "status": "current",
        "relevance": "Empowers an aggrieved woman living in a shared household to seek immediate interim protection orders, residence orders restraining dispossession, and emergency monetary maintenance.",
        "applicability_conditions": ["domestic relationship in a shared household", "economic, emotional, or physical deprivation"],
        "key_excerpt": "The Magistrate may, on an application, pass an order directing the respondent to pay monetary relief to meet the expenses incurred and losses suffered by the aggrieved person and any child.",
    },

    # ── Contract & Money Recovery ─────────────────────────────
    {
        "source": "Indian Contract Act, 1872",
        "authority_type": "statute",
        "domain": "contract",
        "jurisdiction": "Central / India",
        "provision": "Section 73",
        "status": "current",
        "relevance": "Right to compensation for loss or damage caused by breach of contract: the party who suffers by breach is entitled to receive compensation for any loss naturally arising.",
        "applicability_conditions": ["existence of valid contract (written or oral)", "breach by opposing party causing loss"],
        "key_excerpt": "When a contract has been broken, the party who suffers by such breach is entitled to receive, from the party who has broken the contract, compensation for any loss or damage caused to him thereby.",
    },
    {
        "source": "Limitation Act, 1963",
        "authority_type": "statute",
        "domain": "contract",
        "jurisdiction": "Central / India",
        "provision": "Schedule, Article 18 & Article 55",
        "status": "current",
        "relevance": "General statutory limitation for money recovery and breach of contract is 3 years from the date the money was due or the date the contract was broken.",
        "applicability_conditions": ["civil suit for money recovery / damages"],
        "key_excerpt": "The period of limitation for a suit for the recovery of money or breach of any contract is three years from the time when the money became due or when the contract is broken.",
    },
    {
        "source": "Negotiable Instruments Act, 1881",
        "authority_type": "statute",
        "domain": "contract",
        "jurisdiction": "Central / India",
        "provision": "Section 138",
        "status": "current",
        "relevance": "Dishonour of cheque for insufficiency of funds is a criminal offense punishable with up to 2 years imprisonment or twice the cheque amount. Strict procedure: notice within 30 days of dishonour memo.",
        "applicability_conditions": ["cheque issued for discharge of legally enforceable debt", "bank memo showing dishonour", "statutory legal notice within 30 days"],
        "key_excerpt": "Where any cheque drawn by a person is returned by the bank unpaid, such person shall be deemed to have committed an offence and shall be punished with imprisonment or fine which may extend to twice the amount of the cheque.",
    },

    # ── Government, Administrative & RTI ──────────────────────
    {
        "source": "Constitution of India",
        "authority_type": "statute",
        "domain": "government",
        "jurisdiction": "Central / India",
        "provision": "Article 226",
        "status": "current",
        "relevance": "Power of High Courts to issue prerogative writs (Mandamus, Certiorari, Prohibition, Quo Warranto, Habeas Corpus) against arbitrary, unlawful, or unconstitutional actions of the State or government authorities.",
        "applicability_conditions": ["arbitrary government/civic action", "violation of fundamental rights or principles of natural justice"],
        "key_excerpt": "Every High Court shall have powers throughout the territories in relation to which it exercise jurisdiction to issue to any person or authority directions, orders or writs for the enforcement of rights or for any other purpose.",
    },
    {
        "source": "Right to Information Act, 2005",
        "authority_type": "statute",
        "domain": "government",
        "jurisdiction": "Central / India",
        "provision": "Section 6 & Section 19",
        "status": "current",
        "relevance": "Right of citizens to secure information under the control of public authorities within 30 days (or 48 hours where life or liberty is involved), with right of First and Second Appeal.",
        "applicability_conditions": ["information held by or under control of any public authority"],
        "key_excerpt": "A person, who desires to obtain any information under this Act, shall make a request in writing to the Public Information Officer. The PIO shall provide information within thirty days of receipt.",
    },
]


class LegalResearchLayer:
    """
    Research and retrieval engine that provides authoritative legal provisions
    and filters precedents appropriately.
    """

    def __init__(self, catalog: list[dict[str, Any]] | None = None) -> None:
        self._catalog = catalog or STATUTORY_CATALOG

    def research_authorities(
        self,
        case_state: UniversalCaseState,
        limit: int = 4,
    ) -> list[RetrievedAuthority]:
        """
        Identify applicable statutory provisions based on the CaseState.
        Takes into account jurisdiction, domain, active issues, and parties.
        Enforces strict domain isolation (e.g. Rent Control cannot appear in employment disputes).
        """
        raw_domain = (case_state.case_domain or case_state.primary_category or "").lower()
        state = (case_state.jurisdiction.state or "").lower()
        city = (case_state.jurisdiction.city or "").lower()
        active_issue_texts = [i.issue.lower() for i in case_state.issues if i.status != "ruled_out"]
        all_text = f"{raw_domain} {' '.join(active_issue_texts)} {state} {city} {case_state.user_goal or ''}".lower()

        # Canonical domain normalization
        target_domain = None
        if any(w in raw_domain for w in ["employ", "wage", "salary", "labor", "labour"]):
            target_domain = "employment"
        elif any(w in raw_domain for w in ["tenant", "evict", "landlord", "rent", "property"]):
            target_domain = "property"
        elif any(w in raw_domain for w in ["cyber", "scam", "fraud"]):
            target_domain = "cybercrime"
        elif "consumer" in raw_domain:
            target_domain = "consumer"
        elif any(w in raw_domain for w in ["crime", "crim"]):
            target_domain = "criminal"
        elif any(w in raw_domain for w in ["divorce", "matrimon", "custody", "family"]):
            target_domain = "family"
        elif any(w in raw_domain for w in ["contract", "debt", "loan", "money"]):
            target_domain = "contract"
        elif any(w in raw_domain for w in ["govt", "government", "admin", "rti"]):
            target_domain = "government"

        # Government employment check
        is_govt = False
        if isinstance(case_state.domain_extensions, dict):
            ext_emp = case_state.domain_extensions.get("employment", {})
            if isinstance(ext_emp, dict) and ext_emp.get("employment_type") == "government":
                is_govt = True
        if not is_govt:
            for f in case_state.known_facts:
                f_str = str(f).lower()
                if "government" in f_str or "civil servant" in f_str:
                    is_govt = True
                    break
        if not is_govt and isinstance(case_state.parties, list):
            for p in case_state.parties:
                p_str = str(p).lower()
                if "government" in p_str or "public service" in p_str:
                    is_govt = True
                    break

        scored: list[tuple[float, dict[str, Any]]] = []

        for item in self._catalog:
            item_domain = item.get("domain", "").lower()

            # STRICT DOMAIN ISOLATION:
            # An authority belonging to a different domain must NEVER be cited.
            # e.g., Rent Control Act must NEVER be cited in an Employment dispute.
            if target_domain and item_domain and item_domain != target_domain:
                continue

            score = 0.0
            item_source = item["source"].lower()
            item_jur = item["jurisdiction"].lower()
            item_relevance = item["relevance"].lower()
            item_excerpt = item["key_excerpt"].lower()

            # Employment subtype alignment: Government vs Private
            if target_domain == "employment":
                if is_govt:
                    if "administrative tribunals" in item_source or "constitution of india" in item_source:
                        score += 15.0
                    elif "industrial disputes" in item_source or "shops" in item_source:
                        score -= 25.0  # Civil servants under Union/State are governed by Service Rules & Article 311, not IDA/Shops
                else:
                    if "administrative tribunals" in item_source or "constitution of india" in item_source:
                        score -= 20.0  # Private employees cannot approach CAT or invoke Article 311

            # State-specific matching
            if item_jur != "central / india":
                if (state and item_jur in state) or (city and item_jur in city) or item_jur in all_text:
                    score += 10.0  # Perfect local statute match (e.g. Maharashtra Shops Act for Mumbai)
                elif state and item_jur not in state:
                    # Specific to another state (e.g. Karnataka law for Maharashtra dispute)
                    score -= 10.0
            else:
                score += 2.0  # Central statutes applicable pan-India

            # Domain affinity bonus
            if target_domain and item_domain == target_domain:
                score += 5.0

            # Issue keyword matching
            for issue in active_issue_texts:
                for word in issue.split():
                    if len(word) > 3:
                        if word in item_relevance or word in item_excerpt or word in item_source:
                            score += 2.0

            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[RetrievedAuthority] = []
        for _, item in scored[:limit]:
            results.append(
                RetrievedAuthority(
                    source=item["source"],
                    authority_type=item["authority_type"],
                    jurisdiction=item["jurisdiction"],
                    provision=item["provision"],
                    status=item["status"],
                    relevance=item["relevance"],
                    applicability_conditions=item.get("applicability_conditions", []),
                    key_excerpt=item.get("key_excerpt", ""),
                )
            )

        return results

    def process_retrieved_cases(
        self,
        cases: list[dict[str, str]],
        case_state: UniversalCaseState,
    ) -> list[RetrievedAuthority]:
        """
        Process past cases returned by Legal_QA into structured precedents.
        Enforces that precedents are persuasive context, NOT user facts.
        """
        precedents: list[RetrievedAuthority] = []
        current_state = (case_state.jurisdiction.state or "").lower()

        for c in cases:
            q = c.get("question", "")
            ans = c.get("answer", "")

            # Detect jurisdiction mentioned in precedent
            prec_jur = "India"
            for st in ["Delhi", "Karnataka", "Maharashtra", "Tamil Nadu", "Uttar Pradesh", "West Bengal", "Gujarat", "Telangana"]:
                if st.lower() in q.lower() or st.lower() in ans.lower():
                    prec_jur = st
                    break

            precedents.append(
                RetrievedAuthority(
                    source=f"Precedent / Legal_QA Case Archive ({prec_jur})",
                    authority_type="precedent",
                    jurisdiction=prec_jur,
                    provision="Similar Factual Query",
                    status="persuasive_context_only",
                    relevance="Past judicial / legal forum approach for similar scenario. Not binding facts about this client.",
                    applicability_conditions=["Applies by analogy only; factual differences must be distinguished."],
                    key_excerpt=f"Past Query: {q[:150]}... | Solution: {ans[:200]}...",
                )
            )

        return precedents


legal_research_layer = LegalResearchLayer()
