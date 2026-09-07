"""
Central case taxonomy and registry.

Provides a modular, extensible catalogue of legal case modules and fast,
zero-heavy-dependency routing.
"""

from __future__ import annotations

import re
from typing import Any
from conversation.cases.models import CaseModule


class CaseRegistry:
    """
    Central registry of legal case modules organized hierarchically.
    """

    def __init__(self) -> None:
        self._modules: dict[str, CaseModule] = {}
        self._register_default_modules()

    def register(self, module: CaseModule) -> None:
        """Register or override a case module in the taxonomy."""
        key = self._make_key(module.category, module.subcategory, module.case_type)
        self._modules[key] = module

    def get(self, category: str, subcategory: str, case_type: str) -> CaseModule | None:
        """Retrieve a specific case module."""
        key = self._make_key(category, subcategory, case_type)
        return self._modules.get(key)

    def get_by_name(self, name: str) -> CaseModule | None:
        """Find a module by case_type name or key substring."""
        name_lower = name.lower()
        for key, mod in self._modules.items():
            if mod.case_type.lower() == name_lower or key.endswith(f"/{name_lower}"):
                return mod
        return None

    def list_all(self) -> list[CaseModule]:
        """Return all registered case modules."""
        return list(self._modules.values())

    def find_candidate_modules(
        self,
        text: str,
        current_state: dict[str, Any] | None = None,
        limit: int = 3,
    ) -> list[CaseModule]:
        """
        Identify the most relevant case modules for a conversation turn.

        Scores candidates based on keyword matches, current classification,
        and related modules. Never loads the entire catalogue into prompts.
        """
        text_lower = text.lower()
        scored: list[tuple[float, CaseModule]] = []

        # Extract current state indicators if available
        curr_case_type = ""
        curr_subcat = ""
        curr_cat = ""
        related_types: list[str] = []

        if current_state:
            curr_case_type = (current_state.get("case_type") or "").lower()
            curr_subcat = (current_state.get("subcategory") or "").lower()
            curr_cat = (current_state.get("primary_category") or "").lower()
            related_types = [
                r.lower() for r in current_state.get("related_case_types", [])
            ]

        for mod in self._modules.values():
            if mod.case_type == "Unknown / Needs Further Classification":
                continue

            score = 0.0

            # 1. Direct keyword match scoring
            for kw in mod.keywords:
                # Whole-phrase or word-boundary match
                if kw in text_lower:
                    # Longer/multi-word keywords get higher weight
                    score += 2.0 if " " in kw else 1.0

            # 2. Affinity with current classification state
            if curr_case_type and mod.case_type.lower() == curr_case_type:
                score += 3.0
            elif curr_subcat and mod.subcategory.lower() == curr_subcat:
                score += 1.5
            elif curr_cat and mod.category.lower() == curr_cat:
                score += 0.5

            # 3. Affinity with marked related types
            if mod.case_type.lower() in related_types:
                score += 2.0

            if score > 0:
                scored.append((score, mod))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        candidates = [mod for _, mod in scored[:limit]]

        # If no specific module matched, fallback to default/general modules
        if not candidates:
            # If current classification has a module, use it
            if curr_case_type:
                existing = self.get_by_name(curr_case_type)
                if existing:
                    candidates.append(existing)

            # Otherwise return default civil/criminal catch-all
            if not candidates:
                fallback = self.get_by_name("Other Civil Dispute") or self.get_by_name("Unknown / Needs Further Classification")
                if fallback:
                    candidates.append(fallback)

        return candidates

    @staticmethod
    def _make_key(category: str, subcategory: str, case_type: str) -> str:
        return f"{category.strip().lower()}/{subcategory.strip().lower()}/{case_type.strip().lower()}"

    def _register_default_modules(self) -> None:
        """Populate initial comprehensive legal case hierarchy."""

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # CIVIL -> Property
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.register(
            CaseModule(
                category="Civil",
                subcategory="Property",
                case_type="Ownership Dispute",
                description="Dispute over legal title, deeds, or true ownership of land, flat, or immovable property.",
                keywords=["title", "deed", "sale deed", "owner", "ownership", "fake deed", "encroachment", "patta", "khata", "mutation", "land dispute"],
                relevant_facts=[
                    "Exact property type and location",
                    "How title was acquired (purchase, gift, inheritance)",
                    "Who holds registered sale deed or title documents",
                    "Current possession status",
                    "Basis of opposing party's claim to title",
                    "Any mutation or revenue records status",
                    "Desired outcome",
                ],
                priority_info=["Nature of title deed", "Who has physical possession", "State where property is situated"],
                potential_evidence=["Sale deed", "Revenue records / Patta / Khata", "Tax receipts", "Encumbrance certificate"],
                timeline_info=["Date of acquisition", "When opposing party challenged ownership"],
                financial_info=["Estimated property value"],
                parties_involved=["Claimant owner", "Opposing claimant", "Registrar/revenue authorities if relevant"],
                urgency_indicators=["Threat of third-party sale", "Illegal construction starting", "Dispossession threats"],
                desired_outcomes=["Declaration of title", "Injunction restraining sale or interference"],
                related_modules=["Possession Dispute", "Fraud", "Property Sale Dispute"],
            )
        )

        self.register(
            CaseModule(
                category="Civil",
                subcategory="Property",
                case_type="Possession Dispute",
                description="Dispute concerning physical possession, trespass, illegal dispossession, or injunction.",
                keywords=["possession", "dispossessed", "trespass", "encroached", "boundary", "vacate", "occupying", "illegal possession"],
                relevant_facts=[
                    "Who currently has physical possession",
                    "How long the current party has held possession",
                    "Whether dispossession was forcible or without due process",
                    "Utility bills or municipal tax under whose name",
                    "Police complaints or local panchayat/civic reports",
                ],
                priority_info=["Current physical possession", "Basis of possession", "Date of alleged dispossession"],
                potential_evidence=["Electricity/water bills", "Possession letter", "Police diary / complaint", "Photographs/video"],
                timeline_info=["Date of entering possession", "Date of threat/dispossession"],
                parties_involved=["Possessor", "Trespasser / Dispossessing party"],
                urgency_indicators=["Physical force being used", "Immediate eviction without court order"],
                desired_outcomes=["Restoration of possession", "Permanent or temporary injunction"],
                related_modules=["Ownership Dispute", "Landlord Tenant"],
            )
        )

        self.register(
            CaseModule(
                category="Civil",
                subcategory="Property",
                case_type="Partition",
                description="Division and separation of shares in ancestral or joint family property.",
                keywords=["partition", "ancestral property", "joint family", "share in property", "coparcener", "family settlement", "division of land"],
                relevant_facts=[
                    "Whether property is ancestral or self-acquired",
                    "Genealogy / family tree of coparceners or heirs",
                    "Whether a prior partition or family settlement occurred",
                    "Status of female heirs' shares",
                    "Whether any portion was alienated without consent",
                ],
                priority_info=["Source of property (ancestral vs self-acquired)", "Relationship to original owner", "Whether prior partition exists"],
                potential_evidence=["Genealogy certificate / family tree", "Original title deeds of ancestor", "Prior partition deed/khatiyan"],
                timeline_info=["Date of death of original owner", "When dispute over division arose"],
                parties_involved=["Co-sharers", "Legal heirs", "Karta/head of family"],
                urgency_indicators=["Attempt by one co-owner to sell entire property"],
                desired_outcomes=["Preliminary decree of partition", "Demarcation of physical share"],
                related_modules=["Inheritance", "Inheritance Related Family Matters"],
            )
        )

        self.register(
            CaseModule(
                category="Civil",
                subcategory="Property",
                case_type="Inheritance",
                description="Inheritance of property, testamentary vs intestate succession, and probate.",
                keywords=["inheritance", "will", "probate", "legal heir", "succession", "grandfather property", "father property", "bequeathed", "intestate"],
                relevant_facts=[
                    "Did deceased leave a valid Will, or was succession intestate",
                    "Religious personal law applicable (Hindu, Muslim, Christian, Indian Succession Act)",
                    "Surviving Class I / Class II legal heirs",
                    "Whether Will is registered or contested for forgery/coercion",
                    "Whether legal heir certificate or succession certificate was issued",
                ],
                priority_info=["Existence of a Will", "Relationship to deceased", "Surviving heirs"],
                potential_evidence=["Will document", "Death certificate", "Legal heir certificate", "Family tree"],
                timeline_info=["Date of demise", "Date of execution of Will"],
                parties_involved=["Heirs", "Executor of Will", "Contesting relatives"],
                urgency_indicators=["Other heirs actively alienating or transferring the estate"],
                desired_outcomes=["Probate/Letters of Administration", "Succession Certificate", "Mutation of name"],
                related_modules=["Partition", "Inheritance Related Family Matters", "Ownership Dispute"],
            )
        )

        self.register(
            CaseModule(
                category="Civil",
                subcategory="Property",
                case_type="Landlord Tenant",
                description="Tenancy issues including eviction, unpaid rent, security deposit refund, lease termination.",
                keywords=["tenant", "landlord", "rent", "lease", "eviction", "security deposit", "rent agreement", "rent control", "subletting"],
                relevant_facts=[
                    "Existence of a written rent agreement/lease deed and its term",
                    "Monthly rent and security deposit amount",
                    "Ground of dispute (arrears of rent, personal necessity, expiry of lease, damage)",
                    "Whether statutory legal notice to vacate was served",
                    "State and applicable Rent Control Act or Model Tenancy Act",
                ],
                priority_info=["Written lease agreement status", "Rent amount & arrears", "Notice served status", "State/City"],
                potential_evidence=["Rent agreement", "Rent payment bank transfers / receipts", "Eviction notice", "WhatsApp/emails regarding rent"],
                timeline_info=["Start and end date of lease", "Months of default"],
                financial_info=["Monthly rent", "Security deposit amount", "Arrears amount"],
                parties_involved=["Landlord", "Tenant", "Guarantor"],
                urgency_indicators=["Lock changing / illegal eviction attempts", "Disconnecting electricity/water"],
                desired_outcomes=["Eviction order", "Recovery of arrears", "Deposit refund", "Protection against illegal eviction"],
                related_modules=["Breach of Contract", "Possession Dispute"],
            )
        )

        self.register(
            CaseModule(
                category="Civil",
                subcategory="Property",
                case_type="Property Sale Dispute",
                description="Disputes arising from agreement to sell property, builder-buyer delays, or RERA disputes.",
                keywords=["builder", "rera", "agreement to sell", "booking amount", "possession delayed", "allotment", "builder buyer agreement", "flat possession"],
                relevant_facts=[
                    "Agreement to sell or builder-buyer agreement signed",
                    "Total consideration, amount paid, and balance due",
                    "Promised delivery/possession date vs actual status",
                    "RERA registration status of project",
                    "Whether refund with interest or possession is demanded",
                ],
                priority_info=["Total paid vs agreed price", "Agreed possession date", "RERA registration"],
                potential_evidence=["Builder-buyer agreement", "Payment receipts", "Demand letters", "Completion certificate / OC"],
                timeline_info=["Date of booking", "Due date of possession"],
                financial_info=["Total price", "Paid amount", "Interest claimed"],
                parties_involved=["Buyer / Allottee", "Builder / Developer / Seller"],
                urgency_indicators=["Insolvency / NCLT notice against builder", "Project abandoned"],
                desired_outcomes=["RERA complaint", "Consumer forum complaint", "Refund with interest or specific performance"],
                related_modules=["Breach of Contract", "Consumer Dispute"],
            )
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # CIVIL -> Contract
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.register(
            CaseModule(
                category="Civil",
                subcategory="Contract",
                case_type="Breach of Contract",
                description="Failure to perform obligations under a valid agreement, causing damages or loss.",
                keywords=["contract", "breach", "agreement", "violation of contract", "terms violated", "defaulted on agreement", "mou", "non-performance"],
                relevant_facts=[
                    "Whether written, oral, or implied contract exists",
                    "Specific terms or clauses allegedly breached",
                    "Whether notice of breach or cure period was provided",
                    "Damages or financial loss suffered",
                    "Dispute resolution / arbitration clause in contract",
                ],
                priority_info=["Existence of written agreement", "Specific breach alleged", "Quantifiable financial loss"],
                potential_evidence=["Signed contract / MOU", "Email communications", "Invoices", "Termination notice"],
                timeline_info=["Date contract executed", "Date breach occurred"],
                financial_info=["Contract value", "Damages/loss claimed"],
                parties_involved=["Contracting parties", "Guarantors"],
                urgency_indicators=["Limitation period nearing expiry (3 years)", "Bank guarantee invocation"],
                desired_outcomes=["Specific performance", "Damages / compensation", "Injunction"],
                related_modules=["Money Recovery", "Service Agreement", "Sale Agreement"],
            )
        )

        self.register(
            CaseModule(
                category="Civil",
                subcategory="Contract",
                case_type="Service Agreement",
                description="Disputes over delivery of professional, commercial, or IT services.",
                keywords=["service agreement", "vendor", "client dispute", "milestone", "scope of work", "deliverables", "service contract", "freelance"],
                relevant_facts=[
                    "Scope of work and milestones agreed",
                    "Services delivered vs accepted",
                    "Outstanding invoices and payment schedule",
                    "Dispute over quality vs outright non-delivery",
                ],
                priority_info=["Scope of work agreement", "Pending payment / invoices", "Proof of delivery/milestones"],
                potential_evidence=["Master services agreement / SOW", "Timesheets", "Emails approving milestones", "Invoices"],
                timeline_info=["Milestone deadlines", "Payment due dates"],
                financial_info=["Invoice amounts outstanding"],
                parties_involved=["Service provider / Vendor", "Client"],
                urgency_indicators=["Withholding critical data or access credentials"],
                desired_outcomes=["Recovery of service dues", "Termination of agreement"],
                related_modules=["Breach of Contract", "Money Recovery"],
            )
        )

        self.register(
            CaseModule(
                category="Civil",
                subcategory="Contract",
                case_type="Sale Agreement",
                description="Disputes over contracts for sale of goods, commercial transactions, or supply agreements.",
                keywords=["sale of goods", "supply contract", "delivery of goods", "purchase order", "invoice dispute", "supplier dispute"],
                relevant_facts=[
                    "Purchase order and acceptance terms",
                    "Delivery status of goods and inspection/acceptance",
                    "Payment terms and credit period",
                    "Whether defective goods were rejected within reasonable time",
                ],
                priority_info=["Purchase order / Invoice", "Delivery challan / proof of dispatch", "Amount due"],
                potential_evidence=["Purchase order", "Invoices", "Lorry receipt / transport bill", "Debit/credit notes"],
                timeline_info=["Date of dispatch", "Credit period expiry"],
                financial_info=["Total invoice value", "Balance unpaid"],
                parties_involved=["Buyer", "Seller / Supplier"],
                urgency_indicators=["Commercial court limitation window"],
                desired_outcomes=["Summary suit under Order 37 CPC", "MSME Samadhaan complaint"],
                related_modules=["Breach of Contract", "Money Recovery"],
            )
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # CIVIL -> Money Recovery
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.register(
            CaseModule(
                category="Civil",
                subcategory="Money Recovery",
                case_type="Money Recovery",
                description="Recovery of friendly loans, personal debts, advances, or unpaid commercial debts.",
                keywords=["lent money", "loan", "borrowed", "recover money", "return money", "friend money", "advance paid", "cheque bounce", "debt", "promissory note"],
                relevant_facts=[
                    "Who gave the money and who received it",
                    "Exact amount and date(s) of payment",
                    "Nature of transaction (friendly loan, investment, advance, gift)",
                    "Whether repayment was agreed and specific deadline/terms",
                    "Method of transfer (UPI, bank transfer, cash, cheque)",
                    "Whether debtor acknowledges debt (written messages, partial payment)",
                    "Previous attempts or legal notice sent",
                ],
                priority_info=["Nature of transaction (loan vs gift)", "Repayment agreement & deadline", "Evidence of transfer & acknowledgment", "Current debtor response"],
                potential_evidence=["Bank statement / UPI transaction screenshot", "WhatsApp / SMS / email acknowledgment", "Promissory note / loan agreement", "Bounced cheque"],
                timeline_info=["Date loan given", "Agreed repayment deadline", "Last date of acknowledgment (for limitation)"],
                financial_info=["Principal amount", "Interest agreed", "Amount repaid if any"],
                parties_involved=["Creditor / Lender", "Debtor / Borrower"],
                urgency_indicators=["3-year limitation period expiring soon", "Debtor disposing of assets"],
                desired_outcomes=["Legal notice under Section 138 NI Act (if cheque)", "Summary suit Order 37 CPC", "Money suit"],
                related_modules=["Breach of Contract", "Fraud"],
            )
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # CIVIL -> Consumer Dispute
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.register(
            CaseModule(
                category="Civil",
                subcategory="Consumer",
                case_type="Consumer Dispute",
                description="Deficiency in goods or services, unfair trade practices, misleading advertisements, warranty disputes.",
                keywords=["defective product", "consumer complaint", "refund", "warranty", "consumer forum", "e-commerce dispute", "amazon", "flipkart", "deficiency of service"],
                relevant_facts=[
                    "Product or service purchased and date",
                    "Price paid and receipt/invoice details",
                    "Specific defect or deficiency experienced",
                    "Complaint raised with customer care / company response",
                    "Remedy sought (replacement, full refund, compensation for mental harassment)",
                ],
                priority_info=["Purchase invoice & payment proof", "Nature of defect", "Company's refusal/delay to rectify"],
                potential_evidence=["Tax invoice", "Warranty card", "Customer support emails/tickets", "Photographs/video of defect"],
                timeline_info=["Purchase date", "Complaint date (2-year consumer court limitation)"],
                financial_info=["Product price", "Compensation claimed"],
                parties_involved=["Consumer", "Manufacturer / Seller / Platform"],
                urgency_indicators=["Warranty about to expire"],
                desired_outcomes=["Consumer commission complaint (e-Daakhil)", "National Consumer Helpline complaint"],
                related_modules=["Breach of Contract", "Sale Agreement"],
            )
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # CIVIL -> Business Dispute
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.register(
            CaseModule(
                category="Civil",
                subcategory="Business",
                case_type="Business Dispute",
                description="Partnership disputes, shareholder conflicts, director deadlocks, joint venture disputes.",
                keywords=["partnership", "partner dispute", "shareholder", "director", "startup dispute", "joint venture", "llp dispute", "profit sharing"],
                relevant_facts=[
                    "Business structure (Partnership firm, LLP, Private Limited)",
                    "Partnership deed or Shareholders' Agreement (SHA) terms",
                    "Profit sharing ratio and capital contribution",
                    "Nature of conflict (siphoning funds, exclusion from management, non-payment of profits)",
                    "Arbitration clause existence",
                ],
                priority_info=["Entity type & incorporation status", "Written agreement terms", "Financial irregularities alleged"],
                potential_evidence=["Partnership deed / SHA", "Bank account statements", "Audited accounts", "Board resolutions"],
                timeline_info=["Date of entering partnership", "Onset of dispute"],
                financial_info=["Capital invested", "Alleged siphoned / owed profits"],
                parties_involved=["Partners / Co-founders / Shareholders"],
                urgency_indicators=["Partner draining company bank accounts", "Siphoning clients/IP"],
                desired_outcomes=["Dissolution of firm", "NCLT oppression/mismanagement petition", "Arbitration"],
                related_modules=["Breach of Contract", "Fraud"],
            )
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # CIVIL -> Employment Dispute
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.register(
            CaseModule(
                category="Civil",
                subcategory="Employment",
                case_type="Employment Dispute",
                description="Unpaid salary, wrongful termination, gratuity, PF default, notice period dispute, non-compete.",
                keywords=["salary", "unpaid wage", "fired", "terminated", "layoff", "provident fund", "gratuity", "relieving letter", "notice pay", "workplace dispute"],
                relevant_facts=[
                    "Nature of employer (private, public, startup, contractor)",
                    "Designation and nature of duties (workman vs managerial)",
                    "Written employment contract or appointment letter",
                    "Duration of unpaid salary or terms of termination",
                    "State where employed",
                ],
                priority_info=["Employment type (private/govt)", "State of employment", "Unpaid duration / termination grounds"],
                potential_evidence=["Appointment letter / contract", "Salary slips / bank credits", "Termination email", "Relieving request emails"],
                timeline_info=["Date of joining", "Date of termination", "Months salary withheld"],
                financial_info=["Monthly salary", "Total arrears claimed", "Gratuity amount"],
                parties_involved=["Employee", "Employer / HR / Directors"],
                urgency_indicators=["Relieving letter withheld preventing new job"],
                desired_outcomes=["Labour commissioner complaint", "Payment of Wages Act claim", "Legal notice"],
                related_modules=["Breach of Contract", "Money Recovery"],
            )
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # CIVIL -> Defamation
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.register(
            CaseModule(
                category="Civil",
                subcategory="Defamation",
                case_type="Defamation",
                description="Civil and/or criminal defamation for false and damaging statements made publicly.",
                keywords=["defamation", "slander", "libel", "defamed", "reputation damaged", "false accusation online", "whatsapp defaming", "character assassination"],
                relevant_facts=[
                    "Exact defamatory statement made and medium (social media, print, verbal)",
                    "Whether publication was made to third parties",
                    "Whether statement is factually untrue",
                    "Reputational and financial harm caused",
                ],
                priority_info=["Exact statement and medium", "Third-party publication proof", "Harm caused"],
                potential_evidence=["Screenshots of posts/messages", "Witnesses who heard/read statement", "Proof of falsity"],
                timeline_info=["Date of publication"],
                parties_involved=["Complainant", "Defamer"],
                urgency_indicators=["Ongoing viral social media campaign"],
                desired_outcomes=["Legal cease-and-desist notice", "Injunction to take down content", "Civil damages suit", "Section 499/500 IPC complaint"],
                related_modules=["Cybercrime", "Other Criminal Matter"],
            )
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # CIVIL -> Intellectual Property
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.register(
            CaseModule(
                category="Civil",
                subcategory="Intellectual Property",
                case_type="Intellectual Property",
                description="Infringement of trademark, copyright, patent, trade secrets, or brand passing off.",
                keywords=["trademark", "copyright", "patent", "piracy", "brand name copied", "counterfeit", "ip infringement", "plagiarism"],
                relevant_facts=[
                    "Type of IP (Trademark, Copyright, Patent)",
                    "Registration status and certificate number",
                    "Date of prior use",
                    "Nature of infringement or passing off by opposing party",
                ],
                priority_info=["Registration certificate", "Prior user proof", "Infringing product/website"],
                potential_evidence=["Registration certificate", "Invoices showing prior use", "Screenshots / samples of infringing goods"],
                timeline_info=["Date of first use", "Date infringement noticed"],
                financial_info=["Estimated revenue loss"],
                parties_involved=["IP Owner", "Infringer"],
                urgency_indicators=["Widespread counterfeit sales"],
                desired_outcomes=["Cease and desist notice", "Commercial court suit with interim ex-parte injunction"],
                related_modules=["Breach of Contract", "Fraud"],
            )
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # CIVIL -> Other Civil Dispute
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.register(
            CaseModule(
                category="Civil",
                subcategory="General Civil",
                case_type="Other Civil Dispute",
                description="Other civil matters including torts, easements, nuisance, and general civil claims.",
                keywords=["civil suit", "injunction", "nuisance", "neighbour dispute", "damages", "civil dispute"],
                relevant_facts=["Nature of civil wrong or dispute", "Parties involved", "Location and jurisdiction", "Remedy sought"],
                priority_info=["Core grievance", "Location/State", "Parties"],
                potential_evidence=["Correspondence", "Notices", "Photographs"],
                timeline_info=["Occurrence date"],
                parties_involved=["Plaintiff", "Defendant"],
                urgency_indicators=["Imminent harm or property damage"],
                desired_outcomes=["Civil court injunction", "Damages"],
                related_modules=["Ownership Dispute", "Breach of Contract"],
            )
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # CRIMINAL -> Theft
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.register(
            CaseModule(
                category="Criminal",
                subcategory="Property Crimes",
                case_type="Theft",
                description="Theft, burglary, robbery, or stolen movable property.",
                keywords=["theft", "stolen", "burglary", "robbery", "stole", "pickpocket", "housebreak"],
                relevant_facts=[
                    "What was stolen and approximate value",
                    "Date, time, and location of incident",
                    "Known suspect or unknown perpetrator",
                    "Whether FIR has been lodged with police",
                ],
                priority_info=["Item stolen & value", "FIR status", "State/City where incident occurred"],
                potential_evidence=["CCTV footage", "Purchase bills of stolen items", "FIR copy / complaint receipt"],
                timeline_info=["Date and time of incident"],
                financial_info=["Value of stolen property"],
                parties_involved=["Victim / Complainant", "Accused / Suspect"],
                urgency_indicators=["FIR not being registered by police station"],
                desired_outcomes=["FIR under Section 378/379 IPC (or BNS)", "Section 156(3) CrPC court direction"],
                related_modules=["Fraud", "Other Criminal Matter"],
            )
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # CRIMINAL -> Fraud
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.register(
            CaseModule(
                category="Criminal",
                subcategory="Economic Crimes",
                case_type="Fraud",
                description="Cheating, forgery, embezzlement, Ponzi schemes, fake document creation.",
                keywords=["fraud", "cheating", "forgery", "fake document", "scam", "duped", "impersonation", "embezzlement", "420"],
                relevant_facts=[
                    "Manner of deception or fraudulent inducement",
                    "Financial amount parted with due to inducement",
                    "Whether forged documents were created or used",
                    "Communication trail showing fraudulent promises",
                    "Whether police complaint or FIR lodged",
                ],
                priority_info=["Deceptive inducement details", "Amount defrauded", "Evidence of forgery/misrepresentation"],
                potential_evidence=["Forged documents vs genuine records", "Bank transfer records", "WhatsApp/emails showing fraudulent promises"],
                timeline_info=["Date of misrepresentation", "Date money paid"],
                financial_info=["Defrauded amount"],
                parties_involved=["Victim", "Accused fraudster", "Intermediaries"],
                urgency_indicators=["Accused attempting to flee country or empty accounts"],
                desired_outcomes=["FIR under Section 420/467/468/471 IPC (or BNS)", "Account freeze request via police"],
                related_modules=["Money Recovery", "Cybercrime", "Property Sale Dispute"],
            )
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # CRIMINAL -> Assault
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.register(
            CaseModule(
                category="Criminal",
                subcategory="Offences Against Person",
                case_type="Assault",
                description="Physical violence, criminal intimidation, hurt, grievous hurt, battery.",
                keywords=["assault", "beaten", "hit", "physical violence", "threat to kill", "criminal intimidation", "hurt", "injured"],
                relevant_facts=[
                    "Date, time, and location of assault",
                    "Injuries sustained and whether medical treatment/MLC was done",
                    "Identity of assailants and weapons used if any",
                    "Whether FIR was registered",
                ],
                priority_info=["Injuries & MLC report", "FIR lodged", "Threat level / immediate safety"],
                potential_evidence=["Medico-Legal Certificate (MLC) / hospital report", "Photographs of injuries", "Eyewitnesses", "CCTV"],
                timeline_info=["Date of assault"],
                parties_involved=["Victim", "Assailants"],
                urgency_indicators=["Ongoing threat to life or safety", "Severe injuries"],
                desired_outcomes=["Immediate FIR registration", "Protection order / bail opposition"],
                related_modules=["Domestic Disputes", "Other Criminal Matter"],
            )
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # CRIMINAL -> Cybercrime
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.register(
            CaseModule(
                category="Criminal",
                subcategory="Cyber Offences",
                case_type="Cybercrime",
                description="Online financial fraud, phishing, identity theft, hacking, cyberstalking, deepfakes.",
                keywords=["cybercrime", "online fraud", "phishing", "hacked", "cyber fraud", "otp scam", "cyber police", "cyberstalking", "1930"],
                relevant_facts=[
                    "Specific nature of cybercrime (banking fraud, phishing, blackmail, stalking)",
                    "Amount lost and payment gateway / UPI / bank used",
                    "Time elapsed since fraudulent transaction (golden hour window)",
                    "Whether complaint filed on cybercrime.gov.in or helpline 1930",
                ],
                priority_info=["Transaction time & amount", "Bank/UPI details", "1930 helpline report status"],
                potential_evidence=["Bank debit SMS / statement", "Transaction reference numbers (UTR)", "Fraudulent website/link screenshot", "Chat history with scammer"],
                timeline_info=["Timestamp of transaction"],
                financial_info=["Amount stolen"],
                parties_involved=["Complainant", "Unknown online fraudsters", "Remittance banks"],
                urgency_indicators=["Incident within last 24 hours (immediate account freeze possible)"],
                desired_outcomes=["Reporting to 1930 / National Cyber Crime Reporting Portal", "Bank dispute form submission", "Cyber cell FIR"],
                related_modules=["Fraud", "Money Recovery"],
            )
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # CRIMINAL -> Other Criminal Matter
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.register(
            CaseModule(
                category="Criminal",
                subcategory="General Criminal",
                case_type="Other Criminal Matter",
                description="Bail, police harassment, summons, general penal code matters.",
                keywords=["fir", "police station", "anticipatory bail", "arrest", "chargesheet", "summons", "warrant", "crpc", "bns"],
                relevant_facts=["Whether arrested or fearing imminent arrest", "Police station involved", "Sections invoked", "Court hearing date"],
                priority_info=["Arrest status or threat", "Sections invoked", "Court dates"],
                potential_evidence=["Copy of FIR", "Notice under Section 41A CrPC", "Court summons"],
                timeline_info=["Date of FIR / notice / hearing"],
                parties_involved=["Accused / Complainant", "Investigating officer"],
                urgency_indicators=["Imminent arrest", "Upcoming remand hearing"],
                desired_outcomes=["Anticipatory bail under Section 438 CrPC", "Quashing petition under Section 482 CrPC"],
                related_modules=["Fraud", "Assault", "Theft"],
            )
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FAMILY -> Divorce
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.register(
            CaseModule(
                category="Family",
                subcategory="Matrimonial",
                case_type="Divorce",
                description="Mutual consent divorce, contested divorce, cruelty, desertion, adultery, annulment.",
                keywords=["divorce", "mutual consent divorce", "separation", "matrimonial dispute", "cruelty", "desertion", "annulment of marriage"],
                relevant_facts=[
                    "Date and place of marriage",
                    "Applicable marriage act (Hindu Marriage Act, Special Marriage Act, etc.)",
                    "Whether parties agree to separate mutually or if contested",
                    "Duration of living separately",
                    "Whether children are involved",
                ],
                priority_info=["Mutual consent vs contested", "Marriage Act applicable", "Duration of separation"],
                potential_evidence=["Marriage certificate / wedding photos", "Proof of separate residence", "Messages/records of cruelty or desertion"],
                timeline_info=["Marriage date", "Separation start date"],
                parties_involved=["Husband", "Wife"],
                urgency_indicators=["Restraining order needed against harassment"],
                desired_outcomes=["Mutual consent petition (Section 13B HMA)", "Contested petition on grounds of cruelty/desertion"],
                related_modules=["Child Custody", "Maintenance", "Domestic Disputes"],
            )
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FAMILY -> Child Custody
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.register(
            CaseModule(
                category="Family",
                subcategory="Matrimonial",
                case_type="Child Custody",
                description="Physical custody, legal custody, visitation rights, guardianship of minor children.",
                keywords=["child custody", "custody", "visitation rights", "guardianship", "minor child", "welfare of child", "access to child"],
                relevant_facts=[
                    "Ages and gender of children",
                    "Current residence of children (with mother or father)",
                    "Schooling and health needs",
                    "Whether other parent is being denied visitation or access",
                ],
                priority_info=["Age of children", "Current physical custody", "Visitation arrangement sought"],
                potential_evidence=["Birth certificates", "School records", "Evidence of parent-child relationship"],
                timeline_info=["Date parents separated", "Last date of contact with child"],
                parties_involved=["Father", "Mother", "Minor children"],
                urgency_indicators=["Threat of removing child out of jurisdiction or country"],
                desired_outcomes=["Interim custody application", "Visitation schedule", "Guardians and Wards Act petition"],
                related_modules=["Divorce", "Maintenance"],
            )
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FAMILY -> Maintenance
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.register(
            CaseModule(
                category="Family",
                subcategory="Matrimonial",
                case_type="Maintenance",
                description="Interim and permanent maintenance, alimony, financial support under Section 125 CrPC / HMA / DV Act.",
                keywords=["maintenance", "alimony", "interim maintenance", "125 crpc", "financial support", "child maintenance", "spousal support"],
                relevant_facts=[
                    "Income, financial assets, and employment of both spouses",
                    "Monthly expenses of dependent spouse and children",
                    "Whether applicant spouse has independent source of livelihood",
                    "Whether any previous maintenance order exists",
                ],
                priority_info=["Applicant's income status", "Respondent's known income/lifestyle", "Minor children needs"],
                potential_evidence=["Income tax returns (ITR)", "Salary slips / bank statements", "Expense proofs (school fees, medical)"],
                timeline_info=["Duration since maintenance stopped"],
                financial_info=["Maintenance amount claimed per month"],
                parties_involved=["Claimant spouse/child", "Respondent spouse"],
                urgency_indicators=["Complete destitution / inability to pay basic rent or food"],
                desired_outcomes=["Application under Section 125 CrPC", "Interim maintenance under Section 24 HMA / DV Act"],
                related_modules=["Divorce", "Domestic Disputes"],
            )
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FAMILY -> Domestic Disputes
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.register(
            CaseModule(
                category="Family",
                subcategory="Domestic Safety",
                case_type="Domestic Disputes",
                description="Protection under Protection of Women from Domestic Violence Act (DV Act), 498A IPC, dowry harassment.",
                keywords=["domestic violence", "dv act", "498a", "dowry", "in-laws harassment", "protection order", "shared household", "stridhan"],
                relevant_facts=[
                    "Nature of violence (physical, verbal, emotional, economic)",
                    "Whether living in shared household or evicted",
                    "Whether medical treatment was required",
                    "Whether stridhan or jewellery was retained by in-laws",
                ],
                priority_info=["Immediate physical safety", "Current shelter/housing", "Police or protection officer complaints"],
                potential_evidence=["Medical records", "Photographs/recordings of abuse", "List of Stridhan items", "Police complaints"],
                timeline_info=["Duration of marriage", "Most recent violent incident"],
                parties_involved=["Aggrieved woman", "Husband and in-laws"],
                urgency_indicators=["Immediate risk of physical harm or homelessness"],
                desired_outcomes=["Protection order under DV Act", "Residence order", "Monetary relief", "Return of Stridhan"],
                related_modules=["Divorce", "Maintenance", "Assault"],
            )
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FAMILY -> Inheritance Related Family Matters
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.register(
            CaseModule(
                category="Family",
                subcategory="Family Estate",
                case_type="Inheritance Related Family Matters",
                description="Family settlements, relinquishment deeds, family disputes over ancestral distributions.",
                keywords=["family settlement", "relinquishment deed", "release deed", "family estate", "brother sister property dispute", "ancestral distribution"],
                relevant_facts=[
                    "Family structure and nature of ancestral assets",
                    "Whether family settlement agreement was drafted or executed",
                    "Whether relinquishment deed was signed under duress or fraud",
                ],
                priority_info=["Settlement document status", "Assets involved", "Parties agreeing vs disputing"],
                potential_evidence=["Family settlement draft/deed", "Genealogy tree", "Asset titles"],
                timeline_info=["Date of agreement / dispute"],
                parties_involved=["Family members / Legal heirs"],
                urgency_indicators=["Execution of third-party conveyance during dispute"],
                desired_outcomes=["Enforcement of family settlement", "Cancellation of fraudulent relinquishment"],
                related_modules=["Partition", "Inheritance"],
            )
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # OTHER -> Administrative, Tax, Constitutional, Unknown
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.register(
            CaseModule(
                category="Other",
                subcategory="Public Law",
                case_type="Administrative",
                description="Government notices, municipal permissions, licensing, civil servant disputes (CAT), RTI.",
                keywords=["government notice", "municipal", "panchayat", "demolition notice", "rti", "cat", "central administrative tribunal", "tender dispute"],
                relevant_facts=["Authority issuing notice/order", "Grounds of notice", "Statutory appeal period", "Relief sought"],
                priority_info=["Authority name", "Notice date & deadline", "Hearing dates"],
                potential_evidence=["Copy of government notice/order", "RTI responses", "Challan / receipt"],
                timeline_info=["Date of notice and response deadline"],
                parties_involved=["Citizen / Enterprise", "Government Department / Municipal Authority"],
                urgency_indicators=["Demolition notice with short deadline (e.g. 24-48 hours)"],
                desired_outcomes=["Stay of demolition or coercive action", "Writ petition under Article 226"],
                related_modules=["Constitutional", "Other Civil Dispute"],
            )
        )

        self.register(
            CaseModule(
                category="Other",
                subcategory="Revenue",
                case_type="Tax",
                description="Income tax notices, GST assessments, scrutiny, penalties, appeals.",
                keywords=["income tax", "gst", "tax notice", "tax penalty", "scrutiny", "tax assessment", "gst notice", "148 notice"],
                relevant_facts=["Assessment year and section under which notice was issued", "Amount of tax or penalty demanded", "Filing status of original return", "Deadline to respond"],
                priority_info=["Section of notice (e.g., 148, 143(2))", "Deadline to file response", "Tax demanded"],
                potential_evidence=["Notice copy", "Original return acknowledgment", "Audit report / financials"],
                timeline_info=["Notice date and response deadline"],
                financial_info=["Demand amount"],
                parties_involved=["Assessee", "Income Tax / GST Department"],
                urgency_indicators=["Statutory deadline expiring within days"],
                desired_outcomes=["Filing reply to notice", "Appeal before CIT(A) / GST Appellate Authority"],
                related_modules=["Administrative"],
            )
        )

        self.register(
            CaseModule(
                category="Other",
                subcategory="Public Law",
                case_type="Constitutional",
                description="Violation of fundamental rights, writ petitions (Habeas Corpus, Mandamus, Certiorari), PIL.",
                keywords=["fundamental right", "writ", "habeas corpus", "mandamus", "article 32", "article 226", "high court writ", "supreme court", "pil"],
                relevant_facts=["Specific constitutional right violated (Article 14, 19, 21)", "State action or inaction complained of", "High Court or Supreme Court jurisdiction"],
                priority_info=["State agency involved", "Right violated", "Urgency of relief"],
                potential_evidence=["Government orders", "Correspondence", "Affidavits"],
                timeline_info=["Date of impugned action"],
                parties_involved=["Petitioner", "State / Union of India / Authority"],
                urgency_indicators=["Illegal detention (Habeas Corpus)", "Impending eviction/demolition"],
                desired_outcomes=["Writ of Mandamus / Certiorari / Habeas Corpus", "Interim stay"],
                related_modules=["Administrative"],
            )
        )

        self.register(
            CaseModule(
                category="Other",
                subcategory="General",
                case_type="Unknown / Needs Further Classification",
                description="Initial or ambiguous legal situation requiring conversational scoping before classification.",
                keywords=[],
                relevant_facts=["General nature of concern", "Parties involved", "Location/State", "Desired outcome"],
                priority_info=["Core grievance", "Location"],
                potential_evidence=["Any documents available"],
                timeline_info=["When issue began"],
                parties_involved=["User", "Opposing party"],
                urgency_indicators=["Court summons or police intervention mentioned"],
                desired_outcomes=["Initial legal clarity and categorization"],
                related_modules=[],
            )
        )


# Global singleton instance of registry
case_registry = CaseRegistry()
