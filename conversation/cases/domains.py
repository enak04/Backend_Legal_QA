"""
Domain-specific extension schemas and extensible DomainRegistry.

Supports the initial core legal domains:
  1. Employment
  2. Consumer
  3. Property / Tenancy
  4. Criminal
  5. Cybercrime
  6. Family
  7. Contract / Money Recovery
  8. Government / Administrative / RTI

Adding a new domain is as simple as defining an extension schema and registering it!
"""

from __future__ import annotations

from typing import Any, Type
from pydantic import BaseModel, Field


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Domain-Specific Extension Schemas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EmploymentExtension(BaseModel):
    """Employment and labor law extension."""
    employment_type: str | None = None          # "private", "government", "psu", "contractor", "gig"
    employee_or_contractor: str | None = None   # "regular_employee", "independent_contractor", "probationer"
    employer_type: str | None = None            # "startup", "mnc", "factory", "shop_or_establishment"
    job_title: str | None = None
    salary: str | None = None                   # Monthly / Annual salary
    employment_duration: str | None = None      # e.g., "2 years"
    termination_date: str | None = None
    termination_reason: str | None = None       # Reason stated by employer, or "none given"
    notice_period: str | None = None            # Contractual / statutory notice


class ConsumerExtension(BaseModel):
    """Consumer protection and defective goods/services extension."""
    product_or_service: str | None = None       # Name/type of item or service
    purchase_date: str | None = None
    seller_or_provider: str | None = None       # Store, e-commerce platform, manufacturer
    amount_paid: str | None = None
    defect_or_grievance: str | None = None      # Deficiency in service, unfair trade practice
    complaint_made: bool | None = None          # Written grievance sent to seller/company
    response_received: str | None = None        # Refusal, silence, partial refund offer


class PropertyExtension(BaseModel):
    """Property, land, and landlord-tenant extension."""
    property_type: str | None = None            # "residential_flat", "commercial_shop", "agricultural_land"
    ownership_status: str | None = None         # "title_holder", "co_owner", "tenant", "licensee"
    possession_status: str | None = None        # "in_possession", "dispossessed", "locked_out"
    agreement_exists: bool | None = None        # Written lease / sale deed / license
    registration_status: str | None = None      # "registered", "notarized", "unregistered_oral"
    dispute_type: str | None = None             # "illegal_eviction", "title_cloud", "deposit_withholding", "trespass"


class CriminalExtension(BaseModel):
    """Criminal offense and police complaint extension."""
    offense_type: str | None = None             # "assault", "theft", "cheating", "threat", "harassment"
    incident_date: str | None = None
    accused_known: bool | None = None           # Known person or unknown
    injury_or_loss: str | None = None           # Bodily harm, property damage, monetary loss
    police_contacted: bool | None = None        # Dialed 112 / visited police station
    fir_registered: bool | None = None          # FIR, NCR, or refusal to register
    arrest_made: bool | None = None


class CybercrimeExtension(BaseModel):
    """Cybercrime, financial fraud, and electronic dispute extension."""
    incident_type: str | None = None            # "unauthorized_bank_transfer", "upi_fraud", "identity_theft", "blackmail"
    platform_or_channel: str | None = None      # "banking_app", "whatsapp", "telegram", "phishing_site"
    financial_loss: str | None = None           # Amount stolen
    transaction_id: str | None = None           # UTR / Transaction reference
    reported_to_portal: bool | None = None      # Reported on cybercrime.gov.in / 1930
    bank_notified: bool | None = None           # Golden hour notification (< 3 days for zero liability)


class FamilyExtension(BaseModel):
    """Matrimonial, custody, maintenance, and family dispute extension."""
    relation_or_marriage_date: str | None = None
    children: str | None = None                 # Number and ages of minor children
    separation_date: str | None = None
    maintenance_or_custody_sought: str | None = None
    cruelty_or_dowry_allegations: bool | None = None
    existing_court_cases: str | None = None


class ContractExtension(BaseModel):
    """Commercial contract, money recovery, and breach extension."""
    agreement_type: str | None = None           # "service_agreement", "vendor_contract", "promissory_note", "oral"
    contract_value: str | None = None
    breach_type: str | None = None              # "non_payment", "defective_work", "abandonment"
    payment_defaulted: str | None = None        # Amount outstanding
    notice_sent: bool | None = None             # Formal demand notice sent
    limitation_date: str | None = None          # Typically 3 years from cause of action under Limitation Act 1963


class GovernmentExtension(BaseModel):
    """Administrative law, RTI, and government notices extension."""
    department: str | None = None               # Civic body, tax dept, labor dept, revenue authority
    notice_type: str | None = None              # "show_cause_notice", "penalty_order", "demolition_notice"
    date_received: str | None = None
    order_challenged: str | None = None
    rti_filed: bool | None = None
    limitation_deadline: str | None = None      # Statutory time limit to appeal (e.g. 15, 30, or 60 days)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Domain Definition & Metadata Registry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DomainDefinition(BaseModel):
    """Metadata, core statutes, and question-ranking rules for a legal domain."""
    name: str
    display_name: str
    keywords: list[str]
    primary_statutes: list[str]
    core_issues: list[str]
    typical_evidence: list[str]
    risk_indicators: list[str]
    emergency_relief: str | None = None
    high_value_questions: list[dict[str, Any]] = Field(default_factory=list)
    required_documents: list[str] = Field(default_factory=list)
    evidentiary_fallback_strategy: list[str] = Field(default_factory=list)

    def get_jurisdiction_question(self) -> str:
        for q in self.high_value_questions:
            if q.get("fact_key") in ("jurisdiction_state", "jurisdiction"):
                return q.get("question", "In which State or City did this occur?")
        return "To determine the proper legal forum and applicable state laws, could you please confirm which State or City you are located in?"


class DomainRegistry:
    """Registry managing domain definitions and schemas."""

    def __init__(self) -> None:
        self._domains: dict[str, DomainDefinition] = {}
        self._extension_map: dict[str, Type[BaseModel]] = {
            "employment": EmploymentExtension,
            "consumer": ConsumerExtension,
            "property": PropertyExtension,
            "criminal": CriminalExtension,
            "cybercrime": CybercrimeExtension,
            "family": FamilyExtension,
            "contract": ContractExtension,
            "government": GovernmentExtension,
        }
        self._register_default_domains()

    def register(self, domain_def: DomainDefinition, extension_cls: Type[BaseModel] | None = None) -> None:
        self._domains[domain_def.name.lower()] = domain_def
        if extension_cls:
            self._extension_map[domain_def.name.lower()] = extension_cls

    def get(self, name: str) -> DomainDefinition | None:
        if not name:
            return None
        norm = name.lower()
        if any(w in norm for w in ["employ", "wage", "salary", "labor", "labour"]):
            norm = "employment"
        elif any(w in norm for w in ["tenant", "evict", "landlord", "rent", "property"]):
            norm = "property"
        elif any(w in norm for w in ["cyber", "online_scam"]):
            norm = "cybercrime"
        elif "consumer" in norm:
            norm = "consumer"
        elif any(w in norm for w in ["crime", "crim"]):
            norm = "criminal"
        elif any(w in norm for w in ["divorce", "matrimon", "custody", "family"]):
            norm = "family"
        elif any(w in norm for w in ["contract", "debt", "loan", "money"]):
            norm = "contract"
        elif any(w in norm for w in ["govt", "government", "admin", "rti"]):
            norm = "government"
        return self._domains.get(norm)

    def list_domains(self) -> list[DomainDefinition]:
        return list(self._domains.values())

    def get_extension_class(self, domain_name: str) -> Type[BaseModel] | None:
        return self._extension_map.get(domain_name.lower())

    def detect_domains(self, text: str) -> list[tuple[str, float]]:
        """
        Spot relevant domains for the given text with confidence scores.
        Can return multiple domains when issues overlap.
        """
        text_lower = text.lower()
        scored: list[tuple[str, float]] = []

        for name, domain in self._domains.items():
            matches = 0
            for kw in domain.keywords:
                if kw in text_lower:
                    matches += 2 if " " in kw else 1

            if matches > 0:
                confidence = min(0.95, 0.4 + (matches * 0.15))
                scored.append((name, round(confidence, 2)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _register_default_domains(self) -> None:
        # 1. Employment
        self.register(
            DomainDefinition(
                name="employment",
                display_name="Employment & Labour Law",
                keywords=[
                    "salary", "wage", "employer", "employment", "fired", "terminated", "resign",
                    "layoff", "retrench", "pf", "provident fund", "gratuity", "bonus", "boss",
                    "unpaid salary", "job", "offer letter", "notice period"
                ],
                primary_statutes=[
                    "Payment of Wages Act, 1936",
                    "Industrial Disputes Act, 1947",
                    "Karnataka Shops and Commercial Establishments Act, 1961 (and state equivalents)",
                    "Payment of Gratuity Act, 1972",
                ],
                core_issues=["unpaid wages", "wrongful termination", "notice pay default", "gratuity withholding"],
                typical_evidence=["Offer letter / Employment contract", "Payslips / Bank statements", "Termination email/letter", "Resignation acceptance"],
                risk_indicators=["Statutory limitation for wage recovery", "Destruction of workplace communications"],
                high_value_questions=[
                    {
                        "fact_key": "jurisdiction_state",
                        "importance": "HIGH",
                        "reason": "Labor and Shops & Establishments regulations vary strictly by state",
                        "question": "In which State or City were you employed?",
                    },
                    {
                        "fact_key": "financial_dues",
                        "importance": "HIGH",
                        "reason": "Determines statutory wage claims under Payment of Wages Act or Section 33C Industrial Disputes Act",
                        "question": "Are there any unpaid salary, notice pay, gratuity, or pending dues, and what is the approximate amount remaining?",
                    },
                    {
                        "fact_key": "written_contract",
                        "importance": "MEDIUM",
                        "reason": "Dictates contractual notice terms and binding dispute resolution clauses",
                        "question": "Do you have a written employment contract, offer letter, or appointment letter?",
                    },
                    {
                        "fact_key": "employment_type",
                        "importance": "HIGH",
                        "reason": "Determines applicable statutory regime (Shops & Establishments Act vs Industrial Disputes Act vs Civil Court)",
                        "question": "Was this a private company/firm, or a government/PSU organization?",
                    },
                    {
                        "fact_key": "termination_notice_given",
                        "importance": "MEDIUM",
                        "reason": "Shows whether employer complied with statutory 30-day notice or wages in lieu thereof",
                        "question": "Did your employer give you written termination notice or reasons, or was it verbal?",
                    },
                ],
                required_documents=[
                    "Formal written Offer Letter, Appointment Letter, or Employment Agreement setting out terms of service and notice period",
                    "Official written Termination Letter, Dismissal Order, or Show Cause Notice (if issued)",
                    "Monthly Payslips or Salary Slips for the last 3 to 6 months of employment",
                    "Bank Account Statements highlighting salary credit history and the date of cessation of pay",
                    "All written communications, termination emails, resignation correspondence, or WhatsApp/Slack chat records",
                    "Proof of statutory deductions: Provident Fund (PF/UAN) passbook, Form 16, and Gratuity nomination records",
                ],
                evidentiary_fallback_strategy=[
                    "If formal employment contract is missing: Rely on appointment emails, monthly bank salary credit descriptions, company ID card, PF/EPFO member passbook records, or work email threads to establish employer-employee relationship.",
                    "If termination was verbal without written notice: Reconstruct chronological contemporaneous records by sending an immediate written email/letter recording the date, time, and exact verbal words spoken by the employer, and preserve all access-revocation timestamps (email deactivation, biometric card blocking).",
                    "If salary slips are withheld by employer: Bank account statements demonstrating regular monthly credits from the employer's entity account serve as primary secondary proof of wages under the Payment of Wages Act.",
                ],
            )
        )

        # 2. Consumer
        self.register(
            DomainDefinition(
                name="consumer",
                display_name="Consumer Protection Law",
                keywords=[
                    "defect", "laptop", "mobile", "bought", "purchase", "seller", "refund",
                    "replacement", "warranty", "consumer", "amazon", "flipkart", "defective",
                    "deficiency in service", "unfair trade", "store refused", "repair"
                ],
                primary_statutes=[
                    "Consumer Protection Act, 2019",
                    "Consumer Protection (E-Commerce) Rules, 2020",
                ],
                core_issues=["defective goods", "deficiency in service", "refusal of refund/replacement", "misleading advertisement"],
                typical_evidence=["Tax invoice / Bill of purchase", "Warranty card", "Service center inspection report", "Emails / Support chat transcripts"],
                risk_indicators=["2-year limitation period under Section 69 of Consumer Protection Act 2019"],
                high_value_questions=[
                    {
                        "fact_key": "purchase_date_and_warranty",
                        "importance": "HIGH",
                        "reason": "Determines if product was under warranty and within statutory 2-year limitation",
                        "question": "When did you purchase the item, and is it still within the warranty period?",
                    },
                    {
                        "fact_key": "purchase_amount",
                        "importance": "HIGH",
                        "reason": "Determines pecuniary jurisdiction (District Commission up to 50L vs State Commission)",
                        "question": "What was the purchase price or amount paid for the product/service?",
                    },
                    {
                        "fact_key": "formal_complaint_sent",
                        "importance": "MEDIUM",
                        "reason": "Establishes deficiency in service before filing before the Consumer Commission",
                        "question": "Have you sent a formal written complaint or legal notice demanding replacement/refund?",
                    },
                ],
                required_documents=[
                    "Tax Invoice, Cash Memo, or Bill of Purchase with GSTIN, seller details, date, and description",
                    "Manufacturer / Seller Warranty Card or Extended Warranty Policy Certificate",
                    "Authorized Service Center Job Sheets, Inspection Reports, or Repair Invoices detailing defects",
                    "Written grievance emails, ticket numbers, or customer care escalation logs sent to seller/manufacturer",
                    "Photographs, unboxing videos, or video recordings clearly demonstrating the defect or deficiency",
                    "Payment transaction receipt, credit card charge slip, or UPI confirmation",
                ],
                evidentiary_fallback_strategy=[
                    "If invoice is lost: Request duplicate invoice from seller/e-commerce platform, or submit bank/credit card transaction statements coupled with delivery confirmation emails/SMS.",
                    "If service center refuses job sheet: Record interactions, capture video of product malfunction with date stamp, and send a registered email/letter demanding inspection within 48 hours.",
                    "For e-commerce transactions: Archive order page screenshots, delivery tracking records, and return request rejection logs.",
                ],
            )
        )

        # 3. Property & Tenancy
        self.register(
            DomainDefinition(
                name="property",
                display_name="Property & Tenancy Law",
                keywords=[
                    "landlord", "tenant", "rent", "eviction", "lease", "leave and license",
                    "security deposit", "changed locks", "vacate", "property", "possession",
                    "electricity cut", "land dispute", "flat", "ownership"
                ],
                primary_statutes=[
                    "Transfer of Property Act, 1882",
                    "Specific Relief Act, 1963 (Section 6 - summary suit for dispossession)",
                    "State Rent Control & Tenancy Acts (e.g. Maharashtra Rent Control Act, Karnataka Rent Act)",
                ],
                core_issues=["illegal lockout/dispossession", "withholding of security deposit", "utility disconnection", "unlawful eviction without due process"],
                typical_evidence=["Rental / Lease agreement", "Rent payment receipts / UPI logs", "Security deposit transaction record", "Photos / Video of lockout"],
                risk_indicators=["Immediate physical lockout or utility disconnection", "6-month limitation under Section 6 Specific Relief Act"],
                emergency_relief="Approach jurisdictional Civil Court or Rent Authority immediately for interim mandatory injunction / Section 6 Specific Relief Act suit.",
                high_value_questions=[
                    {
                        "fact_key": "agreement_and_registration",
                        "importance": "MEDIUM",
                        "reason": "Determines applicable statutory protection (registered lease vs leave and license)",
                        "question": "Do you have a written or registered rental/lease agreement, and what is the remaining duration?",
                    },
                    {
                        "fact_key": "jurisdiction_state",
                        "importance": "HIGH",
                        "reason": "Tenancy laws and rent courts are state-specific under Indian law",
                        "question": "In which city and state is the property located?",
                    },
                    {
                        "fact_key": "current_possession_status",
                        "importance": "HIGH",
                        "reason": "Urgent dispossession requires immediate emergency injunction within 6 months",
                        "question": "Are you currently locked out, or are you still inside the premises facing eviction threats?",
                    },
                    {
                        "fact_key": "security_deposit_or_rent",
                        "importance": "MEDIUM",
                        "reason": "Determines financial recovery claims, interest on withheld security deposits, and court pecuniary limits",
                        "question": "What is the monthly rent and the amount of security deposit remaining with the landlord?",
                    },
                ],
                required_documents=[
                    "Registered Lease / Rent Agreement or Leave and License Agreement (or notarized tenancy deed)",
                    "Security Deposit transfer receipt, cheque record, or bank transaction confirmation",
                    "Monthly rent receipts or bank/UPI transaction statements showing consistent rent payments",
                    "Official Written Notice to Vacate, Demand Notice, or Landlord's eviction communication",
                    "Utility bills (Electricity / Water / Municipal Tax receipts) establishing continuous physical possession",
                    "Photographs or video evidence of any illegal lockout, property damage, or utility disconnection",
                ],
                evidentiary_fallback_strategy=[
                    "If tenancy agreement is oral or unregistered: Rely on rent payment bank credits/UPI logs, utility bills in occupant's name, postal receipts, courier deliveries, or neighbor affidavits establishing settled possession.",
                    "If locked out or dispossessed without due process: File immediate police diary entry/112 call record, obtain timestamped video of locked doors, and file a summary suit under Section 6 of the Specific Relief Act within 6 months.",
                    "For security deposit withholding: Reconstruct move-in and move-out condition via handover inspection photos, keys handover receipt, and WhatsApp messages acknowledging return of premises.",
                ],
            )
        )

        # 4. Criminal
        self.register(
            DomainDefinition(
                name="criminal",
                display_name="Criminal Law & Offenses",
                keywords=[
                    "assault", "assaulted", "beaten", "threat", "police", "fir", "arrest",
                    "theft", "stolen", "extortion", "blackmail", "cheating", "fraud", "criminal",
                    "beating", "injuries", "danger", "police complaint"
                ],
                primary_statutes=[
                    "Bharatiya Nyaya Sanhita (BNS), 2023 / Indian Penal Code (IPC)",
                    "Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023 / Code of Criminal Procedure (CrPC)",
                    "Bharatiya Sakshya Adhiniyam (BSA), 2023 / Indian Evidence Act",
                ],
                core_issues=["cognizable offense", "bodily injury/assault", "criminal intimidation", "refusal to register FIR"],
                typical_evidence=["Medico-Legal Certificate (MLC) / Hospital discharge", "CCTV footage / Video recording", "Police complaint receipt / General Diary entry", "Threatening messages / Call logs"],
                risk_indicators=["Immediate physical danger", "Ongoing threat of violence", "Destruction of physical/forensic evidence"],
                emergency_relief="Call 112 (National Emergency Helpline) immediately. If injured, report directly to a government hospital casualty for an MLC.",
                high_value_questions=[
                    {
                        "fact_key": "incident_date_and_medical",
                        "importance": "HIGH",
                        "reason": "Medical evidence (MLC) must be documented immediately after injury for criminal charges",
                        "question": "When did this incident occur, and did you undergo a medical examination (MLC)?",
                    },
                    {
                        "fact_key": "police_report_status",
                        "importance": "HIGH",
                        "reason": "Determines whether to approach Station House Officer, Superintendent of Police, or Magistrate under BNSS 175(3)",
                        "question": "Have you reported this to the police, and was an FIR or written complaint receipt provided?",
                    },
                    {
                        "fact_key": "identity_of_accused",
                        "importance": "MEDIUM",
                        "reason": "Affects whether investigation proceeds against named perpetrators or unknown persons",
                        "question": "Do you know the identity, names, or contact details of the perpetrators?",
                    },
                ],
                required_documents=[
                    "Medico-Legal Certificate (MLC) or Emergency Hospital Discharge Summary documenting injuries",
                    "First Information Report (FIR) copy or written Police Complaint acknowledgment / General Diary (GD) entry receipt",
                    "CCTV camera footage, mobile video recordings, or audio recordings of the incident/altercation",
                    "Screenshots, call detail records (CDR), WhatsApp chats, or letters demonstrating criminal intimidation or threats",
                    "Independent eyewitness names, contact information, and written statements",
                    "Proof of financial loss, stolen property ownership documents, or valuation bills (in theft/cheating)",
                ],
                evidentiary_fallback_strategy=[
                    "If police station refuses to register FIR: Submit written complaint to Superintendent of Police / Commissioner under Section 175(3) BNSS (154(3) CrPC) by Registered Post AD, followed by Section 175(4) BNSS application before the Judicial Magistrate.",
                    "If private medical clinic did not issue MLC: Immediately visit a government district hospital emergency department stating cause of injury was physical assault to create official government medical record.",
                    "For electronic evidence (audio/video/chats): Preserve original recording device without alteration and prepare Section 63 Bharatiya Sakshya Adhiniyam (BSA 2023) / 65B Indian Evidence Act certificate.",
                ],
            )
        )

        # 5. Cybercrime
        self.register(
            DomainDefinition(
                name="cybercrime",
                display_name="Cybercrime & Online Financial Fraud",
                keywords=[
                    "transferred from my bank", "bank account", "without permission", "otp",
                    "unauthorized transaction", "cyber", "phishing", "scam", "hacked",
                    "upi fraud", "money stolen online", "crypto fraud", "sim swap"
                ],
                primary_statutes=[
                    "Information Technology Act, 2000 (Sections 43, 66C, 66D)",
                    "RBI Master Circular on Customer Protection – Limiting Liability in Unauthorized Electronic Banking Transactions (2017)",
                    "Bharatiya Nyaya Sanhita, 2023 (Section 318 - Cheating)",
                ],
                core_issues=["unauthorized electronic fund transfer", "zero liability under RBI guidelines", "cyber fraud", "account freeze"],
                typical_evidence=["Bank statement showing debit", "SMS alert / email notification", "Transaction UTR number", "Screenshots of phishing links/chats"],
                risk_indicators=["First 24-72 hours ('Golden Hours') are critical to freeze recipient bank accounts and claim zero customer liability"],
                emergency_relief="Dial 1930 (National Cybercrime Helpline) immediately and report on cybercrime.gov.in. Notify your bank within 72 hours in writing.",
                high_value_questions=[
                    {
                        "fact_key": "financial_loss",
                        "importance": "HIGH",
                        "reason": "Determines police cyber cell jurisdiction threshold and compensation claim under IT Act",
                        "question": "What was the total amount debited or lost, and from which bank account or wallet?",
                    },
                    {
                        "fact_key": "transaction_date_and_time",
                        "importance": "HIGH",
                        "reason": "RBI rules grant ZERO liability if reported to the bank within 3 calendar days of unauthorized debit",
                        "question": "When exactly was the money transferred, and how many hours/days have passed?",
                    },
                    {
                        "fact_key": "bank_and_cybercell_notified",
                        "importance": "HIGH",
                        "reason": "Immediate reporting to 1930 can freeze the fraudulent money in the recipient account before withdrawal",
                        "question": "Have you called the 1930 cybercrime helpline and filed a written fraud complaint with your bank?",
                    },
                    {
                        "fact_key": "otp_or_credential_sharing",
                        "importance": "MEDIUM",
                        "reason": "Distinguishes pure bank system negligence (zero liability) from shared credential liability under RBI norms",
                        "question": "Was an OTP or password shared with anyone, or did the transaction happen without any sharing?",
                    },
                ],
                required_documents=[
                    "Bank / Credit Card Statement showing exact debit timestamp, transaction amount, and recipient account/UPI ID/UTR number",
                    "SMS alerts and email notifications received from bank regarding unauthorized debit",
                    "Complaint acknowledgment receipt from National Cybercrime Reporting Portal (cybercrime.gov.in) or 1930 Helpline Citizen Copy",
                    "Written fraud intimation submitted to Home Bank branch with official receiving stamp and date",
                    "Screenshots of fraudulent phishing links, payment gateway prompts, fake caller IDs, or WhatsApp/Telegram chats",
                    "Device details (IMEI number, IP address logs, browser history) if device was compromised by remote desktop apps",
                ],
                evidentiary_fallback_strategy=[
                    "If bank delays dispute acknowledgement: Send written notice via registered email to the Principal Nodal Officer of the bank within 72 hours, explicitly quoting RBI Master Circular 2017 for Zero Customer Liability.",
                    "If recipient account is in another bank: Immediately escalate complaint number 1930 to National Cybercrime Portal to trigger CFCFRMS (Citizen Financial Cyber Fraud Reporting & Management System) interstate account freeze.",
                    "If remote access software (AnyDesk/TeamViewer) was used: Uninstall app, preserve connection session ID logs, and obtain device technical diagnostic certificate for evidentiary submission.",
                ],
            )
        )

        # 6. Family Law
        self.register(
            DomainDefinition(
                name="family",
                display_name="Family & Matrimonial Law",
                keywords=[
                    "spouse", "husband", "wife", "divorce", "maintenance", "financial support",
                    "custody", "child support", "alimony", "domestic violence", "separated",
                    "marriage", "dowry", "498a"
                ],
                primary_statutes=[
                    "Protection of Women from Domestic Violence Act, 2005 (PWDVA)",
                    "Section 125 CrPC / Section 144 BNSS (Statutory right to interim maintenance)",
                    "Personal laws (Hindu Marriage Act 1955, Special Marriage Act 1954, Muslim Personal Law)",
                ],
                core_issues=["interim maintenance", "refusal of financial support", "child custody", "domestic violence / protection orders"],
                typical_evidence=["Marriage certificate / wedding photos", "Bank statements showing lack of maintenance / income disparity", "Proof of minor children's expenses", "Police complaints / medical proof if any"],
                risk_indicators=["Destitution / deprivation of basic living necessities", "Threats to forcibly remove minor children"],
                emergency_relief="File for immediate interim monetary relief and residence orders under Section 12/23 of the Protection of Women from Domestic Violence Act.",
                high_value_questions=[
                    {
                        "fact_key": "personal_law_and_marriage_status",
                        "importance": "HIGH",
                        "reason": "Determines statutory grounds and applicable court (Family Court vs Magistrate)",
                        "question": "Under which law/religion was the marriage solemnized, and are you currently living together or separated?",
                    },
                    {
                        "fact_key": "children_and_expenses",
                        "importance": "HIGH",
                        "reason": "Affects statutory child maintenance and custody jurisdiction under Guardians and Wards Act",
                        "question": "Do you have any minor children living with you who require schooling and living support?",
                    },
                    {
                        "fact_key": "spouse_income_status",
                        "importance": "MEDIUM",
                        "reason": "Forms the basis of interim maintenance calculation under Supreme Court guidelines (Rajnesh v. Neha)",
                        "question": "What is your spouse's occupation or estimated income, and do you have an independent source of income?",
                    },
                ],
                required_documents=[
                    "Marriage Certificate (under Hindu Marriage Act / Special Marriage Act / relevant personal law) or wedding invitations/photographs",
                    "Birth certificates and school fee receipts for minor children",
                    "Bank statements of both parties for the last 12 to 24 months demonstrating standard of living and income disparity",
                    "Income Tax Returns (ITR), Form 16, or salary slips of respondent spouse (if available)",
                    "Medical records, MLC reports, or police complaint receipts (in cases involving domestic violence or cruelty)",
                    "Detailed monthly household expenditure statement supported by grocery, rent, and medical receipts",
                ],
                evidentiary_fallback_strategy=[
                    "If spouse's true income is concealed: File Application for Comprehensive Affidavit of Assets and Liabilities pursuant to Supreme Court guidelines in Rajnesh v. Neha (2020), mandating disclosure of all bank accounts, investments, and tax filings under oath.",
                    "If marriage certificate was not registered: Establish valid marriage through joint bank accounts, passport entries naming spouse, Aadhaar card address endorsements, and wedding photograph albums.",
                    "If domestic cruelty occurred within shared household: Rely on WhatsApp chats, contemporaneous audio/video recordings, diary entries, and neighbor/relative corroborating statements under Section 12 PWDVA.",
                ],
            )
        )

        # 7. Contract & Money Recovery
        self.register(
            DomainDefinition(
                name="contract",
                display_name="Contract & Money Recovery Law",
                keywords=[
                    "contractor", "agreement", "paid money", "breach", "completed work",
                    "recover money", "promissory note", "unpaid invoice", "lakh", "cheque bounce",
                    "defaulted", "refund money", "contract"
                ],
                primary_statutes=[
                    "Indian Contract Act, 1872 (Sections 73, 74 for damages/breach)",
                    "Limitation Act, 1963 (Schedule Article 18, 55 - 3-year limitation for recovery)",
                    "Commercial Courts Act, 2015 (pre-institution mediation for commercial disputes)",
                    "Negotiable Instruments Act, 1881 (Section 138 for dishonored cheques)",
                ],
                core_issues=["breach of contract", "recovery of advance money", "failure to perform services", "unjust enrichment"],
                typical_evidence=["Written agreement / Quotation / Work order", "Bank transfer / UPI payment proofs", "WhatsApp / Email communications acknowledging receipt", "Photographs of incomplete work"],
                risk_indicators=["3-year limitation period running from date of breach / last payment"],
                high_value_questions=[
                    {
                        "fact_key": "agreement_form",
                        "importance": "HIGH",
                        "reason": "Determines evidentiary standard under Indian Contract Act (express terms vs oral agreement via electronic records)",
                        "question": "Was there a written contract, work order, or quotation, or was it agreed via WhatsApp/email?",
                    },
                    {
                        "fact_key": "payment_proof_and_amount",
                        "importance": "HIGH",
                        "reason": "Determines exact claim amount and whether Summary Suit under Order XXXVII CPC applies",
                        "question": "What was the total agreed value, and how much did you pay with bank/UPI proof?",
                    },
                    {
                        "fact_key": "date_of_breach_or_default",
                        "importance": "HIGH",
                        "reason": "Statutory limitation period is 3 years from the date the work was due or payment refused",
                        "question": "When was the work supposed to be completed, and when did they stop responding?",
                    },
                ],
                required_documents=[
                    "Executed Contract, Service Level Agreement (SLA), Master Services Agreement (MSA), or Purchase Order (PO)",
                    "Signed Invoices, Delivery Challans, or Milestone Completion Certificates accepted by opposite party",
                    "Bank account statements or RTGS/NEFT/UPI transfer counterfoils proving payments made or unpaid invoices",
                    "Written Formal Demand Notice / Legal Notice along with Postal Dispatch Slip and Delivery Tracking Report",
                    "Email correspondence, WhatsApp messages, or minutes of meeting showing clear acknowledgment of debt or liability",
                    "Dishonored Cheque (original) with Bank Return Memo stating 'Funds Insufficient' (if Negotiable Instruments Act Section 138 claim)",
                ],
                evidentiary_fallback_strategy=[
                    "If contract is oral: Reconstruct binding agreement using purchase orders, invoice acceptances, part-payments credited to bank accounts, WhatsApp confirmation of terms, and delivery receipts under Section 8 of the Indian Contract Act.",
                    "To prevent claim being barred by limitation: Identify any email, letter, or part-payment made within 3 years that constitutes a valid written acknowledgment of liability under Section 18 of the Limitation Act, 1963.",
                    "For commercial disputes exceeding ₹3 Lakhs: Initiate mandatory Pre-Institution Mediation under Section 12A of the Commercial Courts Act, 2015 prior to filing suit.",
                ],
            )
        )

        # 8. Government, Administrative & RTI
        self.register(
            DomainDefinition(
                name="government",
                display_name="Government, Administrative & RTI Law",
                keywords=[
                    "government notice", "govt notice", "municipal", "panchayat", "show cause notice",
                    "order", "department", "rti", "authority", "demolition notice", "tax notice",
                    "licence cancelled", "sealed"
                ],
                primary_statutes=[
                    "Constitution of India (Articles 226 & 32 - Writ jurisdiction for arbitrary state action)",
                    "Right to Information Act, 2005",
                    "Applicable Municipal / Civic Authority Acts (e.g. BBMP Act, BMC Act, Delhi Municipal Corporation Act)",
                ],
                core_issues=["violation of natural justice (audi alteram partem)", "arbitrary administrative action", "failure to respond to statutory representation", "imminent coercive action"],
                typical_evidence=["Copy of the government notice / order received", "Proof of delivery / postal tracking showing date of receipt", "Prior approvals / sanctions / licenses", "Written reply submitted"],
                risk_indicators=["Strict statutory deadline to reply or appeal (often 7, 15, or 30 days) to prevent demolition/sealing"],
                emergency_relief="Immediately approach the jurisdictional High Court under Article 226 for a stay on coercive action if statutory remedy is inadequate or absent.",
                high_value_questions=[
                    {
                        "fact_key": "issuing_authority_and_subject",
                        "importance": "HIGH",
                        "reason": "Determines applicable statutory appellate tribunal (RERA, NGT, Municipal Tribunal, High Court)",
                        "question": "Which specific government department issued the notice, and under which Act/Section?",
                    },
                    {
                        "fact_key": "date_of_receipt_and_deadline",
                        "importance": "HIGH",
                        "reason": "Statutory deadlines to reply or appeal are strict; failure to respond can result in immediate adverse orders",
                        "question": "When did you physically or digitally receive this notice, and what response deadline does it mention?",
                    },
                    {
                        "fact_key": "threat_of_coercive_action",
                        "importance": "HIGH",
                        "reason": "Threats of demolition, arrest, or property sealing require emergency writ petition for stay",
                        "question": "Does the notice threaten immediate action like sealing, demolition, or penalties?",
                    },
                ],
                required_documents=[
                    "Original Government Show Cause Notice, Sealing Notice, Demolition Order, or Penalty Memo received",
                    "Postal envelope / Speed Post tracking slip / Email header showing exact date of receipt of notice",
                    "Sanctioned Building Plan, Trade License, Occupancy Certificate, or statutory NOCs issued by municipal authorities",
                    "Written Representation / Reply submitted to the government department with official inward/acknowledgment seal",
                    "RTI Application filed and Public Information Officer (PIO) / First Appellate Authority replies received",
                    "Relevant Gazetted Notification, Rules, or Bye-laws under which authority claims power",
                ],
                evidentiary_fallback_strategy=[
                    "If government files/reasons are withheld: Immediately file Urgent Right to Information (RTI) application under Section 7(1) RTI Act (or standard application) to inspect official file notings and departmental processing sheets.",
                    "If threatened with immediate arbitrary demolition/sealing without hearing: Draft emergency Writ Petition under Article 226 of the Constitution of India citing violation of natural justice (audi alteram partem) and seek ad-interim status quo.",
                    "If department refuses to accept written reply: Send reply by Speed Post AD, email to designated official email ID, and upload on departmental grievance portal, preserving electronic transmission logs.",
                ],
            )
        )


domain_registry = DomainRegistry()
