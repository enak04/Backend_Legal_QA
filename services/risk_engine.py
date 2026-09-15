"""
Risk and urgency engine.

Detects high-risk, time-critical legal situations (arrest, physical danger,
ongoing financial cyber fraud, illegal eviction/lockout, imminent court deadlines)
and surfaces immediate interim protective guidance without false certainty.
"""

from __future__ import annotations

import re
from typing import Any
from conversation.cases.models import Risk, UniversalCaseState


class RiskEngine:
    """
    Dedicated risk and urgency detector for legal matters.
    """

    # Trigger patterns for various critical categories
    _PATTERNS = [
        (
            "immediate_physical_danger",
            "emergency",
            re.compile(
                r"\b(assaulted|beaten|hitting|physically abused|threatened to kill|kill me|life is in danger|violence|domestic violence|weapon|attacked)\b",
                re.IGNORECASE,
            ),
            "Physical safety is at immediate risk. Under Indian law, immediate police protection and medical documentation take precedence.",
            "Call 112 (National Emergency Helpline) or 181 (Women Helpline) immediately. If injured, visit the nearest government hospital casualty for an official Medico-Legal Certificate (MLC).",
        ),
        (
            "arrest_or_detention_threat",
            "urgent",
            re.compile(
                r"\b(threat of arrest|police at my door|going to arrest me|police custody|detained|in lockup|arrest warrant)\b",
                re.IGNORECASE,
            ),
            "Imminent risk of custodial action or arrest. Constitutional safeguards under Article 22 of the Constitution of India and Section 35 of BNSS / 41A CrPC apply.",
            "Contact a practicing criminal defense advocate immediately to move an urgent Anticipatory Bail application under Section 482 of the BNSS, 2023 (or Section 438 CrPC).",
        ),
        (
            "ongoing_financial_fraud",
            "urgent",
            re.compile(
                r"\b(transferred from my bank|without permission|without my consent|otp scam|money stolen online|fraudulent transaction|unauthorized transfer|hacked my account|sim swap)\b",
                re.IGNORECASE,
            ),
            "Time-critical electronic financial fraud. RBI rules provide zero customer liability if reported within 3 days, and cyber cell can freeze funds if alerted immediately.",
            "Immediately dial 1930 (National Cybercrime Helpline) and register on cybercrime.gov.in. Submit a formal written fraud dispute letter to your bank within 72 hours.",
        ),
        (
            "illegal_eviction_or_lockout",
            "urgent",
            re.compile(
                r"\b(changed the locks|locks changed|locked me out|thrown my belongings|electricity cut|water cut|forcibly evicted|threat of immediate eviction|throwing me out)\b",
                re.IGNORECASE,
            ),
            "Landlords are strictly prohibited from taking the law into their own hands, locking out tenants, or disconnecting essential utilities without due process of law.",
            "File a complaint with the local police for criminal trespass/intimidation and approach the jurisdictional Civil Court / Rent Controller for immediate mandatory injunction and possession under Section 6 of the Specific Relief Act, 1963.",
        ),
        (
            "imminent_court_or_statutory_deadline",
            "urgent",
            re.compile(
                r"\b(court notice|summons|warrant|demolition notice|government notice|notice from (?:a )?government|show cause notice|reply in \d+ days|hearing tomorrow|hearing is tomorrow|ex-parte|interim stay|penalty order)\b",
                re.IGNORECASE,
            ),
            "Formal judicial or statutory deadline is running. Failure to appear or reply within the prescribed time can result in ex-parte orders or forfeiture of legal defenses.",
            "Immediately consult an advocate to enter an appearance, inspect the court file, and file an urgent reply or application for extension/stay.",
        ),
        (
            "destruction_of_evidence",
            "potentially_urgent",
            re.compile(
                r"\b(deleting emails|formatting phone|destroying documents|erasing cctv|tampering with evidence)\b",
                re.IGNORECASE,
            ),
            "Risk of material evidence spoliation. Preserving electronic and documentary evidence is essential for admissibility under Section 63/65B of Bharatiya Sakshya Adhiniyam / Evidence Act.",
            "Take certified digital screenshots, backup all email headers and WhatsApp chats, and issue a formal Legal Preservation Notice.",
        ),
        (
            "limitation_deadline_approaching",
            "potentially_urgent",
            re.compile(
                r"\b(cheque bounced|dishonour memo|30 days from notice|almost 3 years|limitation period|statutory notice period expiring)\b",
                re.IGNORECASE,
            ),
            "Statutory limitation period is approaching expiry. For example, Section 138 NI Act requires legal notice within 30 days of dishonour and complaint within 30 days thereafter.",
            "Issue the statutory demand notice or file the complaint immediately to prevent your claim from becoming time-barred.",
        ),
    ]

    def detect_risk(
        self,
        text: str,
        current_state: UniversalCaseState | None = None,
    ) -> Risk:
        """
        Evaluate user message and current case state for risks and urgency.
        """
        text_lower = text.lower()
        active_flags: list[str] = []
        highest_level = "normal"
        reasons: list[str] = []
        actions: list[str] = []

        level_weights = {
            "low": 1,
            "normal": 2,
            "potentially_urgent": 3,
            "urgent": 4,
            "emergency": 5,
        }

        # Check existing state flags
        if current_state and current_state.risk.flags:
            active_flags.extend(current_state.risk.flags)
            if level_weights.get(current_state.risk.level, 0) > level_weights.get(highest_level, 0):
                highest_level = current_state.risk.level

        for flag, level, pattern, reason, action in self._PATTERNS:
            if pattern.search(text_lower):
                if flag not in active_flags:
                    active_flags.append(flag)
                if level_weights[level] > level_weights.get(highest_level, 1):
                    highest_level = level
                reasons.append(reason)
                actions.append(action)

        combined_reason = " | ".join(reasons) if reasons else None
        combined_action = " | ".join(actions) if actions else None

        return Risk(
            level=highest_level,
            flags=active_flags,
            reason=combined_reason or (current_state.risk.reason if current_state else None),
            recommended_emergency_action=combined_action or (current_state.risk.recommended_emergency_action if current_state else None),
        )


risk_engine = RiskEngine()
