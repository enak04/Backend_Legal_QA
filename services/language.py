"""
Multilingual language detection and utilities for Indian legal consultation.

Supports major Indian languages (Telugu, Hindi, Tamil, Kannada, Marathi, Bengali,
Gujarati, Malayalam, Punjabi, Odia, English) and mixed script/transliteration.
"""

from __future__ import annotations

import re

# Supported ISO-639-1 language codes mapped to human names
SUPPORTED_LANGUAGES: dict[str, str] = {
    "te": "Telugu",
    "hi": "Hindi",
    "ta": "Tamil",
    "kn": "Kannada",
    "mr": "Marathi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
    "en": "English",
}

# Regex ranges for Indian native scripts
_SCRIPT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"[\u0C00-\u0C7F]"), "te"),  # Telugu
    (re.compile(r"[\u0B80-\u0BFF]"), "ta"),  # Tamil
    (re.compile(r"[\u0C80-\u0CFF]"), "kn"),  # Kannada
    (re.compile(r"[\u0980-\u09FF]"), "bn"),  # Bengali
    (re.compile(r"[\u0A80-\u0AFF]"), "gu"),  # Gujarati
    (re.compile(r"[\u0D00-\u0D7F]"), "ml"),  # Malayalam
    (re.compile(r"[\u0A00-\u0A7F]"), "pa"),  # Gurmukhi / Punjabi
    (re.compile(r"[\u0B00-\u0B7F]"), "or"),  # Odia
    (re.compile(r"[\u0900-\u097F]"), "hi"),  # Devanagari (Hindi / Marathi)
    (re.compile(r"[\u0600-\u06FF\u0750-\u077F]"), "hi"),  # Perso-Arabic (Urdu/Hindi)
]

# Common Latin-script transliterated Indic tokens
_TELUGLISH_TOKENS = {
    "nenu", "naa", "naaku", "chesanu", "chesaru", "ivvaledhu", "ledhu",
    "jeetham", "cheyali", "meeru", "undhi", "unnaru", "teliyadhu", "daggara",
    "pani", "manishi", "raavadam", "emiti", "evaru", "ippudu",
}

_HINGLISH_TOKENS = {
    "mera", "meri", "mere", "mujhe", "kya", "nahin", "nahi", "diya", "kiye",
    "karein", "karna", "hoga", "raha", "rahe", "chahiye", "paisa", "paise",
    "nokri", "naukri", "kaam", "saal", "mahine", "aap", "hum",
}

_TAMGLISH_TOKENS = {
    "naan", "enakku", "pannanum", "pannanga", "sambalam", "theriyathu",
    "irukku", "illa", "illai", "kudukkala", "velai", "solranga",
}


def detect_language(text: str, default: str = "en") -> str:
    """
    Detect the primary language of the input text.
    
    1. Checks for Indic Unicode script characters first (highest confidence).
    2. Checks for transliterated Indic tokens in Latin script.
    3. Falls back to default (English).
    """
    if not text or not text.strip():
        return default

    cleaned = text.strip()

    # 1. Check Unicode script ranges
    for pattern, lang_code in _SCRIPT_PATTERNS:
        # If there are at least 2 script characters or significant presence
        matches = pattern.findall(cleaned)
        if len(matches) >= 2 or (len(cleaned) <= 10 and len(matches) >= 1):
            return lang_code

    # 2. Check transliterated tokens in Latin script
    words = set(re.findall(r"\b[a-zA-Z]+\b", cleaned.lower()))
    if not words:
        return default

    telugu_overlap = words & _TELUGLISH_TOKENS
    if len(telugu_overlap) >= 2 or (len(words) <= 5 and len(telugu_overlap) >= 1):
        return "te"

    hinglish_overlap = words & _HINGLISH_TOKENS
    if len(hinglish_overlap) >= 2 or (len(words) <= 5 and len(hinglish_overlap) >= 1):
        return "hi"

    tamglish_overlap = words & _TAMGLISH_TOKENS
    if len(tamglish_overlap) >= 2 or (len(words) <= 5 and len(tamglish_overlap) >= 1):
        return "ta"

    return default


def normalize_language_code(code: str | None, default: str = "en") -> str:
    """Normalize input language code or return default."""
    if not code or code.strip().lower() in ("auto", "none", ""):
        return default
    c = code.strip().lower()
    return c if c in SUPPORTED_LANGUAGES else default


def get_language_name(code: str) -> str:
    """Get human-readable language name from ISO code."""
    return SUPPORTED_LANGUAGES.get(code, "English")


def is_indic_language(code: str) -> bool:
    """Check if the language is an Indian regional language."""
    return code in SUPPORTED_LANGUAGES and code != "en"


def resolve_turn_language(text: str, current_lang: str = "en") -> str:
    """
    Resolve the active language for the current turn, supporting dynamic language switching.
    
    1. If user uses native Indic script (Telugu, Hindi, Tamil, etc.), switch to that language.
    2. If user uses transliterated tokens (Hinglish, Teluglish), switch to that language.
    3. If user sends a short factual reply (1-3 words like 'Yes', 'No', '2 months', 'Hyderabad'),
       retain the current conversation language so flow is not disrupted.
    4. If user writes a longer English message (> 3 words) without Indic script/tokens, switch to English.
    """
    valid_current = normalize_language_code(current_lang, default="en")
    if not text or not text.strip():
        return valid_current

    cleaned = text.strip()

    # 1. Native Indic script check (immediate switch)
    for pattern, lang_code in _SCRIPT_PATTERNS:
        matches = pattern.findall(cleaned)
        if len(matches) >= 2 or (len(cleaned) <= 10 and len(matches) >= 1):
            return lang_code

    # 2. Check transliterated Indic tokens
    words = re.findall(r"\b[a-zA-Z]+\b", cleaned.lower())
    words_set = set(words)

    if words_set & _TELUGLISH_TOKENS:
        return "te"
    if words_set & _HINGLISH_TOKENS:
        return "hi"
    if words_set & _TAMGLISH_TOKENS:
        return "ta"

    # 3. Short answers (e.g., "Yes", "2 months", "Bangalore") retain active regional language
    if len(words) <= 3 and valid_current != "en":
        return valid_current

    # 4. Otherwise user wrote in English
    return "en"


LOCALIZED_SYSTEM_QUESTIONS: dict[str, dict[str, str]] = {
    "employment_type": {
        "en": "Was this a private company/firm, or a government/PSU organization?",
        "hi": "क्या आप किसी निजी कंपनी/फर्म में काम कर रहे थे, या यह कोई सरकारी विभाग या पीएसयू संगठन था?",
        "te": "మీరు ప్రైవేట్ కంపెనీ/సంస్థలో పనిచేస్తున్నారా లేక ప్రభుత్వ విభాగం లేదా పీఎస్‌యూ సంస్థలో పనిచేస్తున్నారా?",
        "ta": "நீங்கள் ஒரு தனியார் நிறுவனத்தில் பணிபுரிந்தீர்களா அல்லது அரசுத் துறை அல்லது பொதுத்துறை நிறுவனத்திலா?",
        "kn": "ನೀವು ಖಾಸಗಿ ಕಂಪನಿ/ಸಂಸ್ಥೆಯಲ್ಲಿ ಕೆಲಸ ಮಾಡುತ್ತಿದ್ದೀರಾ ಅಥವಾ ಸರ್ಕಾರಿ ಇಲಾಖೆ ಅಥವಾ ಪಿಎಸ್‌ಯು ಸಂಸ್ಥೆಯಲ್ಲಿದ್ದೀರಾ?",
        "bn": "আপনি কি কোনো বেসরকারি কোম্পানি/ফার্মে কাজ করতেন, নাকি এটি কোনো সরকারি বিভাগ বা পিএসইউ ছিল?",
        "mr": "तुम्ही खाजगी कंपनी/फर्ममध्ये काम करत होतात की सरकारी विभाग किंवा पीएसयू संस्थेमध्ये?",
        "gu": "શું તમે કોઈ ખાનગી કંપની/ફર્મમાં કામ કરતા હતા, કે પછી આ કોઈ સરકારી વિભાગ કે પીએસયુ હતું?",
        "ml": "നിങ്ങൾ ഒരു സ്വകാര്യ കമ്പനിയിലാണോ അതോ സർക്കാർ വകുപ്പിലോ പി.എസ്.യു-വിലോ ആണോ ജോലി ചെയ്തിരുന്നത്?",
    },
    "jurisdiction": {
        "en": "To identify the correct legal forum and applicable state acts, could you please confirm which State or City you are located in?",
        "hi": "उचित कानूनी मंच और लागू राज्य कानूनों की पहचान करने के लिए, क्या आप पुष्टि कर सकते हैं कि आप किस राज्य या शहर में स्थित हैं?",
        "te": "సరైన న్యాయ వేదిక మరియు వర్తించే రాష్ట్ర చట్టాలను గుర్తించడానికి, మీరు ఏ రాష్ట్రం లేదా నగరంలో ఉన్నారో దయచేసి ధృవీకరించగలరా?",
        "ta": "சரியான சட்ட மன்றம் மற்றும் பொருந்தக்கூடிய மாநில சட்டங்களை அடையாளம் காண, நீங்கள் எந்த மாநிலம் அல்லது நகரத்தில் உள்ளீர்கள் என்பதை உறுதிப்படுத்த முடியுமா?",
        "kn": "ಸರಿಯಾದ ಕಾನೂನು ವೇದಿಕೆ ಮತ್ತು ಅನ್ವಯವಾಗುವ ರಾಜ್ಯ ಕಾಯಿದೆಗಳನ್ನು ಗುರುತಿಸಲು, ನೀವು ಯಾವ ರಾಜ್ಯ ಅಥವಾ ನಗರದಲ್ಲಿದ್ದೀರಿ ಎಂಬುದನ್ನು ದಯವಿಟ್ಟು ದೃಢೀಕರಿಸಬಹುದೇ?",
        "bn": "সঠিক আইনি ফোরাম এবং প্রযোজ্য রাজ্য আইন চিহ্নিত করতে, আপনি কোন রাজ্য বা শহরে অবস্থিত তা কি নিশ্চিত করতে পারেন?",
        "mr": "योग्य कायदेशीर मंच आणि लागू राज्य कायदे ओळखण्यासाठी, आपण कोणत्या राज्यात किंवा शहरात आहात याची पुष्टी करू शकता का?",
        "gu": "યોગ્ય કાનૂની મંચ અને લાગુ પડતા રાજ્યના કાયદા ઓળખવા માટે, તમે કયા રાજ્ય કે શહેરમાં છો તેની પુષ્ટિ કરી શકો છો?",
        "ml": "ശരിയായ നിയമ ഫോറവും ബാധകമായ സംസ്ഥാന നിയമങ്ങളും തിരിച്ചറിയാൻ, നിങ്ങൾ ഏത് സംസ്ഥാനത്തോ നഗരത്തിലോ ആണെന്ന് സ്ഥിരീകരിക്കാമോ?",
    },
    "financial_dues": {
        "en": "What is the approximate amount of unpaid salary, notice pay, or other dues owed to you?",
        "hi": "आपके बकाया वेतन, नोटिस वेतन या अन्य देय राशि लगभग कितनी है?",
        "te": "మీకు రావాల్సిన బకాయి జీతం, నోటీసు పే లేదా ఇతర బకాయిల మొత్తం సుమారుగా ఎంత?",
        "ta": "உங்களுக்கு வரவேண்டிய நிலுவை சம்பளம், நோட்டீஸ் சம்பளம் அல்லது பிற நிலுவைத் தொகையின் தோராயமான மதிப்பு என்ன?",
        "kn": "ನಿಮಗೆ ಬರಬೇಕಾದ ಬಾಕಿ ಸಂಬಳ, ನೋಟಿಸ್ ವೇತನ ಅಥವಾ ಇತರ ಬಾಕಿ ಮೊತ್ತ ಸರಿಸುಮಾರು ಎಷ್ಟು?",
        "bn": "আপনার বকেয়া বেতন, নোটিশ পে বা অন্যান্য পাওনার আনুমানিক পরিমাণ কত?",
        "mr": "तुमचे थकीत वेतन, नोटीस पे किंवा इतर देयकांची अंदाजे रक्कम किती आहे?",
        "gu": "તમારો બાકી પગાર, નોટિસ પે અથવા અન્ય બાકી રકમ આશરે કેટલી છે?",
        "ml": "നിങ്ങൾക്ക് ലഭിക്കാനുള്ള ശമ്പള കുടിശ്ശിക, നോട്ടീസ് പേ അല്ലെങ്കിൽ മറ്റ് കുടിശ്ശികകൾ എത്രയാണ്?",
    },
    "written_contract": {
        "en": "Do you have a written employment contract, offer letter, or appointment letter?",
        "hi": "क्या आपके पास लिखित रोजगार अनुबंध, ऑफर लेटर या नियुक्ति पत्र है?",
        "te": "మీ వద్ద రాతపూర్వక ఉపాధి ఒప్పందం, ఆఫర్ లెటర్ లేదా నియామక పత్రం ఉందా?",
        "ta": "உங்களிடம் எழுத்துப்பூர்வ வேலை ஒப்பந்தம், ஆஃபர் லெட்டர் அல்லது நியமனக் கடிதம் உள்ளதா?",
        "kn": "ನಿಮ್ಮ ಬಳಿ ಲಿಖಿತ ಉದ್ಯೋಗ ಒಪ್ಪಂದ, ಆಫರ್ ಲೆಟರ್ ಅಥವಾ ನೇಮಕಾತಿ ಪತ್ರವಿದೆಯೇ?",
        "bn": "আপনার কাছে কি লিখিত চাকরির চুক্তি, অফার লেটার বা নিয়োগপত্র রয়েছে?",
        "mr": "तुमच्याकडे लेखी रोजगार करार, ऑफर लेटर किंवा नियुक्ती पत्र आहे का?",
        "gu": "શું તમારી પાસે લેખિત રોજગાર કરાર, ઑફર લેટર અથવા નિમણૂક પત્ર છે?",
        "ml": "നിങ്ങളുടെ പക്കൽ രേഖാമൂലമുള്ള തൊഴിൽ കരാറോ ഓഫർ ലെറ്ററോ നിയമന ഉത്തരവോ ഉണ്ടോ?",
    },
    "tenancy_lockout": {
        "en": "Are you currently locked out, or are you still inside the premises facing eviction threats?",
        "hi": "क्या आपको परिसर से बाहर निकाल दिया गया है (ताला लगा दिया गया है), या आप अभी भी अंदर हैं और बेदखली की धमकियों का सामना कर रहे हैं?",
        "te": "మిమ్మల్ని ప్రాంగణం నుండి బయటకు పంపేశారా (లాకౌట్ చేశారా), లేదా మీరు ఇంకా లోపలే ఉండి ఖాళీ చేయాలనే బెదిరింపులను ఎదుర్కొంటున్నారా?",
        "ta": "நீங்கள் வளாகத்திலிருந்து வெளியேற்றப்பட்டுவிட்டீர்களா, அல்லது இன்னும் உள்ளேயே இருந்து வெளியேற்றும் அச்சுறுத்தல்களை எதிர்கொள்கிறீர்களா?",
        "kn": "ನಿಮ್ಮನ್ನು ಆವರಣದಿಂದ ಹೊರಹಾಕಲಾಗಿದೆಯೇ (ಲಾಕ್‌ಔಟ್ ಮಾಡಲಾಗಿದೆಯೇ), ಅಥವಾ ನೀವು ಇನ್ನೂ ಒಳಗಿದ್ದು ಹೊರಹಾಕುವ ಬೆದರಿಕೆಗಳನ್ನು ಎದುರಿಸುತ್ತಿದ್ದೀರಾ?",
        "bn": "আপনাকে কি প্রাঙ্গণ থেকে বের করে তালা ঝুলিয়ে দেওয়া হয়েছে, নাকি আপনি এখনও ভেতরে আছেন এবং উচ্ছেদের হুমকির মুখোমুখি হচ্ছেন?",
        "mr": "तुम्हाला जागेतून बाहेर काढले आहे का, की तुम्ही अजूनही आत आहात आणि बेदखल करण्याच्या धमक्यांचा सामना करत आहात?",
        "gu": "શું તમને જગ્યામાંથી બહાર કાઢી મૂકવામાં આવ્યા છે, કે તમે હજુ પણ અંદર છો અને ખાલી કરાવવાની ધમકીઓનો સામનો કરી રહ્યા છો?",
        "ml": "നിങ്ങളെ മുറിയിൽ നിന്ന് പുറത്താക്കിയോ, അതോ നിങ്ങൾ ഇപ്പോഴും അകത്തുണ്ടായിരിക്കെ ഒഴിപ്പിക്കൽ ഭീഷണി നേരിടുകയാണോ?",
    },
}


def localize_system_question(text: str, target_lang: str) -> str:
    """
    Ensure system-generated follow-up questions match the client's language.
    If the question is in English and target_lang is an Indic language,
    translate or map it to the corresponding localized version.
    """
    if not text or not target_lang or target_lang == "en":
        return text

    # If the text is already written in the target language's script, do not alter it
    detected = detect_language(text)
    if detected == target_lang:
        return text

    clean_text = text.split("\n*(")[0].strip().lower()

    if any(k in clean_text for k in ["private company", "government/psu", "government department"]):
        return LOCALIZED_SYSTEM_QUESTIONS["employment_type"].get(target_lang, text)
    if any(k in clean_text for k in ["state or city", "which state", "jurisdiction", "located in"]):
        return LOCALIZED_SYSTEM_QUESTIONS["jurisdiction"].get(target_lang, text)
    if any(k in clean_text for k in ["unpaid salary", "notice pay", "other dues", "pending salary", "dues owed"]):
        return LOCALIZED_SYSTEM_QUESTIONS["financial_dues"].get(target_lang, text)
    if any(k in clean_text for k in ["contract", "offer letter", "appointment letter"]):
        return LOCALIZED_SYSTEM_QUESTIONS["written_contract"].get(target_lang, text)
    if any(k in clean_text for k in ["locked out", "eviction threat", "inside the premises"]):
        return LOCALIZED_SYSTEM_QUESTIONS["tenancy_lockout"].get(target_lang, text)

    return text

