# Legal Assistance â€” Application Backend

Conversational application backend for legal assistance with native **Multilingual Indic Language Support** (Telugu, Hindi, Tamil, Kannada, etc.) and **Voice Consultation** (Speech-to-Text).  
Manages multi-turn conversations, collects relevant facts through an intake process, and communicates with the **Legal_QA** inference service.

## Key Capabilities

- **Intelligent Intake Flow:** Dynamic multi-turn fact collection and domain scoping across 8 legal domains (employment, consumer, property, criminal, cybercrime, family, contract, government).
- **Multilingual Indic Support:** Native Unicode script and romanized transliteration (Teluglish, Hinglish) detection. Dynamic mid-chat language switching, dual-track statutory grounding (official English legal nomenclature + regional language explanations).
- **Voice & Audio Consultation:** Speech-to-text integration via IIT Madras Indic ASR (`asr.iitm.ac.in`) with automatic OpenAI Whisper fallback.
- **Real-time Streaming:** Server-Sent Events (SSE) for low time-to-first-token token streaming.

## Architecture

```
Frontend (Text / Voice Audio)
    â†“
Application Backend (:8080)        â†� this project
  â”œâ”€â”€ Multilingual Engine (Indic Unicode + Transliteration Detection)
  â”œâ”€â”€ Voice Service (IIT Madras Indic ASR + OpenAI Whisper)
  â””â”€â”€ Conversational Intake & Grounded Generator
    â†“ HTTP POST /predict
Legal_QA Service (:8000)           â†� separate service
    â†“
HKG + PPO + DSSM + Reranking + GPT-4o
    â†“
Mode-specific legal response (in User's Selected/Spoken Language)
```

This backend communicates with Legal_QA for legal retrieval and HKG inference, while orchestrating intake, speech transcription, and language synthesis.

## Three Modes

| Mode | Goal |
|------|------|
| **Actionable** | Guide the user toward a legal remedy. Collects facts via conversational intake before querying Legal_QA. |
| **Informative** | Help the user understand a legal issue. Direct answers with minimal follow-up. |
| **Readable** | Make legal information easy to understand. No follow-up questions. |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and configure environment
cp .env.example .env
# Edit .env if needed (Legal_QA URL, port, etc.)

# 3. Start the backend
python app.py
# or: uvicorn app:app --host 0.0.0.0 --port 8080

# 4. Run tests
python -m pytest tests/ -v
```

## API Endpoints & Request/Response Flow

Below is the detailed specification of all available endpoints. You can also view the auto-generated documentation at `http://localhost:8080/docs` (Swagger UI) when the application is running.

---

### 1. Health Check
* **Endpoint:** `GET /health`
* **Description:** Verifies that the server is up and running.
* **Response (200 OK):**
  ```json
  {
    "status": "ok"
  }
  ```

---

### 2. List Available Modes
* **Endpoint:** `GET /api/modes`
* **Description:** Retrieves the list of conversational legal assistance modes.
* **Response (200 OK):**
  ```json
  [
    {
      "id": "actionable",
      "name": "Actionable",
      "description": "Guides you toward an appropriate legal remedy or action..."
    },
    {
      "id": "informative",
      "name": "Informative",
      "description": "Helps you understand a legal issue, law, section..."
    },
    {
      "id": "readable",
      "name": "Readable",
      "description": "Makes legal information easier for a normal user to understand..."
    }
  ]
  ```

---

### 3. Create a New Conversation
* **Endpoint:** `POST /api/conversations`
* **Request Headers:** `Content-Type: application/json`
* **Request Body:**
  ```json
  {
    "mode": "actionable",
    "language": "te"
  }
  ```
  *(Note: `language` is optional, defaults to `"en"`. Supports `"te"`, `"hi"`, `"ta"`, `"kn"`, `"mr"`, `"bn"`, `"gu"`, `"ml"`, `"auto"`, etc.)*
* **Response (200 OK):**
  ```json
  {
    "conversation_id": "9f3b128c74de",
    "mode": "actionable",
    "stage": "initial",
    "created_at": "2026-08-30T04:09:50.000000+00:00"
  }
  ```

---

### 4. Send Message (Intake & QA Flow)
* **Endpoint:** `POST /api/conversations/{conversation_id}/messages`
* **Request Headers:** `Content-Type: application/json`
* **Request Body:**
  ```json
  {
    "message": "à°®à°¾ à°¯à°œà°®à°¾à°¨à°¿ 3 à°¨à±†à°²à°²à±�à°—à°¾ à°œà±€à°¤à°‚ à°‡à°µà±�à°µà°²à±‡à°¦à±�.",
    "language": "te"
  }
  ```
  *(Note: `language` is optional. If omitted or set to `"auto"`, the system automatically detects Telugu, Hindi, Tamil, Hinglish/Teluglish, etc. Users can switch languages at any turn.)*

* **Response Scenario A (More facts required â€” Regional Follow-Up):**
  The engine determines that essential details are missing and responds in the client's language:
  ```json
  {
    "type": "follow_up",
    "conversation_id": "9f3b128c74de",
    "mode": "actionable",
    "message": "à°®à±€ à°¸à°®à°¸à±�à°¯ à°µà°¿à°šà°¾à°°à°•à°°à°‚. à°®à±€à°°à±� à°� à°­à°¾à°°à°¤à±€à°¯ à°°à°¾à°·à±�à°Ÿà±�à°°à°‚à°²à±‹ à°ªà°¨à°¿à°šà±‡à°¸à±�à°¤à±�à°¨à±�à°¨à°¾à°°à±�, à°®à°°à°¿à°¯à±� à°®à±€ à°•à°‚à°ªà±†à°¨à±€ à°ªà±�à°°à±ˆà°µà±‡à°Ÿà±� à°¸à°‚à°¸à±�à°¥à°¾ à°²à±‡à°• à°ªà±�à°°à°­à±�à°¤à±�à°µ à°µà°¿à°­à°¾à°—à°®à°¾?",
    "language": "te",
    "timestamp": "2026-08-30T04:10:02.000000+00:00"
  }
  ```

* **Response Scenario B (Intake complete â€” Grounded Multilingual Legal Answer):**
  Synthesizes facts into canonical legal queries, executes legal research, and provides the mode-specific remedy in the client's chosen language with dual-track statutory naming:
  ```json
  {
    "type": "final_answer",
    "conversation_id": "9f3b128c74de",
    "mode": "actionable",
    "language": "te",
    "answer": "Payment of Wages Act, 1936 (à°µà±‡à°¤à°¨ à°šà±†à°²à±�à°²à°¿à°‚à°ªà±� à°šà°Ÿà±�à°Ÿà°‚) à°¸à±†à°•à±�à°·à°¨à±� 15 à°ªà±�à°°à°•à°¾à°°à°‚...",
    "reasoning_chain": [
      "Step 1: Analyzed jurisdiction (Telangana / Andhra Pradesh)",
      "Step 2: Identified applicable labor legislation..."
    ],
    "sources": [
      {
        "question": "What remedies exist for unpaid wages?",
        "answer": "Under Section 15 of the Payment of Wages Act..."
      }
    ],
    "timestamp": "2026-08-30T04:10:05.000000+00:00"
  }
  ```

---

### 5. Send Message with Streaming (Server-Sent Events)
* **Endpoint:** `POST /api/conversations/{conversation_id}/messages/stream`
* **Request Headers:** `Content-Type: application/json`
* **Request Body:**
  ```json
  {
    "message": "Mere boss ne 3 mahine se salary nahi di.",
    "language": "hi"
  }
  ```
* **Response Content-Type:** `text/event-stream`
* **Description:** Streams response tokens and status events with low time-to-first-token.
* **Stream Event Formats:**
  ```
  data: {"type": "status", "status": "analyzing"}

  data: {"type": "token", "content": "Payment of "}
  data: {"type": "token", "content": "Wages Act à¤•à¥‡ "}
  data: {"type": "token", "content": "à¤¤à¤¹à¤¤... "}

  data: {"type": "complete", "data": { "type": "final_answer", "conversation_id": "...", "answer": "...", "language": "hi", "reasoning_chain": [...], "sources": [...] }}
  ```

---

### 6. Get Full Conversation State
* **Endpoint:** `GET /api/conversations/{conversation_id}`
* **Description:** Retrieves the complete state of the conversation, including metadata, conversation history, cumulative extracted facts, and any QA results.
* **Response (200 OK):**
  ```json
  {
    "conversation_id": "9f3b128c74de",
    "mode": "actionable",
    "stage": "answered",
    "facts": {
      "detected_domain": "employment_wage",
      "state": "Karnataka",
      "employment_type": "private",
      "duration_or_dates": "3 months"
    },
    "messages": [
      {
        "role": "user",
        "content": "My boss has not paid my salary...",
        "timestamp": "2026-08-30T04:10:00Z"
      }
    ],
    "legal_qa_results": [...],
    "files": [],
    "created_at": "2026-08-30T04:09:50Z",
    "updated_at": "2026-08-30T04:10:05Z"
  }
  ```

---

### 7. Upload a Document Attachment
* **Endpoint:** `POST /api/conversations/{conversation_id}/files`
* **Request Content-Type:** `multipart/form-data`
* **Form Parameters:**
  * `file`: (Binary File) e.g., `contract.pdf`
* **Response (200 OK):**
  ```json
  {
    "file_id": "c1f7b8d9e2a3",
    "filename": "contract.pdf",
    "content_type": "application/pdf",
    "size_bytes": 1048576,
    "uploaded_at": "2026-08-30T04:15:00.000000+00:00"
  }
  ```

---

### 8. Download/Retrieve a Document Attachment
* **Endpoint:** `GET /api/conversations/{conversation_id}/files/{file_id}`
* **Description:** Downloads the raw binary of the uploaded file with original content-type and filename headers.
* **Response:** Binary file stream.

---

### 9. Transcribe Audio (Speech-to-Text)
* **Endpoint:** `POST /api/audio/transcribe`
* **Request Content-Type:** `multipart/form-data`
* **Form Parameters:**
  * `file`: (Binary Audio File) e.g., `.wav`, `.mp3`, `.m4a`, `.webm`, `.ogg`
  * `language`: *(Optional)* ISO language code (e.g., `"te"`, `"hi"`, `"ta"`, `"kn"`, `"mr"`, `"bn"`, `"gu"`, `"ml"`, `"auto"`)
  * `provider`: *(Optional)* Preferred ASR engine: `"iitm"` (IIT Madras Indic ASR, default) or `"whisper"` (OpenAI Whisper)
* **Description:** Transcribes spoken audio into native Indic text. IIT Madras Indic ASR is specifically calibrated for regional Indian languages, with automatic fallback to OpenAI Whisper.
* **Response (200 OK):**
  ```json
  {
    "text": "à°¨à°¾à°•à±� à°®à±‚à°¡à±� à°¨à±†à°²à°²à±�à°—à°¾ à°œà±€à°¤à°‚ à°°à°¾à°²à±‡à°¦à±�, à°�à°®à°¿ à°šà±‡à°¯à°¾à°²à°¿?",
    "language": "te",
    "provider": "iitm",
    "filename": "query.wav"
  }
  ```

---

### 10. Send Audio Message (Voice Consultation Turn)
* **Endpoint:** `POST /api/conversations/{conversation_id}/audio`
* **Request Content-Type:** `multipart/form-data`
* **Form Parameters:**
  * `file`: (Binary Audio File) The user's spoken voice recording
  * `language`: *(Optional)* Expected language code (e.g., `"te"`, `"hi"`, `"auto"`). If omitted, language is automatically detected from the transcription.
  * `provider`: *(Optional)* Preferred ASR engine: `"iitm"` or `"whisper"`
* **Description:** Performs full end-to-end voice consultation: transcribes spoken audio, processes the legal intake or legal research turn in the user's spoken language, and returns both the transcribed query and the legal response.
* **Response (200 OK):**
  ```json
  {
    "type": "follow_up",
    "conversation_id": "9f3b128c74de",
    "mode": "actionable",
    "language": "te",
    "transcription": "à°¨à°¾à°•à±� à°®à±‚à°¡à±� à°¨à±†à°²à°²à±�à°—à°¾ à°œà±€à°¤à°‚ à°°à°¾à°²à±‡à°¦à±�, à°�à°®à°¿ à°šà±‡à°¯à°¾à°²à°¿?",
    "transcription_provider": "iitm",
    "message": "à°®à±€à°°à±� à°� à°­à°¾à°°à°¤à±€à°¯ à°°à°¾à°·à±�à°Ÿà±�à°°à°‚à°²à±‹ à°ªà°¨à°¿à°šà±‡à°¸à±�à°¤à±�à°¨à±�à°¨à°¾à°°à±�, à°®à°°à°¿à°¯à±� à°®à±€ à°•à°‚à°ªà±†à°¨à±€ à°ªà±�à°°à±ˆà°µà±‡à°Ÿà±� à°¸à°‚à°¸à±�à°¥à°¾ à°²à±‡à°• à°ªà±�à°°à°­à±�à°¤à±�à°µ à°µà°¿à°­à°¾à°—à°®à°¾?",
    "timestamp": "2026-08-30T04:10:02.000000+00:00"
  }
  ```

---

## Multilingual & Voice Capabilities

### Supported Languages & Dialects
- **Telugu (`te`):** Full native Telugu script & Teluglish (Telugu in Roman alphabet)
- **Hindi (`hi`):** Devanagari script & Hinglish
- **Tamil (`ta`), Kannada (`kn`), Malayalam (`ml`):** South Indian regional scripts
- **Bengali (`bn`), Marathi (`mr`), Gujarati (`gu`), Odia (`or`), Punjabi (`pa`):** North/East/West Indic scripts
- **English (`en`):** Standard English
- **Code-Mixing & Transliteration:** Automatically recognizes Romanized transliterations (e.g. *"salary nahi mila"*, *"jeetham raledu"*).
- **Dynamic Mid-Chat Switching:** A user can start in English, switch to Telugu on turn 2, and switch to Hindi on turn 3. The system matches the client's current language while retaining cross-turn legal facts.
- **Dual-Track Statutory Grounding:** Acts and sections are referenced with both English legal naming and regional language explanations (e.g. *Payment of Wages Act, 1936 (à°µà±‡à°¤à°¨ à°šà±†à°²à±�à°²à°¿à°‚à°ªà±� à°šà°Ÿà±�à°Ÿà°‚)*).

---

## Frontend Integration Guide

### 1. Typical Lifecycle in Frontend UI
1. **Mode & Language Selection Screen:**
   - Call `GET /api/modes` to render mode choices (`Actionable`, `Informative`, `Readable`).
   - (Optional) Provide language selector dropdown (`"en"`, `"te"`, `"hi"`, etc., or `"auto"`).
2. **Conversation Initialization:**
   - User selects mode â†’ Call `POST /api/conversations` with `{ "mode": "actionable", "language": "te" }`.
   - Store `conversation_id` in frontend state.
3. **Text Chat Interface:**
   - User types message â†’ Call `POST /api/conversations/{conversation_id}/messages` (or `.../stream`).
   - If response `type === "follow_up"`: Display the AI question in the chat and prompt the user for the next reply.
   - If response `type === "final_answer"`: Display `answer`, collapsible `reasoning_chain`, and citation `sources`.
4. **Voice / Audio Input Interface:**
   - User taps Microphone icon to record audio via browser `MediaRecorder` API:
     ```javascript
     const formData = new FormData();
     formData.append("file", audioBlob, "voice_input.webm");
     formData.append("language", "te"); // or "auto"
     
     const response = await fetch(`/api/conversations/${conversationId}/audio`, {
       method: "POST",
       body: formData
     });
     const data = await response.json();
     // data.transcription contains user's spoken words
     // data.message or data.answer contains AI advocate response
     ```
5. **File Attachments (Optional):**
   - User attaches PDF/evidence â†’ Call `POST /api/conversations/{conversation_id}/files` with `FormData`.
6. **Session Resume / Reload:**
   - Call `GET /api/conversations/{conversation_id}` to repopulate message history, extracted facts, and attachments.

---

### 2. TypeScript Data Types
Copy these interfaces into your frontend project:

```typescript
export type LegalMode = "actionable" | "informative" | "readable";

export interface ModeInfo {
  id: LegalMode;
  name: string;
  description: string;
}

export interface CreateConversationPayload {
  mode: LegalMode;
  language?: string; // e.g. "te", "hi", "en", "auto"
}

export interface ConversationCreated {
  conversation_id: string;
  mode: LegalMode;
  language?: string;
  stage: string;
  created_at: string;
}

export interface SendMessagePayload {
  message: string;
  language?: string; // optional override or "auto"
}

export interface FollowUpResponse {
  type: "follow_up";
  conversation_id: string;
  mode: LegalMode;
  message: string;
  language?: string;
  transcription?: string; // present when sent via voice endpoint
  transcription_provider?: "iitm" | "whisper";
  timestamp: string;
}

export interface LegalSource {
  question: string;
  answer: string;
}

export interface FinalAnswerResponse {
  type: "final_answer";
  conversation_id: string;
  mode: LegalMode;
  answer: string;
  language?: string;
  transcription?: string; // present when sent via voice endpoint
  transcription_provider?: "iitm" | "whisper";
  reasoning_chain: string[];
  sources: LegalSource[];
  timestamp: string;
}

export interface TranscriptionResponse {
  text: string;
  language: string;
  provider: "iitm" | "whisper";
  filename: string;
}

export type ChatMessageResponse = FollowUpResponse | FinalAnswerResponse;
```

---

## Configuration

All settings are loaded from environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `LEGAL_QA_BASE_URL` | `http://localhost:8000` | Legal_QA inference service URL |
| `LEGAL_QA_TIMEOUT` | `120` | Request timeout (seconds) |
| `DATABASE_URL` | `sqlite:///./conversations.db` | Database URL. Supports SQLite (`sqlite:///...`) and MongoDB (`mongodb://...` or `mongodb+srv://...`) |
| `CORS_ORIGINS` | `http://localhost:3000,http://localhost:5173` | Allowed CORS origins (Vite, React, Next.js) |
| `APP_HOST` | `0.0.0.0` | Server bind host |
| `APP_PORT` | `8080` | Server bind port |
| `OPENAI_API_KEY` | `None` | OpenAI API key for conversational intake & Whisper ASR fallback |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI Model for intake and synthesis |
| `IITM_ASR_ENDPOINT` | `https://asr.iitm.ac.in/asr/v1/recognize/` | IIT Madras Indic Speech-to-Text API |

## Project Structure

```
Backend_Legal_QA/
├── app.py                    # FastAPI application & startup
├── config.py                 # Environment configuration
├── api/routes/
│   ├── conversations.py      # Conversation text & stream endpoints
│   ├── audio.py              # Audio transcription & voice consultation endpoints
│   ├── files.py              # File attachment upload/download routes
│   ├── modes.py              # Mode listing endpoint
│   └── health.py             # Health check
├── conversation/
│   ├── manager.py            # Central orchestrator with dynamic language tracking
│   ├── state.py              # Universal Case State mutation helpers
│   ├── followup.py           # Follow-up engine & domain rules
│   └── cases/models.py       # UniversalCaseState & FactItem models
├── legal_qa/
│   ├── client.py             # HTTP client for Legal_QA
│   ├── grounded_generator.py # Dual-track grounded legal answer synthesizer
│   └── query_builder.py      # Query construction from facts
├── database/
│   ├── models.py             # Data models (Pydantic) with language field
│   └── repository.py         # Storage abstraction + SQLite & MongoDB impls
├── services/
│   ├── intake_service.py     # OpenAI conversational intake & domain scoping
│   ├── language.py           # Indic Unicode & transliteration language detector
│   ├── voice_service.py      # Speech-to-text (IIT Madras Indic ASR + Whisper)
│   └── response_formatter.py # Response formatting
└── tests/                    # Comprehensive test suite (127 tests passing)
    ├── test_audio.py         # Voice transcription & audio endpoints tests
    └── test_multilingual.py  # Language detection & dynamic switching tests
```
