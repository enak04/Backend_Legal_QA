# Legal Assistance — Application Backend

Conversational application backend for legal assistance.  
Manages multi-turn conversations, collects relevant facts through an intake process, and communicates with the **Legal_QA** inference service.

## Architecture

```
Frontend
    ↓
Application Backend (:8080)        ← this project
    ↓ HTTP POST /predict
Legal_QA Service (:8000)           ← separate service
    ↓
HKG + PPO + DSSM + Reranking + GPT-4o
    ↓
Mode-specific legal response
```

This backend does **NOT** perform any ML inference.  It communicates with Legal_QA exclusively through its HTTP API.

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
    "mode": "actionable"
  }
  ```
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
    "message": "My boss hasn't paid my salary for 3 months."
  }
  ```

* **Response Scenario A (More facts required):**
  The engine determines that essential details (e.g. jurisdiction, employment contract type) are missing and generates a conversational follow-up question:
  ```json
  {
    "type": "follow_up",
    "conversation_id": "9f3b128c74de",
    "mode": "actionable",
    "message": "I'm sorry to hear that. Could you let me know which Indian state you are employed in, and if you work for a private company or a government department?",
    "timestamp": "2026-08-30T04:10:02.000000+00:00"
  }
  ```

* **Response Scenario B (Intake complete - Legal QA result):**
  Once all required details are extracted or the mode is direct, the engine synthesizes the facts and queries the `Legal_QA` service to obtain the legal remedies:
  ```json
  {
    "type": "final_answer",
    "conversation_id": "9f3b128c74de",
    "mode": "actionable",
    "answer": "Under the Payment of Wages Act, 1936...",
    "reasoning_chain": [
      "Step 1: Analyzed jurisdiction (Karnataka)",
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
    "message": "My boss hasn't paid my salary for 3 months."
  }
  ```
* **Response Content-Type:** `text/event-stream`
* **Description:** Streams response tokens and status events with low time-to-first-token.
* **Stream Event Formats:**
  ```
  data: {"type": "status", "status": "analyzing"}

  data: {"type": "token", "content": "Under the "}
  data: {"type": "token", "content": "Payment of "}
  data: {"type": "token", "content": "Wages Act... "}

  data: {"type": "complete", "data": { "type": "final_answer", "conversation_id": "...", "answer": "...", "reasoning_chain": [...], "sources": [...] }}
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

## Frontend Integration Guide

### 1. Typical Lifecycle in Frontend UI
1. **Mode Selection Screen:**
   - Call `GET /api/modes` to render mode choices (`Actionable`, `Informative`, `Readable`).
2. **Conversation Initialization:**
   - User selects a mode → Call `POST /api/conversations` with `{ "mode": "actionable" }`.
   - Store `conversation_id` in React/Vue state or URL params.
3. **Chat Interface:**
   - User enters message → Call `POST /api/conversations/{conversation_id}/messages` (or `.../stream`).
   - If response `type === "follow_up"`: Display the AI question in the chat and allow the user to reply.
   - If response `type === "final_answer"`: Display the synthesized `answer`, render `reasoning_chain` (as step-by-step accordions/cards), and show `sources` (retrieved legal cases/sections).
4. **File Attachments (Optional):**
   - User attaches PDF/evidence → Call `POST /api/conversations/{conversation_id}/files` with `FormData`.
5. **Session Resume / Reload:**
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

export interface ConversationCreated {
  conversation_id: string;
  mode: LegalMode;
  stage: string;
  created_at: string;
}

export interface FollowUpResponse {
  type: "follow_up";
  conversation_id: string;
  mode: LegalMode;
  message: string;
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
  reasoning_chain: string[];
  sources: LegalSource[];
  timestamp: string;
}

export type ChatMessageResponse = FollowUpResponse | FinalAnswerResponse;
```

---

## Configuration

All settings are loaded from environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `LEGAL_QA_BASE_URL` | `http://localhost:8000` | Legal_QA service URL |
| `LEGAL_QA_TIMEOUT` | `120` | Request timeout (seconds) |
| `DATABASE_URL` | `sqlite:///./conversations.db` | Database URL. Supports SQLite (`sqlite:///...`) and MongoDB (`mongodb://...` or `mongodb+srv://...`) |
| `CORS_ORIGINS` | `http://localhost:3000,http://localhost:5173` | Allowed CORS origins (Vite, React, Next.js) |
| `APP_HOST` | `0.0.0.0` | Server bind host |
| `APP_PORT` | `8080` | Server bind port |
| `OPENAI_API_KEY` | `None` | OpenAI API key for conversational intake |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI Model |

## Project Structure

```
Backend_Legal_QA/
├── app.py                    # FastAPI application & startup
├── config.py                 # Environment configuration
├── api/routes/
│   ├── conversations.py      # Conversation endpoints
│   ├── files.py              # File attachment upload/download routes
│   ├── modes.py              # Mode listing endpoint
│   └── health.py             # Health check
├── conversation/
│   ├── manager.py            # Central orchestrator
│   ├── state.py              # State mutation helpers
│   └── followup.py           # Follow-up engine & domain rules
├── legal_qa/
│   ├── client.py             # HTTP client for Legal_QA
│   └── query_builder.py      # Query construction from facts
├── database/
│   ├── models.py             # Data models (Pydantic)
│   └── repository.py         # Storage abstraction + SQLite & MongoDB impls
├── services/
│   ├── intake_service.py     # OpenAI conversational intake service
│   └── response_formatter.py # Response formatting
└── tests/                    # Test suite (pytest)
```
