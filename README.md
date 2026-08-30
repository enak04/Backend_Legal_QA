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
  The AI determines that essential details (like the jurisdiction or employment type) are missing and generates a conversational follow-up question:
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
  Once all required details are conversationally extracted, the engine synthesizes the facts and queries the `Legal_QA` service to obtain the legal remedies:
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

### 5. Get Full Conversation State
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

### 6. Upload a Document Attachment
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

### 7. Download/Retrieve a Document Attachment
* **Endpoint:** `GET /api/conversations/{conversation_id}/files/{file_id}`
* **Description:** Downloads the raw binary of the uploaded file with original content-type and filename headers.
* **Response:** Binary file stream.

---

## Configuration

All settings are loaded from environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `LEGAL_QA_BASE_URL` | `http://localhost:8000` | Legal_QA service URL |
| `LEGAL_QA_TIMEOUT` | `120` | Request timeout (seconds) |
| `DATABASE_URL` | `sqlite:///./conversations.db` | Database URL. Supports SQLite (`sqlite:///...`) and MongoDB (`mongodb://...` or `mongodb+srv://...`) |
| `CORS_ORIGINS` | `http://localhost:3000,http://localhost:5173` | Allowed CORS origins |
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
