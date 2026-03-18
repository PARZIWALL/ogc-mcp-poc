# GSoC PoC: Bridging OGC APIs and LLMs via MCP & LangGraph

## Background

The GSoC project (org: OSGeo / pygeoapi) aims to expose OGC APIs as LLM-callable tools via a **Model Context Protocol (MCP)** server, orchestrated by a **LangGraph** state machine. This PoC targets the hardest problem in the space: **asynchronous job management** — where an LLM submits a long-running OGC Process and must wait for the result through a structured workflow.

The backend is a **pygeoapi** instance already running in Docker, exposing an OGC Processes endpoint.

---

## Two-Layer Architecture

```
User prompt
    │
    ▼
┌─────────────────────────────────────┐
│  Layer 2 · LangGraph Agent          │
│  - Gemini LLM decides tool calls    │
│  - State: IDLE→SUBMITTED→POLLING→  │
│           SUCCESS / FAILED          │
└──────────────┬──────────────────────┘
               │  MCP tool calls
               ▼
┌─────────────────────────────────────┐
│  Layer 1 · FastMCP Server           │
│  - list_processes                   │
│  - execute_process (→ jobID)        │
│  - get_job_status                   │
│  - get_job_result                   │
└──────────────┬──────────────────────┘
               │  HTTP
               ▼
   pygeoapi Docker  (localhost:5000)
```

---

## Proposed Changes

### Project Root — New Files

#### [NEW] `pyproject.toml` (or `requirements.txt`)

Dependencies:
- `fastmcp` — MCP server framework
- `langgraph` — state machine orchestration  
- `langchain-google-genai` — Gemini LLM binding
- `httpx` — async HTTP client for pygeoapi calls
- `python-dotenv` — `.env` for API keys

#### [NEW] `.env.example`

```
GOOGLE_API_KEY=your_gemini_api_key_here
PYGEOAPI_BASE_URL=http://localhost:5000
```

---

### Layer 1 — MCP Server

#### [NEW] `src/ogc_mcp/server.py`

A **FastMCP** server that exposes these four tools:

| Tool | OGC Endpoint | Description |
|---|---|---|
| `list_processes` | `GET /processes` | Returns all available process IDs and summaries |
| `execute_process` | `POST /processes/{id}/execution` | Submits job, returns `jobID` |
| `get_job_status` | `GET /jobs/{jobID}` | Returns `status` (accepted / running / successful / failed) |
| `get_job_result` | `GET /jobs/{jobID}/results` | Returns final results when `status=successful` |

Each tool is a thin async HTTP wrapper over `pygeoapi`.

---

### Layer 2 — LangGraph Orchestration Agent

#### [NEW] `src/ogc_mcp/agent.py`

A **LangGraph** `StateGraph` with these nodes & edges:

```
[__start__]
     │
     ▼
 [call_llm] ──── no tool call ──► [__end__]
     │
     │ tool call
     ▼
 [execute_tool] ─── not a job submission ──► [call_llm]
     │
     │ job submitted (jobID in result)
     ▼
 [poll_status] ──── still running ──► [poll_status]  (loop with sleep)
     │
     │ successful
     ▼
 [fetch_result] ──► [call_llm] ──► [__end__]
```

**State schema:**
```python
class AgentState(TypedDict):
    messages: list        # conversation history
    job_id: str | None    # active job being tracked
    poll_count: int       # prevent infinite loops
```

---

### Demo Entry Point

#### [NEW] `demo.py`

A simple CLI script that:
1. Starts the MCP server in a background thread
2. Initialises the LangGraph agent
3. Sends the prompt: *"Run the `hello-world` process with name='GSoC Mentor' and return the result"*
4. Prints each LangGraph state transition with timestamps to show the async polling loop

---

### Documentation

#### [MODIFY] [README.md](file:///D:/PROJECTS/odc-mcp-poc/ogc-mcp-poc/README.md)

Full setup guide, architecture diagram (ASCII), and demo instructions.

---

## Verification Plan

### Prerequisites
- Docker running with pygeoapi on `localhost:5000`
- `GOOGLE_API_KEY` set in `.env`
- Python 3.11+

### Automated / Script Tests

```bash
# 1. Confirm pygeoapi is healthy
curl http://localhost:5000/processes | python -m json.tool

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test MCP server tools in isolation
python -m pytest tests/ -v
```

> **Note:** `tests/test_server.py` will be written to mock `httpx` and verify each of the 4 MCP tools returns the correct shape.

### End-to-End Demo

```bash
# Run the full demo
python demo.py
```

Expected terminal output shows:
```
[State: IDLE]        User: "Run hello-world for GSoC Mentor"
[State: SUBMITTED]   Tool: execute_process → jobID=abc-123
[State: POLLING]     Tool: get_job_status  → running
[State: POLLING]     Tool: get_job_status  → running
[State: SUCCESS]     Tool: get_job_result  → "Hello GSoC Mentor!"
[Agent Final]        LLM: "The process completed. Result: Hello GSoC Mentor!"
```

### Manual Visual Verification

1. Open browser to `http://localhost:5000/processes` — confirm processes listed
2. Run `python demo.py` — observe state transitions printed to terminal
3. Confirm final LLM answer includes the `hello-world` output string
