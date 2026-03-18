# OGC-MCP PoC — GSoC 2026

> **Proof of Concept:** Bridging OGC APIs and LLMs via MCP & LangGraph

## What This Proves

An LLM can discover, submit, and **asynchronously manage** OGC Process jobs through a structured two-layer architecture — without any hardcoded workflow logic.

```
User prompt
    │
    ▼
┌─────────────────────────────────────┐
│  Layer 2 · LangGraph Agent          │
│  Gemini LLM drives the workflow     │
│  State: IDLE→SUBMITTED→POLLING      │
│              →SUCCESS / FAILED      │
└──────────────┬──────────────────────┘
               │  tool calls
               ▼
┌─────────────────────────────────────┐
│  Layer 1 · FastMCP Server           │
│  list_processes                     │
│  execute_process  (→ jobID)         │
│  get_job_status   (poll loop)       │
│  get_job_result                     │
└──────────────┬──────────────────────┘
               │  HTTP
               ▼
   pygeoapi Docker  (localhost:5000)
```

## Project Structure

```
ogc-mcp-poc/
├── src/ogc_mcp/
│   ├── server.py   # Layer 1: FastMCP server (4 OGC tools)
│   └── agent.py    # Layer 2: LangGraph state machine
├── demo.py         # End-to-end demo script
├── requirements.txt
└── .env.example
```

## Setup

### 1. Prerequisites

- Python 3.11+
- Docker with `pygeoapi` running on `localhost:5000`
- Google Gemini API key

### 2. Start pygeoapi

```bash
docker run -p 5000:80 geopython/pygeoapi:latest
```

Verify: http://localhost:5000/processes

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env and set your GOOGLE_API_KEY
```

## Run the Demo

```bash
python demo.py
```

### Expected Output

```
╔══════════════════════════════════════════════════════════════╗
║       GSoC PoC: OGC APIs × MCP × LangGraph                  ║
╚══════════════════════════════════════════════════════════════╝

USER  ▶  I want to run the 'hello-world' OGC process with the
         input name='GSoC Mentor'. Please execute it, wait for
         it to finish, and tell me the result.

[Agent · LLM]    Thinking... (phase=IDLE)
[Agent · Tool]   Calling: execute_process({"process_id": "hello-world", ...})
[Agent · State]  SUBMITTED — jobID=abc-123-xyz
[Agent · Poll]   Waiting 2s before next status check (attempt 1/20)...
[Agent · Tool]   Calling: get_job_status({"job_id": "abc-123-xyz"})
[Agent · State]  POLLING — status=running
[Agent · Poll]   Waiting 2s before next status check (attempt 2/20)...
[Agent · Tool]   Calling: get_job_status({"job_id": "abc-123-xyz"})
[Agent · State]  SUCCESS — job completed!
[Agent · Tool]   Calling: get_job_result({"job_id": "abc-123-xyz"})

AGENT ▶  The hello-world process completed successfully!
         Result: "Hello GSoC Mentor! This is a greeting from pygeoapi."

✅  Done in 8.3s
```

## Architecture Notes

- **Layer 1 (FastMCP):** Each tool is a thin async HTTP wrapper over pygeoapi. Zero process-specific logic — the same 4 tools work for any OGC process.
- **Layer 2 (LangGraph):** The state machine adds a `poll_wait` node between POLLING cycles to avoid hammering the server. The LLM decides *when* to poll and *when* to fetch results — no hardcoded logic.
- **Async-first:** All tool calls use `httpx.AsyncClient` and LangGraph's `ainvoke`.

## GSoC Proposal Context

This PoC demonstrates the core architectural claim: a universal MCP-to-OGC mapping layer plus a LangGraph orchestration layer can handle the hardest problem in geospatial AI workflows — asynchronous job state management — in a process-agnostic way.

The full GSoC project extends this with:
- Dynamic process discovery and tool generation (no hardcoded process IDs)
- OGC Features, Maps, and Tiles API support
- A reusable mapping specification