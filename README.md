# OGC-MCP PoC - GSoC 2026

Proof of Concept: Bridging OGC APIs and LLMs via MCP and LangGraph.

## What This Proves

An LLM can discover, submit, and asynchronously manage OGC Process jobs through a structured two-layer architecture without hardcoded workflow logic.

High-level flow:
- Layer 2 (LangGraph agent) decides which tool to call and manages state.
- Layer 1 (FastMCP tools) wraps pygeoapi OGC endpoints.
- pygeoapi executes the process and returns job status or results.

## Project Structure

```
ogc-mcp-poc/
├── src/ogc_mcp/
│   ├── server.py   # Layer 1: FastMCP tools for OGC Processes
│   └── agent.py    # Layer 2: LangGraph state machine
├── tests/
│   └── test_async_flow.py  # Mocked async polling test (no real HTTP/LLM calls)
├── demo.py         # End-to-end demo script
├── requirements.txt
└── .env.example
```

## File Responsibilities

- `src/ogc_mcp/server.py`: Layer 1 tools (list/execute/status/result) wrapped over pygeoapi HTTP endpoints.
- `src/ogc_mcp/agent.py`: Layer 2 LangGraph state machine, tool routing, and polling loop.
- `demo.py`: End-to-end demo using the real LLM and pygeoapi.
- `tests/test_async_flow.py`: Mocked async flow test that forces jobID -> polling -> success -> result.
- `requirements.txt`: Runtime dependencies.
- `.env.example`: Environment template for tokens and base URLs.

## Setup

### 1. Prerequisites

- Python 3.11+
- Docker with pygeoapi running on http://localhost:5000
- Hugging Face token (HF_TOKEN or HUGGINGFACE_HUB_TOKEN)

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
# Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN
# Optionally set HF_MODEL_NAME (default: meta-llama/Llama-3.1-8B-Instruct:sambanova)
```

## Run the Demo

```bash
python demo.py
```

### Example Output (synchronous run)

```
--------------------------------------------------------------
       GSoC PoC: OGC APIs x MCP x LangGraph
       Layer 1: FastMCP  |  Layer 2: LangGraph Agent
--------------------------------------------------------------

USER  ->  I want to run the 'hello-world' OGC process with the input name='GSoC Mentor'.

[Agent . LLM] Thinking... (phase=IDLE)
[Agent . LLM] Wants to call: execute_process({"process_id": "hello-world", "inputs": {"name": "GSoC Mentor"}})
[Agent . Tool] Calling: execute_process({"process_id": "hello-world", "inputs": {"name": "GSoC Mentor"}})
[Agent . State] SUCCESS - synchronous execution!

[Agent . LLM] Thinking... (phase=SUCCESS)
[Agent . LLM] Final answer ready.

AGENT ->  Hello GSoC Mentor!
```

### Async Polling Behavior

If the process returns a jobID, the agent will enter the polling loop:
- call get_job_status every POLL_INTERVAL seconds
- move to SUCCESS when status is "successful"
- fetch results via get_job_result

If the process completes synchronously (jobID is null), the polling loop is skipped.

## Tests

Run the async flow test (mocked, no network calls):

```bash
python tests/test_async_flow.py
```

Why mocked: this keeps the PoC lightweight while still proving the async polling loop and state transitions without requiring a real long-running OGC process.

## Architecture Notes

- Layer 1 (FastMCP): Thin async HTTP wrappers over pygeoapi. No process-specific logic.
- Layer 2 (LangGraph): A state machine that manages tool calls, polling, and final response.
- Tool calling: The model is instructed to emit a strict JSON tool call; the agent extracts JSON even if the model includes extra text.

## GSoC Proposal Context

This PoC demonstrates the core architectural claim: a universal MCP-to-OGC mapping layer plus a LangGraph orchestration layer can handle asynchronous job state management in a process-agnostic way.

The full project extends this with:
- Dynamic process discovery and tool generation
- OGC Features, Maps, and Tiles API support
- A reusable mapping specification
