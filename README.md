# OGC-MCP PoC - GSoC 2026

Proof of Concept: Bridging OGC APIs and LLMs via MCP and LangGraph.

## What This Proves

An LLM can discover, submit, and asynchronously manage OGC Process jobs through a structured two-layer architecture without hardcoded workflow logic.

High-level flow:
- Layer 2 (`LangGraph`) decides which tool to call and manages state.
- Layer 1 (`FastMCP`) wraps pygeoapi OGC Process endpoints as callable tools.
- pygeoapi executes the process and returns status or final results.

## Project Structure

```text
ogc-mcp-poc/
|-- src/ogc_mcp/
|   |-- server.py
|   `-- agent.py
|-- tests/
|   |-- test_async_flow.py
|   `-- stress_test_prompts.py
|-- demo.py
|-- requirements.txt
`-- .env.example
```

## File Responsibilities

- `src/ogc_mcp/server.py`: Layer 1 MCP-style tool wrappers over pygeoapi endpoints.
- `src/ogc_mcp/agent.py`: Layer 2 LangGraph state machine, tool-call extraction, routing, and polling logic.
- `demo.py`: End-to-end demo using the real model and pygeoapi.
- `tests/test_async_flow.py`: Minimal deterministic proof that the async polling loop works.
- `tests/stress_test_prompts.py`: Larger stress harness for prompt robustness, tool use, edge cases, and mocked async flow.
- `requirements.txt`: Runtime dependencies.
- `.env.example`: Environment template for model and backend configuration.

## Setup

### 1. Prerequisites

- Python 3.11+
- Docker with pygeoapi running on `http://localhost:5000`
- Hugging Face token in `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`

### 2. Start pygeoapi

```bash
docker run -p 5000:80 geopython/pygeoapi:latest
```

Verify in the browser:

```text
http://localhost:5000/processes
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
```

Set:
- `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`
- `HF_MODEL_NAME` if you want to override the default
- `PYGEOAPI_BASE_URL` if your backend is not running at `http://localhost:5000`

## Run the Demo

```bash
python demo.py
```

### Example Output (Synchronous Run)

```text
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

## Async Polling Behavior

If the process returns a `jobID`, the agent will enter the polling loop:
- call `get_job_status` every `POLL_INTERVAL` seconds
- move to `SUCCESS` when status becomes `successful`
- call `get_job_result` only after success

If the process completes synchronously and `jobID` is `null`, the polling loop is skipped.

## Tests

This PoC includes two complementary test styles:
- a focused async-proof test for the LangGraph state machine
- a broader stress suite for prompt behavior, extraction robustness, and edge cases

### 1. Minimal Async Proof

Run:

```bash
python tests/test_async_flow.py
```

What it does:
- replaces the real LLM with fixed mock responses
- replaces the real backend calls with mocked tool results
- forces the exact lifecycle `execute_process -> get_job_status -> get_job_status -> get_job_result`
- verifies that polling actually happened before the final answer was returned

What it proves:
- the LangGraph graph wiring is correct
- the agent can move through `SUBMITTED -> POLLING -> SUCCESS`
- the final answer is produced only after the success path completes

Why this test is mocked:
- the built-in `hello-world` process usually completes synchronously in pygeoapi
- the PoC goal is to prove async orchestration without having to build a brand-new long-running process backend

### 2. Stress Prompt Suite

Run:

```bash
python tests/stress_test_prompts.py
```

What it does:
- runs a set of prompt categories against the agent
- captures `[DEBUG]` logs and `[Agent . ...]` transitions
- checks whether the agent hallucinates, skips tool calls, mishandles errors, or breaks sequential control
- includes one mocked async case inside the suite so the LangGraph polling path is still tested even if the live backend is synchronous

What it covers:
- discovery and introspection
- process listing
- sequential execution behavior
- extraction robustness when the model is chatty
- error handling for missing processes
- error handling for fake job IDs
- multi-step instructions
- mocked async LangGraph proof

How to interpret it:
- live prompt cases measure how reliable the real model is under pressure
- the mocked async case measures whether the orchestration layer itself is correct
- if the live cases fail but the mocked async case passes, the weak point is usually model behavior rather than LangGraph

## Architecture Notes

- Layer 1 (`FastMCP`): Thin async HTTP wrappers over pygeoapi. No process-specific logic.
- Layer 2 (`LangGraph`): A state machine that manages tool calls, polling, and final response.
- Tool calling: The model is instructed to emit a strict JSON tool call, and the agent includes extraction and fallback logic for chatty reasoning-model outputs.

## GSoC Proposal Context

This PoC demonstrates the core architectural claim: a universal MCP-to-OGC mapping layer plus a LangGraph orchestration layer can handle asynchronous job state management in a process-agnostic way.

The full project extends this with:
- dynamic process discovery and tool generation
- OGC Features, Maps, and Tiles API support
- a reusable mapping specification
