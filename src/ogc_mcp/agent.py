"""
Layer 2: LangGraph Orchestration Agent
Uses Hugging Face Inference Providers + LangGraph for state management.

State machine: IDLE -> SUBMITTED -> POLLING -> SUCCESS / FAILED
"""

import asyncio
import json
import os
import re
from typing import Literal

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from huggingface_hub import InferenceClient
from typing_extensions import Annotated, TypedDict

from ogc_mcp.server import execute_process, get_job_result, get_job_status, list_processes

load_dotenv()

# Hugging Face Inference Providers setup
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct:sambanova")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

SYSTEM_PROMPT = """\
You are a geospatial assistant connected to an OGC Processes API.

You have access to these tools:
- list_processes(): Lists all available processes on the OGC server.
- execute_process(process_id, inputs): Submits a process for execution. 'process_id' is a string, 'inputs' is a JSON object.
- get_job_status(job_id): Checks the status of a submitted job.
- get_job_result(job_id): Retrieves the result of a completed job.

IMPORTANT RULES:
1. When you need to call a tool, you MUST respond with ONLY a JSON object in this exact format (no other text before or after):
{"tool_call": "<tool_name>", "args": {<arguments>}}

2. When you have the final answer for the user, respond with plain text only (no JSON).

3. Never make up or fabricate results. Always use the tools and wait for real data.

4. Follow this workflow for executing a process:
   Step 1: Call execute_process with the process_id and inputs
   Step 2: If you get a jobID back, call get_job_status with that jobID
   Step 3: If status is "successful", call get_job_result with the jobID
   Step 4: Present the result to the user in plain text

Example tool call:
{"tool_call": "execute_process", "args": {"process_id": "hello-world", "inputs": {"name": "Test"}}}
"""

POLL_INTERVAL = 2   # seconds between status polls
MAX_POLLS = 20      # safety limit

# Client once at import time
_client = InferenceClient(api_key=HF_TOKEN)


class Message(TypedDict):
    role: str      # "user" | "assistant" | "tool"
    content: str   # serialized text or JSON


class AgentState(TypedDict):
    messages: Annotated[list[Message], lambda a, b: a + b]
    job_id: str | None
    poll_count: int
    phase: str      # IDLE | SUBMITTED | POLLING | SUCCESS | FAILED
    final_answer: str | None


def _hf_call(history: list[Message]) -> str:
    """Send the conversation history to HF Inference and return its text response."""
    if not HF_TOKEN:
        return "Missing HF_TOKEN/HUGGINGFACE_HUB_TOKEN in environment."

    chat = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        if role == "tool":
            # Feed tool results as user messages so the model sees the data
            chat.append({"role": "user", "content": f"Tool result:\n{content}"})
        elif role == "assistant":
            chat.append({"role": "assistant", "content": content})
        else:
            # user messages
            chat.append({"role": "user", "content": content})

    print(f"\n[DEBUG] Sending {len(chat)} messages to HF model")
    for i, m in enumerate(chat):
        preview = m["content"][:120].replace("\n", "\\n")
        print(f"  [{i}] {m['role']}: {preview}...")

    completion = _client.chat.completions.create(
        model=HF_MODEL_NAME,
        messages=chat,
        max_tokens=512,
        temperature=0.1,  # Low temperature for more deterministic tool calls
    )
    response = completion.choices[0].message.content.strip()
    print(f"\n[DEBUG] Model response: {response[:300]}")
    return response


def _extract_tool_call(text: str) -> dict | None:
    """Extract the first JSON tool_call object from a mixed response."""
    # Try to find a JSON object with tool_call and args
    # First, try the entire text as JSON
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict) and "tool_call" in obj and "args" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    # Try to find JSON in code blocks
    code_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if code_block:
        try:
            obj = json.loads(code_block.group(1))
            if isinstance(obj, dict) and "tool_call" in obj and "args" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    # Scan for embedded JSON objects
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "tool_call" in obj and "args" in obj:
            return obj
    return None


async def call_llm(state: AgentState) -> dict:
    """Ask the HF model what to do next."""
    print(f"\n[Agent . LLM] Thinking... (phase={state['phase']})")
    response_text = await asyncio.to_thread(_hf_call, state["messages"])

    data = _extract_tool_call(response_text)
    if data:
        print(f"[Agent . LLM] Wants to call: {data['tool_call']}({data['args']})")
        return {
            "messages": [{
                "role": "assistant",
                "content": json.dumps({
                    "tool_call": data["tool_call"],
                    "args": data["args"],
                }),
            }]
        }

    print("[Agent . LLM] Final answer ready.")
    return {
        "messages": [{"role": "assistant", "content": response_text}],
        "final_answer": response_text,
    }


async def execute_tools(state: AgentState) -> dict:
    """Execute the tool call the LLM requested."""
    last_msg = state["messages"][-1]
    call_data = json.loads(last_msg["content"])
    tool_name = call_data["tool_call"]
    tool_args = call_data["args"]

    print(f"[Agent . Tool] Calling: {tool_name}({json.dumps(tool_args)})")

    new_job_id = state.get("job_id")
    new_phase = state.get("phase", "IDLE")

    try:
        if tool_name == "list_processes":
            result = await list_processes()
        elif tool_name == "execute_process":
            result = await execute_process(**tool_args)
            if result.get("jobID"):
                new_job_id = result["jobID"]
                new_phase = "SUBMITTED"
                print(f"[Agent . State] SUBMITTED - jobID={new_job_id}")
            elif result.get("status") == "successful":
                new_phase = "SUCCESS"
                print("[Agent . State] SUCCESS - synchronous execution!")
        elif tool_name == "get_job_status":
            result = await get_job_status(**tool_args)
            status = result.get("status", "unknown")
            if status == "successful":
                new_phase = "SUCCESS"
                print("[Agent . State] SUCCESS - job completed!")
            elif status == "failed":
                new_phase = "FAILED"
                print("[Agent . State] FAILED.")
            else:
                new_phase = "POLLING"
                print(f"[Agent . State] POLLING - status={status}")
        elif tool_name == "get_job_result":
            result = await get_job_result(**tool_args)
            new_phase = "SUCCESS"
        else:
            result = {"error": f"Unknown tool: {tool_name}"}
    except Exception as exc:
        result = {"error": str(exc)}
        print(f"[Agent . Error] {tool_name} raised: {exc}")

    new_poll_count = state.get("poll_count", 0) + (1 if tool_name == "get_job_status" else 0)

    return {
        "messages": [{"role": "tool", "content": json.dumps(result)}],
        "job_id": new_job_id,
        "phase": new_phase,
        "poll_count": new_poll_count,
    }


async def poll_wait(state: AgentState) -> dict:
    """Brief sleep between poll attempts."""
    attempt = state.get("poll_count", 0)
    print(f"[Agent . Poll] Waiting {POLL_INTERVAL}s... (attempt {attempt}/{MAX_POLLS})")
    await asyncio.sleep(POLL_INTERVAL)
    return {
        "messages": [{
            "role": "user",
            "content": f"Please check the job status for jobID={state['job_id']} again.",
        }]
    }


def route_after_llm(state: AgentState) -> Literal["execute_tools", "__end__"]:
    last = state["messages"][-1]
    try:
        data = json.loads(last["content"])
        if "tool_call" in data:
            return "execute_tools"
    except (json.JSONDecodeError, TypeError):
        pass
    return "__end__"


def route_after_tools(state: AgentState) -> Literal["poll_wait", "call_llm"]:
    phase = state.get("phase", "IDLE")
    poll_count = state.get("poll_count", 0)
    if phase in ("SUBMITTED", "POLLING") and poll_count < MAX_POLLS:
        return "poll_wait"
    return "call_llm"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("call_llm", call_llm)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("poll_wait", poll_wait)

    graph.add_edge(START, "call_llm")

    graph.add_conditional_edges("call_llm", route_after_llm, {
        "execute_tools": "execute_tools",
        "__end__": END,
    })
    graph.add_conditional_edges("execute_tools", route_after_tools, {
        "poll_wait": "poll_wait",
        "call_llm": "call_llm",
    })
    graph.add_edge("poll_wait", "call_llm")

    return graph.compile()


async def run_agent(user_prompt: str) -> str:
    """Run the LangGraph agent with a user prompt and return the final answer."""
    agent = build_graph()

    initial_state: AgentState = {
        "messages": [{"role": "user", "content": user_prompt}],
        "job_id": None,
        "poll_count": 0,
        "phase": "IDLE",
        "final_answer": None,
    }

    final_state = await agent.ainvoke(initial_state)
    return final_state.get("final_answer") or "Agent completed without a text answer."
