import asyncio
import io
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ogc_mcp.agent as agent_module
from ogc_mcp.agent import run_agent


class Tee(io.StringIO):
    """Capture logs while still streaming them to the console."""

    def __init__(self, target):
        super().__init__()
        self.target = target

    def write(self, s):
        self.target.write(s)
        return super().write(s)

    def flush(self):
        self.target.flush()
        return super().flush()


@dataclass
class StressCase:
    category: str
    name: str
    prompt: str
    evaluator: Callable[[str, str], tuple[bool, str]]
    mock_setup: Callable[[], Callable[[], None]] | None = None


def eval_tool_awareness(logs: str, answer: str) -> tuple[bool, str]:
    banned = ["delete_job", "cancel_process"]
    if any(item in answer for item in banned):
        return False, "Agent hallucinated unsupported tools."
    if "get_job_status" not in answer:
        return False, "Agent did not identify get_job_status in the answer."
    if "[Agent . Tool]" in logs:
        return False, "This prompt should be answered from tool descriptions, not by calling a tool."
    return True, "Agent described available tools without hallucinating."


def eval_list_processes(logs: str, answer: str) -> tuple[bool, str]:
    if "[Agent . Tool] Calling: list_processes" not in logs:
        return False, "Agent did not call list_processes."
    if not any(token in answer.lower() for token in ["hello-world", "echo", "yes", "no"]):
        return False, "Answer did not summarize the process listing result."
    return True, "Agent listed processes before answering."


def eval_sequential_execution(logs: str, answer: str) -> tuple[bool, str]:
    if "[Agent . Tool] Calling: execute_process" not in logs:
        return False, "Agent never executed the process."

    synchronous_success = "[Agent . State] SUCCESS - synchronous execution!" in logs
    async_success = "[Agent . State] SUCCESS - job completed!" in logs

    if synchronous_success:
        if "[Agent . Tool] Calling: get_job_status" in logs:
            return False, "Agent polled even though the backend completed synchronously."
        if "[Agent . Tool] Calling: get_job_result" in logs:
            return False, "Agent fetched a job result even though no async job was created."
    else:
        if "[Agent . Tool] Calling: get_job_status" not in logs:
            return False, "Agent did not poll job status for an async execution."
        if "[Agent . Tool] Calling: get_job_result" not in logs:
            return False, "Agent did not fetch the final result."
        if not async_success:
            return False, "Agent never observed a successful async job completion state."
    if "GSoC Validator" not in answer:
        return False, "Final answer did not include the requested name."
    if synchronous_success:
        return True, "Agent correctly skipped polling because the backend completed synchronously."
    return True, "Agent performed submit -> poll -> result in sequence."


def eval_chatty_extraction(logs: str, answer: str) -> tuple[bool, str]:
    if "[Agent . LLM] Wants to call:" not in logs:
        return False, "Extractor did not recover a tool call from a chatty model response."
    if "[Agent . Tool] Calling: execute_process" not in logs:
        return False, "Agent never reached tool execution."
    if "Final answer ready." in logs and "[Agent . Tool]" not in logs:
        return False, "Agent treated chatter as a final answer instead of extracting JSON."
    if "Resilience Test" not in answer:
        return False, "Final answer did not include the requested name."
    return True, "Extractor handled a preamble and still triggered tool execution."


def eval_missing_process(logs: str, answer: str) -> tuple[bool, str]:
    if "[Agent . Tool] Calling: list_processes" not in logs:
        return False, "Agent did not verify process availability first."
    if "[Agent . Tool] Calling: execute_process" in logs:
        return False, "Agent attempted to execute a process after discovery showed it was unavailable."
    error_signals = ["404", "not found", "couldn't find", "cannot find", "error"]
    if not any(signal in answer.lower() for signal in error_signals):
        return False, "Agent did not clearly report the missing process."
    return True, "Agent surfaced the missing-process error without calling an unavailable tool."


def eval_fake_job(logs: str, answer: str) -> tuple[bool, str]:
    if "[Agent . Tool] Calling: get_job_status" not in logs:
        return False, "Agent did not attempt to check the fake job ID."
    if "still running" in answer.lower():
        return False, "Agent incorrectly claimed the fake job is still running."
    error_signals = ["invalid", "not found", "couldn't", "cannot", "error"]
    if not any(signal in answer.lower() for signal in error_signals):
        return False, "Agent did not explain that the job ID is invalid."
    return True, "Agent handled the invalid job ID without hallucinating."


def eval_final_boss(logs: str, answer: str) -> tuple[bool, str]:
    if "[Agent . Tool] Calling: list_processes" not in logs:
        return False, "Agent skipped process discovery."
    if "[Agent . Tool] Calling: execute_process" not in logs:
        return False, "Agent never performed the conditional execution step."
    if '"tool_call": "list_processes"' in answer and '"tool_call": "execute_process"' in answer:
        return False, "Agent dumped multiple tool calls into the final answer."
    return True, "Agent handled the multi-step instruction without collapsing into one malformed response."


def eval_mocked_async_langgraph(logs: str, answer: str) -> tuple[bool, str]:
    expected_logs = [
        "[Agent . Tool] Calling: execute_process",
        "[Agent . State] SUBMITTED - jobID=job-async-123",
        "[Agent . Poll] Waiting",
        "[Agent . Tool] Calling: get_job_status",
        "[Agent . State] POLLING - status=running",
        "[Agent . State] SUCCESS - job completed!",
        "[Agent . Tool] Calling: get_job_result",
    ]
    for marker in expected_logs:
        if marker not in logs:
            return False, f"Missing expected async marker: {marker}"
    if "Hello Async Stress Test!" not in answer:
        return False, "Final answer did not include the mocked async result."
    return True, "LangGraph completed the mocked async submit -> poll -> result flow."


def setup_mock_async_case() -> Callable[[], None]:
    original_execute_process = agent_module.execute_process
    original_get_job_status = agent_module.get_job_status
    original_get_job_result = agent_module.get_job_result
    original_hf_call = agent_module._hf_call

    status_calls = {"count": 0}
    responses = [
        '{"tool_call": "execute_process", "args": {"process_id": "hello-world", "inputs": {"name": "Async Stress Test"}}}',
        '{"tool_call": "get_job_status", "args": {"job_id": "job-async-123"}}',
        '{"tool_call": "get_job_status", "args": {"job_id": "job-async-123"}}',
        '{"tool_call": "get_job_result", "args": {"job_id": "job-async-123"}}',
        'The async process completed successfully. Result: Hello Async Stress Test!',
    ]

    async def mock_execute_process(process_id: str, inputs: dict) -> dict:
        return {"jobID": "job-async-123", "status": "accepted"}

    async def mock_get_job_status(job_id: str) -> dict:
        status_calls["count"] += 1
        if status_calls["count"] == 1:
            return {"jobID": job_id, "status": "running"}
        return {"jobID": job_id, "status": "successful"}

    async def mock_get_job_result(job_id: str) -> dict:
        return {"id": "echo", "value": "Hello Async Stress Test!"}

    def mock_hf_call(_history: list[dict]) -> str:
        if not responses:
            raise RuntimeError("Mock LLM called more times than expected")
        return responses.pop(0)

    agent_module.execute_process = mock_execute_process
    agent_module.get_job_status = mock_get_job_status
    agent_module.get_job_result = mock_get_job_result
    agent_module._hf_call = mock_hf_call

    def restore() -> None:
        agent_module.execute_process = original_execute_process
        agent_module.get_job_status = original_get_job_status
        agent_module.get_job_result = original_get_job_result
        agent_module._hf_call = original_hf_call

    return restore


CASES = [
    StressCase(
        category="Category 1: Discovery & Introspection",
        name="Tool Awareness",
        prompt="What tools do you have access to, and which one should I use to see the status of a background job?",
        evaluator=eval_tool_awareness,
    ),
    StressCase(
        category="Category 1: Discovery & Introspection",
        name="List Processes",
        prompt="List the processes on the server. Does any process mention 'echo' in its description?",
        evaluator=eval_list_processes,
    ),
    StressCase(
        category="Category 2: Sequential Execution",
        name="Poll Until Finished",
        prompt="Execute the 'hello-world' process with name='GSoC Validator'. Don't just submit it-poll the status until it's finished and then tell me the final greeting.",
        evaluator=eval_sequential_execution,
    ),
    StressCase(
        category="Category 3: Robustness & Extraction",
        name="Chatty Preamble",
        prompt="Hey, can you help me? I really need to run that hello-world thing. Use the name 'Resilience Test'. Be very descriptive in your internal thinking but give me the result.",
        evaluator=eval_chatty_extraction,
    ),
    StressCase(
        category="Category 4: Error Handling & Edge Cases",
        name="Missing Process",
        prompt="Run a process called 'non-existent-tool' with name='Test'. If list_processes confirms it does not exist, do not try to execute it anyway; tell me immediately.",
        evaluator=eval_missing_process,
    ),
    StressCase(
        category="Category 4: Error Handling & Edge Cases",
        name="Fake Job ID",
        prompt="Check the status of job-999999.",
        evaluator=eval_fake_job,
    ),
    StressCase(
        category="Category 5: Multi-Part Logic",
        name="Final Boss",
        prompt="First, list all processes. Then, if 'hello-world' is available, run it with my name. If it's not available, tell me what else I can run instead.",
        evaluator=eval_final_boss,
    ),
    StressCase(
        category="Category 6: Async LangGraph Proof",
        name="Mocked Async Polling",
        prompt="Execute the 'hello-world' process with name='Async Stress Test'. Poll until it finishes and then return the final greeting.",
        evaluator=eval_mocked_async_langgraph,
        mock_setup=setup_mock_async_case,
    ),
]


async def run_case(case: StressCase) -> tuple[bool, str, str]:
    tee = Tee(sys.stdout)
    restore = case.mock_setup() if case.mock_setup else None
    try:
        with redirect_stdout(tee):
            answer = await run_agent(case.prompt)
        logs = tee.getvalue()
        passed, reason = case.evaluator(logs, answer)
        return passed, reason, answer
    finally:
        if restore:
            restore()


async def main() -> int:
    failures = 0

    print("Running PoC stress prompts")
    print("=" * 72)

    for index, case in enumerate(CASES, start=1):
        print(f"\n[{index}/{len(CASES)}] {case.category} :: {case.name}")
        print(f"Prompt: {case.prompt}\n")

        try:
            passed, reason, answer = await run_case(case)
        except Exception as exc:
            failures += 1
            print("\nResult: FAIL")
            print(f"Reason: {exc}")
            continue

        print(f"\nFinal Answer: {answer}")
        print(f"Result: {'PASS' if passed else 'FAIL'}")
        print(f"Reason: {reason}")

        if not passed:
            failures += 1

        print("-" * 72)

    print("\nSummary")
    print("=" * 72)
    print(f"Total cases: {len(CASES)}")
    print(f"Passed: {len(CASES) - failures}")
    print(f"Failed: {failures}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
