import asyncio
import json
import sys
from pathlib import Path


def main() -> int:
    # Ensure src/ is on path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    # Import inside to allow monkeypatching module globals
    import ogc_mcp.agent as agent

    # --- Mock tool functions ---
    status_calls = {"count": 0}

    async def mock_execute_process(process_id: str, inputs: dict) -> dict:
        assert process_id == "hello-world"
        assert inputs.get("name") == "GSoC Mentor"
        return {"jobID": "job-123", "status": "accepted"}

    async def mock_get_job_status(job_id: str) -> dict:
        assert job_id == "job-123"
        status_calls["count"] += 1
        if status_calls["count"] < 2:
            return {"jobID": job_id, "status": "running"}
        return {"jobID": job_id, "status": "successful"}

    async def mock_get_job_result(job_id: str) -> dict:
        assert job_id == "job-123"
        return {"id": "echo", "value": "Hello GSoC Mentor!"}

    agent.execute_process = mock_execute_process
    agent.get_job_status = mock_get_job_status
    agent.get_job_result = mock_get_job_result

    # --- Mock LLM outputs ---
    responses = [
        json.dumps({
            "tool_call": "execute_process",
            "args": {"process_id": "hello-world", "inputs": {"name": "GSoC Mentor"}},
        }),
        json.dumps({"tool_call": "get_job_status", "args": {"job_id": "job-123"}}),
        json.dumps({"tool_call": "get_job_status", "args": {"job_id": "job-123"}}),
        json.dumps({"tool_call": "get_job_result", "args": {"job_id": "job-123"}}),
        "The hello-world process completed successfully! Result: Hello GSoC Mentor!",
    ]

    def mock_hf_call(_history):
        if not responses:
            raise RuntimeError("LLM called more times than expected")
        return responses.pop(0)

    agent._hf_call = mock_hf_call

    # --- Run agent ---
    prompt = (
        "I want to run the 'hello-world' OGC process with the input name='GSoC Mentor'. "
        "Please execute it, wait for it to finish, and tell me the result."
    )

    final_answer = asyncio.run(agent.run_agent(prompt))

    expected = "The hello-world process completed successfully! Result: Hello GSoC Mentor!"
    if final_answer != expected:
        print("TEST FAILED")
        print(f"Expected: {expected}")
        print(f"Got:      {final_answer}")
        return 1

    if status_calls["count"] < 2:
        print("TEST FAILED: polling did not happen as expected")
        return 1

    print("TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())