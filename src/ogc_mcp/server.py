"""
Layer 1: FastMCP Server — OGC Processes API Wrapper
Exposes 4 OGC tools the LangGraph agent can call:
  - list_processes
  - execute_process
  - get_job_status
  - get_job_result
"""

import os
import httpx
from fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

PYGEOAPI_BASE_URL = os.getenv("PYGEOAPI_BASE_URL", "http://localhost:5000")

mcp = FastMCP(
    name="ogc-processes-server",
    instructions=(
        "You are connected to an OGC Processes API. "
        "Use list_processes to discover available processes, "
        "execute_process to submit a job, get_job_status to poll progress, "
        "and get_job_result to retrieve the final output."
    ),
)


@mcp.tool()
async def list_processes() -> dict:
    """List all processes available on the OGC server with their IDs and descriptions."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{PYGEOAPI_BASE_URL}/processes")
        response.raise_for_status()
        data = response.json()

    processes = data.get("processes", [])
    return {
        "count": len(processes),
        "processes": [
            {
                "id": p.get("id"),
                "title": p.get("title"),
                "description": p.get("description", "No description"),
                "version": p.get("version"),
                "keywords": p.get("keywords", []),
            }
            for p in processes
        ],
    }


@mcp.tool()
async def execute_process(process_id: str, inputs: dict) -> dict:
    """
    Submit an OGC process for execution and return the job ID.

    Args:
        process_id: The ID of the process to execute (e.g. 'hello-world').
        inputs:     A dict of input parameters required by the process.

    Returns:
        A dict containing the jobID and initial status.
    """
    payload = {"inputs": inputs}
    headers = {"Content-Type": "application/json", "Prefer": "respond-async"}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{PYGEOAPI_BASE_URL}/processes/{process_id}/execution",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()

    # Async response: 201 Created with Location header or body with jobID
    if response.status_code == 201:
        location = response.headers.get("location", "")
        job_id = location.rstrip("/").split("/")[-1] if location else None
        body = response.json() if response.content else {}
        job_id = job_id or body.get("jobID") or body.get("job_id")
        return {"jobID": job_id, "status": "accepted", "location": location}

    # Synchronous response (200): process completed immediately
    return {"jobID": None, "status": "successful", "result": response.json()}


@mcp.tool()
async def get_job_status(job_id: str) -> dict:
    """
    Poll the status of a submitted OGC job.

    Args:
        job_id: The job ID returned by execute_process.

    Returns:
        A dict with 'status' (accepted | running | successful | failed | dismissed)
        and the percentage progress if available.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{PYGEOAPI_BASE_URL}/jobs/{job_id}")
        response.raise_for_status()
        data = response.json()

    return {
        "jobID": data.get("jobID", job_id),
        "status": data.get("status"),
        "progress": data.get("progress"),
        "message": data.get("message"),
        "created": data.get("created"),
        "updated": data.get("updated"),
    }


@mcp.tool()
async def get_job_result(job_id: str) -> dict:
    """
    Retrieve the result of a successfully completed OGC job.

    Args:
        job_id: The job ID of a job whose status is 'successful'.

    Returns:
        The process output as returned by the OGC server.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{PYGEOAPI_BASE_URL}/jobs/{job_id}/results")
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    # Run as a standalone MCP server (stdio transport for LangGraph integration)
    mcp.run(transport="stdio")
