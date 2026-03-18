"""
GSoC PoC Demo — OGC Processes + MCP + LangGraph
Run: python demo.py

Prerequisites:
  1. pygeoapi running at http://localhost:5000 (Docker)
  2. .env file with GOOGLE_API_KEY set
  3. pip install -r requirements.txt
"""

import asyncio
import sys
import time
from pathlib import Path

# Ensure src/ is on the path when running from project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ogc_mcp.agent import run_agent

BANNER = """
--------------------------------------------------------------
       GSoC PoC: OGC APIs x MCP x LangGraph                  
       Layer 1: FastMCP  |  Layer 2: LangGraph Agent          
--------------------------------------------------------------
"""

DEMO_PROMPT = (
    "I want to run the 'hello-world' OGC process with the input name='GSoC Mentor'. "
    "Please execute it, wait for it to finish, and tell me the result."
)


async def main():
    print(BANNER)
    print("-" * 64)
    print(f"USER  ->  {DEMO_PROMPT}")
    print("-" * 64)

    start = time.perf_counter()
    answer = await run_agent(DEMO_PROMPT)
    elapsed = time.perf_counter() - start

    print("\n" + "-" * 64)
    print(f"AGENT ->  {answer}")
    print("-" * 64)
    print(f"\n   Done in {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
