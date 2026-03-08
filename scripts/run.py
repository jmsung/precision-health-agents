"""Entry point for the multi-agent healthcare system."""

import asyncio

from bioai.orchestrator import Orchestrator


async def main():
    orchestrator = Orchestrator()
    # Register agents here
    print("BioAI system ready. No agents registered yet.")


if __name__ == "__main__":
    asyncio.run(main())
