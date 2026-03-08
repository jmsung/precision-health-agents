"""Multi-agent orchestrator."""

import asyncio

from bioai.agents.base import BaseAgent


class Orchestrator:
    """Coordinates specialized agents to analyze patient data."""

    def __init__(self, agents: list[BaseAgent] | None = None):
        self.agents: list[BaseAgent] = agents or []

    def register(self, agent: BaseAgent) -> None:
        self.agents.append(agent)

    async def run(self, query: str, context: dict | None = None) -> dict:
        """Run all agents concurrently and aggregate results."""
        tasks = [agent.analyze(query, context) for agent in self.agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {
            agent.name: result
            for agent, result in zip(self.agents, results)
        }
