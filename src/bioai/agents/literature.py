"""Literature review agent for medical publications."""

from bioai.agents.base import BaseAgent


class LiteratureAgent(BaseAgent):
    name = "literature"
    role = "Review and synthesis of latest medical publications"

    async def analyze(self, query: str, context: dict | None = None) -> dict:
        raise NotImplementedError
