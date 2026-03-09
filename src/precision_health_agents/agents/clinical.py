"""Clinical guidelines agent for guideline interpretation."""

from precision_health_agents.agents.base import BaseAgent


class ClinicalAgent(BaseAgent):
    name = "clinical"
    role = "Clinical guideline interpretation and recommendations"

    async def analyze(self, query: str, context: dict | None = None) -> dict:
        raise NotImplementedError
