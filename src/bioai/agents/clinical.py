"""Clinical guidelines agent for guideline interpretation."""

from bioai.agents.base import BaseAgent


class ClinicalAgent(BaseAgent):
    name = "clinical"
    role = "Clinical guideline interpretation and recommendations"

    async def analyze(self, query: str, context: dict | None = None) -> dict:
        raise NotImplementedError
