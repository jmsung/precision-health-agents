"""Proteomics agent for biomarker inference."""

from bioai.agents.base import BaseAgent


class ProteomicsAgent(BaseAgent):
    name = "proteomics"
    role = "Biomarker inference from proteomic data"

    async def analyze(self, query: str, context: dict | None = None) -> dict:
        raise NotImplementedError
