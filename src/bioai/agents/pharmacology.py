"""Pharmacology agent for drug interaction reasoning."""

from bioai.agents.base import BaseAgent


class PharmacologyAgent(BaseAgent):
    name = "pharmacology"
    role = "Drug interaction reasoning and medication safety"

    async def analyze(self, query: str, context: dict | None = None) -> dict:
        raise NotImplementedError
