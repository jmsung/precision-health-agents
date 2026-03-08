"""Transcriptomics agent for gene expression signals."""

from bioai.agents.base import BaseAgent


class TranscriptomicsAgent(BaseAgent):
    name = "transcriptomics"
    role = "Gene expression signal analysis"

    async def analyze(self, query: str, context: dict | None = None) -> dict:
        raise NotImplementedError
