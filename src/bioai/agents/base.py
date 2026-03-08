"""Base agent interface."""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Base class for all specialized biomedical agents."""

    name: str
    role: str

    @abstractmethod
    async def analyze(self, query: str, context: dict | None = None) -> dict:
        """Analyze a query and return structured results."""
