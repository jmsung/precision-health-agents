"""Smoke tests for project setup."""

import precision_health_agents
from precision_health_agents.agents.base import BaseAgent


def test_version():
    assert precision_health_agents.__version__ == "0.1.0"


def test_base_agent_is_abstract():
    import pytest

    with pytest.raises(TypeError):
        BaseAgent()
