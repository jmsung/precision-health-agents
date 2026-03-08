"""Smoke tests for project setup."""

import bioai
from bioai.agents.base import BaseAgent


def test_version():
    assert bioai.__version__ == "0.1.0"


def test_base_agent_is_abstract():
    import pytest

    with pytest.raises(TypeError):
        BaseAgent()
