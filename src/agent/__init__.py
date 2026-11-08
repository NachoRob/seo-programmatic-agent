"""Sub-agentes, orquestador y agente conversacional del SEO programático.

`uv run adk web` lo encuentra como app `src` (`src.agent.root_agent`). El import
explícito deja en `src.agent.root_agent` la instancia del agente y no el
submódulo homónimo `src/agent/root_agent.py`.
"""

from src.agent.root_agent import root_agent  # noqa: F401

__all__ = ["root_agent"]
