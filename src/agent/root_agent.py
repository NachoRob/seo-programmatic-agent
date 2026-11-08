"""Agente raíz (ADK) del SEO programático: el mismo agente conversacional de Google Chat.

Compatible con `uv run adk web` (app `src`) y Agent Runtime. Las herramientas y la
instrucción viven en `src.agent.chat_agent`.
"""

from __future__ import annotations

from src.agent.chat_agent import build_chat_agent

root_agent = build_chat_agent()
