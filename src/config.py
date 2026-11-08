"""Configuración central del agente.

Todos los valores tienen un default para D2B y se pueden sobreescribir con
variables de entorno (ver `Settings.from_env`).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, model_validator

SHEETS_SCOPES = ("https://www.googleapis.com/auth/spreadsheets",)


class McpAuth(BaseModel):
    """Estrategia de autenticación hacia un MCP.

    - secret_token: Bearer estático guardado en Secret Manager (`MCP_AUTH_TOKEN` del servicio).
    - id_token: ID token de Google con audiencia = URL del servicio (Cloud Run con IAM).
    """

    mode: Literal["secret_token", "id_token"]
    secret: str | None = None  # nombre del secreto (modo secret_token)
    secret_project: str | None = None  # por defecto, Settings.project_id


class Settings(BaseModel):
    # GCP / Agent Registry
    project_id: str = "your-gcp-project"
    location: str = "us-central1"

    # Servicio MCP requerido en Agent Registry (sufijo del mcpServerId)
    search_console_mcp: str = "search-console-readonly"

    # Cómo autenticarse con cada MCP (clave = nombre del servicio). Sin entrada:
    # Cloud Run -> ID token (IAM); googleapis -> access token de ADC (lo hace ADK).
    mcp_auth: dict[str, McpAuth] = Field(default_factory=lambda: {
        "search-console-readonly": McpAuth(mode="secret_token", secret="search-console-mcp-auth-token"),
    })

    # Vertex AI (Gemini y embeddings)
    vertex_project: str | None = None  # por defecto, project_id
    vertex_location: str = "us-central1"

    # Modelos. Regla de oro: el juez nunca es el modelo que redactó.
    orchestrator_model: str = "gemini-2.5-pro"  # agente conversacional (Google Chat / adk web)
    writer_model: str = "gemini-2.5-pro"  # PR-01, PR-02, PR-03, P1-16, PR-06
    judge_model: str = "gemini-2.5-flash"  # PR-04
    extractor_model: str = "gemini-2.5-flash"  # PR-05, clasificación de intención (P1-06)
    embedding_model: str = "text-multilingual-embedding-002"

    # Estado de los runs y lock de Google Chat: "firestore" (sobrevive a Cloud Run)
    # o "json" (archivos locales en `local_state_dir`, solo desarrollo).
    state_backend: Literal["firestore", "json"] = "firestore"
    firestore_project: str | None = None  # por defecto, project_id
    firestore_database: str = "(default)"
    local_state_dir: str = ".seo_agent_state"
    # Memory Bank de Vertex AI (y sesiones conversacionales de `adk web` / Agent
    # Runtime). Sin Agent Engine la memoria por cliente queda en Firestore.
    agent_engine_id: str | None = None

    # Identidad del agente. Sheets y Google Chat se usan suplantándola desde ADC
    # (evita pedir esos scopes en `gcloud auth application-default login`); en
    # Cloud Run / Agent Runtime es la cuenta de ejecución. Cada Sheet de cliente
    # debe compartirse con ella como Editor.
    service_account: str | None = "seo-programmatic-agent@your-gcp-project.iam.gserviceaccount.com"

    # Google Chat: audiencia del token que envía Chat (número de proyecto o URL del
    # endpoint, según la config de la app). Sin valor no se verifica (solo local).
    chat_audience: str | None = None
    # Chat exige responder en < 30 s: se espera la respuesta del agente hasta este
    # tope; si no alcanza, se responde "trabajando..." y el resultado se publica después.
    chat_sync_seconds: float = 22.0
    # Plazo total desde que Chat envía el evento (corta en 30 s; margen para la red y el arranque)
    chat_deadline_seconds: float = 25.0
    chat_history_events: int = 60  # eventos de la conversación que se guardan (los más recientes)

    # Pestañas de la Sheet del cliente (entrada)
    parameters_tab: str = "Parámetros"
    brief_tab: str = "Brief"
    seeds_tab: str = "Semillas"
    own_data_tab: str = "Datos propios"

    # Search Console (P1-04): ventana de días, terminando hace `gsc_lag_days`
    gsc_days: int = 90
    gsc_lag_days: int = 3

    # Localización por defecto si el Brief no la define
    default_country: str = "Chile"
    default_language: str = "es-CL"
    default_treatment: str = "tuteo"

    # Defaults si la Sheet no define el parámetro
    default_min_judge_score: float = 4.0  # escala 1-5 del juez (PR-04)
    # P1-18 calibrado con 10 ciudades (scripts/calibrate_similarity.py), texto con
    # los datos de la fila enmascarados: trigramas -> páginas buenas 0.12-0.35,
    # copias 0.67-1.0. Con "embeddings" (coseno) no hay umbral que las separe.
    similarity_method: Literal["trigramas", "embeddings"] = "trigramas"
    default_similarity_threshold: float = 0.5
    default_min_keyword_score: float = 0.0  # 0-100
    default_min_search_volume: float = 0.0

    # Margen para renovar el ID token antes de que expire (segundos)
    id_token_refresh_margin_s: int = Field(default=300, ge=0)

    @model_validator(mode="after")
    def _judge_differs_from_writer(self) -> Settings:
        if self.judge_model == self.writer_model:
            raise ValueError(
                f"judge_model ({self.judge_model}) no puede ser igual a writer_model: "
                "el juez nunca es el modelo que generó el texto."
            )
        return self

    @property
    def vertex_project_id(self) -> str:
        return self.vertex_project or self.project_id

    @property
    def mcp_servers(self) -> tuple[str, ...]:
        return (self.search_console_mcp,)

    @classmethod
    def from_env(cls) -> Settings:
        env_map = {
            "project_id": "SEO_AGENT_PROJECT_ID",
            "location": "SEO_AGENT_LOCATION",
            "search_console_mcp": "SEO_AGENT_SEARCH_CONSOLE_MCP",
            "vertex_project": "SEO_AGENT_VERTEX_PROJECT",
            "vertex_location": "SEO_AGENT_VERTEX_LOCATION",
            "orchestrator_model": "SEO_AGENT_ORCHESTRATOR_MODEL",
            "writer_model": "SEO_AGENT_WRITER_MODEL",
            "judge_model": "SEO_AGENT_JUDGE_MODEL",
            "extractor_model": "SEO_AGENT_EXTRACTOR_MODEL",
            "embedding_model": "SEO_AGENT_EMBEDDING_MODEL",
            "similarity_method": "SEO_AGENT_SIMILARITY_METHOD",
            "agent_engine_id": "SEO_AGENT_AGENT_ENGINE_ID",
            "service_account": "SEO_AGENT_SERVICE_ACCOUNT",
            "chat_audience": "SEO_AGENT_CHAT_AUDIENCE",
            "chat_sync_seconds": "SEO_AGENT_CHAT_SYNC_SECONDS",
            "chat_deadline_seconds": "SEO_AGENT_CHAT_DEADLINE_SECONDS",
            "local_state_dir": "SEO_AGENT_LOCAL_STATE_DIR",
            "state_backend": "SEO_AGENT_STATE_BACKEND",
            "firestore_database": "SEO_AGENT_FIRESTORE_DATABASE",
            "parameters_tab": "SEO_AGENT_PARAMETERS_TAB",
            "brief_tab": "SEO_AGENT_BRIEF_TAB",
        }
        overrides = {
            field: os.environ[var] for field, var in env_map.items() if os.environ.get(var)
        }
        return cls(**overrides)


def default_parameter_rows(settings: Settings | None = None) -> list[list]:
    """Plantilla de la pestaña Parámetros con los defaults vigentes."""
    settings = settings or Settings()
    return [
        ["parametro", "valor"],
        ["peso_volumen", 0.30], ["peso_dificultad", 0.20], ["peso_intencion", 0.25],
        ["peso_datos_propios", 0.10], ["peso_valor_negocio", 0.15],
        ["score_minimo_juez", settings.default_min_judge_score],
        ["umbral_similitud", settings.default_similarity_threshold],
        ["score_minimo_keyword", settings.default_min_keyword_score],
        ["volumen_minimo", settings.default_min_search_volume],
    ]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
