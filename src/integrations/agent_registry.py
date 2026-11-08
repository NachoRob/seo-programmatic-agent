"""Descubrimiento de servidores MCP registrados en Google Cloud Agent Registry.

Usa el cliente nativo de ADK (`google.adk.integrations.agent_registry`) para
listar los MCP del proyecto y construir sus `McpToolset`.

Autenticación hacia el MCP (por servicio, ver `Settings.mcp_auth`):
- secret_token: el MCP valida un Bearer estático (`MCP_AUTH_TOKEN`) guardado en Secret Manager. Se lee con ADC
  y se inyecta con `SecretBearerProvider`.
- id_token: Cloud Run con IAM exige un *ID token* con audiencia = URL del
  servicio (`CloudRunIdTokenProvider`). Es el default para `*.run.app`.
- Endpoints `*.googleapis.com`: ADK ya inyecta el access token de ADC.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Callable
from urllib.parse import urlparse

import google.auth
import google.auth.exceptions
import google.auth.transport.requests
from google.auth import jwt
from google.oauth2 import id_token as google_id_token
from pydantic import BaseModel

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.integrations.agent_registry import AgentRegistry
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

from src.config import McpAuth, Settings, get_settings
from src.integrations.secret_manager import access_secret

logger = logging.getLogger(__name__)

RegistryFactory = Callable[..., AgentRegistry]


class McpServerInfo(BaseModel):
    """Datos mínimos de un MCP registrado."""

    name: str  # projects/.../locations/.../mcpServers/{id}
    display_name: str  # p.ej. "D2B - Google Search Console MCP"
    server_id: str | None = None  # urn:mcp:...:services:search-console-readonly
    url: str | None = None

    @property
    def service_name(self) -> str | None:
        """Nombre del servicio al final del `mcpServerId` (p.ej. `search-console-readonly`)."""
        return self.server_id.rsplit(":", 1)[-1] if self.server_id else None

    @property
    def is_cloud_run(self) -> bool:
        return bool(self.url) and (urlparse(self.url).hostname or "").endswith(".run.app")

    @property
    def audience(self) -> str | None:
        """Audiencia del ID token: la URL base del servicio, sin path (`/mcp`)."""
        if not self.url:
            return None
        parsed = urlparse(self.url)
        return f"{parsed.scheme}://{parsed.netloc}"


def _extract_url(server: dict[str, Any]) -> str | None:
    """Primera URL declarada en `protocols[].interfaces[]` o `interfaces[]`."""
    protocols = list(server.get("protocols") or [])
    if server.get("interfaces"):
        protocols.append({"interfaces": server["interfaces"]})
    for protocol in protocols:
        for interface in protocol.get("interfaces", []):
            if url := interface.get("url"):
                return url
    return None


def _to_info(server: dict[str, Any]) -> McpServerInfo:
    name = server["name"]
    return McpServerInfo(
        name=name,
        display_name=server.get("displayName") or name.rsplit("/", 1)[-1],
        server_id=server.get("mcpServerId"),
        url=_extract_url(server),
    )


def _token_env_var(service_name: str) -> str:
    return "SEO_AGENT_MCP_TOKEN_" + re.sub(r"[^A-Z0-9]", "_", service_name.upper())


class SecretBearerProvider:
    """`header_provider` de ADK con un Bearer estático leído de Secret Manager.

    Si existe `SEO_AGENT_MCP_TOKEN_<SERVICIO>` en el entorno se usa ese valor
    (útil en local/CI). El token se carga en el primer uso y no se serializa
    al hacer pickle, para no viajar dentro del agente desplegado.
    """

    def __init__(
        self,
        service_name: str,
        project_id: str,
        secret: str,
        secret_fetcher: Callable[[str, str], str] = access_secret,
    ):
        self.service_name = service_name
        self.project_id = project_id
        self.secret = secret
        self._secret_fetcher = secret_fetcher
        self._token: str | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_token"] = None
        return state

    def get_token(self) -> str:
        if not self._token:
            self._token = os.environ.get(_token_env_var(self.service_name)) or self._secret_fetcher(
                self.project_id, self.secret
            )
        return self._token

    def __call__(self, context: Any = None) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.get_token()}"}


def fetch_id_token(audience: str) -> str:
    """Obtiene un ID token de Google para `audience` usando ADC.

    `fetch_id_token` cubre service accounts, metadata server (Cloud Run / Agent
    Runtime) e impersonación. Con credenciales de usuario
    (`gcloud auth application-default login`) cae al `id_token` del refresh.
    """
    request = google.auth.transport.requests.Request()
    try:
        return google_id_token.fetch_id_token(request, audience)
    except google.auth.exceptions.DefaultCredentialsError:
        credentials, _ = google.auth.default()
        credentials.refresh(request)
        token = getattr(credentials, "id_token", None)
        if not token:
            raise RuntimeError(
                "No se pudo obtener un ID token para Cloud Run con las ADC actuales. "
                "Usa una service account o `gcloud auth application-default login`."
            ) from None
        return token


class CloudRunIdTokenProvider:
    """`header_provider` de ADK que agrega `Authorization: Bearer <id_token>`.

    Cachea el token hasta `refresh_margin_s` segundos antes de su expiración.
    Es picklable (necesario si el agente se despliega en Agent Runtime).
    """

    def __init__(
        self,
        audience: str,
        refresh_margin_s: int = 300,
        token_fetcher: Callable[[str], str] = fetch_id_token,
    ):
        self.audience = audience
        self.refresh_margin_s = refresh_margin_s
        self._token_fetcher = token_fetcher
        self._token: str | None = None
        self._expires_at: float = 0.0

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_token"] = None
        state["_expires_at"] = 0.0
        return state

    def _token_is_fresh(self) -> bool:
        return bool(self._token) and time.time() < self._expires_at - self.refresh_margin_s

    def get_token(self) -> str:
        if not self._token_is_fresh():
            token = self._token_fetcher(self.audience)
            claims = jwt.decode(token, verify=False)
            self._token = token
            self._expires_at = float(claims.get("exp", time.time() + 3600))
        return self._token

    def __call__(self, context: Any = None) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.get_token()}"}


async def _discovery_context() -> ReadonlyContext:
    """Contexto mínimo para usar un toolset fuera de un Runner.

    `McpToolset` solo llama al `header_provider` si recibe un contexto; dentro de
    un agente ADK lo entrega siempre, pero en descubrimiento hay que crearlo.
    """
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name="seo_discovery", user_id="discovery")
    return ReadonlyContext(InvocationContext(
        session_service=session_service, invocation_id="discovery", session=session
    ))


class AgentRegistryLoader:
    """Descubre e instancia los toolsets MCP registrados en Agent Registry."""

    def __init__(
        self,
        settings: Settings | None = None,
        registry_factory: RegistryFactory = AgentRegistry,
    ):
        self.settings = settings or get_settings()
        self._registry_factory = registry_factory
        self._registry = registry_factory(
            project_id=self.settings.project_id,
            location=self.settings.location,
        )

    @property
    def registry(self) -> AgentRegistry:
        return self._registry

    def list_servers(self) -> list[McpServerInfo]:
        """Lista todos los MCP del proyecto/región (con paginación)."""
        servers: list[McpServerInfo] = []
        page_token: str | None = None
        while True:
            data = self._registry.list_mcp_servers(page_token=page_token)
            servers.extend(_to_info(s) for s in data.get("mcpServers", []))
            page_token = data.get("nextPageToken")
            if not page_token:
                return servers

    def find_server(self, display_name: str) -> McpServerInfo:
        """Busca un MCP por nombre de servicio, `displayName` o id del resource name."""
        for server in self.list_servers():
            if display_name in (
                server.service_name,
                server.display_name,
                server.name.rsplit("/", 1)[-1],
            ):
                return server
        raise LookupError(
            f"MCP '{display_name}' no está registrado en "
            f"projects/{self.settings.project_id}/locations/{self.settings.location}"
        )

    def _auth_for(self, server: McpServerInfo) -> McpAuth | None:
        auth = self.settings.mcp_auth.get(server.service_name or "")
        if auth is None and server.is_cloud_run:
            return McpAuth(mode="id_token")
        return auth

    def header_provider_for(self, server: McpServerInfo) -> Callable[[Any], dict[str, str]] | None:
        auth = self._auth_for(server)
        if auth is None:
            return None
        if auth.mode == "secret_token":
            if not auth.secret:
                raise ValueError(f"mcp_auth de '{server.service_name}' es secret_token pero no define `secret`.")
            return SecretBearerProvider(
                server.service_name or server.display_name,
                auth.secret_project or self.settings.project_id,
                auth.secret,
            )
        if auth.mode == "id_token":
            return CloudRunIdTokenProvider(
                server.audience, refresh_margin_s=self.settings.id_token_refresh_margin_s
            )
        raise ValueError(f"Modo de auth desconocido: {auth.mode}")

    def get_toolset(self, display_name: str) -> McpToolset:
        """Construye el `McpToolset` de un MCP registrado, listo para un `Agent`."""
        server = self.find_server(display_name)
        header_provider = self.header_provider_for(server)
        if header_provider is None:
            return self._registry.get_mcp_toolset(server.name)
        logger.info("MCP %s: auth %s", display_name, type(header_provider).__name__)
        # El header_provider es por registry, así que cada MCP con auth propia
        # usa su propia instancia de AgentRegistry.
        registry = self._registry_factory(
            project_id=self.settings.project_id,
            location=self.settings.location,
            header_provider=header_provider,
        )
        return registry.get_mcp_toolset(server.name)

    def load_toolsets(self, display_names: tuple[str, ...] | None = None) -> dict[str, McpToolset]:
        """Toolsets de los MCP configurados (por defecto solo search-console-readonly)."""
        names = display_names or self.settings.mcp_servers
        return {name: self.get_toolset(name) for name in names}

    async def list_tool_names(self, display_name: str) -> list[str]:
        """Conecta al MCP y devuelve los nombres de sus herramientas."""
        toolset = self.get_toolset(display_name)
        try:
            tools = await toolset.get_tools(await _discovery_context())
            return [tool.name for tool in tools]
        finally:
            await toolset.close()
