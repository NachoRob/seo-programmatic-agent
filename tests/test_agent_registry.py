"""Descubrimiento de MCPs en Agent Registry.

Por defecto usa un registry falso. Con `pytest --live` consulta el Agent Registry
real de `your-gcp-project` y lista las herramientas de `search-console-readonly`.
"""

import base64
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import Settings
import pickle

from src.integrations.agent_registry import (
    AgentRegistryLoader,
    CloudRunIdTokenProvider,
    SecretBearerProvider,
)

BASE = "projects/your-gcp-project/locations/us-central1/mcpServers"

URN = "urn:mcp:projects-123456789012:projects:123456789012:locations:us-central1:agentregistry:services"

# Formato real devuelto por Agent Registry
SEARCH_CONSOLE = {
    "name": f"{BASE}/agentregistry-00000000-0000-0000-47e3-3b43cd1a0a64",
    "displayName": "D2B - Google Search Console MCP",
    "mcpServerId": f"{URN}:search-console-readonly",
    "protocols": None,
    "interfaces": [
        {"url": "https://search-console-readonly-abc-uc.a.run.app/mcp", "protocolBinding": "JSONRPC"}
    ],
}
GOOGLE_ADS = {
    "name": f"{BASE}/ads-456",
    "displayName": "D2B - Google Ads MCP",
    "mcpServerId": f"{URN}:google-ads-mcp",
    "interfaces": [{"url": "https://google-ads-mcp-123456789012.us-central1.run.app/mcp", "protocolBinding": "JSONRPC"}],
}
PRIVATE_RUN = {
    "name": f"{BASE}/priv-1",
    "displayName": "Servicio privado",
    "mcpServerId": f"{URN}:private-mcp",
    "interfaces": [{"url": "https://private-mcp-xyz-uc.a.run.app/mcp"}],
}
GOOGLEAPIS = {
    "name": f"{BASE}/gapi-1",
    "displayName": "BigQuery MCP",
    "mcpServerId": f"{URN}:bigquery",
    "protocols": [{"interfaces": [{"url": "https://bigquery.googleapis.com/mcp"}]}],
}


def fake_jwt(exp: float) -> str:
    def b64(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()
    return f"{b64({'alg': 'RS256', 'typ': 'JWT'})}.{b64({'exp': exp, 'aud': 'x'})}.c2ln"


@pytest.fixture
def registry_factory():
    """Fábrica que devuelve registries falsos y registra con qué args se crearon."""
    created = []

    def factory(**kwargs):
        registry = MagicMock(name=f"AgentRegistry{len(created)}")
        registry.init_kwargs = kwargs
        registry.list_mcp_servers.side_effect = [
            {"mcpServers": [SEARCH_CONSOLE], "nextPageToken": "p2"},
            {"mcpServers": [GOOGLE_ADS, PRIVATE_RUN, GOOGLEAPIS]},
        ] * 10
        registry.get_mcp_toolset.side_effect = lambda name: SimpleNamespace(server=name, registry=registry)
        created.append(registry)
        return registry

    factory.created = created
    return factory


@pytest.fixture
def loader(registry_factory):
    return AgentRegistryLoader(Settings(), registry_factory=registry_factory)


def test_registry_client_uses_project_and_location(loader, registry_factory):
    assert registry_factory.created[0].init_kwargs == {
        "project_id": "your-gcp-project",
        "location": "us-central1",
    }


def test_list_servers_follows_pagination(loader):
    servers = loader.list_servers()
    assert [s.service_name for s in servers] == [
        "search-console-readonly", "google-ads-mcp", "private-mcp", "bigquery",
    ]
    assert loader.registry.list_mcp_servers.call_args_list[1].kwargs == {"page_token": "p2"}


def test_find_server_parses_url_and_cloud_run_audience(loader):
    server = loader.find_server("search-console-readonly")
    assert server.name == SEARCH_CONSOLE["name"]
    assert server.url.endswith("/mcp")
    assert server.is_cloud_run
    assert server.audience == "https://search-console-readonly-abc-uc.a.run.app"


def test_find_server_by_display_name_or_resource_id(loader):
    assert loader.find_server("D2B - Google Search Console MCP").service_name == "search-console-readonly"
    assert loader.find_server("ads-456").service_name == "google-ads-mcp"


def test_find_server_missing_raises(loader):
    with pytest.raises(LookupError, match="no-existe"):
        loader.find_server("no-existe")


def test_search_console_uses_secret_manager_bearer(loader, registry_factory):
    toolset = loader.get_toolset("search-console-readonly")

    dedicated = registry_factory.created[1]
    provider = dedicated.init_kwargs["header_provider"]
    assert isinstance(provider, SecretBearerProvider)
    assert (provider.project_id, provider.secret) == ("your-gcp-project", "search-console-mcp-auth-token")
    assert toolset.server == SEARCH_CONSOLE["name"]
    assert toolset.registry is dedicated


def test_unconfigured_cloud_run_defaults_to_id_token(loader, registry_factory):
    loader.get_toolset("private-mcp")
    provider = registry_factory.created[1].init_kwargs["header_provider"]
    assert isinstance(provider, CloudRunIdTokenProvider)
    assert provider.audience == "https://private-mcp-xyz-uc.a.run.app"


def test_googleapis_toolset_uses_default_registry_auth(loader, registry_factory):
    toolset = loader.get_toolset("bigquery")
    assert toolset.registry is registry_factory.created[0]
    assert len(registry_factory.created) == 1


def test_default_toolsets_only_search_console(loader):
    assert set(loader.load_toolsets()) == {"search-console-readonly"}


def test_load_toolsets_accepts_explicit_names(loader):
    toolsets = loader.load_toolsets(("search-console-readonly", "private-mcp"))
    assert set(toolsets) == {"search-console-readonly", "private-mcp"}


async def test_list_tool_names_lists_and_closes(loader, monkeypatch):
    fake_toolset = MagicMock()
    fake_toolset.get_tools = AsyncMock(return_value=[
        SimpleNamespace(name="search_console_readonly_list_sites"),
        SimpleNamespace(name="search_console_readonly_query_search_analytics"),
    ])
    fake_toolset.close = AsyncMock()
    monkeypatch.setattr(loader, "get_toolset", lambda name: fake_toolset)

    names = await loader.list_tool_names("search-console-readonly")

    assert names == [
        "search_console_readonly_list_sites",
        "search_console_readonly_query_search_analytics",
    ]
    fake_toolset.close.assert_awaited_once()
    # Sin contexto, McpToolset no llama al header_provider: debe recibir uno
    assert fake_toolset.get_tools.await_args.args[0] is not None


def test_secret_bearer_provider_caches_and_is_not_pickled_with_token(monkeypatch):
    monkeypatch.delenv("SEO_AGENT_MCP_TOKEN_SEARCH_CONSOLE_READONLY", raising=False)
    fetcher = MagicMock(return_value="s3cr3t")
    provider = SecretBearerProvider("search-console-readonly", "proj", "sc-token", secret_fetcher=fetcher)

    assert provider(None) == {"Authorization": "Bearer s3cr3t"}
    provider(None)
    fetcher.assert_called_once_with("proj", "sc-token")

    deployable = SecretBearerProvider("search-console-readonly", "proj", "sc-token")
    deployable._token = "s3cr3t"
    restored = pickle.loads(pickle.dumps(deployable))
    assert restored._token is None and restored.secret == "sc-token"


def test_secret_bearer_provider_env_override(monkeypatch):
    monkeypatch.setenv("SEO_AGENT_MCP_TOKEN_SEARCH_CONSOLE_READONLY", "desde-env")
    fetcher = MagicMock()
    provider = SecretBearerProvider("search-console-readonly", "proj", "sc-token", secret_fetcher=fetcher)
    assert provider(None)["Authorization"] == "Bearer desde-env"
    fetcher.assert_not_called()


def test_id_token_provider_caches_until_near_expiry():
    fetcher = MagicMock(side_effect=[fake_jwt(time.time() + 3600), fake_jwt(time.time() + 3600)])
    provider = CloudRunIdTokenProvider("https://svc.a.run.app", token_fetcher=fetcher)

    first = provider(None)
    second = provider(None)
    assert first == second
    assert first["Authorization"].startswith("Bearer ")
    fetcher.assert_called_once_with("https://svc.a.run.app")

    provider._expires_at = time.time() + 60  # dentro del margen de 300s
    provider(None)
    assert fetcher.call_count == 2


# ------------------------------------------------------------------- live ----

@pytest.mark.live
async def test_live_search_console_tools_discovery():
    loader = AgentRegistryLoader(Settings())
    server = loader.find_server("search-console-readonly")
    assert server.url, "el MCP registrado no declara URL"

    names = await loader.list_tool_names("search-console-readonly")
    print(f"\n{server.url} -> {names}")
    assert names, "search-console-readonly no expuso herramientas"

