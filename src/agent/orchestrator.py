"""Orquestador end-to-end del Flujo 1 (P1-01 a P1-25) con frenos humanos.

Máquina de estados (persistida por `spreadsheet_id` + `thread_id`):

    NEW ──run_phase_1_and_2──▶ WAITING_KEYWORD_APPROVAL   (P1-10: el equipo marca
                                                           "Aprobado" en Keywords)
        ──run_phase_3────────▶ WAITING_URL_APPROVAL       (P1-13: el equipo marca
                                                           "Aprobado" en Plan de URLs)
        ──run_phase_4_to_7───▶ DONE

Cada etapa valida el estado de entrada, guarda el estado al terminar y, si falla,
conserva el estado anterior con `last_error` para poder reintentarla.

Pestañas de entrada: Brief, Parametros, Semillas, Datos propios.
Pestañas de salida (P1-25, cruzadas por `url_id`): Resumen, Keywords, Plan de URLs,
Contenido, On-page, Schema, Enlaces internos, Descartados.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import uuid
from collections import Counter
from enum import Enum
from typing import Any, Callable, Protocol
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from src.agent.content_pipeline import (
    ContentPipeline,
    PageJob,
    PipelineParams,
    PipelineReport,
)
from src.agent.memory_manager import ClientMemoryStore, RunStateStore, build_stores
from src.agent.prompts import PR01, PR02
from src.agent.schemas import PR01Output, PR02Output
from src.agent.subagents import Embedder, StructuredLLM, classify_and_cluster_keywords
from src.config import Settings, get_settings
from src.tools.seo_deterministic import (
    dedupe_key,
    generate_and_validate_slugs,
    mark_gsc_overlap,
    nodes_from_pr02,
    normalize_and_deduplicate_keywords,
    normalize_text,
    score_and_route_keywords,
    slugify,
)
from src.tools.site_reader import (
    hub_for_intent,
    match_existing_pages,
    site_pages,
    url_rules,
)

logger = logging.getLogger(__name__)

TAB_RESUMEN = "Resumen"
TAB_KEYWORDS = "Keywords"
TAB_PLAN = "Plan de URLs"
TAB_CONTENIDO = "Contenido"
TAB_ONPAGE = "On-page"
TAB_SCHEMA = "Schema"
TAB_ENLACES = "Enlaces internos"
TAB_DESCARTADOS = "Descartados"
FINAL_TABS = (TAB_RESUMEN, TAB_KEYWORDS, TAB_PLAN, TAB_CONTENIDO, TAB_ONPAGE, TAB_SCHEMA, TAB_ENLACES, TAB_DESCARTADOS)

SITE_INVENTORY_KEY = "inventario-sitio"  # thread_id con el que se guarda el inventario del sitio por Sheet
MAX_EXISTING_FOR_PROMPT = 300

APPROVED_COL = "Aprobado"
REJECT_REASON_COL = "Motivo rechazo"
KEYWORD_COLUMNS = [
    "keyword", "original_keyword", "fuentes", "cluster_id", "intencion", "patron", "search_volume",
    "difficulty", "coverage_status", "existing_url", "gsc_clicks", "gsc_impressions", "gsc_position",
    "score", "ruta", "url_id", APPROVED_COL, REJECT_REASON_COL,
]
PLAN_COLUMNS = [
    "url_id", "url", "url_propuesta", "coincide_con_propuesta", "keyword_principal", "keywords_secundarias",
    "hub", "patron", "variables", "slug_valid", "slug_issues", "datos_propios", APPROVED_COL, REJECT_REASON_COL,
]
DISCARDED_COLUMNS = ["etapa", "paso", "tipo", "elemento", "url_id", "motivo"]
MAX_CELL = 49_000  # límite de Sheets: 50.000 caracteres por celda
SHEET_RANGE = "A1:AZ20000"

_TRUE = {"true", "verdadero", "si", "sí", "x", "1", "ok", "aprobado", "aprobada"}
_FALSE = {"false", "falso", "no", "0", "rechazado", "rechazada"}

# Alias de columnas de la pestaña Semillas (normalizadas) -> clave interna
_SEED_ALIASES = {
    "keyword": "keyword", "keywords": "keyword", "palabra_clave": "keyword",
    "fuente": "source", "source": "source", "origen": "source",
    "volumen": "search_volume", "search_volume": "search_volume", "volumen_busqueda": "search_volume",
    "dificultad": "difficulty", "difficulty": "difficulty", "kd": "difficulty",
    "intencion": "intent", "intent": "intent",
    "valor_negocio": "business_value", "valor_de_negocio": "business_value", "business_value": "business_value",
    "datos_propios": "has_own_data", "has_own_data": "has_own_data",
}
_SITE_KEYS = ("sitio", "sitio_web", "site_url", "url_sitio", "url", "dominio", "propiedad_gsc")


class RunStatus(str, Enum):
    NEW = "NEW"
    WAITING_KEYWORD_APPROVAL = "WAITING_KEYWORD_APPROVAL"
    WAITING_URL_APPROVAL = "WAITING_URL_APPROVAL"
    DONE = "DONE"


class InvalidRunState(RuntimeError):
    pass


class SearchConsoleError(RuntimeError):
    """Search Console (vía MCP) no devolvió datos para el sitio del Brief."""

    def __init__(self, site_url: str, detail: str):
        super().__init__(f"Search Console falló para {site_url}: {detail}")
        self.site_url = site_url


class Localizacion(BaseModel):
    pais: str
    idioma: str
    trato: str
    regla: str


class RunState(BaseModel):
    run_id: str
    spreadsheet_id: str
    thread_id: str
    status: RunStatus = RunStatus.NEW
    dominio: str | None = None
    site_url: str | None = None
    parameters: dict = {}
    brief: dict = {}
    localizacion: Localizacion | None = None
    memoria_cliente: list[str] = []
    existing_urls: list[str] = []  # Search Console + colecciones del sitio
    site_urls: list[str] = []  # todo el sitemap: solo para detectar colisiones de URL
    plataforma: str | None = None  # None: el sitio no se exploró antes del análisis
    reglas_url: dict = {}
    seed_counts: dict[str, int] = {}
    clusters: list[dict] = []
    patrones: list[dict] = []
    keywords: list[dict] = []  # salida de P1-08/09 con patrón y ruta
    routing_summary: dict[str, int] = {}
    approved_keywords: list[str] = []
    plan: list[dict] = []  # salida de P1-12 con url_id
    descartados: list[list] = []
    pipeline_summary: dict[str, int] = {}
    content_summary: dict[str, Any] = {}
    history: list[dict] = []
    last_error: str | None = None

    def log(self, event: str, **data: Any) -> None:
        self.history.append({"at": dt.datetime.now(dt.timezone.utc).isoformat(), "event": event, **data})


# -------------------------------------------------------------- gateways ----

class SheetsGateway(Protocol):
    async def read_parameters_and_brief(self, spreadsheet_id: str) -> dict: ...
    async def read_tab(self, spreadsheet_id: str, tab_name: str) -> list[dict]: ...
    async def write_tabs(self, spreadsheet_id: str, tabs: dict[str, list[list]]) -> dict: ...
    async def read_raw(self, spreadsheet_id: str, tab_name: str) -> list[list]: ...


class SearchConsoleGateway(Protocol):
    async def get_rows(self, site_url: str, start_date: str, end_date: str) -> list[dict]: ...
    async def list_sites(self) -> list[dict]: ...


class GoogleSheetsGateway:
    """Sheets API v4 vía `src.tools.sheets_tool` (llamadas síncronas en un thread)."""

    async def read_parameters_and_brief(self, spreadsheet_id: str) -> dict:
        from src.tools.sheets_tool import read_parameters_and_brief
        return await asyncio.to_thread(read_parameters_and_brief, spreadsheet_id)

    async def read_tab(self, spreadsheet_id: str, tab_name: str) -> list[dict]:
        from src.tools.sheets_tool import read_sheet_tab
        return await asyncio.to_thread(read_sheet_tab, spreadsheet_id, tab_name, SHEET_RANGE)

    async def write_tabs(self, spreadsheet_id: str, tabs: dict[str, list[list]]) -> dict:
        from src.tools.sheets_tool import batch_write_sheet_tabs
        return await asyncio.to_thread(batch_write_sheet_tabs, spreadsheet_id, tabs)

    async def read_raw(self, spreadsheet_id: str, tab_name: str) -> list[list]:
        from src.tools.sheets_tool import read_sheet_values
        return await asyncio.to_thread(read_sheet_values, spreadsheet_id, tab_name)


class McpSearchConsoleGateway:
    """MCP `search-console-readonly`: `get_search_analytics` (dimensiones query + page) y `list_sites`."""

    ROW_LIMIT = 25_000

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    async def get_rows(self, site_url: str, start_date: str, end_date: str) -> list[dict]:
        try:
            return await self._get_rows(site_url, start_date, end_date)
        except Exception as e:
            raise SearchConsoleError(site_url, f"{type(e).__name__}: {e}") from e

    async def list_sites(self) -> list[dict]:
        """Propiedades visibles para la cuenta del MCP: `[{"siteUrl", "permissionLevel"}]`."""
        async with self._client() as client:
            result = await client.call_tool("list_sites", {})
        return json.loads(result.content[0].text).get("siteEntry", [])

    def _client(self):
        from fastmcp import Client
        from fastmcp.client.transports import StreamableHttpTransport

        from src.integrations.agent_registry import AgentRegistryLoader

        loader = AgentRegistryLoader(self.settings)
        server = loader.find_server(self.settings.search_console_mcp)
        provider = loader.header_provider_for(server)
        headers = provider(None) if provider else {}
        return Client(StreamableHttpTransport(server.url, headers=headers))

    async def _get_rows(self, site_url: str, start_date: str, end_date: str) -> list[dict]:
        rows: list[dict] = []
        async with self._client() as client:
            start_row = 0
            while True:
                result = await client.call_tool("get_search_analytics", {
                    "site_url": site_url, "start_date": start_date, "end_date": end_date,
                    "dimensions": ["query", "page"], "row_limit": self.ROW_LIMIT, "start_row": start_row,
                })
                page = json.loads(result.content[0].text).get("rows", [])
                rows.extend(page)
                if len(page) < self.ROW_LIMIT:
                    return rows
                start_row += self.ROW_LIMIT


# ----------------------------------------------------------------- helpers ----

def is_approved(value: Any) -> bool | None:
    """True/False si la celda es una decisión explícita; None si está vacía o no es clara."""
    if isinstance(value, bool):
        return value
    text = normalize_text(value)
    if text in _TRUE or str(value).strip().lower() in _TRUE:
        return True
    if text in _FALSE:
        return False
    return None


def localization_from_brief(brief: dict, settings: Settings) -> Localizacion:
    pais = str(brief.get("pais") or settings.default_country)
    idioma = str(brief.get("idioma") or settings.default_language)
    trato_raw = normalize_text(brief.get("trato") or brief.get("tratamiento") or settings.default_treatment)
    trato = "usted" if "usted" in trato_raw else "tuteo"
    forma = "de usted" if trato == "usted" else "de tú"
    regla = (
        f"Escribe en español de {pais} ({idioma}) tratando al lector {forma}. "
        "Está prohibido el voseo (por ejemplo «cotizá», «contratá», «tenés», «podés», «sos»)."
    )
    return Localizacion(pais=pais, idioma=idioma, trato=trato, regla=regla)


def site_from_brief(brief: dict) -> tuple[str, str]:
    """(site_url para GSC, dominio). Acepta https://..., sc-domain:... o un dominio suelto."""
    raw = next((str(brief[k]).strip() for k in _SITE_KEYS if brief.get(k)), "")
    if not raw:
        raise ValueError(f"El Brief no define el sitio (usa una de estas claves: {', '.join(_SITE_KEYS)}).")
    if raw.startswith("sc-domain:"):
        return raw, raw.split(":", 1)[1]
    if "://" not in raw:
        raw = f"https://{raw.strip('/')}/"
    host = urlparse(raw).hostname or ""
    return raw, host[4:] if host.startswith("www.") else host


def _seed_row(row: dict) -> dict:
    seed: dict[str, Any] = {}
    for column, value in row.items():
        key = _SEED_ALIASES.get(normalize_text(column).replace(" ", "_"))
        if key and value not in (None, ""):
            seed[key] = value
    if "has_own_data" in seed:
        seed["has_own_data"] = bool(is_approved(seed["has_own_data"]))
    return seed


def _own_data_summary(rows: list[dict], max_values: int = 50) -> dict:
    columns = list(rows[0].keys()) if rows else []
    values = {c: sorted({str(r[c]) for r in rows if r.get(c) not in (None, "")})[:max_values] for c in columns}
    return {"columnas": columns, "total_filas": len(rows), "filas_muestra": rows[:20], "valores_por_columna": values}


def _column_for(variable: str, rows: list[dict]) -> str | None:
    wanted = normalize_text(variable).replace(" ", "_")
    for column in (rows[0].keys() if rows else []):
        if normalize_text(column).replace(" ", "_") == wanted:
            return column
    return None


def match_own_data(variables: dict[str, str], rows: list[dict]) -> dict | None:
    """Fila de datos propios cuyas columnas coinciden con todas las variables.

    Compara con `slugify` (sin tildes ni ñ), igual que el slug de la URL, para
    tolerar "Vina del Mar" vs "Viña del Mar".
    """
    if not variables:
        return None
    columns = {var: _column_for(var, rows) for var in variables}
    if any(c is None for c in columns.values()):
        return None
    for row in rows:
        if all(slugify(row.get(columns[var]) or "") == slugify(val) for var, val in variables.items()):
            return row
    return None


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False)
    if isinstance(value, str) and len(value) > MAX_CELL:
        return value[:MAX_CELL] + "…"
    return value


def _table(columns: list[str], rows: list[dict]) -> list[list]:
    return [columns] + [[_cell(r.get(c)) for c in columns] for r in rows]


def _today() -> dt.date:
    return dt.date.today()


# ------------------------------------------------------------ orquestador ----

class SeoProgrammaticOrchestrator:
    def __init__(
        self,
        llm: StructuredLLM,
        embedder: Embedder,
        sheets: SheetsGateway | None = None,
        search_console: SearchConsoleGateway | None = None,
        state_store: RunStateStore | None = None,
        memory: ClientMemoryStore | None = None,
        settings: Settings | None = None,
        today: Callable[[], dt.date] = _today,
        batch_size: int = 15,
    ):
        self.settings = settings or get_settings()
        if state_store is None or memory is None:
            default_state, default_memory = build_stores(self.settings)
            state_store = state_store or default_state
            memory = memory or default_memory
        self.llm = llm
        self.embedder = embedder
        self.sheets = sheets or GoogleSheetsGateway()
        self.search_console = search_console or McpSearchConsoleGateway(self.settings)
        self.state_store = state_store
        self.memory = memory
        self.today = today
        self.batch_size = batch_size

    # ------------------------------------------------------------ estado ----

    async def get_state(self, spreadsheet_id: str, thread_id: str = "default") -> RunState | None:
        raw = await self.state_store.load(spreadsheet_id, thread_id)
        return RunState.model_validate(raw) if raw else None

    async def _save(self, state: RunState) -> None:
        await self.state_store.save(state.spreadsheet_id, state.thread_id, state.model_dump(mode="json"))

    async def _require(self, spreadsheet_id: str, thread_id: str, expected: RunStatus) -> RunState:
        state = await self.get_state(spreadsheet_id, thread_id)
        if state is None or state.status != expected:
            current = state.status.value if state else "sin run"
            raise InvalidRunState(f"Se esperaba {expected.value} y el run está en {current}.")
        return state

    async def _guarded(self, state: RunState, stage: str, body) -> RunState:
        try:
            await body(state)
            state.last_error = None
        except Exception as e:
            state.last_error = f"{stage}: {type(e).__name__}: {e}"
            state.log("error", etapa=stage, error=str(e))
            await self._save(state)
            raise
        await self._save(state)
        return state

    async def resume(self, spreadsheet_id: str, thread_id: str = "default") -> RunState:
        """Ejecuta la etapa que corresponde al estado guardado."""
        state = await self.get_state(spreadsheet_id, thread_id)
        if state is None or state.status == RunStatus.NEW:
            return await self.run_phase_1_and_2(spreadsheet_id, thread_id)
        if state.status == RunStatus.WAITING_KEYWORD_APPROVAL:
            return await self.run_phase_3(spreadsheet_id, thread_id)
        if state.status == RunStatus.WAITING_URL_APPROVAL:
            return await self.run_phase_4_to_7(spreadsheet_id, thread_id)
        return state

    async def rewind(self, spreadsheet_id: str, thread_id: str, to: RunStatus) -> RunState:
        """Vuelve el run a una pausa anterior para rehacer las etapas siguientes.

        Solo retrocede: a WAITING_KEYWORD_APPROVAL (rehacer B y C) o a
        WAITING_URL_APPROVAL (rehacer C). Las pestañas ya escritas se reescriben
        cuando la etapa vuelve a correr.
        """
        order = [RunStatus.NEW, RunStatus.WAITING_KEYWORD_APPROVAL, RunStatus.WAITING_URL_APPROVAL, RunStatus.DONE]
        state = await self.get_state(spreadsheet_id, thread_id)
        if state is None or order.index(state.status) < order.index(to) or to == RunStatus.NEW:
            current = state.status.value if state else "sin run"
            raise InvalidRunState(f"No se puede volver a {to.value} desde {current}.")
        if state.status != to:
            state.status = to
            redo = {"B", "C"} if to == RunStatus.WAITING_KEYWORD_APPROVAL else {"C"}
            state.descartados = [d for d in state.descartados if d[0] not in redo]
            if to == RunStatus.WAITING_KEYWORD_APPROVAL:
                state.plan, state.approved_keywords = [], []
            state.content_summary, state.pipeline_summary = {}, {}
            state.log("rewind", a=to.value)
            await self._save(state)
        return state

    # ---------------------------------------------- Etapa A: P1-01 a P1-09 ----

    async def run_phase_1_and_2(self, spreadsheet_id: str, thread_id: str = "default", force: bool = False) -> RunState:
        """Etapa A. Con `force=True` descarta un run en pausa y empieza de nuevo."""
        existing = await self.get_state(spreadsheet_id, thread_id)
        if existing and existing.status not in (RunStatus.NEW, RunStatus.DONE) and not force:
            raise InvalidRunState(f"Ya hay un run en {existing.status.value}; usa resume() o un thread_id nuevo.")
        state = RunState(run_id=uuid.uuid4().hex[:12], spreadsheet_id=spreadsheet_id, thread_id=thread_id)
        return await self._guarded(state, "A", self._phase_a)

    async def _phase_a(self, state: RunState) -> None:
        sid = state.spreadsheet_id
        # P1-01: Brief + Parámetros
        pb = await self.sheets.read_parameters_and_brief(sid)
        state.parameters, state.brief = pb["parameters"], pb["brief"]
        state.site_url, state.dominio = site_from_brief(state.brief)
        state.localizacion = localization_from_brief(state.brief, self.settings)
        state.memoria_cliente = await self.memory.recall(
            state.dominio, "reglas de tono, estructura y aprendizajes de SEO programático"
        )

        # P1-03: semillas
        seeds = [s for s in (_seed_row(r) for r in await self.sheets.read_tab(sid, self.settings.seeds_tab)) if s.get("keyword")]
        dedup = normalize_and_deduplicate_keywords(seeds)
        state.seed_counts = {k: dedup[k] for k in ("before_count", "after_count", "empty_removed")}

        # P1-04 + P1-05: Search Console
        end = self.today() - dt.timedelta(days=self.settings.gsc_lag_days)
        start = end - dt.timedelta(days=self.settings.gsc_days - 1)
        gsc_rows = await self.search_console.get_rows(state.site_url, start.isoformat(), end.isoformat())
        state.existing_urls = sorted({r["keys"][1] if "keys" in r else r.get("page") for r in gsc_rows} - {None})
        covered = mark_gsc_overlap(dedup["keywords"], gsc_rows)

        # Inventario del sitio (si el agente lo exploró): plataforma, sitemap y colecciones
        inventory = await self.state_store.load(sid, SITE_INVENTORY_KEY) or {}
        pages = site_pages(inventory)
        if inventory:
            state.plataforma = inventory.get("plataforma") or "desconocida"
            state.reglas_url = inventory.get("reglas_url") or url_rules(state.plataforma)
            state.site_urls = list(inventory.get("urls") or [])
            collection_urls = [c["url"] for c in inventory.get("colecciones") or [] if c.get("url")]
            state.existing_urls = sorted(set(state.existing_urls) | set(collection_urls))

        # P1-06: clústeres + intención
        p106 = await classify_and_cluster_keywords(covered, self.llm, self.embedder)
        state.clusters = p106["clusters"]
        keywords = p106["keywords"]

        # P1-07: PR-01 patrones
        own_rows = await self.sheets.read_tab(sid, self.settings.own_data_tab)
        clusters_payload = [{
            "cluster_id": c["cluster_id"],
            "keywords": [
                {k: kw.get(k) for k in ("keyword", "search_volume", "difficulty", "intent")}
                for kw in keywords if kw["cluster_id"] == c["cluster_id"]
            ],
        } for c in state.clusters]
        pr01: PR01Output = await self.llm.generate(PR01.id, {
            "clusters": clusters_payload,
            "datos_propios": _own_data_summary(own_rows),
            "memoria_cliente": state.memoria_cliente,
        })
        state.patrones = [p.model_dump() for p in pr01.patrones]
        pattern_by_key: dict[str, dict] = {}
        for patron in state.patrones:
            for kw in patron["keywords"]:
                pattern_by_key.setdefault(dedupe_key(kw), patron)
        for kw in keywords:
            patron = pattern_by_key.get(kw["dedupe_key"])
            kw["patron"] = patron["nombre"] if patron else None
            kw["patron_descartado"] = bool(patron and patron["descartado"])
            if "has_own_data" not in kw:
                kw["has_own_data"] = bool(patron and not patron["descartado"])

        # Keywords que Search Console ve libres pero que ya tienen página en el sitio (ej. una
        # colección sin tráfico): van a optimizar esa página, no a crear otra
        if pages:
            free = [kw["keyword"] for kw in keywords if kw.get("coverage_status") == "libre"]
            # Respaldo para keywords solas en su patrón: la parte que comparten todas las keywords
            groups = [p["keywords"] for p in state.patrones] + [[kw["keyword"] for kw in keywords]]
            matches = match_existing_pages(free, pages, groups)
            for kw in keywords:
                found = matches.get(kw["keyword"]) if kw.get("coverage_status") == "libre" else None
                if found:
                    kw.update(coverage_status="ya_cubierta", existing_url=found["url"],
                              existe_en_sitio=found.get("titulo") or found["url"])

        # P1-08 / P1-09: scoring y ruteo
        routed = score_and_route_keywords(keywords, state.parameters)
        state.routing_summary = routed["summary"]
        listed: list[dict] = []
        for kw in routed["create_new_url"]:
            if not kw["patron"] or kw["patron_descartado"]:
                patron = pattern_by_key.get(kw["dedupe_key"])
                motivo = f"patrón descartado: {patron['motivo']}" if patron else "sin patrón programático"
                state.descartados.append(["A", "P1-07", "keyword", kw["keyword"], "", motivo])
                continue
            listed.append({**kw, "ruta": "create_new_url", APPROVED_COL: ""})
        for kw in routed["optimize_existing"]:
            listed.append({**kw, "ruta": "optimize_existing", APPROVED_COL: "flujo 2"})
        for kw in routed["below_threshold"]:
            state.descartados.append(["A", "P1-09", "keyword", kw["keyword"], "", kw["route_reason"]])
        state.keywords = listed

        await self.sheets.write_tabs(sid, {
            TAB_KEYWORDS: _table(KEYWORD_COLUMNS, [self._keyword_row(k) for k in listed]),
            TAB_DESCARTADOS: [DISCARDED_COLUMNS] + state.descartados,
        })
        state.status = RunStatus.WAITING_KEYWORD_APPROVAL  # P1-10
        state.log("etapa_A", keywords=len(listed), descartadas=len(state.descartados))

    @staticmethod
    def _keyword_row(kw: dict, url_id: str = "") -> dict:
        return {
            **kw,
            "fuentes": ", ".join(kw.get("sources") or ([kw["source"]] if kw.get("source") else [])),
            "intencion": kw.get("intent"),
            "url_id": url_id,
        }

    # --------------------------------------------- Etapa B: P1-10 a P1-13 ----

    async def run_phase_3(self, spreadsheet_id: str, thread_id: str = "default") -> RunState:
        state = await self._require(spreadsheet_id, thread_id, RunStatus.WAITING_KEYWORD_APPROVAL)
        return await self._guarded(state, "B", self._phase_b)

    async def _phase_b(self, state: RunState) -> None:
        sid = state.spreadsheet_id
        # Idempotente: si la etapa se reintenta, no duplica sus propios descartes
        state.descartados = [d for d in state.descartados if d[0] != "B"]
        rows = await self.sheets.read_tab(sid, TAB_KEYWORDS)
        approved, decisions = [], {}
        for row in rows:
            if row.get("ruta") != "create_new_url":
                continue
            decision = is_approved(row.get(APPROVED_COL))
            decisions[row["keyword"]] = row
            if decision:
                approved.append(row["keyword"])
            else:
                motivo = str(row.get(REJECT_REASON_COL) or "").strip()
                reason = f"rechazada: {motivo or 'sin motivo'}" if decision is False else "sin aprobación"
                state.descartados.append(["B", "P1-10", "keyword", row["keyword"], "", reason])
        if not approved:
            # Sin aprobaciones no se avanza: el run queda esperando (se puede volver a correr)
            state.descartados = [d for d in state.descartados if d[0] != "B"]
            raise ValueError("No hay keywords con Aprobado = TRUE en la pestaña Keywords.")
        state.approved_keywords = approved

        # Guarda en el estado las decisiones humanas (para la pestaña final)
        for kw in state.keywords:
            row = decisions.get(kw["keyword"])
            if row:
                kw[APPROVED_COL] = row.get(APPROVED_COL, "")
                kw[REJECT_REASON_COL] = row.get(REJECT_REASON_COL, "")

        # P1-11: PR-02 con patrones aprobados
        own_rows = await self.sheets.read_tab(sid, self.settings.own_data_tab)
        approved_keys = {dedupe_key(k) for k in approved}
        by_key = {kw["dedupe_key"]: kw for kw in state.keywords}
        patterns_payload, variables = [], set()
        for patron in state.patrones:
            if patron["descartado"]:
                continue
            kws = [by_key[dedupe_key(k)] for k in patron["keywords"]
                   if dedupe_key(k) in approved_keys and dedupe_key(k) in by_key]
            if not kws:
                continue
            variables.update(patron["variables"])
            patterns_payload.append({
                "nombre": patron["nombre"], "variables": patron["variables"],
                "dato_diferenciador": patron["dato_diferenciador"],
                "keywords": [{"keyword": k["keyword"], "intencion": k.get("intent"),
                              "search_volume": k.get("search_volume")} for k in kws],
            })
        valores = {}
        for var in sorted(variables):
            column = _column_for(var, own_rows)
            valores[var] = sorted({str(r[column]) for r in own_rows if r.get(column)})[:200] if column else []
        reglas = state.reglas_url or url_rules(state.plataforma or "desconocida")
        pr02: PR02Output = await self.llm.generate(PR02.id, {
            "patrones": patterns_payload,
            "valores_variables": valores,
            "urls_existentes": state.existing_urls[:MAX_EXISTING_FOR_PROMPT],
            "plataforma": state.plataforma or "desconocida",
            "reglas_url": reglas,
        })

        # P1-12: slugs por fórmula. Si la plataforma fija los hubs (Shopify), el hub lo decide
        # la intención de la keyword principal, no el modelo.
        known_urls = state.existing_urls + state.site_urls
        url_tree: list[dict] | dict = pr02.model_dump()
        if not reglas.get("subcarpetas_libres", True):
            url_tree = nodes_from_pr02(url_tree)
            for node in url_tree:
                kw = by_key.get(dedupe_key(node.get("keyword")), {})
                node["hub"] = hub_for_intent(reglas, kw.get("intent")) or node["hub"]
        slugs = generate_and_validate_slugs(url_tree, known_urls,
                                            allowed_hubs=reglas.get("hubs_permitidos"))
        plan = []
        for i, page in enumerate(slugs, start=1):
            fila = match_own_data(page.get("variables") or {}, own_rows)
            plan.append({
                **page,
                "url_id": f"U{i:03d}",
                "url": page["url_path"],
                "patron": page.get("pattern"),
                "datos_propios": "ok" if fila else "sin_datos_propios",
                APPROVED_COL: "",
            })
        state.plan = plan

        await self.sheets.write_tabs(sid, {
            TAB_PLAN: _table(PLAN_COLUMNS, [self._plan_row(p) for p in plan]),
            TAB_DESCARTADOS: [DISCARDED_COLUMNS] + state.descartados,
        })
        state.status = RunStatus.WAITING_URL_APPROVAL  # P1-13
        state.log("etapa_B", aprobadas=len(approved), urls=len(plan))

    @staticmethod
    def _plan_row(page: dict, **extra: Any) -> dict:
        return {
            **page,
            "keywords_secundarias": "; ".join(page.get("keywords_secundarias") or []),
            "variables": page.get("variables") or {},
            "slug_issues": ", ".join(page.get("slug_issues") or []),
            **extra,
        }

    # -------------------------------------------- Etapa C: P1-14 a P1-25 ----

    async def run_phase_4_to_7(self, spreadsheet_id: str, thread_id: str = "default") -> RunState:
        state = await self._require(spreadsheet_id, thread_id, RunStatus.WAITING_URL_APPROVAL)
        return await self._guarded(state, "C", self._phase_c)

    async def _phase_c(self, state: RunState) -> None:
        sid = state.spreadsheet_id
        sheet_rows = {r.get("url_id"): r for r in await self.sheets.read_tab(sid, TAB_PLAN)}
        own_rows = await self.sheets.read_tab(sid, self.settings.own_data_tab)
        descartados = list(state.descartados)

        jobs: list[PageJob] = []
        for page in state.plan:
            row = sheet_rows.get(page["url_id"], {})
            page[APPROVED_COL] = row.get(APPROVED_COL, "")
            page[REJECT_REASON_COL] = row.get(REJECT_REASON_COL, "")
            decision = is_approved(row.get(APPROVED_COL))
            if not decision:
                motivo = str(row.get(REJECT_REASON_COL) or "").strip()
                reason = f"rechazada: {motivo or 'sin motivo'}" if decision is False else "sin aprobación"
                descartados.append(["C", "P1-13", "url", page["url"], page["url_id"], reason])
                page["estado"] = "no_aprobada"
                continue
            if not page["slug_valid"]:
                descartados.append(["C", "P1-13", "url", page["url"], page["url_id"],
                                    "slug inválido: " + ", ".join(page["slug_issues"])])
                page["estado"] = "slug_invalido"
                continue
            fila = match_own_data(page.get("variables") or {}, own_rows)
            if fila is None:
                descartados.append(["C", "P1-14", "url", page["url"], page["url_id"], "sin datos propios para sus variables"])
                page["estado"] = "sin_datos_propios"
                continue
            jobs.append(PageJob(
                url=page["url"], keyword=page["keyword_principal"],
                keywords_secundarias=page.get("keywords_secundarias") or [],
                hub=page["hub"], fila=fila, subtemas=page.get("keywords_secundarias") or [],
                tipo_schema_sugerido=state.brief.get("tipo_schema"),
            ))

        if not any(is_approved(sheet_rows.get(p["url_id"], {}).get(APPROVED_COL)) for p in state.plan):
            # Sin URLs aprobadas no se avanza: el run queda esperando (se puede volver a correr)
            raise ValueError("No hay URLs con Aprobado = TRUE en la pestaña Plan de URLs.")

        # P1-14 a P1-24
        pipeline = ContentPipeline(
            self.llm, self.embedder,
            PipelineParams.from_sheet(state.parameters, batch_size=self.batch_size,
                                      similarity_method=self.settings.similarity_method),
            existing_urls=state.existing_urls,
            generation_context={
                "localizacion": state.localizacion.model_dump() if state.localizacion else None,
                "memoria_cliente": state.memoria_cliente,
                "marca": state.brief.get("marca") or state.brief.get("cliente"),
            },
        )
        report = await pipeline.run(jobs)
        state.pipeline_summary = report.summary
        scores = [r.juicio.score for r in report.rows if r.juicio]
        state.content_summary = {
            "paginas_generadas": len(report.rows),
            "aprobadas": report.summary.get("aprobada", 0),
            "score_promedio_juez": round(sum(scores) / len(scores), 2) if scores else None,
            "revision_humana": [r.url for r in report.rows if r.status == "requiere_revision_humana"],
            "errores": [r.url for r in report.rows if r.status == "error"],
        }
        by_url = {r.url: r for r in report.rows}
        for page in state.plan:
            if page["url"] in by_url and "estado" not in page:
                page["estado"] = by_url[page["url"]].status
        for row in report.rows:
            if row.status != "aprobada":
                url_id = next(p["url_id"] for p in state.plan if p["url"] == row.url)
                motivo = row.error or (row.intentos[-1].motivo if row.intentos else row.status)
                descartados.append(["C", "P1-19", "url", row.url, url_id, f"{row.status}: {motivo}"])
        state.descartados = descartados

        # P1-25: escritura atómica de las 8 pestañas
        await self.sheets.write_tabs(sid, self._final_tabs(state, report))
        await self.memory.remember(state.dominio, self._learnings(state, report))
        state.status = RunStatus.DONE
        state.log("etapa_C", **report.summary)

    # ------------------------------------------------------ pestañas finales ----

    def _final_tabs(self, state: RunState, report: PipelineReport) -> dict[str, list[list]]:
        url_id_by_url = {p["url"]: p["url_id"] for p in state.plan}
        url_id_by_kw: dict[str, str] = {}
        for p in state.plan:
            for kw in [p["keyword_principal"], *(p.get("keywords_secundarias") or [])]:
                url_id_by_kw.setdefault(dedupe_key(kw), p["url_id"])

        content_rows, onpage_rows, schema_rows, link_rows = [], [], [], []
        for r in report.rows:
            url_id = url_id_by_url[r.url]
            c, j = r.contenido, r.juicio
            content_rows.append({
                "url_id": url_id, "url": r.url, "keyword": r.keyword, "estado": r.status,
                "score_juez": j.score if j else None, "motivo_juez": j.motivo if j else None,
                "intentos": len(r.intentos), "max_similarity": r.max_similarity, "similar_a": r.similar_a,
                "h1": c.h1 if c else None, "intro": c.intro if c else None,
                "secciones": "\n\n".join(f"## {s.h2}\n{s.texto}" for s in c.secciones) if c else None,
                "faq": "\n\n".join(f"P: {f.pregunta}\nR: {f.respuesta}" for f in c.faq) if c else None,
                "cta": c.cta if c else None,
                "bloques_omitidos": ", ".join(c.bloques_omitidos) if c else None,
                "issues": ", ".join(r.issues), "error": r.error,
            })
            if r.title_meta:
                sel = r.title_meta["seleccionada"] or {}
                onpage_rows.append({
                    "url_id": url_id, "url": r.url, "title": sel.get("title"),
                    "meta_description": sel.get("meta_description"), "title_length": sel.get("title_length"),
                    "meta_length": sel.get("meta_length"), "valido": bool(sel),
                    "variantes": r.title_meta["variantes"],
                })
            if r.schema_jsonld:
                schema_rows.append({
                    "url_id": url_id, "url": r.url, "tipo_schema": r.schema_jsonld["tipo_schema"],
                    "valido": r.schema_jsonld["validacion"]["valid"],
                    "issues": ", ".join(r.schema_jsonld["validacion"]["issues"]),
                    "campos_faltantes": ", ".join(r.schema_jsonld["campos_faltantes"]),
                    "jsonld": r.schema_jsonld["jsonld"],
                })
            if r.enlaces:
                for link in r.enlaces["valid_links"]:
                    link_rows.append({"url_id_origen": url_id, "url_origen": r.url,
                                      "url_id_destino": url_id_by_url.get(link["destino"], ""),
                                      **link, "estado": "aceptado", "issues": ""})
                for rejected in r.enlaces["rejected"]:
                    link = rejected["link"]
                    link_rows.append({"url_id_origen": url_id, "url_origen": r.url,
                                      "url_id_destino": url_id_by_url.get(link.get("destino"), ""),
                                      **link, "estado": "rechazado", "issues": ", ".join(rejected["issues"])})

        summary = [
            ["run_id", state.run_id], ["dominio", state.dominio], ["sitio", state.site_url],
            ["estado", RunStatus.DONE.value],
            ["localizacion", f"{state.localizacion.pais} / {state.localizacion.idioma} / {state.localizacion.trato}"
             if state.localizacion else ""],
            ["modelo_redactor", self.settings.writer_model], ["modelo_juez", self.settings.judge_model],
            ["semillas_antes", state.seed_counts.get("before_count")],
            ["semillas_despues", state.seed_counts.get("after_count")],
            ["keywords_crear", state.routing_summary.get("create_new_url")],
            ["keywords_optimizar", state.routing_summary.get("optimize_existing")],
            ["keywords_bajo_umbral", state.routing_summary.get("below_threshold")],
            ["keywords_aprobadas", len(state.approved_keywords)],
            ["urls_planificadas", len(state.plan)],
            ["urls_procesadas", report.summary.get("total")],
            ["urls_aprobadas", report.summary.get("aprobada")],
            ["urls_revision_humana", report.summary.get("requiere_revision_humana")],
            ["urls_error", report.summary.get("error")],
            ["descartados", len(state.descartados)],
            ["generado", dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")],
        ]
        return {
            TAB_RESUMEN: [["campo", "valor"]] + [[k, _cell(v)] for k, v in summary],
            TAB_KEYWORDS: _table(KEYWORD_COLUMNS, [
                self._keyword_row(k, url_id_by_kw.get(k["dedupe_key"], "")) for k in state.keywords
            ]),
            TAB_PLAN: _table(PLAN_COLUMNS + ["estado"], [self._plan_row(p) for p in state.plan]),
            TAB_CONTENIDO: _table(list(content_rows[0]) if content_rows else ["url_id"], content_rows),
            TAB_ONPAGE: _table(["url_id", "url", "title", "meta_description", "title_length", "meta_length",
                                "valido", "variantes"], onpage_rows),
            TAB_SCHEMA: _table(["url_id", "url", "tipo_schema", "valido", "issues", "campos_faltantes", "jsonld"],
                               schema_rows),
            TAB_ENLACES: _table(["url_id_origen", "url_origen", "url_id_destino", "destino", "tipo_destino",
                                 "anchor", "parrafo_literal", "justificacion", "estado", "issues"], link_rows),
            TAB_DESCARTADOS: [DISCARDED_COLUMNS] + [[_cell(v) for v in d] for d in state.descartados],
        }

    @staticmethod
    def _learnings(state: RunState, report: PipelineReport) -> list[str]:
        """Aprendizajes deterministas para el Memory Bank del cliente."""
        facts = []
        if state.localizacion:
            facts.append(f"Localización del cliente: {state.localizacion.regla}")
        rejected = [f"{d[3]} ({d[5]})" for d in state.descartados if d[1] == "P1-10" and d[5].startswith("rechazada")]
        if rejected:
            facts.append("Keywords rechazadas por el equipo: " + "; ".join(rejected[:20]))
        generic = Counter(f for r in report.rows if r.juicio for f in r.juicio.frases_genericas)
        if generic:
            facts.append("Frases genéricas señaladas por el juez (evitar): " + "; ".join(f for f, _ in generic.most_common(10)))
        review = [r.url for r in report.rows if r.status == "requiere_revision_humana"]
        if review:
            facts.append(f"{len(review)} URLs quedaron en revisión humana en el run {state.run_id}.")
        return facts
