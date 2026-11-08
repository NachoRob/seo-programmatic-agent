"""Acciones del agente sobre un run, compartidas por el `root_agent` (ADK) y Google Chat.

Cada acción devuelve un dict serializable con la misma forma:
    {"ok", "sheet_id", "sheet_url", "status", "etapa_ejecutada"?, "aprobacion"?,
     "resumen", "siguiente_paso", "mensaje"?, "error"?, "que_hacer"?, "detalle_tecnico"?,
     "requiere_compartir_sheet"?}

`status` público: IDLE, WAITING_KEYWORD_APPROVAL, WAITING_URL_APPROVAL, COMPLETED.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable
from urllib.parse import urlparse

from src.agent.orchestrator import (
    APPROVED_COL,
    SearchConsoleError,
    REJECT_REASON_COL,
    TAB_KEYWORDS,
    TAB_PLAN,
    InvalidRunState,
    RunState,
    RunStatus,
    SeoProgrammaticOrchestrator,
    is_approved,
)
from src.config import Settings, default_parameter_rows, get_settings

IDLE = "IDLE"
COMPLETED = "COMPLETED"
PUBLIC_STATUS = {
    RunStatus.NEW: IDLE,
    RunStatus.WAITING_KEYWORD_APPROVAL: "WAITING_KEYWORD_APPROVAL",
    RunStatus.WAITING_URL_APPROVAL: "WAITING_URL_APPROVAL",
    RunStatus.DONE: COMPLETED,
}
NEXT_STAGE = {IDLE: "A", "WAITING_KEYWORD_APPROVAL": "B", "WAITING_URL_APPROVAL": "C"}
NEXT_STEP = {
    IDLE: "Si la Sheet está vacía, aprieta «🧩 Preparar la Sheet» y complétala siguiendo la pestaña "
          "Instrucciones. Si ya está completa, aprieta «▶️ Iniciar Etapa A».",
    "WAITING_KEYWORD_APPROVAL": "Abre la pestaña Keywords y escribe TRUE en la columna «Aprobado» de las keywords "
                                "que quieras convertir en páginas (FALSE + «Motivo rechazo» para descartar). "
                                "Después aprieta «Ya aprobé en la Sheet: Avanzar».",
    "WAITING_URL_APPROVAL": "Abre la pestaña Plan de URLs y escribe TRUE en la columna «Aprobado» de las URLs que "
                            "quieras generar. Después aprieta «Ya aprobé en la Sheet: Avanzar».",
    COMPLETED: "Revisa las pestañas Contenido, On-page, Schema y Enlaces internos. Las páginas en «revisión "
               "humana» necesitan una lectura antes de publicarse. Para otro run, escribe «reiniciar».",
}
_SHEET_ID = re.compile(r"/spreadsheets/d/([A-Za-z0-9_-]+)")
_BARE_ID = re.compile(r"^[A-Za-z0-9_-]{25,}$")


class SheetAccessError(RuntimeError):
    """La service account del agente no tiene acceso a la Sheet."""


def parse_sheet_id(value: str) -> str:
    """ID de la Sheet desde una URL de Google Sheets o un ID suelto."""
    value = (value or "").strip()
    if match := _SHEET_ID.search(value):
        return match.group(1)
    if _BARE_ID.match(value):
        return value
    raise ValueError(f"No reconozco una Google Sheet en: {value!r}")


def sheet_url(sheet_id: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit"


def normalize_domain(domain: str) -> str:
    domain = domain.strip().lower()
    host = urlparse(domain if "://" in domain else f"https://{domain}").hostname or domain
    return host[4:] if host.startswith("www.") else host


def is_sheet_permission_error(exc: BaseException) -> bool:
    """403 de la API de Sheets (Sheet no compartida con la service account)."""
    from googleapiclient.errors import HttpError

    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, HttpError) and getattr(current.resp, "status", None) == 403:
            return True
        if "The caller does not have permission" in str(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def access_message(settings: Settings) -> str:
    return (
        "No tengo acceso a la Sheet. Compártela como Editor con "
        f"{settings.service_account} (botón Compartir de la Sheet) y vuelve a intentarlo."
    )


def explain_error(exc: BaseException, settings: Settings) -> dict[str, Any]:
    """Traduce un error a {error, que_hacer, detalle_tecnico} para el equipo de SEO."""
    detail = f"{type(exc).__name__}: {exc}"[:500]
    text = str(exc)
    if is_sheet_permission_error(exc):
        return {"error": "No tengo acceso a la Google Sheet.",
                "que_hacer": f"Abre la Sheet → Compartir → agrega {settings.service_account} como Editor "
                             "(sin notificar) y vuelve a intentarlo.",
                "detalle_tecnico": detail, "requiere_compartir_sheet": True}
    if isinstance(exc, SearchConsoleError) or "Search Console" in text:
        site = getattr(exc, "site_url", "el sitio del Brief")
        return {"error": f"No pude leer Search Console para {site}.",
                "que_hacer": "1) Revisa que «sitio» en la pestaña Brief sea igual a la propiedad de Search "
                             "Console (https://www.cliente.cl/ o sc-domain:cliente.cl). 2) En Search Console → "
                             "Configuración → Usuarios y permisos, agrega "
                             "search-console-mcp-runner@your-gcp-project.iam.gserviceaccount.com como "
                             "usuario Restringido. Después vuelve a apretar Avanzar.",
                "detalle_tecnico": detail}
    if "El Brief no define el sitio" in text:
        return {"error": "Falta el sitio del cliente en la pestaña Brief.",
                "que_hacer": "En la pestaña Brief, escribe «sitio» en la columna A y el dominio en la columna B "
                             "(como está en Search Console, ej. https://www.cliente.cl/).",
                "detalle_tecnico": detail}
    if "pestaña Keywords" in text:
        return {"error": "Todavía no hay keywords aprobadas.",
                "que_hacer": "En la pestaña Keywords, escribe TRUE en la columna «Aprobado» de las keywords que "
                             "quieras convertir en páginas (FALSE + «Motivo rechazo» para descartar) y vuelve a "
                             "apretar Avanzar.",
                "detalle_tecnico": detail}
    if "pestaña Plan de URLs" in text:
        return {"error": "Todavía no hay URLs aprobadas.",
                "que_hacer": "En la pestaña Plan de URLs, escribe TRUE en la columna «Aprobado» de las URLs que "
                             "quieras generar y vuelve a apretar Avanzar.",
                "detalle_tecnico": detail}
    if isinstance(exc, InvalidRunState):
        return {"error": "Ese paso no corresponde a la etapa actual del run.",
                "que_hacer": "Escribe «estado» para ver en qué etapa va, o «reiniciar <url>» para empezar de cero.",
                "detalle_tecnico": detail}
    if any(k in text for k in ("RESOURCE_EXHAUSTED", "429", "503", "UNAVAILABLE", "DEADLINE_EXCEEDED")):
        return {"error": "El modelo de IA (Gemini) está saturado o no respondió a tiempo.",
                "que_hacer": "Espera un par de minutos y vuelve a apretar Avanzar. El avance quedó guardado.",
                "detalle_tecnico": detail}
    return {"error": "Algo falló de forma inesperada.",
            "que_hacer": "Vuelve a intentarlo. Si se repite, avisa al equipo de datos con el detalle técnico.",
            "detalle_tecnico": detail}


TEMPLATE_INSTRUCTIONS = [
    ["Cómo usar esta Sheet con el agente de SEO programático"],
    ["1. Brief: completa «cliente» y «sitio» (la propiedad de Search Console, ej. https://www.cliente.cl/)."],
    ["2. Semillas: una keyword por fila (columna «keyword»); volumen y dificultad si los tienes."],
    ["3. Datos propios: una fila por página posible (ej. por ciudad o producto) con datos REALES del cliente."],
    ["   Cambia los nombres de columna por los tuyos; más columnas = contenido más específico."],
    ["4. Parámetros: los valores por defecto sirven; ajusta pesos o umbrales si lo necesitas."],
    ["5. En Google Chat, pega la URL de esta Sheet y aprieta «Iniciar Etapa A»."],
    ["6. Etapa A escribe «Keywords»: marca TRUE en «Aprobado» las que quieras y aprieta «Avanzar»."],
    ["7. Etapa B escribe «Plan de URLs»: marca TRUE en «Aprobado» las URLs y aprieta «Avanzar»."],
    ["8. Etapa C escribe el contenido en Contenido, On-page, Schema y Enlaces internos."],
    ["Requisito: en Search Console del cliente, agrega search-console-mcp-runner@your-gcp-project.iam.gserviceaccount.com como usuario Restringido."],
]


def template_tabs(settings: Settings) -> dict[str, list[list]]:
    return {
        "Instrucciones": TEMPLATE_INSTRUCTIONS,
        settings.brief_tab: [
            ["clave", "valor", "ayuda"],
            ["cliente", "", "Nombre de la marca (se usa en el contenido y el schema)"],
            ["sitio", "", "Propiedad de Search Console: https://www.cliente.cl/ o sc-domain:cliente.cl"],
            ["pais", "Chile", ""],
            ["idioma", "es-CL", ""],
            ["trato", "tuteo", "tuteo o usted"],
            ["tipo_schema", "Service", "Service, Product, LocalBusiness o Article"],
        ],
        settings.parameters_tab: default_parameter_rows(settings),
        settings.seeds_tab: [["keyword", "fuente", "volumen", "dificultad", "intencion"]],
        settings.own_data_tab: [["ciudad", "precio_desde", "sucursales", "dato_diferenciador"]],
    }


def summarize(state: RunState | None) -> dict[str, Any]:
    """Resumen ejecutivo del run según la etapa alcanzada."""
    if state is None:
        return {}
    summary: dict[str, Any] = {"run_id": state.run_id, "dominio": state.dominio}
    if state.keywords or state.routing_summary:
        create = [k for k in state.keywords if k.get("ruta") == "create_new_url"]
        summary["keywords"] = {
            "total": len(state.keywords),
            "libres": sum(k.get("coverage_status") == "libre" for k in create),
            "parciales": sum(k.get("coverage_status") == "parcial" for k in state.keywords),
            "ya_cubiertas_gsc": sum(k.get("coverage_status") == "ya_cubierta" for k in state.keywords),
            "descartadas": sum(1 for d in state.descartados if d[0] == "A"),
            "patrones": [p["nombre"] for p in state.patrones if not p.get("descartado")],
        }
    if state.plan:
        summary["plan"] = {
            "urls": len(state.plan),
            "slugs_validos": sum(bool(p.get("slug_valid")) for p in state.plan),
            "sin_datos_propios": sum(p.get("datos_propios") == "sin_datos_propios" for p in state.plan),
            "keywords_aprobadas": len(state.approved_keywords),
        }
    if state.content_summary:
        summary["contenido"] = state.content_summary
    return summary


class SeoRunService:
    def __init__(
        self,
        orchestrator: SeoProgrammaticOrchestrator | None = None,
        settings: Settings | None = None,
        orchestrator_factory: Callable[[], SeoProgrammaticOrchestrator] | None = None,
    ):
        self.settings = settings or get_settings()
        self._orchestrator = orchestrator
        self._factory = orchestrator_factory

    @property
    def orchestrator(self) -> SeoProgrammaticOrchestrator:
        if self._orchestrator is None:
            if self._factory:
                self._orchestrator = self._factory()
            else:
                from src.agent.subagents import AdkStructuredLLM, GenAIEmbedder

                self._orchestrator = SeoProgrammaticOrchestrator(
                    AdkStructuredLLM(self.settings), GenAIEmbedder(self.settings), settings=self.settings
                )
        return self._orchestrator

    # ------------------------------------------------------------ helpers ----

    def _base(self, sheet_id: str, state: RunState | None) -> dict[str, Any]:
        status = PUBLIC_STATUS[state.status] if state else IDLE
        result = {
            "ok": True,
            "sheet_id": sheet_id,
            "sheet_url": sheet_url(sheet_id),
            "status": status,
            "resumen": summarize(state),
            "siguiente_paso": NEXT_STEP[status],
        }
        if state and state.last_error:
            result["ultimo_error"] = state.last_error
        return result

    def _error(self, sheet_id: str, message: str, access: bool = False, **extra: Any) -> dict[str, Any]:
        return {"ok": False, "sheet_id": sheet_id, "sheet_url": sheet_url(sheet_id),
                "error": message, "requiere_compartir_sheet": access, **extra}

    def _explain(self, sheet_id: str, exc: BaseException) -> dict[str, Any]:
        info = explain_error(exc, self.settings)
        return self._error(sheet_id, info.pop("error"), access=info.pop("requiere_compartir_sheet", False), **info)

    async def _approval_counts(self, sheet_id: str, status: str) -> dict[str, Any] | None:
        if status == "WAITING_KEYWORD_APPROVAL":
            tab, keep = TAB_KEYWORDS, (lambda r: r.get("ruta") == "create_new_url")
        elif status == "WAITING_URL_APPROVAL":
            tab, keep = TAB_PLAN, (lambda r: True)
        else:
            return None
        rows = [r for r in await self.orchestrator.sheets.read_tab(sheet_id, tab) if keep(r)]
        decisions = [is_approved(r.get(APPROVED_COL)) for r in rows]
        return {
            "pestaña": tab,
            "aprobadas": decisions.count(True),
            "rechazadas": decisions.count(False),
            "pendientes": decisions.count(None),
            "rechazos": [f"{r.get('keyword') or r.get('url')}: {r.get(REJECT_REASON_COL)}"
                         for r, d in zip(rows, decisions) if d is False and r.get(REJECT_REASON_COL)][:10],
        }

    # ------------------------------------------------------------ acciones ----

    async def status(self, sheet: str, thread_id: str = "default") -> dict[str, Any]:
        sheet_id = parse_sheet_id(sheet)
        state = await self.orchestrator.get_state(sheet_id, thread_id)
        result = self._base(sheet_id, state)
        try:
            counts = await self._approval_counts(sheet_id, result["status"])
        except Exception as e:  # noqa: BLE001
            return self._explain(sheet_id, e)
        if counts:
            result["aprobacion"] = counts
        return result

    async def preflight(self, sheet: str) -> dict[str, Any] | None:
        """Revisa que la Sheet tenga lo mínimo para la Etapa A. None si está todo bien."""
        from src.agent.orchestrator import site_from_brief

        sheet_id = parse_sheet_id(sheet)
        sheets = self.orchestrator.sheets
        settings = self.orchestrator.settings
        try:
            brief = (await sheets.read_parameters_and_brief(sheet_id)).get("brief") or {}
            seeds = await sheets.read_tab(sheet_id, settings.seeds_tab)
            own = await sheets.read_tab(sheet_id, settings.own_data_tab)
        except Exception as e:  # noqa: BLE001
            return self._explain(sheet_id, e)
        problems = []
        try:
            site_from_brief(brief)
        except ValueError:
            problems.append(f"Pestaña «{settings.brief_tab}»: falta la clave «sitio» con el dominio del "
                            "cliente tal como está en Search Console (ej. https://www.cliente.cl/).")
        if not any(str(r.get("keyword") or r.get("Keyword") or "").strip() for r in seeds):
            problems.append(f"Pestaña «{settings.seeds_tab}»: vacía o sin columna «keyword».")
        if not own:
            problems.append(f"Pestaña «{settings.own_data_tab}»: vacía (una fila por página posible, con datos reales).")
        if not problems:
            return None
        return self._error(
            sheet_id, "La Sheet todavía no está lista para empezar.", problemas=problems,
            que_hacer="Si la Sheet está vacía, aprieta «🧩 Preparar la Sheet» para crear las pestañas con sus "
                      "columnas. Complétalas y vuelve a apretar «Iniciar Etapa A».",
            puede_preparar=True,
        )

    async def prepare_sheet(self, sheet: str) -> dict[str, Any]:
        """Crea las pestañas de entrada con sus columnas (solo las que falten o estén vacías)."""
        sheet_id = parse_sheet_id(sheet)
        sheets = self.orchestrator.sheets
        try:
            tabs = template_tabs(self.orchestrator.settings)
            # Una pestaña con solo encabezados ya está preparada: no se reescribe
            missing = {name: rows for name, rows in tabs.items()
                       if not any(any(str(c).strip() for c in r) for r in await sheets.read_raw(sheet_id, name))}
            if missing:
                await sheets.write_tabs(sheet_id, missing)
        except Exception as e:  # noqa: BLE001
            return self._explain(sheet_id, e)
        kept = [name for name in tabs if name not in missing]
        return {
            "ok": True, "sheet_id": sheet_id, "sheet_url": sheet_url(sheet_id), "status": "IDLE",
            "creadas": list(missing), "sin_tocar": kept, "resumen": {},
            "mensaje": ("Creé las pestañas: " + ", ".join(missing) + "." if missing else
                        "Las pestañas ya existían con datos; no toqué nada.")
                       + (" Las que ya tenían datos no las toqué: " + ", ".join(kept) + "." if missing and kept else ""),
            "siguiente_paso": "Completa Brief (cliente y sitio), Semillas y Datos propios siguiendo la pestaña "
                              "Instrucciones, y después aprieta «Iniciar Etapa A».",
        }

    async def peek_status(self, sheet: str, thread_id: str = "default") -> str:
        """Estado público sin leer la Sheet (rápido, para decidir qué etapa toca)."""
        state = await self.orchestrator.get_state(parse_sheet_id(sheet), thread_id)
        return PUBLIC_STATUS[state.status] if state else IDLE

    async def _run(self, sheet_id: str, action) -> dict[str, Any]:
        started = time.time()
        try:
            state = await action()
        except Exception as e:  # noqa: BLE001 - se devuelve como resultado legible
            return self._explain(sheet_id, e)
        result = self._base(sheet_id, state)
        result["segundos"] = round(time.time() - started)
        return result

    async def advance(self, sheet: str, thread_id: str = "default") -> dict[str, Any]:
        """Corre la etapa que toca (A, B o C) y devuelve el resumen de lo escrito."""
        sheet_id = parse_sheet_id(sheet)
        current = await self.peek_status(sheet_id, thread_id)
        if current == COMPLETED:
            result = self._base(sheet_id, await self.orchestrator.get_state(sheet_id, thread_id))
            result["mensaje"] = "Este run ya terminó. Usa reiniciar para empezar uno nuevo."
            return result
        stage = NEXT_STAGE[current]
        result = await self._run(sheet_id, lambda: self.orchestrator.resume(sheet_id, thread_id))
        result["etapa_ejecutada"] = stage
        return result

    async def reset(self, sheet: str, thread_id: str = "default") -> dict[str, Any]:
        """Descarta el run de esta Sheet/hilo y corre la Etapa A de nuevo."""
        sheet_id = parse_sheet_id(sheet)
        result = await self._run(
            sheet_id, lambda: self.orchestrator.run_phase_1_and_2(sheet_id, thread_id, force=True)
        )
        result["etapa_ejecutada"] = "A"
        return result

    async def save_preference(self, domain: str, rule: str) -> dict[str, Any]:
        """Guarda una regla editorial o de negocio del cliente para futuros runs."""
        rule = (rule or "").strip()
        if not rule:
            return {"ok": False, "error": "La regla está vacía."}
        dominio = normalize_domain(domain)
        await self.orchestrator.memory.remember(dominio, [f"Preferencia del cliente: {rule}"])
        return {"ok": True, "dominio": dominio, "regla": rule,
                "mensaje": f"Guardé la regla para {dominio}; se usará desde el próximo run."}
