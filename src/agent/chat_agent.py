"""Agente conversacional de SEO programático para Google Chat (ADK + Gemini).

La persona conversa; el agente pide los datos de a uno, llena la Sheet por ella,
corre el motor (scoring, Search Console, embeddings, slugs, validadores) y
explica el resultado en palabras simples. La Sheet queda como entregable.

- `build_chat_agent()`: el `LlmAgent` con sus herramientas (también es el
  `root_agent` de `adk web`).
- `ChatConversation`: corre un turno con el historial guardado por conversación
  (Firestore) y devuelve el texto ya en formato de Google Chat.
- `notify()`: las herramientas largas avisan cuánto van a tardar; la app de Chat
  usa ese aviso como respuesta inmediata (límite de 30 s) o lo publica en el hilo.
"""

from __future__ import annotations

import contextvars
import logging
import re
from typing import Any, Awaitable, Callable

from google.adk.agents import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from src.agent.chat_actions import GSC_ACCOUNT, SeoChatActions
from src.agent.memory_manager import APP_NAME, ConversationStore
from src.agent.service import SeoRunService, normalize_domain
from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

SHEET_KEY = "sheet_id"
CONVERSATION_KEY = "conversation_id"
CHAT_LIMIT = 3900  # Google Chat corta en 4096 caracteres por mensaje

# ------------------------------------------------------------- avisos ----

Notifier = Callable[[str], Awaitable[None]]
_notifier: contextvars.ContextVar[Notifier | None] = contextvars.ContextVar("seo_chat_notifier", default=None)


def set_notifier(notifier: Notifier | None) -> contextvars.Token:
    return _notifier.set(notifier)


async def notify(text: str) -> None:
    """Aviso de una herramienta larga ("tarda ~5 min"). Sin app de Chat, no hace nada."""
    notifier = _notifier.get()
    if notifier:
        try:
            await notifier(text)
        except Exception:  # noqa: BLE001 - un aviso fallido no corta la herramienta
            logger.exception("No pude enviar el aviso al chat")


# ------------------------------------------------------- herramientas ----

_actions: SeoChatActions | None = None


def get_actions() -> SeoChatActions:
    global _actions
    if _actions is None:
        _actions = SeoChatActions(SeoRunService())
    return _actions


def set_actions(actions: SeoChatActions | None) -> None:
    """Para tests: reemplaza las acciones que usan las herramientas."""
    global _actions
    _actions = actions


def _conversation(tool_context: ToolContext) -> str:
    return tool_context.state.get(CONVERSATION_KEY) or tool_context.session.id


def _sheet(tool_context: ToolContext) -> str | None:
    return tool_context.state.get(SHEET_KEY)


NO_SHEET = {"ok": False, "error": "Todavía no hay una Google Sheet en esta conversación.",
            "que_hacer": "Pedir la URL de una Google Sheet en blanco compartida como Editor con la cuenta del agente."}


async def use_sheet(sheet_url: str, tool_context: ToolContext) -> dict[str, Any]:
    """Conecta la Google Sheet donde quedarán los resultados de esta conversación.

    Revisa el acceso y crea las pestañas que falten sin tocar las que tienen datos.

    Args:
        sheet_url: URL (o ID) de la Google Sheet.
    """
    result = await get_actions().use_sheet(sheet_url)
    if result.get("ok"):
        tool_context.state[SHEET_KEY] = result["sheet_id"]
        return {**result, **await get_actions().overview(result["sheet_id"], _conversation(tool_context))}
    return result


async def get_overview(tool_context: ToolContext) -> dict[str, Any]:
    """Qué hay en la Sheet (cliente, keywords, datos propios) y en qué paso va el trabajo."""
    if not (sheet_id := _sheet(tool_context)):
        return NO_SHEET
    return await get_actions().overview(sheet_id, _conversation(tool_context))


async def set_client_info(tool_context: ToolContext, cliente: str = "", sitio: str = "", pais: str = "",
                          trato: str = "", tipo_schema: str = "") -> dict[str, Any]:
    """Guarda los datos del cliente. Solo se actualiza lo que venga con valor.

    Args:
        cliente: Nombre de la marca, como debe aparecer en el contenido.
        sitio: Dominio tal como está en Search Console (https://www.cliente.cl/ o sc-domain:cliente.cl).
        pais: País del público (por defecto Chile).
        trato: "tuteo" o "usted".
        tipo_schema: Service, Product, LocalBusiness, Article o BreadcrumbList (categorías de ecommerce).
    """
    if not (sheet_id := _sheet(tool_context)):
        return NO_SHEET
    return await get_actions().set_client(sheet_id, {"cliente": cliente, "sitio": sitio, "pais": pais,
                                                     "trato": trato, "tipo_schema": tipo_schema})


async def list_search_console_sites(dominio: str = "") -> dict[str, Any]:
    """Propiedades de Search Console a las que la cuenta del agente tiene acceso. Úsala para saber
    cómo se llama exactamente la propiedad del cliente antes de guardar el sitio.

    Args:
        dominio: Dominio del cliente (por ejemplo libreria-demo.cl) para filtrar; vacío lista todas.
    """
    return await get_actions().search_console_sites(dominio)


async def review_site(tool_context: ToolContext) -> dict[str, Any]:
    """Resumen de Search Console de los últimos 28 días: búsquedas top, secciones del sitio
    y búsquedas con impresiones donde el sitio está bajo la posición 10 (oportunidades)."""
    if not (sheet_id := _sheet(tool_context)):
        return NO_SHEET
    return await get_actions().site_snapshot(sheet_id)


async def explore_site(tool_context: ToolContext, dominio: str = "") -> dict[str, Any]:
    """Recorre el sitio del cliente: plataforma (Shopify, WordPress...), cómo se pueden crear URLs ahí,
    secciones del sitemap y colecciones o categorías que ya existen. Lo que encuentra queda guardado y el
    análisis de keywords lo usa para no proponer páginas que ya existen.

    Args:
        dominio: Dominio a explorar; vacío usa el sitio guardado del cliente.
    """
    if not (sheet_id := _sheet(tool_context)):
        return NO_SHEET
    await notify("⏳ Recorriendo el sitio, toma menos de un minuto.")
    return await get_actions().explore_site(sheet_id, dominio)


async def read_page(url: str, tool_context: ToolContext) -> dict[str, Any]:
    """Lee una página del sitio del cliente: title, meta description, H1, H2, texto principal y
    enlaces internos. Úsala para entender qué vende o publica el cliente (portada, una colección,
    una categoría) antes de proponer páginas.

    Args:
        url: URL completa o ruta (por ejemplo /collections/pediatria). Solo del sitio del cliente.
    """
    if not (sheet_id := _sheet(tool_context)):
        return NO_SHEET
    return await get_actions().read_page(sheet_id, url)


async def set_keywords(keywords: str, tool_context: ToolContext, modo: str = "reemplazar",
                       fuente: str = "") -> dict[str, Any]:
    """Guarda la lista de keywords con la que se trabajará.

    Args:
        keywords: Una keyword por línea. Opcional por línea: "keyword | volumen | dificultad | intención".
            Se puede pasar tal cual lo que pegó el usuario.
        modo: "reemplazar" (lista nueva), "agregar" o "quitar".
        fuente: De dónde salieron (Ubersuggest, Search Console, cliente...).
    """
    if not (sheet_id := _sheet(tool_context)):
        return NO_SHEET
    if modo not in ("reemplazar", "agregar", "quitar"):
        return {"ok": False, "error": "modo debe ser reemplazar, agregar o quitar."}
    return await get_actions().set_keywords(sheet_id, keywords, modo, fuente)


async def set_own_data(table: str, tool_context: ToolContext, modo: str = "reemplazar") -> dict[str, Any]:
    """Guarda los datos reales del cliente por variante (una fila por página posible).

    Args:
        table: Tabla con encabezado en la primera línea, tal como la pegó el usuario (copiada desde
            Sheets/Excel, markdown o CSV). La primera columna es la variable (ciudad, comuna, producto).
        modo: "agregar" si la persona dice agregar, sumar, «también» o pega la tabla en partes;
            "reemplazar" solo si pide cambiar los datos (borra los anteriores).
    """
    if not (sheet_id := _sheet(tool_context)):
        return NO_SHEET
    if modo not in ("reemplazar", "agregar"):
        return {"ok": False, "error": "modo debe ser reemplazar o agregar."}
    return await get_actions().set_own_data(sheet_id, table, modo)


async def analyze_keywords(tool_context: ToolContext) -> dict[str, Any]:
    """Analiza las keywords guardadas: cruza con Search Console, agrupa en patrones y decide cuáles
    sirven para páginas nuevas. Tarda varios minutos. Rehace el análisis desde cero (las selecciones
    y el plan anteriores se pierden)."""
    if not (sheet_id := _sheet(tool_context)):
        return NO_SHEET
    await notify("⏳ Analizando las keywords contra Search Console. Toma unos 5 minutos, te aviso aquí.")
    return await get_actions().analyze_keywords(sheet_id, _conversation(tool_context))


async def get_keyword_analysis(tool_context: ToolContext) -> dict[str, Any]:
    """Último análisis de keywords (y el plan de URLs si existe), sin volver a correrlo."""
    if not (sheet_id := _sheet(tool_context)):
        return NO_SHEET
    return await get_actions().keywords_status(sheet_id, _conversation(tool_context))


async def plan_urls(keywords: list[str], todas: bool, rechazar: list[str], tool_context: ToolContext,
                    motivo: str = "") -> dict[str, Any]:
    """Registra qué keywords para páginas nuevas eligió el usuario y arma el plan de URLs.

    Se puede volver a llamar para cambiar la selección (rehace el plan).

    Args:
        keywords: Keywords elegidas, escritas como en el análisis. Vacía si `todas` es true.
        todas: true para usar todas las keywords para páginas nuevas menos las de `rechazar`.
        rechazar: Keywords que el usuario no quiere.
        motivo: Por qué las rechazó, si lo dijo.
    """
    if not (sheet_id := _sheet(tool_context)):
        return NO_SHEET
    await notify("⏳ Armando el plan de URLs, toma un par de minutos.")
    return await get_actions().plan_urls(sheet_id, _conversation(tool_context), None if todas else keywords,
                                         rechazar, motivo)


async def generate_content(url_ids: list[str], todas: bool, rechazar: list[str], tool_context: ToolContext,
                           motivo: str = "") -> dict[str, Any]:
    """Registra qué URLs del plan eligió el usuario y genera su contenido, title, meta, schema y
    enlaces internos (con validadores y juez). Tarda varios minutos. Se puede volver a llamar para
    regenerar.

    Args:
        url_ids: IDs de las URLs elegidas (U001, U002...). Vacía si `todas` es true.
        todas: true para generar todas las URLs del plan menos las de `rechazar`.
        rechazar: IDs que el usuario no quiere.
        motivo: Por qué las rechazó, si lo dijo.
    """
    if not (sheet_id := _sheet(tool_context)):
        return NO_SHEET
    await notify("⏳ Escribiendo y revisando el contenido. Toma varios minutos, te aviso aquí.")
    return await get_actions().generate_content(sheet_id, _conversation(tool_context), None if todas else url_ids,
                                                rechazar, motivo)


async def show_page(url_id: str, tool_context: ToolContext) -> dict[str, Any]:
    """Title, meta description, H1, intro y nota del juez de una URL ya generada.

    Args:
        url_id: ID de la URL (U001...).
    """
    if not (sheet_id := _sheet(tool_context)):
        return NO_SHEET
    return await get_actions().page_detail(sheet_id, url_id)


async def save_client_preference(rule: str, tool_context: ToolContext, dominio: str = "") -> dict[str, Any]:
    """Guarda una regla del cliente para el contenido (tono, palabras prohibidas, formato de URLs).

    Args:
        rule: La regla, en una oración.
        dominio: Dominio del cliente; si se omite, se usa el sitio guardado.
    """
    if not dominio:
        if not (sheet_id := _sheet(tool_context)):
            return NO_SHEET
        dominio = ((await get_actions().overview(sheet_id, _conversation(tool_context))).get("cliente") or {}).get("sitio", "")
    if not dominio:
        return {"ok": False, "error": "No sé el dominio del cliente todavía."}
    return await get_actions().service.save_preference(normalize_domain(dominio.replace("sc-domain:", "")), rule)


TOOLS = [use_sheet, get_overview, list_search_console_sites, set_client_info, review_site, explore_site, read_page,
         set_keywords, set_own_data, analyze_keywords, get_keyword_analysis, plan_urls, generate_content, show_page,
         save_client_preference]


# --------------------------------------------------------- instrucción ----

INSTRUCTION = """Eres el agente de SEO programático de D2B y conversas por Google Chat con alguien del equipo de SEO.
Juntos crean páginas nuevas a escala para un cliente (por ejemplo «servicio + ciudad»), con datos reales del cliente.
Tú pides los datos de a uno, haces el trabajo pesado y explicas el resultado; la persona decide.

Estilo (Google Chat):
- Mensajes cortos: 2 a 6 líneas. Nada de introducciones ni despedidas, ni muletillas como «¡Perfecto!» o «¡Listo!».
- Nunca anuncies lo que vas a hacer: hazlo en el mismo turno (llama la herramienta) y cuenta el resultado.
- Una sola pregunta por mensaje, al final.
- Formato de Chat: *negrita* con un asterisco, _cursiva_, listas con «• ». Sin títulos (#), sin tablas, sin **.
- Links como <URL|texto>, solo con URLs que te dieron las herramientas, tal cual (nunca armes links a pestañas). Para listas largas muestra 5 a 8 ítems y di cuántos más hay.
- Español de Chile con tuteo, sin voseo. Cercano y directo.
- Nunca muestres nombres de herramientas, columnas, códigos internos ni JSON. Usa palabras simples:
  «ya tienes una página que rankea para eso» en vez de estados técnicos.
- No inventes cifras ni datos: usa solo lo que devuelven las herramientas.

Fidelidad (lo más importante):
- Cantidades, URLs, keywords y nombres de pestañas: cópialos tal cual de la última respuesta de la herramienta.
  Si muestras ejemplos de URLs, sácalos de la lista «urls»; si hay 1 URL, di que hay 1.
- Nunca deduzcas cuántas páginas saldrán multiplicando keywords por filas de datos propios.
  Cuántas URLs habrá solo lo sabes después de armar el plan.
- Nombra solo pestañas de «pestanas» y enlaza solo «sheet_url», sin agregarle nada.
- Al proponer keywords, muestra la lista completa que vas a guardar (o di cuántas son y muestra todas si son 15 o menos).
- Si una herramienta dice que se borraron filas, dilo.
- Al proponer keywords o páginas, no propongas las que ya tienen colección o página en el sitio:
  revisa las colecciones que encontraste; si existe, propón mejorar esa página.
- Sobre el sitio, afirma solo lo que viste en las páginas que leíste o en el inventario.

Cómo avanzar (en este orden, pero se puede volver a cualquier paso cuando la persona quiera):
1. Sheet: si no hay una, pide una Google Sheet en blanco compartida como Editor con {sa}. Ahí quedan los resultados.
2. Cliente: pide nombre y dominio. Busca sus propiedades de Search Console y guarda como sitio la sugerida, escrita
   exactamente igual (si no hay sugerida y hay varias, pregunta cuál). Solo si no aparece ninguna, pide agregar
   {gsc} como usuario Restringido. Luego revisa el sitio en Search Console.
   Después recorre el sitio: explóralo y lee las páginas que necesites para entender el negocio (la portada y
   2 o 3 colecciones o categorías clave). Cuenta en 2 a 4 líneas qué vende, en qué plataforma está, qué
   colecciones o secciones ya existen y dónde hay espacio para páginas nuevas.
   Si el sitio no se puede leer, dilo y sigue con Search Console.
3. Keywords: pide la lista que ya tengan (pegada en el chat; volumen y dificultad si los tienen).
   Sirven las búsquedas con variantes que se repiten: «internet hogar en Temuco», «libros de pediatría».
   Las búsquedas genéricas donde el cliente ya rankea no generan páginas nuevas. Si no tienen lista,
   propón una a partir de Search Console y pide el ok antes de guardarla.
4. Datos propios: pide los datos reales que cambian en cada variante (por ciudad: sucursales, precios, cobertura).
   Que los peguen como tabla (copiar desde Sheets o Excel sirve). La primera columna es la variante (ciudad,
   comuna, producto) escrita igual que en las keywords. Sin datos que cambien por página, no hay páginas.
5. Análisis: apenas tengas keywords y datos propios, córrelo en ese mismo turno, sin pedir permiso. Resume: cuáles sirven para páginas nuevas, cuáles ya tienen
   página (conviene mejorar esa, no crear otra) y las descartadas con su motivo en simple. Si ninguna sirve,
   di por qué y propone qué cambiar.
6. Selección: la persona elige keywords conversando. Arma el plan de URLs, muestra unas pocas de ejemplo y pregunta.
7. Contenido: genera las URLs que elija (tarda minutos). Al final: link a la Sheet, cuántas aprobó el juez y cuáles revisar.

Reglas:
- La persona aprueba conversando. Nunca le pidas que marque nada en la Sheet.
- Si ya dijo que sí a un paso largo, córrelo sin volver a preguntar.
- Para volver atrás: si cambian keywords o datos propios, guárdalos y vuelve a analizar; si cambian la selección,
  vuelve a armar el plan. Avisa en una línea qué se rehízo.
- Si la persona saca una variante («saca Temuco»): después del análisis es dejarla fuera de la selección; antes, quitar
  la keyword. No pidas rehacer los datos propios por eso.
- Si no sabes en qué va la conversación, revisa el estado antes de preguntar.
- Si la persona da una regla del cliente (tono, palabras prohibidas), guárdala.
- Si una herramienta falla, explica el error y qué hacer en una o dos líneas.
- No inventes cómo funcionas ni a qué tienes acceso: si no lo sabes, compruébalo con una herramienta.
{contexto}"""


def _instruction(settings: Settings) -> Callable[[ReadonlyContext], str]:
    # Proveedor de instrucción: ADK no reemplaza llaves en el texto y se agrega el contexto vigente
    def provider(ctx: ReadonlyContext) -> str:
        sheet_id = ctx.state.get(SHEET_KEY)
        contexto = f"\nSheet de esta conversación: https://docs.google.com/spreadsheets/d/{sheet_id}/edit" if sheet_id else \
            "\nEsta conversación todavía no tiene Sheet."
        if name := ctx.state.get("user_display_name"):
            contexto += f"\nConversas con {name}."
        return INSTRUCTION.format(sa=settings.service_account, gsc=GSC_ACCOUNT, contexto=contexto)
    return provider


def build_chat_agent(settings: Settings | None = None) -> Agent:
    from src.agent.subagents import configure_vertex_env

    settings = settings or get_settings()
    configure_vertex_env(settings)
    return Agent(
        name="seo_programatico",
        model=settings.orchestrator_model,
        description="Agente conversacional de SEO programático de D2B (Google Chat).",
        instruction=_instruction(settings),
        tools=list(TOOLS),
    )


# -------------------------------------------------- formato de Chat ----

_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
_BULLET = re.compile(r"^(\s*)[*-]\s+")
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}")
_SHEET_TAB_LINK = re.compile(r"(https://docs\.google\.com/spreadsheets/d/[\w-]+/edit)[?#]gid=\d+(?:#gid=\d+)?")


def to_chat_text(text: str) -> str:
    """Markdown común -> formato de Google Chat (por si el modelo igual lo usa)."""
    lines = []
    for line in (text or "").strip().splitlines():
        if _TABLE_RULE.match(line):
            continue
        if line.strip().startswith("|") and line.strip().endswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            line = "• " + " · ".join(c for c in cells if c)
        if m := _HEADING.match(line):
            line = f"*{m.group(1).strip('*')}*"
        line = _BULLET.sub(lambda m: f"{m.group(1)}• ", line)
        lines.append(line)
    out = "\n".join(lines)
    out = _BOLD.sub(lambda m: f"*{m.group(1) or m.group(2)}*", out)
    out = _LINK.sub(lambda m: f"<{m.group(2)}|{m.group(1)}>", out)
    out = _SHEET_TAB_LINK.sub(r"\1", out)  # links a pestañas armados por el modelo: el gid suele ser inventado
    return re.sub(r"\n{3,}", "\n\n", out)


def split_chat_text(text: str, limit: int = CHAT_LIMIT) -> list[str]:
    """Parte un texto largo en mensajes de Chat, cortando en saltos de línea."""
    chunks, current = [], ""
    for line in text.splitlines(keepends=True):
        while len(line) > limit:
            chunks.append(current + line[: limit - len(current)])
            line, current = line[limit - len(current):], ""
        if len(current) + len(line) > limit:
            chunks.append(current)
            current = ""
        current += line
    if current.strip():
        chunks.append(current)
    return [c.strip() for c in chunks if c.strip()]


# ------------------------------------------------------- conversación ----

def _trim(events: list[Event], keep: int) -> list[Event]:
    """Los últimos `keep` eventos, empezando en un mensaje del usuario (sin respuestas de
    herramientas huérfanas, que Gemini rechaza)."""
    if len(events) <= keep:
        return events
    starts = [i for i, e in enumerate(events) if e.author == "user" and i >= len(events) - keep]
    return events[starts[0]:] if starts else events[-1:]


class ChatConversation:
    """Un turno del agente con el historial de la conversación guardado entre instancias."""

    def __init__(self, agent: Agent, store: ConversationStore, settings: Settings | None = None):
        self.agent = agent
        self.store = store
        self.settings = settings or get_settings()

    async def run(self, conversation_id: str, text: str, user_display_name: str | None = None) -> str:
        saved = await self.store.load(conversation_id) or {}
        sessions = InMemorySessionService()
        state = {**(saved.get("state") or {}), CONVERSATION_KEY: conversation_id}
        if user_display_name:
            state["user_display_name"] = user_display_name
        session = await sessions.create_session(app_name=APP_NAME, user_id="chat", session_id="s", state=state)
        for raw in saved.get("events") or []:
            event = Event.model_validate(raw)
            if event.actions:
                event.actions.state_delta = {}  # el estado ya viene completo en `state`
            await sessions.append_event(session, event)

        runner = Runner(agent=self.agent, app_name=APP_NAME, session_service=sessions)
        message = types.Content(role="user", parts=[types.Part(text=text)])
        replies: list[str] = []
        async for event in runner.run_async(user_id="chat", session_id="s", new_message=message):
            if event.author == self.agent.name and event.content and event.content.parts and event.is_final_response():
                replies += [p.text for p in event.content.parts if p.text and not p.thought]

        session = await sessions.get_session(app_name=APP_NAME, user_id="chat", session_id="s")
        await self.store.save(conversation_id, {
            "state": {k: v for k, v in session.state.items() if not k.startswith("temp:")},
            "events": [e.model_dump(mode="json", exclude_none=True)
                       for e in _trim(session.events, self.settings.chat_history_events)],
        })
        return to_chat_text("\n\n".join(r.strip() for r in replies if r.strip())) or "Listo."
