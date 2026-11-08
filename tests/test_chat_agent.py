"""Agente conversacional: acciones sobre la Sheet, formato de Chat e historial por conversación."""

import pytest
from google.adk.agents import Agent
from google.adk.models import BaseLlm, LlmResponse
from google.genai import types

from src.agent import chat_agent
from src.agent.chat_actions import SeoChatActions, parse_keywords, parse_table
from src.agent.chat_agent import ChatConversation, split_chat_text, to_chat_text
from src.agent.memory_manager import InMemoryConversationStore
from src.config import Settings
from tests.test_chat_and_root_agent import SA, SID, URL, AnySheet, EmptySheet, make_service

THREAD = "spaces/AAA/threads/T1"


def actions(tmp_path, sheets=None):
    service, sheets = make_service(tmp_path, sheets=sheets)
    return SeoChatActions(service), sheets


# ------------------------------------------------------------------ parsers ----

def test_parse_keywords_plain_bullets_and_columns():
    text = "- seguro auto temuco | 150 | 20 | transaccional\n• seguro auto santiago\t1.600\n3. seguro auto talca"
    assert parse_keywords(text) == [
        {"keyword": "seguro auto temuco", "search_volume": 150, "difficulty": 20, "intent": "transaccional"},
        {"keyword": "seguro auto santiago", "search_volume": 1600},
        {"keyword": "seguro auto talca"},
    ]


def test_parse_keywords_with_header_and_markdown_table():
    text = "| Keyword | Volumen | KD |\n|---|---|---|\n| libros de pediatría | 320 | 22 |\n| libros de cardiología | 1,600 | 30 |"
    assert parse_keywords(text) == [
        {"keyword": "libros de pediatría", "search_volume": 320, "difficulty": 22},
        {"keyword": "libros de cardiología", "search_volume": 1600, "difficulty": 30},
    ]


def test_parse_table_from_sheets_copy_and_csv():
    tsv = "comuna\tplanes\tvelocidad_max\nMaipú\t4\t940 Mbps\nÑuñoa\t5\t1 Gbps"
    assert parse_table(tsv) == [{"comuna": "Maipú", "planes": 4, "velocidad_max": "940 Mbps"},
                                {"comuna": "Ñuñoa", "planes": 5, "velocidad_max": "1 Gbps"}]
    assert parse_table("ciudad,precio\nTemuco,\"$10.990\"") == [{"ciudad": "Temuco", "precio": "$10.990"}]
    assert parse_table("solo encabezado") == []


# ------------------------------------------------------------------ acciones ----

async def test_use_sheet_creates_missing_tabs_without_example_columns(tmp_path):
    act, sheets = actions(tmp_path, sheets=EmptySheet())
    result = await act.use_sheet(URL)
    assert result == {"ok": True, "sheet_id": SID, "sheet_url": f"https://docs.google.com/spreadsheets/d/{SID}/edit"}
    assert {"Brief", "Semillas", "Parámetros", "Datos propios"} <= set(sheets.raw)
    assert sheets.raw["Datos propios"] == [[]]  # sin columnas de ejemplo que confundan
    assert (await act.use_sheet("no es una sheet"))["ok"] is False


async def test_set_client_updates_brief_rows(tmp_path):
    act, sheets = actions(tmp_path, sheets=EmptySheet())
    await act.use_sheet(URL)
    result = await act.set_client(SID, {"cliente": "ACME Telecom", "sitio": "acme-telecom.cl", "trato": "de usted",
                                        "pais": ""})
    assert result == {"ok": True, "guardado": {"cliente": "ACME Telecom", "sitio": "https://acme-telecom.cl/",
                                               "trato": "usted"}}
    brief = {r[0]: r[1] for r in sheets.raw["Brief"][1:]}
    assert brief["cliente"] == "ACME Telecom" and brief["sitio"] == "https://acme-telecom.cl/" and brief["pais"] == "Chile"
    bad = await act.set_client(SID, {"tipo_schema": "Producto raro"})
    assert bad["ok"] is False and "Service" in bad["error"]


async def test_set_keywords_replace_add_remove(tmp_path):
    act, sheets = actions(tmp_path, sheets=EmptySheet())
    assert (await act.set_keywords(SID, "seguro auto temuco | 150\nseguro auto talca", "reemplazar", "cliente"))["total"] == 2
    added = await act.set_keywords(SID, "Seguro Auto Temuco | 180\nseguro auto arica", "agregar")
    assert added == {"ok": True, "total": 3, "recibidas": 2, "con_volumen": 1}
    rows = sheets.raw["Semillas"]
    assert rows[0] == ["keyword", "fuente", "volumen", "dificultad", "intencion"]
    assert ["Seguro Auto Temuco", "cliente", 180, "", ""] in rows
    removed = await act.set_keywords(SID, "seguro auto talca", "quitar")
    assert removed["total"] == 2 and all(r[0] != "seguro auto talca" for r in sheets.raw["Semillas"])
    assert (await act.set_keywords(SID, "  \n ", "agregar"))["ok"] is False


async def test_set_own_data_replace_and_add(tmp_path):
    act, sheets = actions(tmp_path, sheets=EmptySheet())
    await act.set_own_data(SID, "ciudad\tprecio_desde\nTemuco\t$10.990", "reemplazar")
    result = await act.set_own_data(SID, "ciudad\ttalleres\nTalca\t5", "agregar")
    assert result == {"ok": True, "filas": 2, "columnas": ["ciudad", "precio_desde", "talleres"]}
    assert sheets.raw["Datos propios"][2] == ["Talca", "", 5]
    replaced = await act.set_own_data(SID, "ciudad\tprecio_desde\nArica\t$9.990", "reemplazar")
    assert replaced == {"ok": True, "filas": 1, "columnas": ["ciudad", "precio_desde"], "filas_anteriores_borradas": 2}


async def test_site_snapshot_summarizes_search_console(tmp_path):
    act, _ = actions(tmp_path)
    snap = await act.site_snapshot(SID)
    assert snap["ok"] and snap["sitio"] == "https://www.acme.cl/" and snap["busquedas_distintas"] == 2
    assert snap["top_por_clics"][0]["query"] == "acme seguros"
    assert snap["secciones"] == {"seguros": 1, "(inicio)": 1}
    assert snap["periodo"] == "2026-09-14 a 2026-10-11"


async def test_search_console_sites_filters_and_suggests(tmp_path):
    act, _ = actions(tmp_path)
    med = await act.search_console_sites("www.Libreria-Demo.cl")
    assert med == {"ok": True, "propiedades": [{"sitio": "https://libreria-demo.cl/", "acceso": "completo"}],
                   "total": 1, "sugerida": "https://libreria-demo.cl/"}
    telecom = await act.search_console_sites("sc-domain:acme-telecom.cl")
    assert telecom["total"] == 3 and telecom["sugerida"] == "sc-domain:acme-telecom.cl"  # la de dominio cubre todo
    assert {"sitio": "sc-domain:acme-telecom.cl", "acceso": "restringido"} in telecom["propiedades"]
    acme = await act.search_console_sites("acme.cl")  # sin verificar = sin datos
    assert acme["total"] == 0 and acme["sugerida"] is None and "search-console-mcp-runner@" in acme["que_hacer"]
    assert (await act.search_console_sites())["total"] == 4


async def test_conversational_flow_with_rewinds(tmp_path):
    act, sheets = actions(tmp_path)
    report = await act.analyze_keywords(SID, THREAD)
    assert report["ok"] and report["totales"] == {"nuevas": 4, "ya_tienen_pagina": 1, "descartadas": 2}
    assert report["ya_tienen_pagina"] == [{"keyword": "seguro de auto", "volumen": 5000, "pagina_actual": "/seguros/auto",
                                           "posicion": 3.2}]
    assert report["grupos"][0]["dato_que_cambia"] and "paginas_posibles" not in report["grupos"][0]
    assert report["pestanas"] == ["Keywords", "Descartados"]
    assert "optimize_existing" not in str(report) and "coverage_status" not in str(report)

    # Elegir conversando: el agente marca Aprobado en la Sheet por la persona
    plan = await act.plan_urls(SID, THREAD, ["Seguro auto Santiago"], ["seguro auto temuco"], "sin cobertura")
    assert plan["ok"] and plan["total_urls"] == 1 and plan["urls"][0]["keyword"] == "seguro auto santiago"
    decisions = {r["keyword"]: (r["Aprobado"], r["Motivo rechazo"]) for r in sheets.tabs["Keywords"]}
    assert decisions["seguro auto santiago"] == (True, "") and decisions["seguro auto temuco"] == (False, "sin cobertura")

    # Cambiar la selección rehace el plan (volver atrás sin reiniciar)
    replan = await act.plan_urls(SID, THREAD, None, ["seguro auto temuco"], "")
    assert replan["ok"] and replan["total_urls"] >= 1
    assert (await act.overview(SID, THREAD))["paso_actual"] == "plan de URLs listo, falta elegir cuáles generar"

    done = await act.generate_content(SID, THREAD, ["u001"], [], "")
    assert done["ok"] and done["generadas"] == 1 and done["aprobadas_por_el_juez"] == 1
    page = await act.page_detail(SID, "u001")
    assert page["ok"] and page["title"] and page["h1"]

    # Regenerar después de terminar también funciona
    again = await act.generate_content(SID, THREAD, None, [], "")
    assert again["ok"]
    assert (await act.page_detail(SID, "U999"))["ok"] is False


async def test_plan_without_new_keywords_explains(tmp_path):
    class OnlyCovered(AnySheet):
        def __init__(self):
            super().__init__()
            self.tabs["Semillas"] = [{"Keyword": "seguro de auto", "Volumen": 5000}]

    act, _ = actions(tmp_path, sheets=OnlyCovered())
    report = await act.analyze_keywords(SID, THREAD)
    assert report["totales"]["nuevas"] == 0
    result = await act.plan_urls(SID, THREAD, None, [], "")
    assert result["ok"] is False and "variantes" in result["que_hacer"]


async def test_unknown_keywords_are_reported(tmp_path):
    act, _ = actions(tmp_path)
    await act.analyze_keywords(SID, THREAD)
    result = await act.plan_urls(SID, THREAD, ["seguro de mascotas en marte"], [], "")
    assert result["ok"] is False and result["no_encontradas"] == ["seguro de mascotas en marte"]
    assert "seguro auto santiago" in result["nuevas_disponibles"]


async def test_errors_are_explained_for_chat(tmp_path):
    from tests.test_chat_and_root_agent import Forbidden

    act, _ = actions(tmp_path, sheets=Forbidden())
    result = await act.use_sheet(URL)
    assert result["ok"] is False and result["error"] == "No tengo acceso a la Sheet." and SA in result["que_hacer"]
    assert "Avanzar" not in str(result) and "TRUE" not in str(result)


# ------------------------------------------------------------- formato ----

def test_to_chat_text_converts_markdown():
    md = "## Resultado\n**12 keywords** sirven, ver [la Sheet](https://docs.google.com/x)\n* una\n- dos\n\n\n| a | b |\n|---|---|\n| 1 | 2 |"
    assert to_chat_text(md) == ("*Resultado*\n*12 keywords* sirven, ver <https://docs.google.com/x|la Sheet>\n• una\n"
                                "• dos\n\n• a · b\n• 1 · 2")


def test_to_chat_text_drops_tab_ids_from_sheet_links():
    sheet = "https://docs.google.com/spreadsheets/d/abc-123/edit"
    assert to_chat_text(f"en la <{sheet}#gid=1588667688|pestaña Keywords>") == f"en la <{sheet}|pestaña Keywords>"
    assert to_chat_text(f"[Sheet]({sheet}?gid=0#gid=0)") == f"<{sheet}|Sheet>"
    assert to_chat_text("*ya en formato Chat*") == "*ya en formato Chat*"


def test_split_chat_text_respects_limit():
    text = "\n".join("x" * 50 for _ in range(10))
    chunks = split_chat_text(text, limit=120)
    assert all(len(c) <= 120 for c in chunks) and "".join(chunks).replace("\n", "") == text.replace("\n", "")
    assert split_chat_text("y" * 250, limit=100) == ["y" * 100, "y" * 100, "y" * 50]


# --------------------------------------------------------- conversación ----

class ScriptedLlm(BaseLlm):
    """Modelo falso: cada llamada devuelve la siguiente respuesta del guion."""

    script: list = []
    requests: list = []

    async def generate_content_async(self, llm_request, stream=False):
        self.requests.append(llm_request)
        step = self.script.pop(0)
        if isinstance(step, tuple):
            name, args = step
            part = types.Part(function_call=types.FunctionCall(name=name, args=args))
        else:
            part = types.Part(text=step)
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


def scripted_agent(script):
    llm = ScriptedLlm(model="fake", script=list(script), requests=[])
    agent = Agent(name="seo_programatico", model=llm, instruction=chat_agent._instruction(Settings()),
                  tools=list(chat_agent.TOOLS))
    return agent, llm


@pytest.fixture
def chat_actions(tmp_path):
    act, sheets = actions(tmp_path, sheets=EmptySheet())
    chat_agent.set_actions(act)
    yield act, sheets
    chat_agent.set_actions(None)


async def test_conversation_keeps_sheet_and_history_between_turns(chat_actions):
    _, sheets = chat_actions
    store = InMemoryConversationStore()
    agent, llm = scripted_agent([
        ("use_sheet", {"sheet_url": URL}),
        "**Listo**, conecté la Sheet. ¿Cuál es el cliente?",
        ("set_client_info", {"cliente": "ACME", "sitio": "acme.cl"}),
        "Guardé ACME.",
    ])
    conversation = ChatConversation(agent, store, Settings())

    first = await conversation.run(THREAD, f"hola, usa esta {URL}", "Ignacio")
    assert first == "*Listo*, conecté la Sheet. ¿Cuál es el cliente?"

    # Otra instancia (store compartido): recuerda la Sheet y la conversación
    second = await ChatConversation(agent, store, Settings()).run(THREAD, "ACME, acme.cl", "Ignacio")
    assert second == "Guardé ACME."
    assert {r[0]: r[1] for r in sheets.raw["Brief"][1:]}["sitio"] == "https://acme.cl/"

    saved = await store.load(THREAD)
    assert saved["state"]["sheet_id"] == SID and saved["state"]["conversation_id"] == THREAD
    instruction = llm.requests[-1].config.system_instruction
    assert SID in str(instruction) and "Ignacio" in str(instruction) and SA in str(instruction)
    # El modelo recibe el historial del turno anterior
    history = " ".join(p.text or "" for c in llm.requests[-1].contents for p in c.parts)
    assert "hola, usa esta" in history and "ACME, acme.cl" in history


async def test_history_is_trimmed_at_a_user_message(chat_actions):
    store = InMemoryConversationStore()
    settings = Settings(chat_history_events=4)
    for i in range(4):
        agent, _ = scripted_agent([("get_overview", {}), f"respuesta {i}"])
        await ChatConversation(agent, store, settings).run(THREAD, f"mensaje {i}")
    events = (await store.load(THREAD))["events"]
    assert len(events) <= 4 and events[0]["author"] == "user"


async def test_long_tool_notifies(chat_actions):
    act, _ = chat_actions
    await act.use_sheet(URL)
    seen = []

    async def notifier(text):
        seen.append(text)

    token = chat_agent.set_notifier(notifier)
    try:
        await chat_agent.notify("⏳ prueba")
    finally:
        chat_agent._notifier.reset(token)
    await chat_agent.notify("sin chat: no hace nada")
    assert seen == ["⏳ prueba"]


def test_root_agent_is_the_chat_agent():
    import src.agent
    from google.adk.agents import BaseAgent

    agent = src.agent.root_agent
    assert isinstance(agent, BaseAgent) and agent.name == "seo_programatico"
    assert [t.__name__ for t in agent.tools] == [t.__name__ for t in chat_agent.TOOLS]
