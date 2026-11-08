"""Servicio de runs y app de Google Chat (eventos, conversación y límite de 30 s)."""

import asyncio
import datetime as dt

import httplib2
import pytest
from fastapi.testclient import TestClient
from google.adk.sessions import InMemorySessionService
from googleapiclient.errors import HttpError

from src.agent.chat_agent import notify
from src.agent.memory_manager import AdkSessionStateStore, JsonFileMemoryStore
from src.agent.orchestrator import SeoProgrammaticOrchestrator
from src.agent.service import (
    SeoRunService,
    is_sheet_permission_error,
    normalize_domain,
    parse_sheet_id,
)
from src.chat_app import BUSY_TEXT, ERROR_TEXT, WELCOME_TEXT, WORKING_TEXT, create_app, parse_chat_event
from src.config import Settings
from tests.test_orchestrator import FakeEmbedder, FakeGSC, FakeSheets, FlowLLM

# ID con forma real (25+ caracteres), como el que viene en una URL de Google Sheets
REAL_ID = "1AbCdEfGhIjKlMnOpQrStUvWxYz0123456789_-abcd"  # ID ficticio con formato real
SID = "1AbCdEfGhIjKlMnOpQrStUvWxYz0123456789_-acme"
URL = f"https://docs.google.com/spreadsheets/d/{SID}/edit?gid=0#gid=0"
SA = "seo-programmatic-agent@your-gcp-project.iam.gserviceaccount.com"
SPACE, THREAD = "spaces/AAA", "spaces/AAA/threads/T1"


class AnySheet(FakeSheets):
    """FakeSheets sin atarse al ID del test del orquestador."""

    async def read_parameters_and_brief(self, spreadsheet_id):
        return await super().read_parameters_and_brief("sheet-acme")


class EmptySheet(AnySheet):
    def __init__(self):
        super().__init__()
        self.tabs = {}
        self.raw = {}

    async def read_parameters_and_brief(self, spreadsheet_id):
        return {"parameters": {}, "brief": {}, "raw_parameters": {}}


class Forbidden(FakeSheets):
    """La Sheet no está compartida con la service account."""

    async def read_parameters_and_brief(self, spreadsheet_id):
        raise HttpError(httplib2.Response({"status": 403}), b'{"error": {"message": "The caller does not have permission"}}')

    async def read_tab(self, spreadsheet_id, tab_name):
        raise HttpError(httplib2.Response({"status": 403}), b"forbidden")

    async def read_raw(self, spreadsheet_id, tab_name):
        raise HttpError(httplib2.Response({"status": 403}), b"forbidden")


def make_service(tmp_path, sheets=None):
    sheets = sheets or AnySheet()
    orchestrator = SeoProgrammaticOrchestrator(
        FlowLLM(), FakeEmbedder(), sheets=sheets, search_console=FakeGSC(),
        state_store=AdkSessionStateStore(InMemorySessionService()),
        memory=JsonFileMemoryStore(tmp_path / "memory.json"), settings=Settings(),
        today=lambda: dt.date(2026, 10, 14), batch_size=10,
    )
    return SeoRunService(orchestrator=orchestrator, settings=Settings()), sheets


class FakePoster:
    def __init__(self):
        self.posts = []

    async def post(self, space_name, thread_name, message):
        self.posts.append((space_name, thread_name, message))
        return {"name": f"{space_name}/messages/1"}


class FakeConversation:
    """Turno del agente simulado: puede avisar (herramienta larga), demorarse o fallar."""

    def __init__(self, reply="Hola 👋", delay=0.0, announce=None, fail=False):
        self.reply, self.delay, self.announce, self.fail = reply, delay, announce, fail
        self.calls = []

    async def run(self, conversation_id, text, user_display_name=None):
        self.calls.append((conversation_id, text, user_display_name))
        if self.announce:
            await notify(self.announce)
        await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("Gemini se cayó")
        return self.reply


def message_event(text, thread=THREAD, space_type="SPACE", user_type="HUMAN"):
    return {"type": "MESSAGE", "space": {"name": SPACE, "spaceType": space_type},
            "user": {"name": "users/1", "displayName": "Ignacio", "type": user_type},
            "message": {"text": text, "thread": {"name": thread}, "space": {"name": SPACE}}}


def click_event(function, sheet_id, thread=THREAD):
    return {"type": "CARD_CLICKED", "space": {"name": SPACE}, "user": {"name": "users/1"},
            "message": {"thread": {"name": thread}},
            "action": {"actionMethodName": function,
                       "parameters": [{"key": "sheet_id", "value": sheet_id}, {"key": "thread_name", "value": thread}]},
            "common": {"invokedFunction": function}}


def make_chat(conversation=None, **settings):
    conversation = conversation or FakeConversation()
    poster = FakePoster()
    app = create_app(conversation=conversation, poster=poster, settings=Settings(**settings))
    return TestClient(app), poster, conversation, app


# ------------------------------------------------------------------ helpers ----

def test_parse_sheet_id():
    assert parse_sheet_id(f"https://docs.google.com/spreadsheets/d/{REAL_ID}/edit?gid=0#gid=0") == REAL_ID
    assert parse_sheet_id(REAL_ID) == REAL_ID
    with pytest.raises(ValueError):
        parse_sheet_id("https://acme.cl")


def test_normalize_domain():
    assert normalize_domain("https://www.ACME.cl/seguros") == "acme.cl"
    assert normalize_domain("acme.cl") == "acme.cl"


def test_permission_error_detection():
    assert is_sheet_permission_error(HttpError(httplib2.Response({"status": 403}), b"x"))
    assert not is_sheet_permission_error(HttpError(httplib2.Response({"status": 500}), b"x"))
    wrapped = RuntimeError("falló")
    wrapped.__cause__ = HttpError(httplib2.Response({"status": 403}), b"x")
    assert is_sheet_permission_error(wrapped)


# ------------------------------------------------------------------ servicio ----

async def test_service_full_cycle_and_summaries(tmp_path):
    service, sheets = make_service(tmp_path)

    idle = await service.status(URL, THREAD)
    assert idle["status"] == "IDLE" and "Etapa A" in idle["siguiente_paso"]

    a = await service.advance(URL, THREAD)
    assert a["ok"] and a["etapa_ejecutada"] == "A" and a["status"] == "WAITING_KEYWORD_APPROVAL"
    assert a["resumen"]["keywords"] == {"total": 5, "libres": 4, "parciales": 0, "ya_cubiertas_gsc": 1,
                                        "descartadas": 2, "patrones": ["seguro auto por ciudad"]}

    sheets.approve("Keywords", "keyword", {"seguro auto santiago": True, "seguro auto temuco": (False, "sin cobertura")})
    status = await service.status(URL, THREAD)
    assert status["aprobacion"] == {"pestaña": "Keywords", "aprobadas": 1, "rechazadas": 1, "pendientes": 2,
                                    "rechazos": ["seguro auto temuco: sin cobertura"]}

    b = await service.advance(URL, THREAD)
    assert b["etapa_ejecutada"] == "B" and b["status"] == "WAITING_URL_APPROVAL"
    assert b["resumen"]["plan"]["urls"] == 1

    sheets.approve("Plan de URLs", "url_id", {"U001": True})
    c = await service.advance(URL, THREAD)
    assert c["etapa_ejecutada"] == "C" and c["status"] == "COMPLETED"
    assert c["resumen"]["contenido"] == {"paginas_generadas": 1, "aprobadas": 1, "score_promedio_juez": 5.0,
                                         "revision_humana": [], "errores": []}

    again = await service.advance(URL, THREAD)
    assert again["status"] == "COMPLETED" and "reiniciar" in again["mensaje"]

    reset = await service.reset(URL, THREAD)
    assert reset["status"] == "WAITING_KEYWORD_APPROVAL" and reset["etapa_ejecutada"] == "A"


async def test_reset_discards_paused_run(tmp_path):
    service, _ = make_service(tmp_path)
    first = await service.advance(URL, THREAD)
    second = await service.reset(URL, THREAD)
    assert second["resumen"]["run_id"] != first["resumen"]["run_id"]


async def test_service_reports_missing_approvals(tmp_path):
    service, _ = make_service(tmp_path)
    await service.advance(URL, THREAD)
    result = await service.advance(URL, THREAD)
    assert result["ok"] is False and result["error"] == "Todavía no hay keywords aprobadas."
    assert "TRUE" in result["que_hacer"] and "Keywords" in result["que_hacer"]
    assert "Aprobado = TRUE" in result["detalle_tecnico"]
    assert (await service.status(URL, THREAD))["status"] == "WAITING_KEYWORD_APPROVAL"


async def test_service_403_when_sheet_not_shared(tmp_path):
    service, _ = make_service(tmp_path, sheets=Forbidden())
    result = await service.advance(URL, THREAD)
    assert result["ok"] is False and result["requiere_compartir_sheet"] is True
    assert result["error"] == "No tengo acceso a la Google Sheet."
    assert SA in result["que_hacer"]


async def test_save_preference_feeds_next_run(tmp_path):
    service, _ = make_service(tmp_path)
    saved = await service.save_preference("https://www.acme.cl/", "No usar la palabra 'barato'")
    assert saved == {"ok": True, "dominio": "acme.cl", "regla": "No usar la palabra 'barato'",
                     "mensaje": "Guardé la regla para acme.cl; se usará desde el próximo run."}
    await service.advance(URL, THREAD)
    llm = service.orchestrator.llm
    assert "Preferencia del cliente: No usar la palabra 'barato'" in llm.payloads("PR-01")[-1]["memoria_cliente"]
    assert (await service.save_preference("acme.cl", "  "))["ok"] is False


async def test_prepare_sheet_creates_only_missing_tabs(tmp_path):
    sheets = EmptySheet()
    sheets.tabs = {"Semillas": [{"keyword": "seguro auto"}]}  # ya tiene datos: no se toca
    service, _ = make_service(tmp_path, sheets=sheets)
    result = await service.prepare_sheet(URL)
    assert result["ok"] and result["sin_tocar"] == ["Semillas"]
    assert set(result["creadas"]) == {"Instrucciones", "Brief", "Parámetros", "Datos propios"}
    assert sheets.tabs["Semillas"] == [{"keyword": "seguro auto"}]
    written = sheets.writes[-1]
    assert written["Brief"][0] == ["clave", "valor", "ayuda"]
    assert ["umbral_similitud", 0.5] in written["Parámetros"]
    # El usuario renombra columnas de Datos propios (sin filas aún): Preparar no las pisa
    sheets.raw["Datos propios"] = [["comuna", "precio"]]
    again = await service.prepare_sheet(URL)
    assert again["creadas"] == [] and "no toqué nada" in again["mensaje"]
    assert sheets.raw["Datos propios"] == [["comuna", "precio"]]


async def test_stage_c_without_approved_urls_explains(tmp_path):
    service, sheets = make_service(tmp_path)
    await service.advance(URL, THREAD)
    sheets.approve("Keywords", "keyword", {"seguro auto santiago": True})
    await service.advance(URL, THREAD)
    result = await service.advance(URL, THREAD)
    assert result["ok"] is False and result["error"] == "Todavía no hay URLs aprobadas."
    assert (await service.status(URL, THREAD))["status"] == "WAITING_URL_APPROVAL"


def test_explain_search_console_error():
    from src.agent.orchestrator import SearchConsoleError
    from src.agent.service import explain_error

    info = explain_error(SearchConsoleError("https://www.cliente.cl/", "403 User does not have permission"), Settings())
    assert info["error"] == "No pude leer Search Console para https://www.cliente.cl/."
    assert "search-console-mcp-runner@" in info["que_hacer"] and "403" in info["detalle_tecnico"]


# ------------------------------------------------------------- eventos de Chat ----

def test_parse_message_in_threaded_space():
    req = parse_chat_event(message_event("hola, quiero páginas para ACME"))
    assert (req.command, req.text, req.thread_name, req.space_name) == ("message", "hola, quiero páginas para ACME",
                                                                         THREAD, SPACE)
    assert req.conversation_id == THREAD and req.reply_thread == THREAD and req.user_display_name == "Ignacio"


def test_dm_is_one_conversation_per_space():
    req = parse_chat_event(message_event("hola", space_type="DIRECT_MESSAGE"))
    assert req.is_dm and req.conversation_id == SPACE and req.reply_thread is None


def test_parse_mention_uses_argument_text():
    event = message_event("@SEO bot revisa esto")
    event["message"]["argumentText"] = " revisa esto "
    assert parse_chat_event(event).text == "revisa esto"


def test_old_card_buttons_become_messages():
    req = parse_chat_event(click_event("advance_seo_run", REAL_ID, thread="spaces/AAA/threads/T9"))
    assert req.is_card_click and req.command == "message" and req.thread_name == "spaces/AAA/threads/T9"
    assert req.text.startswith("Sigamos con el siguiente paso.") and REAL_ID in req.text


def test_parse_added_unknown_empty_and_bot_events():
    assert parse_chat_event({"type": "ADDED_TO_SPACE", "space": {"name": SPACE}}).command == "welcome"
    assert parse_chat_event({"type": "REMOVED_FROM_SPACE"}).command == "ignore"
    assert parse_chat_event(message_event("   ")).command == "ignore"
    assert parse_chat_event(message_event("hola", user_type="BOT")).command == "ignore"


# ------------------------------------------------------------- conversación ----

def test_fast_turn_answers_directly_in_thread():
    client, poster, conversation, _ = make_chat(FakeConversation(reply="¿Para qué cliente?"))
    response = client.post("/chat", json=message_event("hola")).json()
    assert response == {"text": "¿Para qué cliente?", "thread": {"name": THREAD}}
    assert conversation.calls == [(THREAD, "hola", "Ignacio")]
    assert poster.posts == []


def test_slow_turn_answers_working_and_posts_later():
    client, poster, _, _ = make_chat(FakeConversation(reply="Listo el análisis", delay=0.3), chat_sync_seconds=0.05)
    response = client.post("/chat", json=message_event("analiza")).json()
    assert response["text"] == WORKING_TEXT
    assert poster.posts == [(SPACE, THREAD, {"text": "Listo el análisis"})]


def test_long_tool_announcement_is_the_immediate_answer():
    aviso = "⏳ Analizando las keywords contra Search Console. Toma unos 5 minutos, te aviso aquí."
    client, poster, _, _ = make_chat(FakeConversation(reply="12 sirven", delay=0.3, announce=aviso),
                                     chat_sync_seconds=5)
    # No espera los 5 s: el aviso sale al tiro y el resultado se publica al terminar
    response = client.post("/chat", json=message_event("dale")).json()
    assert response["text"] == aviso
    assert poster.posts == [(SPACE, THREAD, {"text": "12 sirven"})]


def test_long_reply_is_split_into_several_messages():
    long_reply = "\n".join(f"• keyword {i} " + "x" * 80 for i in range(80))
    client, poster, _, _ = make_chat(FakeConversation(reply=long_reply))
    response = client.post("/chat", json=message_event("lista todo")).json()
    assert len(response["text"]) <= 3900 and poster.posts
    assert response["text"].replace("\n", "") + "".join(p[2]["text"].replace("\n", "") for p in poster.posts) \
        == long_reply.replace("\n", "")


def test_failed_turn_always_answers():
    client, _, _, app = make_chat(FakeConversation(fail=True))
    assert client.post("/chat", json=message_event("hola")).json()["text"] == ERROR_TEXT
    # El lock se liberó: el siguiente mensaje se atiende
    bot = app.state.bot
    assert asyncio.run(bot.lock.acquire(bot.lock_key(THREAD), "x", 60))


def test_busy_conversation_is_not_run_twice():
    client, _, conversation, app = make_chat()
    bot = app.state.bot
    assert asyncio.run(bot.lock.acquire(bot.lock_key(THREAD), "otra-instancia", 600))
    assert client.post("/chat", json=message_event("hola")).json()["text"] == BUSY_TEXT
    assert conversation.calls == []
    # Otro hilo es otra conversación
    assert client.post("/chat", json=message_event("hola", thread="spaces/AAA/threads/T2")).json()["text"] == "Hola 👋"


def test_dm_reply_has_no_thread():
    client, _, conversation, _ = make_chat()
    response = client.post("/chat", json=message_event("hola", space_type="DIRECT_MESSAGE")).json()
    assert response == {"text": "Hola 👋"} and conversation.calls[0][0] == SPACE


def test_welcome_and_card_click():
    client, _, conversation, _ = make_chat()
    assert client.post("/chat", json={"type": "ADDED_TO_SPACE", "space": {"name": SPACE}}).json()["text"] == WELCOME_TEXT
    clicked = client.post("/chat", json=click_event("check_run_status", SID)).json()
    assert clicked["actionResponse"] == {"type": "NEW_MESSAGE"} and clicked["text"] == "Hola 👋"
    assert conversation.calls[-1][1].startswith("¿En qué vamos?")


def test_endpoint_requires_chat_token_when_audience_configured():
    client, _, _, _ = make_chat(chat_audience="123456789012")
    assert client.post("/chat", json=message_event("hola")).status_code == 401
    assert client.post("/chat", json=message_event("hola"),
                       headers={"Authorization": "Bearer no-es-un-jwt"}).status_code == 401


@pytest.mark.parametrize("mode", ["project_number", "endpoint_url"])
def test_verify_chat_token_accepts_both_console_modes(monkeypatch, mode):
    from google.oauth2 import id_token

    from src.chat_app import CHAT_ISSUER, verify_chat_token

    url = "https://seo-programmatic-agent-xyz.a.run.app/chat"
    if mode == "project_number":
        monkeypatch.setattr(id_token, "verify_token",
                            lambda token, request, audience, certs_url: {"iss": CHAT_ISSUER, "aud": audience})
        monkeypatch.setattr(id_token, "verify_oauth2_token", lambda *a, **k: (_ for _ in ()).throw(ValueError("x")))
    else:
        monkeypatch.setattr(id_token, "verify_token", lambda *a, **k: (_ for _ in ()).throw(ValueError("aud")))
        monkeypatch.setattr(id_token, "verify_oauth2_token",
                            lambda token, request, audience: {"email": CHAT_ISSUER, "email_verified": True, "aud": audience})
    verify_chat_token("Bearer t.t.t", f"123456789012;{url}")  # no lanza


def test_verify_chat_token_rejects_other_issuers(monkeypatch, caplog):
    from fastapi import HTTPException
    from google.oauth2 import id_token

    from src.chat_app import verify_chat_token

    monkeypatch.setattr(id_token, "verify_token", lambda *a, **k: {"iss": "otro@x.iam.gserviceaccount.com"})
    monkeypatch.setattr(id_token, "verify_oauth2_token", lambda *a, **k: {"email": "otro@x.com", "email_verified": True})
    with pytest.raises(HTTPException):
        verify_chat_token("Bearer t.t.t", "123456789012,https://x/chat")
    assert "Token de Chat rechazado" in caplog.text


# ----------------------------------------------- app como complemento de Workspace ----

ENDPOINT = "https://seo-programmatic-agent-la26zohasa-uc.a.run.app/chat"


def addon_message(text, thread=THREAD):
    return {
        "commonEventObject": {"hostApp": "CHAT", "userLocale": "es"},
        "chat": {"user": {"name": "users/1", "displayName": "Ignacio"}, "messagePayload": {
            "space": {"name": SPACE},
            "message": {"text": text, "thread": {"name": thread}, "space": {"name": SPACE}},
        }},
    }


def addon_message_out(response):
    return response["hostAppDataAction"]["chatDataAction"]["createMessageAction"]["message"]


def test_parse_addon_events():
    msg = parse_chat_event(addon_message("hola"))
    assert msg.addon and msg.command == "message" and msg.text == "hola"
    assert (msg.thread_name, msg.space_name, msg.user_display_name) == (THREAD, SPACE, "Ignacio")
    added = parse_chat_event({"commonEventObject": {}, "chat": {"addedToSpacePayload": {"space": {"name": SPACE}}}})
    assert added.addon and added.command == "welcome"


def test_addon_endpoint_responds_in_host_app_format():
    client, poster, _, _ = make_chat(FakeConversation(reply="Hecho", delay=0.3), chat_sync_seconds=0.05)
    response = client.post("/chat", json=addon_message("analiza")).json()
    assert addon_message_out(response) == {"text": WORKING_TEXT}
    assert poster.posts == [(SPACE, THREAD, {"text": "Hecho"})]


@pytest.mark.parametrize("email, accepted", [
    ("service-123456789012@gcp-sa-gsuiteaddons.iam.gserviceaccount.com", True),
    ("service-999@gcp-sa-gsuiteaddons.iam.gserviceaccount.com", False),  # otro proyecto
])
def test_verify_chat_token_addon_issuer(monkeypatch, email, accepted):
    from fastapi import HTTPException
    from google.oauth2 import id_token

    from src.chat_app import verify_chat_token

    monkeypatch.setattr(id_token, "verify_token", lambda *a, **k: (_ for _ in ()).throw(ValueError("kid")))
    monkeypatch.setattr(id_token, "verify_oauth2_token", lambda token, request, audience: {
        "iss": "https://accounts.google.com", "aud": audience, "email": email, "email_verified": True})
    audiences = f"123456789012;{ENDPOINT}"
    if accepted:
        verify_chat_token("Bearer t.t.t", audiences)
    else:
        with pytest.raises(HTTPException):
            verify_chat_token("Bearer t.t.t", audiences)


def test_health():
    assert make_chat()[0].get("/health").json() == {"ok": True}


# ------------------------------------------- plazo de 30 s desde el evento ----

from src.chat_app import ChatBot, ChatRequest, parse_event_time  # noqa: E402

NOW = dt.datetime(2026, 10, 27, 14, 10, 0, tzinfo=dt.UTC)


@pytest.mark.parametrize(("value", "expected"), [
    ("2026-10-27T14:09:47.123456789Z", dt.datetime(2026, 10, 27, 14, 9, 47, 123456, tzinfo=dt.UTC)),
    ("2026-10-27T14:09:47Z", dt.datetime(2026, 10, 27, 14, 9, 47, tzinfo=dt.UTC)),
    ("2026-10-27T11:09:47-03:00", dt.datetime(2026, 10, 27, 14, 9, 47, tzinfo=dt.UTC)),
    ({"seconds": NOW.timestamp(), "nanos": 0}, NOW),
    ("", None), (None, None), ("ayer", None), ({"seconds": "x"}, None),
])
def test_parse_event_time(value, expected):
    assert parse_event_time(value) == expected


def bot_budget(seconds_ago, **settings):
    bot = ChatBot(FakeConversation(), FakePoster(), Settings(**settings), lock=None, now=lambda: NOW)
    when = None if seconds_ago is None else NOW - dt.timedelta(seconds=seconds_ago)
    return bot.sync_budget(ChatRequest(command="message", event_type="MESSAGE", event_time=when))


def test_sync_budget_discounts_cold_start():
    # Producción, 27-10: arranque de 13 s; antes se esperaban 22 s más (35 s, Chat ya había cortado)
    assert bot_budget(13) == 12.0
    assert bot_budget(1) == 22.0  # instancia caliente: la espera normal
    assert bot_budget(31) == 0.0  # el plazo ya se fue: responder al tiro
    assert bot_budget(None) == 22.0  # sin hora del evento
    assert bot_budget(-5) == 22.0  # reloj adelantado del lado de Chat


def test_late_event_answers_working_immediately_and_posts_later():
    client, poster, _, _ = make_chat(FakeConversation(reply="Listo el análisis", delay=0.3), chat_sync_seconds=5)
    event = {**message_event("analiza"),
             "eventTime": (dt.datetime.now(dt.UTC) - dt.timedelta(seconds=26)).isoformat()}
    response = client.post("/chat", json=event).json()
    assert response["text"] == WORKING_TEXT
    assert poster.posts == [(SPACE, THREAD, {"text": "Listo el análisis"})]


def test_addon_event_time_is_read():
    event = addon_message("hola")
    event["chat"]["eventTime"] = "2026-10-27T14:09:47Z"
    assert parse_chat_event(event).event_time == dt.datetime(2026, 10, 27, 14, 9, 47, tzinfo=dt.UTC)
    assert parse_chat_event(message_event("hola")).event_time is None
