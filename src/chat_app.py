"""App de Google Chat del agente de SEO programático (FastAPI, lista para Cloud Run).

Cada mensaje va al agente conversacional (`src.agent.chat_agent`): la persona
escribe como en cualquier chat y el agente pide datos, trabaja y responde.

Acepta los dos formatos de app de Chat con endpoint HTTP: el clásico (`type`,
`message`, `action`) y el de complemento de Google Workspace (`chat.messagePayload`,
`commonEventObject`; respuestas en `hostAppDataAction`).

Límite de 30 s de Chat: el turno del agente corre en una tarea y la respuesta HTTP
lo espera hasta `chat_sync_seconds`, sin pasar de `chat_deadline_seconds` contados
desde la hora del evento de Chat (`eventTime`): si la instancia tardó en arrancar,
ese tiempo se descuenta de la espera. Si el turno termina antes, se responde directo. Si no
(o si una herramienta larga avisa "tarda ~5 min"), se responde con ese aviso y el
resultado se publica después en la conversación con `spaces.messages.create`.
En Cloud Run, desplegar con CPU siempre asignada (`--no-cpu-throttling`) para que
la tarea siga corriendo después de responder.

Conversación: en un mensaje directo, todo el espacio; en un espacio con hilos,
cada hilo. El historial vive en Firestore (`seo_chats`) y un lock (`seo_locks`)
evita dos turnos a la vez en la misma conversación.

    uv run uvicorn src.chat_app:app --port 8080
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

from src.agent.memory_manager import InMemoryRunLock, RunLock, build_run_lock
from src.agent.service import sheet_url
from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

CHAT_SCOPES = ("https://www.googleapis.com/auth/chat.bot",)
CHAT_API = "https://chat.googleapis.com/v1"
CHAT_ISSUER = "chat@system.gserviceaccount.com"
# Apps de Chat construidas como complemento de Workspace: el token lo emite esta cuenta
ADDON_ISSUER_RE = re.compile(r"^service-(\d+)@gcp-sa-gsuiteaddons\.iam\.gserviceaccount\.com$")

# Botones de las tarjetas de la versión anterior: se traducen a un mensaje para el agente
CARD_ACTIONS = {
    "advance_seo_run": "Sigamos con el siguiente paso.",
    "check_run_status": "¿En qué vamos?",
    "prepare_seo_sheet": "Prepara la Sheet.",
}

WELCOME_TEXT = ("Hola 👋 Soy el agente de SEO programático de D2B. Cuéntame para qué cliente quieres crear "
                "páginas y te voy guiando.")
WORKING_TEXT = "⏳ Dame un momento, te respondo aquí."
BUSY_TEXT = "⏳ Sigo trabajando en lo anterior. Te aviso aquí apenas termine y seguimos."
ERROR_TEXT = "⚠️ Tuve un problema y no pude terminar. Vuelve a escribirme en un rato; lo avanzado quedó guardado."

Command = Literal["message", "welcome", "ignore"]


# ------------------------------------------------------------------ eventos ----

@dataclass
class ChatRequest:
    command: Command
    event_type: str
    text: str = ""
    thread_name: str | None = None
    space_name: str | None = None
    is_dm: bool = False
    user_name: str | None = None
    user_display_name: str | None = None
    is_card_click: bool = False
    params: dict[str, str] = field(default_factory=dict)
    addon: bool = False  # evento de app de Chat construida como complemento de Workspace
    endpoint_url: str | None = None
    event_time: dt.datetime | None = None  # cuándo lo envió Chat (el plazo de 30 s corre desde ahí)

    @property
    def conversation_id(self) -> str:
        """Mensaje directo: el espacio completo. Espacio con hilos: cada hilo."""
        if self.is_dm or not self.thread_name:
            return self.space_name or self.thread_name or "default"
        return self.thread_name

    @property
    def reply_thread(self) -> str | None:
        return None if self.is_dm else self.thread_name


def _action_params(event: dict) -> dict[str, str]:
    params = {p.get("key"): p.get("value") for p in (event.get("action") or {}).get("parameters") or []}
    params.update((event.get("common") or {}).get("parameters") or {})
    return {k: v for k, v in params.items() if k}


def _addon_to_classic(event: dict) -> dict:
    """Traduce un evento de complemento de Workspace (`chat.*Payload`) al formato clásico."""
    chat = event.get("chat") or {}
    common = event.get("commonEventObject") or {}
    params = common.get("parameters") or {}
    base = {"user": chat.get("user") or {}, "eventTime": chat.get("eventTime") or event.get("eventTime")}
    if payload := chat.get("messagePayload"):
        return {**base, "type": "MESSAGE", "message": payload.get("message") or {}, "space": payload.get("space") or {}}
    if payload := chat.get("buttonClickedPayload"):
        return {**base, "type": "CARD_CLICKED", "message": payload.get("message") or {},
                "space": payload.get("space") or {},
                "action": {"actionMethodName": params.get("action"),
                           "parameters": [{"key": k, "value": v} for k, v in params.items()]}}
    if payload := chat.get("addedToSpacePayload"):
        return {**base, "type": "ADDED_TO_SPACE", "space": payload.get("space") or {}}
    return {**base, "type": "OTHER"}


_FRACTION = re.compile(r"\.(\d{6})\d+")


def parse_event_time(value: Any) -> dt.datetime | None:
    """Hora del evento de Chat como datetime con zona horaria.

    Acepta RFC 3339 ("2026-10-27T14:09:21.123456789Z", con nanosegundos) o un
    Timestamp en JSON ({"seconds": ..., "nanos": ...}). Lo que no se pueda leer
    devuelve None y la espera queda en `chat_sync_seconds`.
    """
    if isinstance(value, dict) and "seconds" in value:
        try:
            seconds = float(value["seconds"]) + float(value.get("nanos") or 0) / 1e9
        except (TypeError, ValueError):
            return None
        return dt.datetime.fromtimestamp(seconds, tz=dt.UTC)
    if not isinstance(value, str) or not value.strip():
        return None
    text = _FRACTION.sub(r".\1", value.strip()).replace("Z", "+00:00").replace("z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)


def _event_time(event: dict) -> dt.datetime | None:
    message = event.get("message") or {}
    return parse_event_time(event.get("eventTime")) or parse_event_time(message.get("createTime"))


def _is_dm(space: dict) -> bool:
    return space.get("type") == "DM" or space.get("spaceType") == "DIRECT_MESSAGE" or bool(space.get("singleUserBotDm"))


def parse_chat_event(event: dict) -> ChatRequest:
    """Normaliza un evento de Google Chat (clásico o complemento) a un `ChatRequest`."""
    if "chat" in event and "type" not in event:
        req = parse_chat_event(_addon_to_classic(event))
        req.addon = True
        return req
    event_type = event.get("type", "")
    message = event.get("message") or {}
    user = event.get("user") or {}
    space = event.get("space") or message.get("space") or {}
    thread = (message.get("thread") or {}).get("name") or (event.get("thread") or {}).get("name")
    space_name = space.get("name") or (thread.split("/threads/")[0] if thread and "/threads/" in thread else None)
    base = dict(
        event_type=event_type, thread_name=thread, space_name=space_name, is_dm=_is_dm(space),
        user_name=user.get("name"), user_display_name=user.get("displayName"), event_time=_event_time(event),
    )
    if user.get("type") == "BOT":
        return ChatRequest(command="ignore", **base)

    if event_type == "ADDED_TO_SPACE":
        return ChatRequest(command="welcome", **base)

    if event_type == "CARD_CLICKED":
        params = _action_params(event)
        function = ((event.get("action") or {}).get("actionMethodName") or params.get("action")
                    or (event.get("common") or {}).get("invokedFunction"))
        if params.get("thread_name"):
            base["thread_name"] = params["thread_name"]
        text = CARD_ACTIONS.get(function, "")
        if text and params.get("sheet_id"):
            text += f" Sheet: {sheet_url(params['sheet_id'])}"
        return ChatRequest(command="message" if text else "ignore", text=text, is_card_click=True,
                           params=params, **base)

    if event_type == "MESSAGE":
        text = (message.get("argumentText") or message.get("text") or "").strip()
        return ChatRequest(command="message" if text else "ignore", text=text, **base)

    return ChatRequest(command="ignore", **base)


# --------------------------------------------------------------- publicación ----

class ChatPoster(Protocol):
    async def post(self, space_name: str, thread_name: str | None, message: dict) -> dict: ...


class GoogleChatPoster:
    """Publica mensajes como la app (`spaces.messages.create`, scope chat.bot)."""

    async def post(self, space_name: str, thread_name: str | None, message: dict) -> dict:
        from google.auth.transport.requests import AuthorizedSession

        from src.integrations.credentials import agent_credentials

        body = dict(message)
        params = {}
        if thread_name:
            body["thread"] = {"name": thread_name}
            params["messageReplyOption"] = "REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD"
        session = AuthorizedSession(agent_credentials(CHAT_SCOPES))
        response = await asyncio.to_thread(
            session.post, f"{CHAT_API}/{space_name}/messages", params=params, json=body, timeout=30
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Chat API {response.status_code}: {response.text[:300]}")
        return response.json()


def _unverified_claims(token: str) -> dict:
    """Emisor/audiencia/email del token sin verificar, solo para diagnosticar en logs."""
    from google.auth import jwt

    try:
        claims = jwt.decode(token, verify=False)
    except Exception:  # noqa: BLE001
        return {"formato": "no es un JWT"}
    return {k: claims.get(k) for k in ("iss", "aud", "email")}


def verify_chat_token(authorization: str | None, audiences: str | list[str]) -> None:
    """Verifica que el request venga de Google Chat.

    Acepta los modos de "Público de autenticación" de la consola de Chat:
    - Número de proyecto: JWT firmado por chat@system.gserviceaccount.com con aud = número.
    - URL del extremo HTTP: ID token OIDC de Google con email = chat@system... y aud = URL.
    - App como complemento de Workspace: ID token OIDC con aud = URL y email =
      service-<número de proyecto>@gcp-sa-gsuiteaddons.iam.gserviceaccount.com (el
      número debe estar entre las audiencias configuradas).
    `audiences` puede traer varios valores (p.ej. "123456789012,https://.../chat").
    """
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token

    if isinstance(audiences, str):
        audiences = [a.strip() for a in re.split(r"[;,]", audiences) if a.strip()]
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Falta el token de Google Chat")
    token = authorization.split(" ", 1)[1]
    request = google_requests.Request()
    errors = []
    for audience in audiences:
        try:
            if audience.isdigit():
                claims = id_token.verify_token(
                    token, request, audience=audience,
                    certs_url=f"https://www.googleapis.com/service_accounts/v1/metadata/x509/{CHAT_ISSUER}",
                )
                if claims.get("iss") == CHAT_ISSUER:
                    return
            else:
                claims = id_token.verify_oauth2_token(token, request, audience=audience)
                email = claims.get("email", "")
                addon = ADDON_ISSUER_RE.match(email)
                project_numbers = {a for a in audiences if a.isdigit()}
                if claims.get("email_verified", False) and (
                    email == CHAT_ISSUER or (addon and addon.group(1) in project_numbers)
                ):
                    return
            errors.append(f"{audience}: emisor distinto de Chat")
        except ValueError as e:
            errors.append(f"{audience}: {e}")
    logger.warning("Token de Chat rechazado. Claims sin verificar: %s. Intentos: %s",
                   _unverified_claims(token), errors)
    raise HTTPException(status_code=401, detail="Token de Google Chat inválido")


# ------------------------------------------------------------------ servidor ----

class Conversation(Protocol):
    async def run(self, conversation_id: str, text: str, user_display_name: str | None = None) -> str: ...


@dataclass
class _Turn:
    """Un turno en curso: el aviso de una herramienta larga puede adelantar la respuesta HTTP."""
    early: asyncio.Event = field(default_factory=asyncio.Event)
    announcement: str | None = None
    detached: bool = False  # la respuesta HTTP ya salió: lo que falte se publica por la API


class ChatBot:
    LOCK_TTL_S = 3600  # un turno con generación de contenido; si la instancia muere, el lock vence solo

    def __init__(self, conversation: Conversation, poster: ChatPoster, settings: Settings, lock: RunLock,
                 now: Callable[[], dt.datetime] | None = None):
        self.conversation = conversation
        self.poster = poster
        self.settings = settings
        self.lock = lock
        self.now = now or (lambda: dt.datetime.now(dt.UTC))

    def sync_budget(self, req: ChatRequest) -> float:
        """Segundos que la respuesta HTTP puede esperar el turno antes de contestar "trabajando".

        El plazo de Chat corre desde que envió el evento: el arranque en frío de la
        instancia (12 a 31 s en producción) se descuenta. Sin hora del evento, o con
        una hora futura por desfase de relojes, se usa `chat_sync_seconds`.
        """
        budget = self.settings.chat_sync_seconds
        if req.event_time is None:
            return budget
        elapsed = (self.now() - req.event_time).total_seconds()
        if elapsed < 0:
            return budget
        return max(0.0, min(budget, self.settings.chat_deadline_seconds - elapsed))

    @staticmethod
    def lock_key(conversation_id: str) -> str:
        return f"chat|{conversation_id}"

    def _reply(self, req: ChatRequest, text: str) -> dict:
        message: dict[str, Any] = {"text": text}
        if req.addon:
            return {"hostAppDataAction": {"chatDataAction": {"createMessageAction": {"message": message}}}}
        if req.reply_thread:
            message["thread"] = {"name": req.reply_thread}
        if req.is_card_click:
            message = {"actionResponse": {"type": "NEW_MESSAGE"}, **message}
        return message

    async def _post(self, req: ChatRequest, text: str) -> None:
        space = req.space_name or (req.thread_name.split("/threads/")[0]
                                   if req.thread_name and "/threads/" in req.thread_name else None)
        if not space:
            logger.error("No sé en qué espacio de Chat publicar (conversación=%s)", req.conversation_id)
            return
        try:
            await self.poster.post(space, req.reply_thread, {"text": text})
        except Exception:  # noqa: BLE001
            logger.exception("No pude publicar en Google Chat")

    async def handle(self, event: dict, background: BackgroundTasks, endpoint_url: str | None = None) -> dict:
        req = parse_chat_event(event)
        req.endpoint_url = endpoint_url
        if req.command == "ignore":
            return {}
        if req.command == "welcome":
            return self._reply(req, WELCOME_TEXT)

        owner = f"{req.user_name or 'chat'}:{id(req)}"
        if not await self.lock.acquire(self.lock_key(req.conversation_id), owner, self.LOCK_TTL_S):
            return self._reply(req, BUSY_TEXT)

        turn = _Turn()
        budget = self.sync_budget(req)
        if budget < self.settings.chat_sync_seconds:
            logger.warning("Evento de Chat con %.1f s de retraso al llegar: espero %.1f s antes de responder",
                           self.settings.chat_deadline_seconds - budget, budget)
        task = asyncio.create_task(self._run_turn(req, turn, owner))
        early = asyncio.create_task(turn.early.wait())
        await asyncio.wait({task, early}, timeout=budget, return_when=asyncio.FIRST_COMPLETED)
        early.cancel()
        if task.done():
            chunks = task.result()
            for extra in chunks[1:]:
                background.add_task(self._post, req, extra)
            return self._reply(req, chunks[0])
        turn.detached = True
        background.add_task(self._finish, req, task)
        return self._reply(req, turn.announcement or WORKING_TEXT)

    async def _run_turn(self, req: ChatRequest, turn: _Turn, owner: str) -> list[str]:
        from src.agent.chat_agent import set_notifier, split_chat_text

        async def notifier(text: str) -> None:
            if not turn.detached and turn.announcement is None:
                turn.announcement = text
                turn.early.set()
            else:
                await self._post(req, text)

        set_notifier(notifier)  # la tarea tiene su propio contexto: no afecta a otros turnos
        try:
            reply = await self.conversation.run(req.conversation_id, req.text, req.user_display_name)
            return split_chat_text(reply) or ["Listo."]
        except Exception:  # noqa: BLE001 - la persona siempre recibe una respuesta
            logger.exception("Falló el turno del agente (conversación=%s)", req.conversation_id)
            return [ERROR_TEXT]
        finally:
            await self.lock.release(self.lock_key(req.conversation_id), owner)

    async def _finish(self, req: ChatRequest, task: asyncio.Task) -> None:
        for chunk in await task:
            await self._post(req, chunk)
        logger.warning("Respuesta publicada en %s", req.reply_thread or req.space_name)


def create_app(
    conversation: Conversation | None = None,
    poster: ChatPoster | None = None,
    settings: Settings | None = None,
    lock: RunLock | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    if lock is None:
        lock = build_run_lock(settings) if conversation is None else InMemoryRunLock()
    if conversation is None:
        from src.agent.chat_agent import ChatConversation, build_chat_agent
        from src.agent.memory_manager import build_conversation_store

        conversation = ChatConversation(build_chat_agent(settings), build_conversation_store(settings), settings)
    bot = ChatBot(conversation, poster or GoogleChatPoster(), settings, lock)
    app = FastAPI(title="SEO programático · Google Chat")
    app.state.bot = bot

    @app.get("/health")
    async def healthz() -> dict:
        return {"ok": True}

    @app.post("/")
    @app.post("/chat")
    async def chat_event(request: Request, background: BackgroundTasks) -> dict:
        if settings.chat_audience:
            verify_chat_token(request.headers.get("authorization"), settings.chat_audience)
        event = await request.json()
        # URL pública del endpoint (los botones de complementos de Workspace la necesitan)
        endpoint_url = f"https://{request.headers.get('host', '')}{request.url.path}"
        return await bot.handle(event, background, endpoint_url)

    return app


def __getattr__(name: str) -> Any:
    # `uvicorn src.chat_app:app` crea la app al primer acceso (los tests usan create_app)
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
