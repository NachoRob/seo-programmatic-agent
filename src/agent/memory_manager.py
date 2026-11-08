"""Persistencia del run entre etapas, lock anti doble clic y memoria por cliente.

Estado transaccional del run (sobrevive al reciclaje de instancias de Cloud Run):
  * `FirestoreStateStore` (colección `seo_runs`), por defecto.
  * `AdkSessionStateStore` sobre cualquier `BaseSessionService` (p.ej. Vertex Sessions).
  * `JsonFileStateStore` para desarrollo local.
Lock distribuido (Google Chat): `FirestoreRunLock` (`seo_locks`) o `InMemoryRunLock`.
Memoria por cliente (`dominio`): reglas de tono, estructura y aprendizajes.
  * `VertexMemoryBankStore` (Memory Bank del Agent Engine) si hay `agent_engine_id`.
  * `FirestoreMemoryStore` (`seo_client_memory`) como respaldo garantizado.
  * `JsonFileMemoryStore` para desarrollo local.

`build_stores(settings)` arma la combinación según `state_backend` y `agent_engine_id`.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import re
import time
import uuid
import zlib
from pathlib import Path
from typing import Any, Protocol

from google.adk.events import Event, EventActions
from google.adk.sessions import BaseSessionService
from google.genai import types

from src.config import Settings, get_settings

APP_NAME = "seo_programatico"
STATE_KEY = "run_state"


def session_id_for(spreadsheet_id: str, thread_id: str) -> str:
    """ID estable y válido para Vertex (`[A-Za-z0-9_-]+`)."""
    digest = hashlib.sha256(f"{spreadsheet_id}|{thread_id}".encode()).hexdigest()[:32]
    return f"seo-{digest}"


# ------------------------------------------------------------ estado del run ----

class RunStateStore(Protocol):
    async def load(self, spreadsheet_id: str, thread_id: str) -> dict | None: ...
    async def save(self, spreadsheet_id: str, thread_id: str, state: dict) -> None: ...


class AdkSessionStateStore:
    """Guarda el estado del run en una sesión de cualquier `BaseSessionService`."""

    def __init__(self, session_service: BaseSessionService, app_name: str = APP_NAME):
        self.session_service = session_service
        self.app_name = app_name

    async def _session(self, spreadsheet_id: str, thread_id: str, create: bool):
        session_id = session_id_for(spreadsheet_id, thread_id)
        session = await self.session_service.get_session(
            app_name=self.app_name, user_id=spreadsheet_id, session_id=session_id
        )
        if session is None and create:
            session = await self.session_service.create_session(
                app_name=self.app_name, user_id=spreadsheet_id, session_id=session_id,
                state={"spreadsheet_id": spreadsheet_id, "thread_id": thread_id},
            )
        return session

    async def load(self, spreadsheet_id: str, thread_id: str) -> dict | None:
        session = await self._session(spreadsheet_id, thread_id, create=False)
        if session is None:
            return None
        return session.state.get(STATE_KEY)

    async def save(self, spreadsheet_id: str, thread_id: str, state: dict) -> None:
        session = await self._session(spreadsheet_id, thread_id, create=True)
        event = Event(
            invocation_id=f"run-{uuid.uuid4().hex[:8]}",
            author="orchestrator",
            actions=EventActions(state_delta={STATE_KEY: state}),
            timestamp=time.time(),
        )
        await self.session_service.append_event(session, event)


class JsonFileStateStore:
    """Un archivo JSON por run en `directory`."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def _path(self, spreadsheet_id: str, thread_id: str) -> Path:
        return self.directory / f"{session_id_for(spreadsheet_id, thread_id)}.json"

    async def load(self, spreadsheet_id: str, thread_id: str) -> dict | None:
        path = self._path(spreadsheet_id, thread_id)
        return json.loads(path.read_text()) if path.exists() else None

    async def save(self, spreadsheet_id: str, thread_id: str, state: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(spreadsheet_id, thread_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1, default=str))
        tmp.replace(path)


# ------------------------------------------------------- memoria por cliente ----

class ClientMemoryStore(Protocol):
    async def recall(self, dominio: str, query: str) -> list[str]: ...
    async def remember(self, dominio: str, facts: list[str]) -> None: ...


def _memory_user_id(dominio: str) -> str:
    return re.sub(r"[^a-z0-9.-]", "-", dominio.lower())


class VertexMemoryBankStore:
    """Memory Bank de Vertex AI, con `user_id` = dominio del cliente."""

    def __init__(self, memory_service: Any, app_name: str = APP_NAME):
        self.memory_service = memory_service
        self.app_name = app_name

    async def recall(self, dominio: str, query: str) -> list[str]:
        response = await self.memory_service.search_memory(
            app_name=self.app_name, user_id=_memory_user_id(dominio), query=query
        )
        facts = []
        for memory in response.memories:
            text = "".join(p.text or "" for p in (memory.content.parts or []))
            if text and text not in facts:
                facts.append(text)
        return facts

    async def remember(self, dominio: str, facts: list[str]) -> None:
        from google.adk.memory.memory_entry import MemoryEntry

        if not facts:
            return
        await self.memory_service.add_memory(
            app_name=self.app_name,
            user_id=_memory_user_id(dominio),
            memories=[MemoryEntry(content=types.Content(role="user", parts=[types.Part(text=f)])) for f in facts],
        )


class JsonFileMemoryStore:
    """Memoria por dominio en un JSON local. `recall` devuelve todo (sin búsqueda semántica)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _read(self) -> dict[str, list[str]]:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    async def recall(self, dominio: str, query: str) -> list[str]:
        return list(self._read().get(_memory_user_id(dominio), []))

    async def remember(self, dominio: str, facts: list[str]) -> None:
        data = self._read()
        bucket = data.setdefault(_memory_user_id(dominio), [])
        bucket.extend(f for f in facts if f not in bucket)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------- Firestore ----

FIRESTORE_DOC_LIMIT = 1_000_000  # Firestore: 1 MiB por documento


def firestore_client(settings: Settings):
    from google.cloud import firestore

    return firestore.AsyncClient(project=settings.firestore_project or settings.project_id,
                                 database=settings.firestore_database)


def _pack(state: dict) -> str:
    """JSON comprimido (zlib + base64): un run grande cabe en el límite de 1 MiB."""
    return base64.b64encode(zlib.compress(json.dumps(state, ensure_ascii=False, default=str).encode(), 9)).decode()


def _unpack(blob: str) -> dict:
    return json.loads(zlib.decompress(base64.b64decode(blob)))


class FirestoreStateStore:
    """Un documento por run en `seo_runs/{session_id}` con el estado comprimido."""

    def __init__(self, client: Any, collection: str = "seo_runs"):
        self.client = client
        self.collection = collection

    def _doc(self, spreadsheet_id: str, thread_id: str):
        return self.client.collection(self.collection).document(session_id_for(spreadsheet_id, thread_id))

    async def load(self, spreadsheet_id: str, thread_id: str) -> dict | None:
        snapshot = await self._doc(spreadsheet_id, thread_id).get()
        if not snapshot.exists:
            return None
        return _unpack(snapshot.to_dict()["run_state"])

    async def save(self, spreadsheet_id: str, thread_id: str, state: dict) -> None:
        packed = _pack(state)
        if len(packed) > FIRESTORE_DOC_LIMIT - 10_000:
            raise ValueError(f"El estado del run ocupa {len(packed)} bytes comprimido: supera el límite de Firestore.")
        await self._doc(spreadsheet_id, thread_id).set({
            "spreadsheet_id": spreadsheet_id,
            "thread_id": thread_id,
            "status": state.get("status"),
            "run_id": state.get("run_id"),
            "run_state": packed,
            "updated_at": dt.datetime.now(dt.timezone.utc),
        })


class RunLock(Protocol):
    async def acquire(self, key: str, owner: str, ttl_s: int = ...) -> bool: ...
    async def release(self, key: str, owner: str) -> None: ...


class InMemoryRunLock:
    """Lock por proceso (desarrollo local y tests)."""

    def __init__(self):
        self._held: dict[str, tuple[str, float]] = {}

    async def acquire(self, key: str, owner: str, ttl_s: int = 3600) -> bool:
        current = self._held.get(key)
        if current and current[1] > time.time():
            return False
        self._held[key] = (owner, time.time() + ttl_s)
        return True

    async def release(self, key: str, owner: str) -> None:
        if self._held.get(key, ("",))[0] == owner:
            self._held.pop(key, None)


class FirestoreRunLock:
    """Lock distribuido con vencimiento en `seo_locks/{key}`.

    `create()` falla si el documento existe, así dos instancias de Cloud Run no
    toman el mismo run. Un lock vencido (instancia caída a mitad de etapa) se
    reemplaza con `update` condicionado a la versión leída, para no pisar a otra
    instancia que lo tome al mismo tiempo.
    """

    def __init__(self, client: Any, collection: str = "seo_locks"):
        self.client = client
        self.collection = collection

    def _doc(self, key: str):
        return self.client.collection(self.collection).document(hashlib.sha256(key.encode()).hexdigest()[:40])

    async def acquire(self, key: str, owner: str, ttl_s: int = 3600) -> bool:
        from google.api_core import exceptions

        now = dt.datetime.now(dt.timezone.utc)
        data = {"key": key, "owner": owner, "expires_at": now + dt.timedelta(seconds=ttl_s), "acquired_at": now}
        doc = self._doc(key)
        try:
            await doc.create(data)
            return True
        except exceptions.AlreadyExists:
            pass
        snapshot = await doc.get()
        if snapshot.exists and snapshot.to_dict().get("expires_at") and snapshot.to_dict()["expires_at"] > now:
            return False
        try:
            option = self.client.write_option(last_update_time=snapshot.update_time) if snapshot.exists else None
            if option is None:
                await doc.create(data)
            else:
                await doc.update(data, option=option)
            return True
        except (exceptions.AlreadyExists, exceptions.FailedPrecondition, exceptions.NotFound):
            return False

    async def release(self, key: str, owner: str) -> None:
        doc = self._doc(key)
        snapshot = await doc.get()
        if snapshot.exists and snapshot.to_dict().get("owner") == owner:
            await doc.delete()


class FirestoreMemoryStore:
    """Memoria por dominio en `seo_client_memory/{dominio}` (`recall` devuelve todo)."""

    def __init__(self, client: Any, collection: str = "seo_client_memory"):
        self.client = client
        self.collection = collection

    def _doc(self, dominio: str):
        return self.client.collection(self.collection).document(_memory_user_id(dominio))

    async def recall(self, dominio: str, query: str) -> list[str]:
        snapshot = await self._doc(dominio).get()
        return list(snapshot.to_dict().get("facts", [])) if snapshot.exists else []

    async def remember(self, dominio: str, facts: list[str]) -> None:
        from google.cloud import firestore

        if not facts:
            return
        await self._doc(dominio).set({
            "dominio": _memory_user_id(dominio),
            "facts": firestore.ArrayUnion(list(facts)),
            "updated_at": dt.datetime.now(dt.timezone.utc),
        }, merge=True)


# -------------------------------------------------------------------- armado ----

def build_stores(settings: Settings | None = None) -> tuple[RunStateStore, ClientMemoryStore]:
    """Estado según `state_backend` (firestore | json) y memoria en Memory Bank si hay
    `agent_engine_id`; si no, en Firestore (o JSON en modo local)."""
    settings = settings or get_settings()
    base = Path(settings.local_state_dir)
    client = firestore_client(settings) if settings.state_backend == "firestore" else None

    state_store: RunStateStore = (FirestoreStateStore(client) if client
                                  else JsonFileStateStore(base / "runs"))
    if settings.agent_engine_id:
        from google.adk.memory import VertexAiMemoryBankService

        memory: ClientMemoryStore = VertexMemoryBankStore(VertexAiMemoryBankService(
            project=settings.vertex_project_id, location=settings.vertex_location,
            agent_engine_id=settings.agent_engine_id,
        ))
    elif client:
        memory = FirestoreMemoryStore(client)
    else:
        memory = JsonFileMemoryStore(base / "memory.json")
    return state_store, memory


# ----------------------------------------------------- conversaciones (Chat) ----

class ConversationStore(Protocol):
    """Historial del agente conversacional por conversación de Google Chat."""

    async def load(self, conversation_id: str) -> dict | None: ...
    async def save(self, conversation_id: str, data: dict) -> None: ...


class FirestoreConversationStore:
    """Un documento por conversación en `seo_chats/{hash}`, comprimido como el run."""

    def __init__(self, client: Any, collection: str = "seo_chats"):
        self.client = client
        self.collection = collection

    def _doc(self, conversation_id: str):
        return self.client.collection(self.collection).document(
            hashlib.sha256(conversation_id.encode()).hexdigest()[:40])

    async def load(self, conversation_id: str) -> dict | None:
        snapshot = await self._doc(conversation_id).get()
        return _unpack(snapshot.to_dict()["data"]) if snapshot.exists else None

    async def save(self, conversation_id: str, data: dict) -> None:
        packed = _pack(data)
        if len(packed) > FIRESTORE_DOC_LIMIT - 10_000:
            raise ValueError(f"La conversación ocupa {len(packed)} bytes comprimida: supera el límite de Firestore.")
        await self._doc(conversation_id).set({
            "conversation_id": conversation_id, "data": packed,
            "updated_at": dt.datetime.now(dt.timezone.utc),
        })


class InMemoryConversationStore:
    def __init__(self):
        self.data: dict[str, dict] = {}

    async def load(self, conversation_id: str) -> dict | None:
        return json.loads(json.dumps(self.data[conversation_id])) if conversation_id in self.data else None

    async def save(self, conversation_id: str, data: dict) -> None:
        self.data[conversation_id] = json.loads(json.dumps(data, default=str))


def build_conversation_store(settings: Settings | None = None) -> ConversationStore:
    settings = settings or get_settings()
    if settings.state_backend == "firestore":
        return FirestoreConversationStore(firestore_client(settings))
    return InMemoryConversationStore()


def build_run_lock(settings: Settings | None = None) -> RunLock:
    settings = settings or get_settings()
    if settings.state_backend == "firestore":
        return FirestoreRunLock(firestore_client(settings))
    return InMemoryRunLock()
