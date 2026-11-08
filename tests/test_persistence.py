"""Persistencia en Firestore: estado de runs, lock distribuido y memoria por cliente."""

import datetime as dt
import uuid

import pytest
from google.api_core import exceptions
from google.cloud.firestore_v1.transforms import ArrayUnion

from src.agent.memory_manager import (
    FirestoreMemoryStore,
    FirestoreRunLock,
    FirestoreStateStore,
    InMemoryRunLock,
    JsonFileMemoryStore,
    JsonFileStateStore,
    _pack,
    _unpack,
    build_stores,
)
from src.config import Settings


# ------------------------------------------------------------- Firestore falso ----

class _Snapshot:
    def __init__(self, data, update_time):
        self._data = data
        self.exists = data is not None
        self.update_time = update_time

    def to_dict(self):
        return dict(self._data) if self._data is not None else None


class _Doc:
    def __init__(self, db, path):
        self.db, self.path = db, path

    async def get(self):
        data, ts = self.db.docs.get(self.path, (None, None))
        return _Snapshot(data, ts)

    def _write(self, data):
        self.db.clock += 1
        self.db.docs[self.path] = (data, self.db.clock)

    async def set(self, data, merge=False):
        current = dict(self.db.docs.get(self.path, ({}, 0))[0] or {}) if merge else {}
        for key, value in data.items():
            if isinstance(value, ArrayUnion):
                existing = list(current.get(key, []))
                current[key] = existing + [v for v in value.values if v not in existing]
            else:
                current[key] = value
        self._write(current)

    async def create(self, data):
        if self.path in self.db.docs:
            raise exceptions.AlreadyExists("existe")
        self._write(dict(data))

    async def update(self, data, option=None):
        if self.path not in self.db.docs:
            raise exceptions.NotFound("no existe")
        if option is not None and option != self.db.docs[self.path][1]:
            raise exceptions.FailedPrecondition("versión distinta")
        self._write({**self.db.docs[self.path][0], **data})

    async def delete(self):
        self.db.docs.pop(self.path, None)


class FakeFirestore:
    def __init__(self):
        self.docs: dict[str, tuple] = {}
        self.clock = 0

    def collection(self, name):
        db = self

        class _Col:
            def document(self, doc_id):
                return _Doc(db, f"{name}/{doc_id}")

        return _Col()

    def write_option(self, last_update_time):
        return last_update_time


# ------------------------------------------------------------------ estado ----

STATE = {"run_id": "abc", "status": "WAITING_KEYWORD_APPROVAL", "keywords": [{"keyword": "seguro auto viña"}] * 500}


def test_pack_roundtrip_and_compresses():
    packed = _pack(STATE)
    assert _unpack(packed) == STATE
    assert len(packed) < len(str(STATE)) / 5


async def test_firestore_state_store_roundtrip():
    db = FakeFirestore()
    store = FirestoreStateStore(db)
    assert await store.load("sheet", "hilo") is None
    await store.save("sheet", "hilo", STATE)
    assert await store.load("sheet", "hilo") == STATE
    (path, (doc, _)), = db.docs.items()
    assert path.startswith("seo_runs/seo-")
    assert doc["status"] == "WAITING_KEYWORD_APPROVAL" and doc["thread_id"] == "hilo"
    assert isinstance(doc["run_state"], str)  # comprimido


async def test_firestore_state_store_rejects_oversized_state(monkeypatch):
    monkeypatch.setattr("src.agent.memory_manager.FIRESTORE_DOC_LIMIT", 100)
    with pytest.raises(ValueError, match="límite de Firestore"):
        await FirestoreStateStore(FakeFirestore()).save("s", "h", STATE)


# -------------------------------------------------------------------- lock ----

@pytest.mark.parametrize("make", [lambda: FirestoreRunLock(FakeFirestore()), InMemoryRunLock])
async def test_lock_blocks_second_owner_until_release(make):
    lock = make()
    assert await lock.acquire("sheet|hilo", "a", 600)
    assert not await lock.acquire("sheet|hilo", "b", 600)
    await lock.release("sheet|hilo", "b")  # otro dueño no lo libera
    assert not await lock.acquire("sheet|hilo", "b", 600)
    await lock.release("sheet|hilo", "a")
    assert await lock.acquire("sheet|hilo", "b", 600)


async def test_firestore_lock_takes_over_expired_lock():
    db = FakeFirestore()
    lock = FirestoreRunLock(db)
    assert await lock.acquire("k", "instancia-caida", 600)
    path = next(iter(db.docs))
    data, ts = db.docs[path]
    db.docs[path] = ({**data, "expires_at": dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)}, ts)
    assert await lock.acquire("k", "nueva", 600)
    assert db.docs[path][0]["owner"] == "nueva"


# ------------------------------------------------------------------ memoria ----

async def test_firestore_memory_store_dedupes_by_domain():
    store = FirestoreMemoryStore(FakeFirestore())
    await store.remember("www.ACME.cl", ["Tono cercano", "No usar 'barato'"])
    await store.remember("www.acme.cl", ["Tono cercano", "Slugs cortos"])
    assert await store.recall("www.acme.cl", "x") == ["Tono cercano", "No usar 'barato'", "Slugs cortos"]
    assert await store.recall("otro.cl", "x") == []


# ------------------------------------------------------------------ armado ----

def test_build_stores_json_backend(tmp_path):
    state, memory = build_stores(Settings(state_backend="json", local_state_dir=str(tmp_path)))
    assert isinstance(state, JsonFileStateStore) and isinstance(memory, JsonFileMemoryStore)


def test_build_stores_firestore_backend(monkeypatch):
    monkeypatch.setattr("src.agent.memory_manager.firestore_client", lambda settings: FakeFirestore())
    state, memory = build_stores(Settings(state_backend="firestore"))
    assert isinstance(state, FirestoreStateStore) and isinstance(memory, FirestoreMemoryStore)


def test_build_stores_memory_bank_when_agent_engine(monkeypatch):
    import google.adk.memory as adk_memory

    from src.agent.memory_manager import VertexMemoryBankStore

    monkeypatch.setattr("src.agent.memory_manager.firestore_client", lambda settings: FakeFirestore())
    monkeypatch.setattr(adk_memory, "VertexAiMemoryBankService", lambda **kw: ("memory-bank", kw))
    state, memory = build_stores(Settings(state_backend="firestore", agent_engine_id="123"))
    assert isinstance(state, FirestoreStateStore)  # Firestore sigue siendo el estado de los runs
    assert isinstance(memory, VertexMemoryBankStore)
    assert memory.memory_service[1]["agent_engine_id"] == "123"


# -------------------------------------------------------------------- live ----

@pytest.mark.live
async def test_live_firestore_state_lock_and_memory():
    from src.agent.memory_manager import firestore_client

    client = firestore_client(Settings())
    suffix = uuid.uuid4().hex[:8]
    state_store = FirestoreStateStore(client, collection=f"seo_test_runs_{suffix}")
    lock = FirestoreRunLock(client, collection=f"seo_test_locks_{suffix}")
    memory = FirestoreMemoryStore(client, collection=f"seo_test_memory_{suffix}")
    try:
        await state_store.save("sheet-live", "hilo", STATE)
        assert await state_store.load("sheet-live", "hilo") == STATE
        assert await lock.acquire("k", "a", 60)
        assert not await lock.acquire("k", "b", 60)
        await lock.release("k", "a")
        assert await lock.acquire("k", "b", 60)
        await memory.remember("acme.cl", ["Tono cercano"])
        await memory.remember("acme.cl", ["Tono cercano", "Slugs cortos"])
        assert await memory.recall("acme.cl", "x") == ["Tono cercano", "Slugs cortos"]
    finally:
        for collection in (state_store.collection, lock.collection, memory.collection):
            async for doc in client.collection(collection).stream():
                await doc.reference.delete()
