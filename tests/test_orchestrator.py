"""Orquestador del Flujo 1: Etapa A -> P1-10 -> Etapa B -> P1-13 -> Etapa C -> 8 pestañas."""

import copy
import datetime as dt

import pytest
from google.adk.sessions import InMemorySessionService

from src.agent.memory_manager import (
    AdkSessionStateStore,
    JsonFileMemoryStore,
    JsonFileStateStore,
    session_id_for,
)
from src.agent.orchestrator import (
    FINAL_TABS,
    InvalidRunState,
    RunStatus,
    SeoProgrammaticOrchestrator,
    is_approved,
    localization_from_brief,
    match_own_data,
    site_from_brief,
)
from src.agent.schemas import (
    EnlacePropuesto,
    HubItem,
    IntentClassificationOutput,
    IntentItem,
    PaginaPlanItem,
    PatronItem,
    PR01Output,
    PR02Output,
    PR03Output,
    PR04Output,
    PR05Output,
    PR06Output,
    SeccionItem,
    TitleMetaOutput,
    TitleMetaVariant,
)
from src.config import Settings, default_parameter_rows
from src.tools.seo_deterministic import normalize_text, slugify

SID = "sheet-acme"

BRIEF = {"cliente": "ACME Seguros", "sitio": "https://www.acme.cl/", "pais": "Chile",
         "idioma": "es-CL", "trato": "tuteo", "tipo_schema": "Service"}
PARAMETERS = {"scoring_weights": {}, "min_judge_score": 4.0, "similarity_threshold": 0.5,
              "min_keyword_score": 0.0, "min_search_volume": 20.0}

SEEDS = [
    {"Keyword": "Seguro auto Viña del Mar", "Fuente": "planner", "Volumen": 300, "Dificultad": 20, "Valor negocio": 5},
    {"Keyword": "seguros auto viña del mar", "Fuente": "gsc", "Volumen": 100},
    {"Keyword": "seguro auto concepción", "Fuente": "planner", "Volumen": 200, "Dificultad": 25},
    {"Keyword": "seguro auto santiago", "Fuente": "planner", "Volumen": 500, "Dificultad": 40},
    {"Keyword": "seguro auto temuco", "Fuente": "planner", "Volumen": 150},
    {"Keyword": "seguro de auto", "Fuente": "planner", "Volumen": 5000},
    {"Keyword": "que es un deducible", "Fuente": "planner", "Volumen": 10},
    {"Keyword": "seguro mascotas", "Fuente": "planner", "Volumen": 400},
    {"Keyword": "", "Fuente": "planner"},
]
OWN_DATA = [
    {"ciudad": "Viña del Mar", "precio_desde": "$12.990", "talleres_convenio": 14},
    {"ciudad": "Concepción", "precio_desde": "$11.490", "talleres_convenio": 9},
    {"ciudad": "Santiago", "precio_desde": "$13.990", "talleres_convenio": 41},
    {"ciudad": "Temuco", "precio_desde": "$10.990", "talleres_convenio": 6},
]
GSC_ROWS = [
    {"keys": ["seguro de auto", "https://www.acme.cl/seguros/auto"], "clicks": 120, "impressions": 3000, "position": 3.2},
    {"keys": ["acme seguros", "https://www.acme.cl/"], "clicks": 500, "impressions": 900, "position": 1.0},
]


# ------------------------------------------------------------------ fakes ----

class FakeSheets:
    """Planilla en memoria: lo que se escribe se puede leer como lista de dicts."""

    def __init__(self):
        self.tabs: dict[str, list[dict]] = {"Semillas": copy.deepcopy(SEEDS), "Datos propios": copy.deepcopy(OWN_DATA)}
        self.raw: dict[str, list[list]] = {}
        self.writes: list[dict[str, list[list]]] = []

    async def read_parameters_and_brief(self, spreadsheet_id):
        assert spreadsheet_id == SID
        return {"parameters": dict(PARAMETERS), "brief": dict(BRIEF), "raw_parameters": {}}

    async def read_tab(self, spreadsheet_id, tab_name):
        return copy.deepcopy(self.tabs.get(tab_name, []))

    async def write_tabs(self, spreadsheet_id, tabs):
        self.writes.append(copy.deepcopy(tabs))
        for name, rows in tabs.items():
            headers = rows[0] if rows else []
            self.tabs[name] = [dict(zip(headers, r)) for r in rows[1:]]
            self.raw[name] = copy.deepcopy(rows)
        return {"tabs": {n: len(r) for n, r in tabs.items()}}

    async def read_raw(self, spreadsheet_id, tab_name):
        if tab_name in self.raw:
            return copy.deepcopy(self.raw[tab_name])
        rows = self.tabs.get(tab_name, [])
        return [list(rows[0])] + [list(r.values()) for r in rows] if rows else []

    def approve(self, tab, key_col, decisions: dict):
        for row in self.tabs[tab]:
            if row[key_col] in decisions:
                value = decisions[row[key_col]]
                row["Aprobado"], row["Motivo rechazo"] = value if isinstance(value, tuple) else (value, "")


class FakeGSC:
    def __init__(self):
        self.calls = []

    async def get_rows(self, site_url, start_date, end_date):
        self.calls.append((site_url, start_date, end_date))
        return copy.deepcopy(GSC_ROWS)

    async def list_sites(self):
        return [{"siteUrl": "https://libreria-demo.cl/", "permissionLevel": "siteFullUser"},
                {"siteUrl": "https://tienda.acme-telecom.cl/", "permissionLevel": "siteFullUser"},
                {"siteUrl": "sc-domain:acme-telecom.cl", "permissionLevel": "siteRestrictedUser"},
                {"siteUrl": "https://www.acme-telecom.cl/", "permissionLevel": "siteFullUser"},
                {"siteUrl": "https://www.acme.cl/", "permissionLevel": "siteUnverifiedUser"}]


class FakeEmbedder:
    def __init__(self):
        self._ids: dict[str, int] = {}

    async def embed(self, texts):
        vectors = []
        for text in texts:
            idx = self._ids.setdefault(text.split("\n\n")[0], len(self._ids))
            vec = [0.0] * 128
            vec[idx % 128] = 1.0
            vectors.append(vec)
        return vectors


class FlowLLM:
    """Responde todos los prompts del Flujo 1. Concepción nunca pasa el juez."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def payloads(self, prompt_id):
        return [p for pid, p in self.calls if pid == prompt_id]

    async def generate(self, prompt_id, payload):
        self.calls.append((prompt_id, payload))
        handler = getattr(self, "_" + prompt_id.replace("-", "_").lower())
        return handler(payload)

    def _p1_06(self, payload):
        return IntentClassificationOutput(clasificaciones=[
            IntentItem(keyword=k, intencion="informativa" if k.startswith("que es") else "transaccional")
            for k in payload["keywords"]
        ])

    def _pr_01(self, payload):
        keywords = [k["keyword"] for c in payload["clusters"] for k in c["keywords"]]
        cities = [k for k in keywords if k.startswith("seguro auto ")]
        return PR01Output(
            patrones=[
                PatronItem(nombre="seguro auto por ciudad", variables=["ciudad"], keywords=cities,
                           dato_diferenciador="talleres_convenio", paginas_estimadas=len(cities), descartado=False),
                PatronItem(nombre="seguro mascota por raza", variables=["raza"], keywords=["seguro mascotas"],
                           dato_diferenciador="ninguno", paginas_estimadas=0, descartado=True,
                           motivo="no hay datos propios por raza"),
            ],
            sin_patron=["que es un deducible", "seguro de auto"],
        )

    def _pr_02(self, payload):
        values = payload["valores_variables"]["ciudad"]
        pages = []
        for patron in payload["patrones"]:
            for kw in patron["keywords"]:
                city = next(v for v in values if normalize_text(v) in kw["keyword"])
                pages.append(PaginaPlanItem(
                    url_propuesta=f"/seguros-auto/seguro-auto-{slugify(city)}", keyword_principal=kw["keyword"],
                    keywords_secundarias=[], hub="/seguros-auto/", variables={"ciudad": city},
                ))
        return PR02Output(hubs=[HubItem(url="/seguros-auto/", patron="seguro auto {ciudad}")], paginas=pages)

    def _pr_03(self, payload):
        fila = payload["fila"]
        return PR03Output(
            h1=f"Seguro de auto en {fila['ciudad']} (intento {payload.get('intento', 1)})",
            intro=f"En {fila['ciudad']} el seguro de auto parte en {fila['precio_desde']} y hay {fila['talleres_convenio']} talleres en convenio.",
            secciones=[SeccionItem(h2="Talleres", texto=f"{fila['talleres_convenio']} talleres en {fila['ciudad']}.")],
            faq=[], cta="Cotiza tu seguro", bloques_omitidos=["faq"],
        )

    def _p1_16(self, payload):
        fila = payload["fila"]
        return TitleMetaOutput(variantes=[
            TitleMetaVariant(title=f"{payload['keyword'].capitalize()} desde {fila['precio_desde']}",
                             meta_description=f"{fila['talleres_convenio']} talleres en convenio en {fila['ciudad']}."),
        ] * 3)

    def _pr_04(self, payload):
        score = 2 if payload["fila"]["ciudad"] == "Concepción" else 5
        return PR04Output(score=score, motivo="x", cifras_no_respaldadas=[],
                          frases_genericas=["Cotiza tu seguro"] if score < 4 else [], cobertura_intencion=True)

    def _pr_05(self, payload):
        return PR05Output(tipo_schema="Service", campos_faltantes=[], jsonld={
            "@context": "https://schema.org", "@type": "Service", "name": f"Seguro de auto en {payload['fila']['ciudad']}",
            "provider": {"@type": "Organization", "name": payload["fila"]["ciudad"]},
        })

    def _pr_06(self, payload):
        intro = payload["contenido"].split("\n\n")[1]
        return PR06Output(enlaces=[EnlacePropuesto(
            destino=payload["hub"], parrafo_literal=intro, anchor=f"seguro de auto", justificacion="hub",
        )], sin_lugar=[h["url"] for h in payload["hermanas"]])


@pytest.fixture
def env(tmp_path):
    return {
        "sheets": FakeSheets(),
        "gsc": FakeGSC(),
        "llm": FlowLLM(),
        "embedder": FakeEmbedder(),
        "state_store": AdkSessionStateStore(InMemorySessionService()),
        "memory": JsonFileMemoryStore(tmp_path / "memory.json"),
    }


def orchestrator(env):
    """Instancia nueva por etapa: simula que el proceso se reinicia entre aprobaciones."""
    return SeoProgrammaticOrchestrator(
        env["llm"], env["embedder"], sheets=env["sheets"], search_console=env["gsc"],
        state_store=env["state_store"], memory=env["memory"], settings=Settings(),
        today=lambda: dt.date(2026, 10, 14), batch_size=10,
    )


# ------------------------------------------------------------ ciclo completo ----

async def test_full_cycle_with_two_human_approvals(env):
    sheets, llm = env["sheets"], env["llm"]

    # ---- Etapa A: P1-01 a P1-09 y pausa P1-10
    state = await orchestrator(env).run_phase_1_and_2(SID, "hilo-1")
    assert state.status == RunStatus.WAITING_KEYWORD_APPROVAL
    assert state.dominio == "acme.cl"
    assert env["gsc"].calls == [("https://www.acme.cl/", "2026-07-14", "2026-10-11")]
    assert state.seed_counts == {"before_count": 8, "after_count": 7, "empty_removed": 0}

    keywords = {r["keyword"]: r for r in sheets.tabs["Keywords"]}
    assert set(keywords) == {"seguro auto viña del mar", "seguro auto concepcion", "seguro auto santiago",
                             "seguro auto temuco", "seguro de auto"}
    assert keywords["seguro de auto"]["ruta"] == "optimize_existing"
    assert keywords["seguro de auto"]["Aprobado"] == "flujo 2"
    assert keywords["seguro auto santiago"]["Aprobado"] == ""
    assert keywords["seguro auto santiago"]["patron"] == "seguro auto por ciudad"
    discarded = {r["elemento"]: r["motivo"] for r in sheets.tabs["Descartados"]}
    assert discarded["que es un deducible"].startswith("volumen 10")
    assert discarded["seguro mascotas"] == "patrón descartado: no hay datos propios por raza"
    # Localización y memoria del cliente llegan a PR-01
    assert llm.payloads("PR-01")[0]["memoria_cliente"] == []

    # ---- Pausa P1-10: aún sin aprobaciones, la Etapa B no avanza
    with pytest.raises(ValueError, match="Aprobado = TRUE"):
        await orchestrator(env).run_phase_3(SID, "hilo-1")
    paused = await orchestrator(env).get_state(SID, "hilo-1")
    assert paused.status == RunStatus.WAITING_KEYWORD_APPROVAL
    assert paused.last_error.startswith("B: ValueError")
    with pytest.raises(InvalidRunState):
        await orchestrator(env).run_phase_4_to_7(SID, "hilo-1")

    # ---- El equipo aprueba en la Sheet
    sheets.approve("Keywords", "keyword", {
        "seguro auto viña del mar": True,
        "seguro auto concepcion": "TRUE",
        "seguro auto santiago": "sí",
        "seguro auto temuco": (False, "sin cobertura en Temuco"),
    })

    # ---- Etapa B: P1-11 a P1-13 y pausa P1-13
    state = await orchestrator(env).run_phase_3(SID, "hilo-1")
    assert state.status == RunStatus.WAITING_URL_APPROVAL
    assert state.last_error is None
    plan = {r["url"]: r for r in sheets.tabs["Plan de URLs"]}
    # Orden del plan = orden por volumen de las keywords aprobadas
    assert [r["url"] for r in sheets.tabs["Plan de URLs"]] == [
        "/seguros-auto/seguro-auto-santiago", "/seguros-auto/seguro-auto-vina-del-mar",
        "/seguros-auto/seguro-auto-concepcion",
    ]
    assert [r["url_id"] for r in sheets.tabs["Plan de URLs"]] == ["U001", "U002", "U003"]
    assert all(r["datos_propios"] == "ok" and r["slug_valid"] is True for r in plan.values())
    assert llm.payloads("PR-02")[0]["valores_variables"]["ciudad"] == ["Concepción", "Santiago", "Temuco", "Viña del Mar"]
    assert any(r["elemento"] == "seguro auto temuco" and r["motivo"] == "rechazada: sin cobertura en Temuco"
               for r in sheets.tabs["Descartados"])

    santiago = plan["/seguros-auto/seguro-auto-santiago"]["url_id"]
    vina = plan["/seguros-auto/seguro-auto-vina-del-mar"]["url_id"]
    conce = plan["/seguros-auto/seguro-auto-concepcion"]["url_id"]

    # ---- El equipo aprueba URLs
    sheets.approve("Plan de URLs", "url_id", {vina: True, conce: True, santiago: (False, "esperar campaña")})

    # ---- Etapa C: P1-14 a P1-25
    state = await orchestrator(env).run_phase_4_to_7(SID, "hilo-1")
    assert state.status == RunStatus.DONE
    assert state.pipeline_summary == {"total": 2, "aprobada": 1, "requiere_revision_humana": 1, "error": 0}

    final = sheets.writes[-1]
    assert tuple(final) == FINAL_TABS  # 8 pestañas en una sola escritura atómica
    tabs = sheets.tabs

    contenido = {r["url_id"]: r for r in tabs["Contenido"]}
    assert set(contenido) == {vina, conce}
    assert contenido[vina]["estado"] == "aprobada"
    assert contenido[conce]["estado"] == "requiere_revision_humana"
    assert contenido[conce]["intentos"] == 3
    assert {r["url_id"] for r in tabs["On-page"]} == {vina, conce}
    assert [r["url_id"] for r in tabs["Schema"]] == [vina]
    assert tabs["Schema"][0]["valido"] is True
    assert tabs["Enlaces internos"][0]["url_id_origen"] == vina
    assert tabs["Enlaces internos"][0]["tipo_destino"] == "hub"

    by_kw = {r["keyword"]: r for r in tabs["Keywords"]}
    assert by_kw["seguro auto viña del mar"]["url_id"] == vina
    assert by_kw["seguro auto temuco"]["url_id"] == ""
    assert by_kw["seguro auto temuco"]["Motivo rechazo"] == "sin cobertura en Temuco"

    plan_final = {r["url_id"]: r for r in tabs["Plan de URLs"]}
    assert plan_final[santiago]["estado"] == "no_aprobada"
    assert plan_final[conce]["estado"] == "requiere_revision_humana"

    discarded = [(r["etapa"], r["paso"], r["url_id"]) for r in tabs["Descartados"]]
    assert ("C", "P1-13", santiago) in discarded
    assert ("C", "P1-19", conce) in discarded

    summary = {r["campo"]: r["valor"] for r in tabs["Resumen"]}
    assert summary["urls_aprobadas"] == 1 and summary["keywords_aprobadas"] == 3
    assert summary["localizacion"] == "Chile / es-CL / tuteo"

    # Localización y marca del Brief en la redacción y el schema
    assert llm.payloads("PR-03")[0]["localizacion"]["idioma"] == "es-CL"
    assert llm.payloads("PR-05")[0]["marca"] == "ACME Seguros"

    # ---- Memory Bank del cliente
    facts = await env["memory"].recall("acme.cl", "tono")
    assert any("tratando al lector de tú" in f for f in facts)
    assert any("seguro auto temuco (rechazada: sin cobertura en Temuco)" in f for f in facts)
    assert any("Cotiza tu seguro" in f for f in facts)

    # ---- Un run nuevo del mismo cliente recibe la memoria
    await orchestrator(env).run_phase_1_and_2(SID, "hilo-2")
    assert llm.payloads("PR-01")[-1]["memoria_cliente"] == facts


async def test_resume_dispatches_by_status(env):
    o = orchestrator(env)
    state = await o.resume(SID, "hilo-r")
    assert state.status == RunStatus.WAITING_KEYWORD_APPROVAL
    env["sheets"].approve("Keywords", "keyword", {"seguro auto santiago": True})
    assert (await o.resume(SID, "hilo-r")).status == RunStatus.WAITING_URL_APPROVAL
    env["sheets"].approve("Plan de URLs", "url_id", {"U001": True})
    assert (await o.resume(SID, "hilo-r")).status == RunStatus.DONE
    assert (await o.resume(SID, "hilo-r")).status == RunStatus.DONE


async def test_cannot_restart_a_paused_run(env):
    await orchestrator(env).run_phase_1_and_2(SID, "hilo-x")
    with pytest.raises(InvalidRunState, match="WAITING_KEYWORD_APPROVAL"):
        await orchestrator(env).run_phase_1_and_2(SID, "hilo-x")


async def test_state_lives_in_adk_session(env):
    service = env["state_store"].session_service
    await orchestrator(env).run_phase_1_and_2(SID, "hilo-s")
    session = await service.get_session(app_name="seo_programatico", user_id=SID,
                                        session_id=session_id_for(SID, "hilo-s"))
    assert session.state["run_state"]["status"] == "WAITING_KEYWORD_APPROVAL"
    assert session.state["thread_id"] == "hilo-s"
    assert session.events[-1].actions.state_delta["run_state"]["spreadsheet_id"] == SID


async def test_json_file_state_store_fallback(env, tmp_path):
    env["state_store"] = JsonFileStateStore(tmp_path / "runs")
    await orchestrator(env).run_phase_1_and_2(SID, "hilo-j")
    files = list((tmp_path / "runs").glob("*.json"))
    assert len(files) == 1
    reloaded = await orchestrator(env).get_state(SID, "hilo-j")
    assert reloaded.status == RunStatus.WAITING_KEYWORD_APPROVAL


# ---------------------------------------------------------------- helpers ----

def test_session_id_is_stable_and_vertex_safe():
    a = session_id_for("1AbC-xyz_123", "spaces/AAA/threads/BBB")
    assert a == session_id_for("1AbC-xyz_123", "spaces/AAA/threads/BBB")
    assert a != session_id_for("1AbC-xyz_123", "otro")
    assert all(c.isalnum() or c in "-_" for c in a)


@pytest.mark.parametrize("value, expected", [
    (True, True), ("TRUE", True), ("Sí", True), ("x", True), (False, False), ("no", False),
    ("", None), (None, None), ("quizás", None),
])
def test_is_approved(value, expected):
    assert is_approved(value) is expected


def test_localization_defaults_and_usted():
    settings = Settings()
    default = localization_from_brief({}, settings)
    assert (default.pais, default.idioma, default.trato) == ("Chile", "es-CL", "tuteo")
    assert "prohibido el voseo" in default.regla
    usted = localization_from_brief({"pais": "México", "idioma": "es-MX", "trato": "Trato de usted"}, settings)
    assert usted.trato == "usted" and "de usted" in usted.regla


@pytest.mark.parametrize("brief, expected", [
    ({"sitio": "https://www.acme.cl/"}, ("https://www.acme.cl/", "acme.cl")),
    ({"dominio": "acme.cl"}, ("https://acme.cl/", "acme.cl")),
    ({"propiedad_gsc": "sc-domain:acme.cl"}, ("sc-domain:acme.cl", "acme.cl")),
])
def test_site_from_brief(brief, expected):
    assert site_from_brief(brief) == expected


def test_site_from_brief_requires_site():
    with pytest.raises(ValueError, match="sitio"):
        site_from_brief({"cliente": "x"})


def test_match_own_data():
    assert match_own_data({"ciudad": "vina del mar"}, OWN_DATA)["talleres_convenio"] == 14
    assert match_own_data({"Ciudad": "Concepción"}, OWN_DATA)["precio_desde"] == "$11.490"
    assert match_own_data({"ciudad": "Arica"}, OWN_DATA) is None
    assert match_own_data({"raza": "labrador"}, OWN_DATA) is None


# ------------------------------------------------------------------- live ----

LIVE_INPUTS = {
    "Brief": [["clave", "valor"],
              ["nota", "Datos de prueba generados por los tests --live (cliente ficticio)"],
              ["cliente", "ACME Seguros"], ["sitio", "https://www.acme.cl/"], ["pais", "Chile"],
              ["idioma", "es-CL"], ["trato", "tuteo"], ["tipo_schema", "Service"]],
    "Parámetros": [row if row[0] != "volumen_minimo" else ["volumen_minimo", 20] for row in default_parameter_rows()],
    "Semillas": [["keyword", "fuente", "volumen", "dificultad"],
                 ["seguro auto viña del mar", "prueba", 300, 20], ["seguro auto concepción", "prueba", 200, 25],
                 ["seguro auto santiago", "prueba", 500, 40], ["seguro auto temuco", "prueba", 150, 15],
                 ["que es un deducible", "prueba", 10, 5]],
    "Datos propios": [["ciudad", "precio_desde", "talleres_convenio", "tiempo_respuesta_siniestro"],
                      ["Viña del Mar", "$12.990 mensual", 14, "48 horas"],
                      ["Concepción", "$11.490 mensual", 9, "72 horas"],
                      ["Santiago", "$13.990 mensual", 41, "24 horas"],
                      ["Temuco", "$10.990 mensual", 6, "72 horas"]],
}


def _approve_first(sheet_id, tab, key_col, n, filter_fn):
    from src.tools.sheets_tool import batch_write_sheet_tabs, read_sheet_tab

    rows = read_sheet_tab(sheet_id, tab, "A1:AZ2000")
    headers = list(rows[0].keys())
    approved = 0
    for row in rows:
        if filter_fn(row) and approved < n:
            row["Aprobado"] = True
            approved += 1
    batch_write_sheet_tabs(sheet_id, {tab: [headers] + [[r.get(h, "") for h in headers] for r in rows]})
    return approved


@pytest.mark.live
async def test_live_full_cycle_against_real_sheet(live_sheet_id, tmp_path):
    """Etapa A -> aprobar 2 keywords -> Etapa B -> aprobar 2 URLs -> Etapa C contra la Sheet real."""
    import time

    from src.agent.subagents import AdkStructuredLLM, GenAIEmbedder
    from src.tools import sheets_tool

    # Solo siembra las pestañas de entrada que falten o estén vacías
    missing = {tab: rows for tab, rows in LIVE_INPUTS.items() if not sheets_tool.read_sheet_tab(live_sheet_id, tab)}
    if missing:
        sheets_tool.batch_write_sheet_tabs(live_sheet_id, missing)

    settings = Settings()
    o = SeoProgrammaticOrchestrator(
        AdkStructuredLLM(settings), GenAIEmbedder(settings), settings=settings,
        state_store=JsonFileStateStore(tmp_path / "runs"), memory=JsonFileMemoryStore(tmp_path / "memory.json"),
        batch_size=10,
    )
    thread = f"live-{int(time.time())}"

    state = await o.run_phase_1_and_2(live_sheet_id, thread)
    assert state.status == RunStatus.WAITING_KEYWORD_APPROVAL
    print(f"\nEtapa A: {state.routing_summary} | patrones: {[p['nombre'] for p in state.patrones]}")
    assert _approve_first(live_sheet_id, "Keywords", "keyword", 2, lambda r: r.get("ruta") == "create_new_url") >= 1

    state = await o.run_phase_3(live_sheet_id, thread)
    assert state.status == RunStatus.WAITING_URL_APPROVAL
    print(f"Etapa B: {[p['url'] for p in state.plan]}")
    assert _approve_first(live_sheet_id, "Plan de URLs", "url_id", 2,
                          lambda r: r.get("slug_valid") is True and r.get("datos_propios") == "ok") >= 1

    state = await o.run_phase_4_to_7(live_sheet_id, thread)
    assert state.status == RunStatus.DONE
    print(f"Etapa C: {state.pipeline_summary}")

    titles = set(sheets_tool._sheet_props(live_sheet_id))
    assert set(FINAL_TABS) <= titles
    contenido = sheets_tool.read_sheet_tab(live_sheet_id, "Contenido", "A1:AZ100")
    assert contenido and all(r["url_id"].startswith("U") for r in contenido)
