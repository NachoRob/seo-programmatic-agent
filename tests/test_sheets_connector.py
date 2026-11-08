"""Conector de Google Sheets con la API mockeada."""

from unittest.mock import MagicMock

import pytest

from src.tools import sheets_tool
from src.tools.sheets_tool import (
    SHEETS_TOOLS,
    batch_write_sheet_tabs,
    read_parameters_and_brief,
    read_sheet_tab,
    sheets_mcp,
)


DEFAULT_TABS = ["Keywords", "Brief d'Cliente", "Parámetros", "Brief"]


@pytest.fixture
def service(monkeypatch):
    svc = MagicMock(name="sheets_service")
    svc.spreadsheets.return_value.get.return_value.execute.return_value = {"sheets": [
        {"properties": {"sheetId": i, "title": t, "gridProperties": {"rowCount": 1000, "columnCount": 26}}}
        for i, t in enumerate(DEFAULT_TABS)
    ]}
    monkeypatch.setattr(sheets_tool, "get_sheets_service", lambda: svc)
    return svc


def values_api(svc):
    return svc.spreadsheets.return_value.values.return_value


# ---------------------------------------------------------- read_sheet_tab ----

def test_read_sheet_tab_maps_headers_and_pads_rows(service):
    values_api(service).get.return_value.execute.return_value = {"values": [
        ["keyword", "volumen", "intencion"],
        ["seguro auto", 1200, "transaccional"],
        ["seguro hogar", 800],          # fila corta
        ["", ""],                       # fila vacía
    ]}

    rows = read_sheet_tab("sheet-1", "Keywords")

    assert rows == [
        {"keyword": "seguro auto", "volumen": 1200, "intencion": "transaccional"},
        {"keyword": "seguro hogar", "volumen": 800, "intencion": ""},
    ]
    values_api(service).get.assert_called_once_with(
        spreadsheetId="sheet-1",
        range="'Keywords'!A1:Z500",
        valueRenderOption="UNFORMATTED_VALUE",
    )


def test_read_sheet_tab_quotes_tab_names(service):
    values_api(service).get.return_value.execute.return_value = {}
    assert read_sheet_tab("s", "Brief d'Cliente", "A1:C10") == []
    assert values_api(service).get.call_args.kwargs["range"] == "'Brief d''Cliente'!A1:C10"


# ------------------------------------------------ read_parameters_and_brief ----

def test_read_parameters_and_brief_parses_kv_tabs(service):
    values_api(service).batchGet.return_value.execute.return_value = {"valueRanges": [
        {"values": [
            ["Parámetro", "Valor"],
            ["Peso Volumen", 0.4],
            ["peso_intencion", "0,35"],
            ["Peso Dificultad", "25%"],
            ["Score mínimo juez", 8],
            ["Umbral similitud", "85%"],
            ["Score mínimo keyword", 40],
            ["Volumen mínimo", 50],
            ["comentario", "texto libre"],
        ]},
        {"values": [
            ["Cliente", "ACME Seguros"],
            ["Dominio", "acme.cl"],
            ["Tono de voz", "cercano"],
        ]},
    ]}

    result = read_parameters_and_brief("sheet-1")

    assert result["parameters"] == {
        "scoring_weights": {"volumen": 0.4, "intencion": 0.35, "dificultad": 0.25},
        "min_judge_score": 8.0,
        "similarity_threshold": 0.85,
        "min_keyword_score": 40.0,
        "min_search_volume": 50.0,
    }
    assert result["brief"] == {"cliente": "ACME Seguros", "dominio": "acme.cl", "tono_de_voz": "cercano"}
    assert result["raw_parameters"]["comentario"] == "texto libre"
    kwargs = values_api(service).batchGet.call_args.kwargs
    # La config dice "Parámetros"; se resuelve contra el título real de la pestaña
    assert kwargs["ranges"] == ["'Parámetros'!A1:B200", "'Brief'!A1:B200"]


def test_tab_names_ignore_accents_and_case(service):
    service.spreadsheets.return_value.get.return_value.execute.return_value = {"sheets": [
        {"properties": {"sheetId": 0, "title": "PARAMETROS"}}, {"properties": {"sheetId": 1, "title": "brief"}},
    ]}
    values_api(service).batchGet.return_value.execute.return_value = {"valueRanges": [{}, {}]}
    read_parameters_and_brief("s")
    assert values_api(service).batchGet.call_args.kwargs["ranges"] == ["'PARAMETROS'!A1:B200", "'brief'!A1:B200"]


def test_missing_tabs_read_as_empty(service):
    service.spreadsheets.return_value.get.return_value.execute.return_value = {"sheets": []}
    assert read_sheet_tab("s", "Semillas") == []
    params = read_parameters_and_brief("s")
    assert params["brief"] == {} and params["parameters"]["min_judge_score"] == 4.0
    values_api(service).get.assert_not_called()
    values_api(service).batchGet.assert_not_called()


def test_read_parameters_uses_defaults_when_missing(service):
    values_api(service).batchGet.return_value.execute.return_value = {"valueRanges": [{}, {}]}
    params = read_parameters_and_brief("s")["parameters"]
    assert params == {
        "scoring_weights": {}, "min_judge_score": 4.0, "similarity_threshold": 0.5,
        "min_keyword_score": 0.0, "min_search_volume": 0.0,
    }


def test_read_parameters_rejects_invalid_similarity(service):
    values_api(service).batchGet.return_value.execute.return_value = {"valueRanges": [
        {"values": [["umbral_similitud", 85]]}, {},
    ]}
    with pytest.raises(ValueError):
        read_parameters_and_brief("s")


# --------------------------------------------------- batch_write_sheet_tabs ----

def test_batch_write_builds_single_atomic_batch_update(service):
    sheets = service.spreadsheets.return_value
    sheets.get.return_value.execute.return_value = {"sheets": [
        {"properties": {"sheetId": 0, "title": "Keywords",
                        "gridProperties": {"rowCount": 2, "columnCount": 2}}},
    ]}
    sheets.batchUpdate.return_value.execute.return_value = {"replies": [{}] * 5}

    result = batch_write_sheet_tabs("sheet-1", {
        "Keywords": [["keyword", "score", "ok"], ["seguro auto", 8.5, True], ["seguro hogar", 7, False]],
        "Resumen": [["total", "=COUNTA(Keywords!A:A)-1"], ["nota", None]],
    })

    sheets.batchUpdate.assert_called_once()
    body = sheets.batchUpdate.call_args.kwargs["body"]
    reqs = body["requests"]

    # Keywords existe: amplía 1 fila y 1 columna, limpia y escribe
    assert reqs[0] == {"appendDimension": {"sheetId": 0, "dimension": "ROWS", "length": 1}}
    assert reqs[1] == {"appendDimension": {"sheetId": 0, "dimension": "COLUMNS", "length": 1}}
    assert reqs[2] == {"updateCells": {"range": {"sheetId": 0}, "fields": "userEnteredValue"}}
    write_kw = reqs[3]["updateCells"]
    assert write_kw["start"] == {"sheetId": 0, "rowIndex": 0, "columnIndex": 0}
    assert write_kw["fields"] == "userEnteredValue"
    assert write_kw["rows"][1]["values"] == [
        {"userEnteredValue": {"stringValue": "seguro auto"}},
        {"userEnteredValue": {"numberValue": 8.5}},
        {"userEnteredValue": {"boolValue": True}},
    ]

    # Resumen no existe: addSheet con id nuevo y escritura en el mismo batch
    assert reqs[4]["addSheet"]["properties"]["sheetId"] == 1
    assert reqs[4]["addSheet"]["properties"]["title"] == "Resumen"
    write_res = reqs[5]["updateCells"]
    assert write_res["start"]["sheetId"] == 1
    assert write_res["rows"][0]["values"][1] == {"userEnteredValue": {"formulaValue": "=COUNTA(Keywords!A:A)-1"}}
    assert write_res["rows"][1]["values"][1] == {}

    assert result == {
        "spreadsheet_id": "sheet-1",
        "tabs": {
            "Keywords": {"rows": 3, "columns": 3, "created": False},
            "Resumen": {"rows": 2, "columns": 2, "created": True},
        },
        "replies": 5,
    }


@pytest.mark.parametrize("payload", [{}, {"Tab": "no-es-lista"}, {"Tab": ["fila-no-lista"]}])
def test_batch_write_validates_payload(service, payload):
    with pytest.raises(ValueError):
        batch_write_sheet_tabs("s", payload)
    service.spreadsheets.return_value.batchUpdate.assert_not_called()


def test_batch_write_reuses_existing_tab_with_accents(service):
    sheets = service.spreadsheets.return_value
    sheets.get.return_value.execute.return_value = {"sheets": [
        {"properties": {"sheetId": 7, "title": "Parámetros", "gridProperties": {"rowCount": 100, "columnCount": 26}}},
    ]}
    sheets.batchUpdate.return_value.execute.return_value = {"replies": []}
    result = batch_write_sheet_tabs("s", {"Parametros": [["clave", "valor"]]})
    reqs = sheets.batchUpdate.call_args.kwargs["body"]["requests"]
    assert not any("addSheet" in r for r in reqs)
    assert reqs[-1]["updateCells"]["start"]["sheetId"] == 7
    assert result["tabs"]["Parametros"]["created"] is False


def test_sheets_credentials_impersonate_service_account(monkeypatch):
    from google.auth import impersonated_credentials

    from src.config import Settings
    from src.integrations import credentials

    source = object()
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", raising=False)
    monkeypatch.setattr(credentials, "get_settings",
                        lambda: Settings(service_account="agente@p.iam.gserviceaccount.com"))
    monkeypatch.setattr(credentials.google.auth, "default", lambda scopes: (source, "p"))
    created = {}
    monkeypatch.setattr(impersonated_credentials, "Credentials", lambda **kw: created.update(kw) or "imp")

    assert sheets_tool.sheets_credentials() == "imp"
    assert created["target_principal"] == "agente@p.iam.gserviceaccount.com"
    assert created["target_scopes"] == ["https://www.googleapis.com/auth/spreadsheets"]
    assert created["source_credentials"] is source


def test_sheets_credentials_use_adc_when_deployed(monkeypatch):
    from src.config import Settings
    from src.integrations import credentials

    monkeypatch.setenv("K_SERVICE", "seo-chat")
    monkeypatch.setattr(credentials, "get_settings",
                        lambda: Settings(service_account="agente@p.iam.gserviceaccount.com"))
    seen = {}
    monkeypatch.setattr(credentials.google.auth, "default", lambda scopes: (seen.update(s=scopes) or "adc", "p"))
    assert sheets_tool.sheets_credentials() == "adc"
    assert seen["s"] == ["https://www.googleapis.com/auth/spreadsheets"]


# ------------------------------------------------------------- wrappers ----

def test_tools_are_exposed_to_adk():
    declarations = {t.name: t._get_declaration() for t in SHEETS_TOOLS}
    assert set(declarations) == {"read_sheet_tab", "read_parameters_and_brief", "batch_write_sheet_tabs"}
    schema = declarations["batch_write_sheet_tabs"].parameters_json_schema
    assert schema["required"] == ["spreadsheet_id", "tabs_data"]
    assert schema["properties"]["tabs_data"]["additionalProperties"]["type"] == "array"


async def test_tools_are_exposed_via_fastmcp():
    tools = await sheets_mcp.list_tools()
    assert {t.name for t in tools} == {"read_sheet_tab", "read_parameters_and_brief", "batch_write_sheet_tabs"}


# ------------------------------------------------------------------- live ----

@pytest.mark.live
def test_live_write_read_roundtrip(live_sheet_id):
    """Crea una pestaña temporal, escribe tipos mixtos, la lee y la borra."""
    import uuid

    tab = f"_prueba_conector_{uuid.uuid4().hex[:6]}"
    rows = [
        ["keyword", "volumen", "ctr", "aprobado", "formula", "vacio"],
        ["seguro auto viña del mar", 1200, 0.25, True, "=B2*2", None],
        ["seguro auto concepción", 300, 0.1, False, "=B3*2", None],
    ]
    result = batch_write_sheet_tabs(live_sheet_id, {tab: rows})
    try:
        assert result["tabs"][tab]["created"] is True
        data = read_sheet_tab(live_sheet_id, tab, "A1:F10")
        assert data[0]["keyword"] == "seguro auto viña del mar"
        assert data[0]["volumen"] == 1200 and data[0]["ctr"] == 0.25
        assert data[0]["aprobado"] is True and data[1]["aprobado"] is False
        assert data[0]["formula"] == 2400  # la fórmula se evalúa en Sheets

        # Reescribir la misma pestaña limpia valores anteriores (no queda la fila 3)
        batch_write_sheet_tabs(live_sheet_id, {tab: rows[:2]})
        assert len(read_sheet_tab(live_sheet_id, tab, "A1:F10")) == 1
    finally:
        sheet_id = sheets_tool._sheet_props(live_sheet_id)[tab]["sheetId"]
        sheets_tool.get_sheets_service().spreadsheets().batchUpdate(
            spreadsheetId=live_sheet_id, body={"requests": [{"deleteSheet": {"sheetId": sheet_id}}]}
        ).execute()


@pytest.mark.live
def test_live_read_parameters_and_brief(live_sheet_id):
    result = read_parameters_and_brief(live_sheet_id)
    assert set(result) == {"parameters", "brief", "raw_parameters"}
    assert 1 <= result["parameters"]["min_judge_score"] <= 5
