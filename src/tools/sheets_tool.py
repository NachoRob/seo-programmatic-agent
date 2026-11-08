"""Herramientas de Google Sheets (API v4 con Application Default Credentials).

Se exponen de dos formas:
- `SHEETS_TOOLS`: lista de `FunctionTool` para usar directo en un `Agent` de ADK.
- `sheets_mcp`: servidor FastMCP (`uv run python -m src.tools.sheets_tool`).
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Any

import google.auth
from fastmcp import FastMCP
from google.adk.tools import FunctionTool
from googleapiclient.discovery import build
from pydantic import BaseModel, Field

from src.config import SHEETS_SCOPES, get_settings

# Alias aceptados en la pestaña de parámetros (claves ya normalizadas)
_MIN_JUDGE_SCORE_KEYS = {"score_minimo_juez", "score_minimo_del_juez", "min_judge_score"}
_SIMILARITY_KEYS = {"umbral_similitud", "umbral_de_similitud", "similarity_threshold"}
_MIN_KEYWORD_SCORE_KEYS = {"score_minimo_keyword", "score_minimo", "min_keyword_score"}
_MIN_VOLUME_KEYS = {"volumen_minimo", "min_search_volume"}
_WEIGHT_PREFIXES = ("peso_", "weight_")
_KV_HEADER_KEYS = {"parametro", "clave", "campo", "key"}


def sheets_credentials():
    """Credenciales con scope de Sheets en nombre de la service account del agente."""
    from src.integrations.credentials import agent_credentials

    return agent_credentials(SHEETS_SCOPES)


@lru_cache(maxsize=1)
def get_sheets_service():
    return build("sheets", "v4", credentials=sheets_credentials(), cache_discovery=False)


# ---------------------------------------------------------------- helpers ----

def _a1_tab(tab_name: str) -> str:
    """Nombre de pestaña citado para notación A1 (`'Mi pestaña'`)."""
    return "'" + tab_name.replace("'", "''") + "'"


def _normalize_key(key: Any) -> str:
    text = unicodedata.normalize("NFKD", str(key)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _to_number(value: Any) -> float | None:
    """Convierte '0,85', '85%', 7 o '7.5' a float. Devuelve None si no es número."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(" ", "")
    if not text:
        return None
    is_percent = text.endswith("%")
    text = text.rstrip("%").replace(",", ".")
    try:
        number = float(text)
    except ValueError:
        return None
    return number / 100 if is_percent else number


def _rows_to_dicts(rows: list[list[Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    headers = [str(h).strip() or f"col_{i + 1}" for i, h in enumerate(rows[0])]
    records = []
    for row in rows[1:]:
        if not any(str(cell).strip() for cell in row):
            continue
        padded = list(row) + [""] * (len(headers) - len(row))
        records.append(dict(zip(headers, padded)))
    return records


def _rows_to_kv(rows: list[list[Any]]) -> dict[str, Any]:
    """Pestaña clave/valor: columna A = clave, columna B = valor."""
    kv: dict[str, Any] = {}
    for row in rows:
        if not row or not str(row[0]).strip():
            continue
        key = _normalize_key(row[0])
        if key in _KV_HEADER_KEYS:
            continue
        kv[key] = row[1] if len(row) > 1 else ""
    return kv


def _cell(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, bool):
        return {"userEnteredValue": {"boolValue": value}}
    if isinstance(value, (int, float)):
        return {"userEnteredValue": {"numberValue": value}}
    text = str(value)
    if text.startswith("="):
        return {"userEnteredValue": {"formulaValue": text}}
    return {"userEnteredValue": {"stringValue": text}}


class SeoParameters(BaseModel):
    scoring_weights: dict[str, float] = Field(default_factory=dict)
    min_judge_score: float
    similarity_threshold: float = Field(ge=0, le=1)
    min_keyword_score: float = Field(default=0.0, ge=0, le=100)
    min_search_volume: float = Field(default=0.0, ge=0)


def _tab_key(name: str) -> str:
    """Nombre de pestaña comparable: "Parámetros" == "parametros" == "PARAMETROS"."""
    return _normalize_key(name)


def _sheet_props(spreadsheet_id: str) -> dict[str, dict]:
    """{título: properties} de todas las pestañas."""
    metadata = get_sheets_service().spreadsheets().get(
        spreadsheetId=spreadsheet_id,
        fields="sheets.properties(sheetId,title,gridProperties)",
    ).execute()
    return {s["properties"]["title"]: s["properties"] for s in metadata.get("sheets", [])}


def resolve_tab_name(spreadsheet_id: str, tab_name: str, titles: list[str] | None = None) -> str | None:
    """Título real de la pestaña que coincide con `tab_name` (ignora tildes y mayúsculas)."""
    titles = titles if titles is not None else list(_sheet_props(spreadsheet_id))
    if tab_name in titles:
        return tab_name
    wanted = _tab_key(tab_name)
    return next((t for t in titles if _tab_key(t) == wanted), None)


# ------------------------------------------------------------------ tools ----

def read_sheet_tab(spreadsheet_id: str, tab_name: str, range_def: str = "A1:Z500") -> list[dict]:
    """Lee una pestaña de Google Sheets y devuelve una lista de filas como diccionarios.

    La primera fila del rango se usa como encabezados. Las filas vacías se omiten
    y las celdas faltantes al final de una fila se rellenan con "". El nombre de la
    pestaña ignora tildes y mayúsculas; si la pestaña no existe devuelve [].

    Args:
        spreadsheet_id: ID de la Google Sheet (el segmento largo de la URL).
        tab_name: Nombre de la pestaña.
        range_def: Rango A1 dentro de la pestaña. Por defecto "A1:Z500".
    """
    tab_name = resolve_tab_name(spreadsheet_id, tab_name)
    if tab_name is None:
        return []
    response = (
        get_sheets_service()
        .spreadsheets()
        .values()
        .get(
            spreadsheetId=spreadsheet_id,
            range=f"{_a1_tab(tab_name)}!{range_def}",
            valueRenderOption="UNFORMATTED_VALUE",
        )
        .execute()
    )
    return _rows_to_dicts(response.get("values", []))


def read_sheet_values(spreadsheet_id: str, tab_name: str, range_def: str = "A1:Z5") -> list[list]:
    """Celdas crudas de una pestaña (incluye encabezados). [] si la pestaña no existe o está vacía."""
    tab_name = resolve_tab_name(spreadsheet_id, tab_name)
    if tab_name is None:
        return []
    response = get_sheets_service().spreadsheets().values().get(
        spreadsheetId=spreadsheet_id, range=f"{_a1_tab(tab_name)}!{range_def}",
        valueRenderOption="UNFORMATTED_VALUE",
    ).execute()
    return response.get("values", [])


def read_parameters_and_brief(spreadsheet_id: str) -> dict:
    """Lee los parámetros configurables y el brief del cliente desde su Google Sheet.

    Ambas pestañas son de formato clave/valor (columna A = parámetro, columna B = valor)
    y se leen en una sola llamada. Parámetros reconocidos:
    - `peso_<criterio>`: pesos de scoring (se devuelven sin el prefijo).
    - `score_minimo_juez`: score mínimo que debe dar el juez para aprobar.
    - `umbral_similitud`: umbral de similitud entre 0 y 1 (acepta "85%" o "0,85").
    - `score_minimo_keyword`: score mínimo (0-100) para crear una URL nueva.
    - `volumen_minimo`: volumen de búsqueda mínimo para crear una URL nueva.

    Args:
        spreadsheet_id: ID de la Google Sheet del cliente.

    Returns:
        {"parameters": {...}, "brief": {...}, "raw_parameters": {...}}
    """
    settings = get_settings()
    titles = list(_sheet_props(spreadsheet_id))
    tabs = {
        "parameters": resolve_tab_name(spreadsheet_id, settings.parameters_tab, titles),
        "brief": resolve_tab_name(spreadsheet_id, settings.brief_tab, titles),
    }
    existing = [(key, title) for key, title in tabs.items() if title]
    values: dict[str, list] = {}
    if existing:
        response = (
            get_sheets_service()
            .spreadsheets()
            .values()
            .batchGet(
                spreadsheetId=spreadsheet_id,
                ranges=[f"{_a1_tab(title)}!A1:B200" for _, title in existing],
                valueRenderOption="UNFORMATTED_VALUE",
            )
            .execute()
        )
        for (key, _), value_range in zip(existing, response.get("valueRanges", [])):
            values[key] = value_range.get("values", [])
    raw_params = _rows_to_kv(values.get("parameters", []))
    brief = _rows_to_kv(values.get("brief", []))

    weights: dict[str, float] = {}
    min_judge_score = settings.default_min_judge_score
    similarity = settings.default_similarity_threshold
    min_keyword_score = settings.default_min_keyword_score
    min_volume = settings.default_min_search_volume
    for key, value in raw_params.items():
        number = _to_number(value)
        if number is None:
            continue
        if key in _MIN_JUDGE_SCORE_KEYS:
            min_judge_score = number
        elif key in _SIMILARITY_KEYS:
            similarity = number
        elif key in _MIN_KEYWORD_SCORE_KEYS:
            min_keyword_score = number
        elif key in _MIN_VOLUME_KEYS:
            min_volume = number
        elif key.startswith(_WEIGHT_PREFIXES):
            weights[key.split("_", 1)[1]] = number

    parameters = SeoParameters(
        scoring_weights=weights,
        min_judge_score=min_judge_score,
        similarity_threshold=similarity,
        min_keyword_score=min_keyword_score,
        min_search_volume=min_volume,
    )
    return {
        "parameters": parameters.model_dump(),
        "brief": brief,
        "raw_parameters": raw_params,
    }


def batch_write_sheet_tabs(spreadsheet_id: str, tabs_data: dict[str, list[list]]) -> dict:
    """Escribe varias pestañas en una sola operación atómica (`spreadsheets.batchUpdate`).

    Para cada pestaña: la crea si no existe, amplía la grilla si hace falta, borra
    los valores anteriores y escribe las filas desde A1. Si ya existe una pestaña
    con el mismo nombre salvo tildes o mayúsculas, se usa esa. Si cualquier parte
    falla, Google Sheets no aplica ningún cambio.

    Args:
        spreadsheet_id: ID de la Google Sheet de destino.
        tabs_data: {nombre_pestaña: [[fila1...], [fila2...]]}. La primera fila
            suele ser el encabezado. Los valores que empiezan con "=" se escriben
            como fórmula.

    Returns:
        {"spreadsheet_id", "tabs": {pestaña: {"rows", "columns", "created"}}, "replies"}
    """
    if not tabs_data:
        raise ValueError("tabs_data está vacío: no hay nada que escribir.")
    for tab, rows in tabs_data.items():
        if not isinstance(rows, list) or not all(isinstance(r, list) for r in rows):
            raise ValueError(f"La pestaña '{tab}' debe ser una lista de filas (list[list]).")

    sheets_api = get_sheets_service().spreadsheets()
    by_title = _sheet_props(spreadsheet_id)
    # "Parámetros" y "Parametros" son la misma pestaña: se escribe sobre la que exista
    existing = {_tab_key(title): props for title, props in by_title.items()}
    next_id = max((p["sheetId"] for p in by_title.values()), default=0) + 1

    requests: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = {}
    for tab, rows in tabs_data.items():
        n_rows = max(len(rows), 1)
        n_cols = max((len(r) for r in rows), default=1) or 1
        created = _tab_key(tab) not in existing

        if created:
            sheet_id = next_id
            next_id += 1
            requests.append({
                "addSheet": {
                    "properties": {
                        "sheetId": sheet_id,
                        "title": tab,
                        "gridProperties": {"rowCount": max(n_rows, 100), "columnCount": max(n_cols, 26)},
                    }
                }
            })
        else:
            props = existing[_tab_key(tab)]
            sheet_id = props["sheetId"]
            grid = props.get("gridProperties", {})
            for dimension, needed, current in (
                ("ROWS", n_rows, grid.get("rowCount", 0)),
                ("COLUMNS", n_cols, grid.get("columnCount", 0)),
            ):
                if needed > current:
                    requests.append({
                        "appendDimension": {"sheetId": sheet_id, "dimension": dimension, "length": needed - current}
                    })
            # Limpia valores previos (mantiene formato)
            requests.append({"updateCells": {"range": {"sheetId": sheet_id}, "fields": "userEnteredValue"}})

        if rows:
            requests.append({
                "updateCells": {
                    "start": {"sheetId": sheet_id, "rowIndex": 0, "columnIndex": 0},
                    "rows": [{"values": [_cell(v) for v in row]} for row in rows],
                    "fields": "userEnteredValue",
                }
            })
        summary[tab] = {"rows": len(rows), "columns": n_cols if rows else 0, "created": created}

    response = sheets_api.batchUpdate(
        spreadsheetId=spreadsheet_id, body={"requests": requests}
    ).execute()
    return {
        "spreadsheet_id": spreadsheet_id,
        "tabs": summary,
        "replies": len(response.get("replies", [])),
    }


# ------------------------------------------------------- ADK / FastMCP ----

SHEETS_TOOLS = [
    FunctionTool(read_sheet_tab),
    FunctionTool(read_parameters_and_brief),
    FunctionTool(batch_write_sheet_tabs),
]

sheets_mcp = FastMCP("d2b-sheets")
for _fn in (read_sheet_tab, read_parameters_and_brief, batch_write_sheet_tabs):
    sheets_mcp.tool(_fn)


if __name__ == "__main__":
    sheets_mcp.run()
