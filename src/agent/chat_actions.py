"""Acciones del agente conversacional sobre la Sheet del cliente y el run.

La persona conversa en Google Chat y el agente llena por ella las pestañas de
entrada (Brief, Semillas, Datos propios), corre las etapas del orquestador y
registra las aprobaciones en las pestañas Keywords y Plan de URLs. Cada acción
devuelve un dict corto, pensado para que el LLM lo explique en palabras simples:
sin códigos internos (`optimize_existing`, `coverage_status`, ...).
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import re
from collections import Counter
from typing import Any, Literal
from urllib.parse import urlparse

from src.agent.orchestrator import (
    _SEED_ALIASES,
    APPROVED_COL,
    REJECT_REASON_COL,
    SITE_INVENTORY_KEY,
    TAB_DESCARTADOS,
    TAB_KEYWORDS,
    TAB_PLAN,
    InvalidRunState,
    RunState,
    RunStatus,
    SearchConsoleError,
    site_from_brief,
)
from src.agent.service import (
    SeoRunService,
    is_sheet_permission_error,
    normalize_domain,
    parse_sheet_id,
    sheet_url,
    template_tabs,
)
from src.tools.seo_deterministic import dedupe_key, normalize_text
from src.tools.site_reader import SiteReader, SiteUnavailable

GSC_ACCOUNT = "search-console-mcp-runner@your-gcp-project.iam.gserviceaccount.com"  # cuenta del MCP
SEED_COLUMNS = ["keyword", "fuente", "volumen", "dificultad", "intencion"]
BRIEF_KEYS = ("cliente", "sitio", "pais", "idioma", "trato", "tipo_schema")
SCHEMA_TYPES = ("Service", "Product", "LocalBusiness", "Article", "BreadcrumbList")
GSC_SNAPSHOT_DAYS = 28
MAX_ITEMS = 25  # tope de filas por lista que vuelve al LLM
MAX_COLLECTIONS_SHOWN = 60  # colecciones que ve el LLM al explorar (el resto queda guardado)

# Problemas del slug en palabras simples (el LLM no debe mostrar los códigos)
SLUG_ISSUES = {
    "hub_invalido": "la carpeta padre no es válida",
    "hub_no_permitido": "la plataforma no admite esa carpeta",
    "variable_faltante": "falta un dato para armar la URL",
    "slug_vacio": "la URL quedó vacía",
    "excede_largo": "la URL es demasiado larga",
    "duplicado_en_plan": "otra página del plan tiene la misma URL",
    "colision_url_existente": "esa URL ya existe en el sitio",
}

Mode = Literal["reemplazar", "agregar", "quitar"]
_BULLET = re.compile(r"^\s*(?:[-•*·]|\d+[.)])\s+")
_MD_RULE = re.compile(r"^\s*\|?\s*:?-{3,}")


# ------------------------------------------------------------------ parsers ----

def _split(line: str) -> list[str]:
    for sep in ("\t", "|", ";"):
        if sep in line:
            return [c.strip() for c in line.strip().strip("|").split(sep)]
    return [line.strip()]


def _number(value: str) -> float | int | str:
    """"1.600" y "1,600" son miles; "12,5" es decimal; lo demás queda como texto."""
    text = value.strip()
    text = re.sub(r"[.,]", "", text) if re.fullmatch(r"\d{1,3}([.,]\d{3})+", text) else text.replace(",", ".")
    try:
        number = float(text)
    except ValueError:
        return value.strip()
    return int(number) if number.is_integer() else number


def parse_keywords(text: str) -> list[dict]:
    """Keywords pegadas en el chat: una por línea, con columnas opcionales.

    Acepta "keyword", "keyword | volumen | dificultad | intención" (también con
    tabulaciones, como se copia desde Sheets) o una tabla con encabezado.
    """
    lines = [_BULLET.sub("", line) for line in (text or "").splitlines()]
    lines = [line for line in lines if line.strip() and not _MD_RULE.match(line)]
    if not lines:
        return []
    header = [normalize_text(c).replace(" ", "_") for c in _split(lines[0])]
    columns = ["keyword", "search_volume", "difficulty", "intent"]
    if any(_SEED_ALIASES.get(h) == "keyword" for h in header):
        columns = [_SEED_ALIASES.get(h, h) for h in header]
        lines = lines[1:]
    out = []
    for line in lines:
        cells = _split(line)
        row = {col: cells[i] for i, col in enumerate(columns) if i < len(cells) and cells[i]}
        if not row.get("keyword"):
            continue
        for key in ("search_volume", "difficulty"):
            if key in row:
                row[key] = _number(row[key])
        out.append(row)
    return out


def parse_table(text: str) -> list[dict]:
    """Tabla pegada en el chat (TSV de Sheets, markdown o CSV); la 1ª línea es el encabezado."""
    lines = [line for line in (text or "").splitlines() if line.strip() and not _MD_RULE.match(line)]
    if len(lines) < 2:
        return []
    if any(sep in lines[0] for sep in ("\t", "|", ";")):
        rows = [_split(line) for line in lines]
    else:
        rows = [[c.strip() for c in r] for r in csv.reader(io.StringIO("\n".join(lines)))]
    header = [h for h in rows[0]]
    return [{h: _number(r[i]) if i < len(r) else "" for i, h in enumerate(header) if h}
            for r in rows[1:] if any(c.strip() for c in r)]


# ------------------------------------------------------------------ helpers ----

def _path(url: str | None) -> str:
    return urlparse(url or "").path or "/"


def _site_domain(site: str) -> str:
    """`sc-domain:x.cl`, `https://www.x.cl/` o `x.cl` -> `x.cl`."""
    return normalize_domain(site.strip().removeprefix("sc-domain:"))


def _gsc_parts(row: dict) -> tuple[str, str | None]:
    keys = row.get("keys") or [row.get("query"), row.get("page")]
    return str(keys[0] or ""), (keys[1] if len(keys) > 1 else None)


def _keyword_view(kw: dict) -> dict:
    view = {"keyword": kw["keyword"], "volumen": kw.get("search_volume")}
    if kw.get("ruta") == "create_new_url":
        view.update(grupo=kw.get("patron"), puntaje=kw.get("score"))
    elif kw.get("existe_en_sitio"):
        view.update(pagina_actual=_path(kw.get("existing_url")), motivo="ya existe esa página en el sitio")
    else:
        view.update(pagina_actual=_path(kw.get("existing_url")), posicion=kw.get("gsc_position"))
    return {k: v for k, v in view.items() if v not in (None, "")}


class SeoChatActions:
    """Lo que el agente conversacional puede hacer. `thread_id` = la conversación."""

    def __init__(self, service: SeoRunService, reader: SiteReader | None = None):
        self.service = service
        self.reader = reader or SiteReader()

    @property
    def orch(self):
        return self.service.orchestrator

    @property
    def settings(self):
        return self.orch.settings

    # ---------------------------------------------------------- errores ----

    def explain(self, sheet_id: str | None, exc: BaseException) -> dict[str, Any]:
        """Error en palabras para el chat (sin botones ni pestañas que llenar a mano)."""
        detail = f"{type(exc).__name__}: {exc}"[:400]
        out: dict[str, Any] = {"ok": False, "detalle_tecnico": detail}
        if sheet_id:
            out["sheet_url"] = sheet_url(sheet_id)
        if is_sheet_permission_error(exc):
            out.update(error="No tengo acceso a la Sheet.",
                       que_hacer=f"Compartirla como Editor con {self.settings.service_account}.")
        elif isinstance(exc, SearchConsoleError) or "Search Console" in str(exc):
            out.update(error=f"No pude leer Search Console de {getattr(exc, 'site_url', 'ese sitio')}.",
                       que_hacer="Buscar la propiedad exacta con la lista de propiedades de Search Console; "
                                 f"si no aparece, agregar {GSC_ACCOUNT} como usuario Restringido en Search Console.")
        elif "El Brief no define el sitio" in str(exc):
            out.update(error="Todavía no sé cuál es el sitio del cliente.", que_hacer="Pedir el dominio.")
        elif isinstance(exc, InvalidRunState):
            out.update(error="Ese paso todavía no corresponde.", que_hacer="Revisar el estado con get_overview.")
        elif any(k in str(exc) for k in ("RESOURCE_EXHAUSTED", "429", "503", "UNAVAILABLE", "DEADLINE_EXCEEDED")):
            out.update(error="Gemini está saturado o no respondió a tiempo.",
                       que_hacer="Reintentar en un par de minutos. El avance quedó guardado.")
        else:
            out.update(error="Algo falló de forma inesperada.", que_hacer="Reintentar; si se repite, avisar a datos.")
        return out

    async def _state(self, sheet_id: str, thread_id: str) -> RunState | None:
        return await self.orch.get_state(sheet_id, thread_id)

    # ------------------------------------------------------------ Sheet ----

    async def use_sheet(self, sheet: str) -> dict[str, Any]:
        """Revisa el acceso y crea las pestañas de entrada que falten (sin pisar datos)."""
        try:
            sheet_id = parse_sheet_id(sheet)
        except ValueError as e:
            return {"ok": False, "error": str(e), "que_hacer": "Pedir la URL completa de la Google Sheet."}
        sheets = self.orch.sheets
        try:
            tabs = template_tabs(self.settings)
            missing = {name: rows for name, rows in tabs.items()
                       if not any(any(str(c).strip() for c in r) for r in await sheets.read_raw(sheet_id, name))}
            # La plantilla de Datos propios trae columnas de ejemplo: se crea vacía
            if self.settings.own_data_tab in missing:
                missing[self.settings.own_data_tab] = [[]]
            if missing:
                await sheets.write_tabs(sheet_id, missing)
        except Exception as e:  # noqa: BLE001
            return self.explain(sheet_id, e)
        return {"ok": True, "sheet_id": sheet_id, "sheet_url": sheet_url(sheet_id)}

    async def overview(self, sheet_id: str, thread_id: str) -> dict[str, Any]:
        """Qué hay hoy en la Sheet y en qué va el trabajo."""
        sheets = self.orch.sheets
        try:
            brief = (await sheets.read_parameters_and_brief(sheet_id)).get("brief") or {}
            seeds = await sheets.read_tab(sheet_id, self.settings.seeds_tab)
            own = await sheets.read_tab(sheet_id, self.settings.own_data_tab)
        except Exception as e:  # noqa: BLE001
            return self.explain(sheet_id, e)
        state = await self._state(sheet_id, thread_id)
        inventory = await self._inventory(sheet_id)
        keywords = [str(r.get("keyword") or r.get("Keyword") or "").strip() for r in seeds]
        keywords = [k for k in keywords if k]
        return {
            "ok": True, "sheet_url": sheet_url(sheet_id),
            "cliente": {k: brief.get(k) for k in BRIEF_KEYS if brief.get(k) not in (None, "")},
            "keywords": {"total": len(keywords), "muestra": keywords[:15]},
            "datos_propios": {"filas": len(own), "columnas": list(own[0].keys()) if own else [],
                              "muestra": own[:3]},
            "paso_actual": self._step(state),
            "sitio_explorado": {"plataforma": inventory.get("plataforma"), "urls": len(inventory.get("urls") or []),
                                "colecciones": len(inventory.get("colecciones") or [])} if inventory else False,
        }

    @staticmethod
    def _step(state: RunState | None) -> str:
        if state is None or state.status == RunStatus.NEW:
            return "sin analizar"
        return {
            RunStatus.WAITING_KEYWORD_APPROVAL: "keywords analizadas, falta elegir cuáles van",
            RunStatus.WAITING_URL_APPROVAL: "plan de URLs listo, falta elegir cuáles generar",
            RunStatus.DONE: "contenido generado",
        }[state.status]

    async def set_client(self, sheet_id: str, values: dict[str, Any]) -> dict[str, Any]:
        """Escribe o actualiza claves del Brief (cliente, sitio, país, trato, tipo de schema)."""
        values = {k: str(v).strip() for k, v in values.items() if k in BRIEF_KEYS and v not in (None, "")}
        if "trato" in values:
            values["trato"] = "usted" if "usted" in normalize_text(values["trato"]) else "tuteo"
        if "tipo_schema" in values:
            match = next((t for t in SCHEMA_TYPES if normalize_text(t) == normalize_text(values["tipo_schema"])), None)
            if not match:
                return {"ok": False, "error": f"tipo_schema debe ser uno de: {', '.join(SCHEMA_TYPES)}."}
            values["tipo_schema"] = match
        if "sitio" in values:
            values["sitio"], _ = site_from_brief({"sitio": values["sitio"]})
        sheets = self.orch.sheets
        try:
            rows = await sheets.read_raw(sheet_id, self.settings.brief_tab) or [["clave", "valor"]]
            seen = set()
            for row in rows:
                key = normalize_text(row[0]).replace(" ", "_") if row else ""
                if key in values:
                    row[1:2] = [values[key]] if len(row) > 1 else [values[key]]
                    seen.add(key)
            rows += [[k, v] for k, v in values.items() if k not in seen]
            await sheets.write_tabs(sheet_id, {self.settings.brief_tab: rows})
        except Exception as e:  # noqa: BLE001
            return self.explain(sheet_id, e)
        return {"ok": True, "guardado": values}

    async def site_snapshot(self, sheet_id: str) -> dict[str, Any]:
        """Resumen de Search Console (últimos 28 días) para conversar sobre el sitio."""
        try:
            brief = (await self.orch.sheets.read_parameters_and_brief(sheet_id)).get("brief") or {}
            site_url, _ = site_from_brief(brief)
            end = self.orch.today() - dt.timedelta(days=self.settings.gsc_lag_days)
            start = end - dt.timedelta(days=GSC_SNAPSHOT_DAYS - 1)
            rows = await self.orch.search_console.get_rows(site_url, start.isoformat(), end.isoformat())
        except Exception as e:  # noqa: BLE001
            return self.explain(sheet_id, e)
        by_query: dict[str, dict] = {}
        pages: Counter[str] = Counter()
        for row in rows:
            query, page = _gsc_parts(row)
            if page:
                pages[_path(page).strip("/").split("/")[0] or "(inicio)"] += 1
            item = by_query.setdefault(query, {"query": query, "clics": 0, "impresiones": 0, "pos": None,
                                               "_best": -1, "pagina": None})
            item["clics"] += int(row.get("clicks") or 0)
            impressions = int(row.get("impressions") or 0)
            item["impresiones"] += impressions
            if impressions > item["_best"]:
                item["_best"], item["pos"], item["pagina"] = impressions, round(float(row.get("position") or 0), 1), _path(page)
        queries = [{k: v for k, v in q.items() if k != "_best"} for q in by_query.values()]
        top = sorted(queries, key=lambda q: -q["clics"])[:10]
        opportunities = sorted((q for q in queries if (q["pos"] or 0) > 10), key=lambda q: -q["impresiones"])[:15]
        return {
            "ok": True, "sitio": site_url, "periodo": f"{start.isoformat()} a {end.isoformat()}",
            "busquedas_distintas": len(queries), "paginas_con_trafico": len({_gsc_parts(r)[1] for r in rows}),
            "secciones": dict(pages.most_common(8)),
            "top_por_clics": top,
            "oportunidades_pos_mayor_10": opportunities,
        }

    # ----------------------------------------------------------- sitio ----

    async def _inventory(self, sheet_id: str) -> dict[str, Any]:
        return await self.orch.state_store.load(sheet_id, SITE_INVENTORY_KEY) or {}

    async def _domain(self, sheet_id: str, domain: str = "") -> str:
        if domain.strip():
            return _site_domain(domain)
        brief = (await self.orch.sheets.read_parameters_and_brief(sheet_id)).get("brief") or {}
        return site_from_brief(brief)[1]

    async def explore_site(self, sheet_id: str, domain: str = "") -> dict[str, Any]:
        """Lee el sitio (plataforma, sitemap, colecciones) y guarda el inventario para el análisis."""
        try:
            domain = await self._domain(sheet_id, domain)
        except Exception as e:  # noqa: BLE001
            return self.explain(sheet_id, e)
        try:
            inventory = await self.reader.explore(domain)
        except SiteUnavailable as e:
            return {"ok": False, "accesible": False, "error": str(e),
                    "que_hacer": ("Seguir con Search Console (review_site) y pedir a la persona las secciones "
                                  "y colecciones principales del sitio.")}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"No pude leer {domain}.", "detalle_tecnico": f"{type(e).__name__}: {e}"[:400]}
        try:
            await self.orch.state_store.save(sheet_id, SITE_INVENTORY_KEY, inventory)
        except Exception as e:  # noqa: BLE001
            return self.explain(sheet_id, e)
        collections = sorted(inventory["colecciones"], key=lambda c: -(c.get("productos") or 0))
        return {
            "ok": True, "sitio": inventory["base_url"], "plataforma": inventory["plataforma"],
            "como_se_crean_urls": inventory["reglas_url"]["nota"],
            "urls_en_sitemap": len(inventory["urls"]), "secciones": inventory["secciones"],
            "colecciones_total": len(collections),
            "colecciones": [{"titulo": c["titulo"], "url": _path(c["url"]),
                             **({"productos": c["productos"]} if c.get("productos") is not None else {})}
                            for c in collections[:MAX_COLLECTIONS_SHOWN]],
            "avisos": inventory["avisos"],
        }

    async def read_page(self, sheet_id: str, url: str) -> dict[str, Any]:
        """Lee una página del sitio del cliente (title, H1, H2, texto, enlaces internos)."""
        try:
            domain = await self._domain(sheet_id)
        except Exception as e:  # noqa: BLE001
            return self.explain(sheet_id, e)
        try:
            return {"ok": True, **await self.reader.read_page(url, domain)}
        except (ValueError, SiteUnavailable) as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": "No pude leer esa página.", "detalle_tecnico": f"{type(e).__name__}: {e}"[:400]}

    async def search_console_sites(self, domain: str = "") -> dict[str, Any]:
        """Propiedades de Search Console con acceso a datos, filtradas por dominio (incluye subdominios)."""
        try:
            sites = await self.orch.search_console.list_sites()
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": "No pude consultar las propiedades de Search Console.",
                    "que_hacer": "Reintentar; si se repite, avisar a datos.",
                    "detalle_tecnico": f"{type(e).__name__}: {e}"[:400]}
        wanted = _site_domain(domain) if domain.strip() else ""
        found = [
            {"sitio": s["siteUrl"], "acceso": "restringido" if s.get("permissionLevel") == "siteRestrictedUser" else "completo"}
            for s in sites
            if s.get("siteUrl") and s.get("permissionLevel") != "siteUnverifiedUser"  # sin acceso a datos
            and (not wanted or _site_domain(s["siteUrl"]) == wanted or _site_domain(s["siteUrl"]).endswith(f".{wanted}"))
        ]
        out: dict[str, Any] = {"ok": True, "propiedades": found[:MAX_ITEMS], "total": len(found)}
        if wanted:
            exact = [p["sitio"] for p in found if _site_domain(p["sitio"]) == wanted]
            # La propiedad de dominio cubre todo el sitio; si no, solo una propiedad sin ambigüedad
            out["sugerida"] = next((s for s in exact if s.startswith("sc-domain:")), exact[0] if len(exact) == 1 else None)
            if not found:
                out["que_hacer"] = f"Agregar {GSC_ACCOUNT} como usuario Restringido en la propiedad del cliente en Search Console."
        return out

    # -------------------------------------------------------------- datos ----

    async def set_keywords(self, sheet_id: str, text: str, mode: Mode, source: str = "") -> dict[str, Any]:
        """Escribe la pestaña Semillas desde texto pegado en el chat."""
        incoming = parse_keywords(text)
        if not incoming:
            return {"ok": False, "error": "No encontré keywords en el texto (una por línea)."}
        for kw in incoming:
            if source and not kw.get("source"):
                kw["source"] = source
        sheets = self.orch.sheets
        try:
            current = [] if mode == "reemplazar" else [
                {"keyword": r.get("keyword") or r.get("Keyword"), "source": r.get("fuente"),
                 "search_volume": r.get("volumen"), "difficulty": r.get("dificultad"), "intent": r.get("intencion")}
                for r in await sheets.read_tab(sheet_id, self.settings.seeds_tab)
                if str(r.get("keyword") or r.get("Keyword") or "").strip()
            ]
            if mode == "quitar":
                drop = {dedupe_key(k["keyword"]) for k in incoming}
                result = [k for k in current if dedupe_key(k["keyword"]) not in drop]
            else:
                by_key = {dedupe_key(k["keyword"]): k for k in current}
                for kw in incoming:
                    by_key[dedupe_key(kw["keyword"])] = {**by_key.get(dedupe_key(kw["keyword"]), {}), **kw}
                result = list(by_key.values())
            rows = [SEED_COLUMNS] + [[k.get("keyword"), k.get("source") or "", k.get("search_volume") or "",
                                      k.get("difficulty") or "", k.get("intent") or ""] for k in result]
            await sheets.write_tabs(sheet_id, {self.settings.seeds_tab: rows})
        except Exception as e:  # noqa: BLE001
            return self.explain(sheet_id, e)
        return {"ok": True, "total": len(result), "recibidas": len(incoming),
                "con_volumen": sum(bool(k.get("search_volume")) for k in result)}

    async def set_own_data(self, sheet_id: str, table: str, mode: Literal["reemplazar", "agregar"]) -> dict[str, Any]:
        """Escribe la pestaña Datos propios desde una tabla pegada en el chat."""
        incoming = parse_table(table)
        if not incoming:
            return {"ok": False, "error": "No encontré una tabla: necesito encabezado y al menos una fila."}
        sheets = self.orch.sheets
        try:
            current = await sheets.read_tab(sheet_id, self.settings.own_data_tab)
            rows_in = incoming if mode == "reemplazar" else current + incoming
            columns = list(dict.fromkeys(c for r in rows_in for c in r))
            await sheets.write_tabs(sheet_id, {
                self.settings.own_data_tab: [columns] + [[r.get(c, "") for c in columns] for r in rows_in],
            })
        except Exception as e:  # noqa: BLE001
            return self.explain(sheet_id, e)
        out = {"ok": True, "filas": len(rows_in), "columnas": columns}
        if mode == "reemplazar" and current:
            out["filas_anteriores_borradas"] = len(current)  # el agente debe contarlo
        return out

    # ------------------------------------------------------------ etapas ----

    async def analyze_keywords(self, sheet_id: str, thread_id: str) -> dict[str, Any]:
        """Etapa A desde cero con lo que hay en la Sheet."""
        try:
            state = await self.orch.run_phase_1_and_2(sheet_id, thread_id, force=True)
        except Exception as e:  # noqa: BLE001
            return self.explain(sheet_id, e)
        return {"ok": True, "sheet_url": sheet_url(sheet_id), "pestanas": [TAB_KEYWORDS, TAB_DESCARTADOS],
                **self.keyword_report(state)}

    @staticmethod
    def keyword_report(state: RunState) -> dict[str, Any]:
        new = [k for k in state.keywords if k.get("ruta") == "create_new_url"]
        existing = [k for k in state.keywords if k.get("ruta") != "create_new_url"]
        # Sin `paginas_estimadas` (= filas de Datos propios): el modelo lo leía como páginas por keyword.
        # Cuántas URLs habrá lo dice el plan de URLs, no el análisis.
        avisos = [] if state.plataforma else ["El sitio no se exploró antes del análisis: no pude descartar keywords "
                                             "que ya tienen colección o página sin tráfico."]
        groups = [{"nombre": p["nombre"], "variables": p["variables"], "dato_que_cambia": p["dato_diferenciador"]}
                  for p in state.patrones if not p.get("descartado")]
        rejected_groups = [{"nombre": p["nombre"], "motivo": p.get("motivo")} for p in state.patrones if p.get("descartado")]
        return {
            "para_paginas_nuevas": [_keyword_view(k) for k in sorted(new, key=lambda k: -(k.get("score") or 0))][:MAX_ITEMS],
            "ya_tienen_pagina": [_keyword_view(k) for k in existing][:MAX_ITEMS],
            "descartadas": [{"keyword": d[3], "motivo": d[5]} for d in state.descartados if d[0] == "A"][:MAX_ITEMS],
            "grupos": groups, "grupos_descartados": rejected_groups,
            "totales": {"nuevas": len(new), "ya_tienen_pagina": len(existing),
                        "descartadas": sum(d[0] == "A" for d in state.descartados)},
            **({"avisos": avisos} if avisos else {}),
        }

    async def _write_decisions(self, sheet_id: str, tab: str, key_col: str, approve: set[str] | None,
                               reject: set[str], reason: str, keep=lambda r: True,
                               norm=dedupe_key) -> tuple[list[str], list[str]]:
        """Marca Aprobado en la pestaña; `approve=None` aprueba todo lo que no se rechaza."""
        rows = await self.orch.sheets.read_raw(sheet_id, tab)
        if not rows:
            raise InvalidRunState(f"La pestaña {tab} está vacía.")
        header = list(rows[0])
        for col in (APPROVED_COL, REJECT_REASON_COL):
            if col not in header:
                header.append(col)
        idx = {c: header.index(c) for c in header}
        approved, unknown = [], set(approve or set()) | set(reject)
        out = [header]
        for raw in rows[1:]:
            row = list(raw) + [""] * (len(header) - len(raw))
            record = dict(zip(header, row))
            value = norm(record.get(key_col))
            unknown.discard(value)
            if keep(record):
                if value in reject:
                    row[idx[APPROVED_COL]], row[idx[REJECT_REASON_COL]] = False, reason
                elif approve is None or value in approve:
                    row[idx[APPROVED_COL]], row[idx[REJECT_REASON_COL]] = True, ""
                    approved.append(str(record.get(key_col)))
                else:
                    row[idx[APPROVED_COL]], row[idx[REJECT_REASON_COL]] = "", ""
            out.append(row)
        await self.orch.sheets.write_tabs(sheet_id, {tab: out})
        return approved, sorted(unknown)

    async def plan_urls(self, sheet_id: str, thread_id: str, approve: list[str] | None,
                        reject: list[str], reason: str) -> dict[str, Any]:
        """Registra qué keywords van y corre la Etapa B (rehace B si ya existía)."""
        try:
            state = await self.orch.rewind(sheet_id, thread_id, RunStatus.WAITING_KEYWORD_APPROVAL)
            new = {k["dedupe_key"] for k in state.keywords if k.get("ruta") == "create_new_url"}
            if not new:
                return {"ok": False, "error": "No hay keywords para páginas nuevas en este análisis.",
                        "que_hacer": "Cambiar o agregar keywords (con variantes, ej. «{servicio} en {ciudad}»)."}
            approved, unknown = await self._write_decisions(
                sheet_id, TAB_KEYWORDS, "keyword",
                None if approve is None else {dedupe_key(k) for k in approve},
                {dedupe_key(k) for k in reject}, reason or "rechazada en el chat",
                keep=lambda r: r.get("ruta") == "create_new_url",
            )
            typed = {dedupe_key(k): k for k in [*(approve or []), *reject]}
            unknown = [typed.get(u, u) for u in unknown]
            if not approved:
                return {"ok": False, "error": "Ninguna de esas keywords es de las que sirven para páginas nuevas.",
                        "no_encontradas": unknown, "nuevas_disponibles": [k["keyword"] for k in state.keywords
                                                                          if k["dedupe_key"] in new][:MAX_ITEMS]}
            state = await self.orch.run_phase_3(sheet_id, thread_id)
        except Exception as e:  # noqa: BLE001
            return self.explain(sheet_id, e)
        report = self.plan_report(state)
        if unknown:
            report["no_encontradas"] = unknown
        return {"ok": True, "sheet_url": sheet_url(sheet_id), "pestanas": [TAB_PLAN], **report}

    @staticmethod
    def plan_report(state: RunState) -> dict[str, Any]:
        urls = [{"id": p["url_id"], "url": _path(p["url"]), "keyword": p["keyword_principal"],
                 "otras_keywords": p.get("keywords_secundarias") or [],
                 "con_datos_propios": p.get("datos_propios") == "ok",
                 **({"problema_url": [SLUG_ISSUES.get(i, i) for i in p.get("slug_issues") or []]}
                    if not p.get("slug_valid") else {})}
                for p in state.plan]
        return {"urls": urls[:MAX_ITEMS], "total_urls": len(urls),
                "sin_datos_propios": sum(not u["con_datos_propios"] for u in urls),
                "descartadas": [{"elemento": d[3], "motivo": d[5]} for d in state.descartados if d[0] == "B"][:MAX_ITEMS]}

    async def generate_content(self, sheet_id: str, thread_id: str, approve: list[str] | None,
                               reject: list[str], reason: str) -> dict[str, Any]:
        """Registra qué URLs van y corre la Etapa C (rehace C si ya existía)."""
        try:
            await self.orch.rewind(sheet_id, thread_id, RunStatus.WAITING_URL_APPROVAL)
            upper = lambda v: str(v or "").strip().upper()  # noqa: E731
            approved, unknown = await self._write_decisions(
                sheet_id, TAB_PLAN, "url_id", None if approve is None else {upper(u) for u in approve},
                {upper(u) for u in reject}, reason or "rechazada en el chat", norm=upper,
            )
            if not approved:
                return {"ok": False, "error": "No quedó ninguna URL aprobada.", "no_encontradas": unknown}
            state = await self.orch.run_phase_4_to_7(sheet_id, thread_id)
        except Exception as e:  # noqa: BLE001
            return self.explain(sheet_id, e)
        summary = state.content_summary or {}
        return {"ok": True, "sheet_url": sheet_url(sheet_id), "generadas": summary.get("paginas_generadas"),
                "aprobadas_por_el_juez": summary.get("aprobadas"), "nota_promedio_juez": summary.get("score_promedio_juez"),
                "para_revision_humana": summary.get("revision_humana"), "errores": summary.get("errores"),
                "descartadas": [{"elemento": d[3], "motivo": d[5]} for d in state.descartados if d[0] == "C"][:MAX_ITEMS]}

    async def page_detail(self, sheet_id: str, url_id: str) -> dict[str, Any]:
        """Title, meta, H1 e intro de una URL generada, para revisarla en el chat."""
        url_id = url_id.strip().upper()
        try:
            content = await self.orch.sheets.read_tab(sheet_id, "Contenido")
            onpage = await self.orch.sheets.read_tab(sheet_id, "On-page")
        except Exception as e:  # noqa: BLE001
            return self.explain(sheet_id, e)
        c = next((r for r in content if str(r.get("url_id")).upper() == url_id), None)
        if c is None:
            return {"ok": False, "error": f"No hay contenido generado para {url_id}.",
                    "disponibles": [r.get("url_id") for r in content][:MAX_ITEMS]}
        o = next((r for r in onpage if str(r.get("url_id")).upper() == url_id), {})
        intro = str(c.get("intro") or "")
        return {"ok": True, "url": c.get("url"), "estado": c.get("estado"), "nota_juez": c.get("score_juez"),
                "comentario_juez": c.get("motivo_juez"), "title": o.get("title"),
                "meta_description": o.get("meta_description"), "h1": c.get("h1"),
                "intro": intro[:600] + ("…" if len(intro) > 600 else "")}

    async def keywords_status(self, sheet_id: str, thread_id: str) -> dict[str, Any]:
        """Último análisis de keywords guardado (sin volver a correrlo)."""
        state = await self._state(sheet_id, thread_id)
        if state is None or not state.keywords and state.status == RunStatus.NEW:
            return {"ok": False, "error": "Todavía no hay un análisis de keywords en esta conversación."}
        return {"ok": True, "paso_actual": self._step(state), **self.keyword_report(state),
                **({"plan": self.plan_report(state)} if state.plan else {})}
