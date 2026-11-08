"""Motor determinista de SEO programático (sin LLM, 100% reproducible).

Pasos de la planilla de diseño:
- P1-03 `normalize_and_deduplicate_keywords`
- P1-05 `mark_gsc_overlap`
- P1-08/P1-09 `score_and_route_keywords`
- P1-12 `generate_and_validate_slugs`
- P1-16 `validate_title_and_meta` (+ `detect_voseo`, `detect_unsupported_claims`)
- P1-18 `mask_row_values` + `compute_cosine_similarity_matrix`
- P1-21 `validate_json_ld_schema`
- P1-23/24 `validate_internal_links`

Formato de keyword que circula entre pasos (claves opcionales salvo `keyword`):
    keyword, source, search_volume, difficulty (0-100), intent, has_own_data,
    business_value (0-`business_value_max`)
"""

from __future__ import annotations

import math
import re
import string
import unicodedata
from typing import Any
from urllib.parse import unquote, urlparse

from google.adk.tools import FunctionTool

# --------------------------------------------------------- normalización ----

_VOWELS = "aeiou"


def _strip_accents(text: str, keep_enie: bool) -> str:
    if keep_enie:
        text = text.replace("ñ", "\0")
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return text.replace("\0", "ñ") if keep_enie else text


def normalize_text(text: Any) -> str:
    """Minúsculas, sin tildes (conserva ñ), sin puntuación y con espacios simples."""
    text = _strip_accents(str(text).lower(), keep_enie=True)
    text = re.sub(r"[^a-z0-9ñ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _singularize(word: str) -> str:
    """Plurales obvios del español. Solo se usa como clave de deduplicación."""
    if len(word) <= 3 or not word.endswith("s") or word.isdigit():
        return word
    if word.endswith("ces") and word[-4] in _VOWELS:  # luces -> luz, veces -> vez
        return word[:-3] + "z"
    if word.endswith("es") and len(word) > 4 and word[-3] in "lrndjy":  # hoteles -> hotel
        return word[:-2]
    if word[-2] in _VOWELS:  # seguros -> seguro, casas -> casa
        return word[:-1]
    return word


def dedupe_key(text: Any) -> str:
    return " ".join(_singularize(w) for w in normalize_text(text).split())


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(" ", "").replace(",", ".")
    try:
        return float(text) if text else None
    except ValueError:
        return None


def _volume(kw: dict) -> float | None:
    for key in ("search_volume", "volume", "volumen"):
        if (v := _number(kw.get(key))) is not None:
            return v
    return None


# ----------------------------------------------------------------- P1-03 ----

def normalize_and_deduplicate_keywords(raw_keywords: list[dict]) -> dict:
    """P1-03: normaliza keywords y elimina duplicados (incluye singular/plural obvio).

    Cada entrada necesita `keyword` y opcionalmente `source` y métricas. Entre
    duplicados se conserva el de mayor `search_volume` (empate: el primero), y
    todas las variantes quedan en `variants` con su `original_keyword` y `source`.

    Args:
        raw_keywords: Lista de dicts con al menos la clave `keyword`.

    Returns:
        {"keywords": [...], "before_count": int, "after_count": int, "empty_removed": int}
    """
    groups: dict[str, list[dict]] = {}
    empty = 0
    for raw in raw_keywords:
        original = str(raw.get("keyword", "")).strip()
        normalized = normalize_text(original)
        if not normalized:
            empty += 1
            continue
        entry = {**raw, "original_keyword": original, "keyword": normalized}
        groups.setdefault(dedupe_key(normalized), []).append(entry)

    keywords = []
    for key, entries in groups.items():
        best = max(enumerate(entries), key=lambda ie: (_volume(ie[1]) or -1.0, -ie[0]))[1]
        keywords.append({
            **best,
            "dedupe_key": key,
            "sources": sorted({str(e.get("source")) for e in entries if e.get("source")}),
            "variants": [
                {"original_keyword": e["original_keyword"], "keyword": e["keyword"], "source": e.get("source")}
                for e in entries
            ],
        })

    return {
        "keywords": keywords,
        "before_count": len(raw_keywords),
        "after_count": len(keywords),
        "empty_removed": empty,
    }


# ----------------------------------------------------------------- P1-05 ----

COVERED_MAX_POSITION = 10.0
PARTIAL_MAX_POSITION = 30.0


def _gsc_row(row: dict) -> tuple[str, str | None, float, float, float] | None:
    """Acepta filas crudas de la API (`keys: [query, page]`) o aplanadas (`query`, `page`)."""
    if "keys" in row:
        keys = row["keys"] or []
        query, page = (keys[0] if keys else None), (keys[1] if len(keys) > 1 else None)
    else:
        query, page = row.get("query"), row.get("page")
    if not query:
        return None
    return (
        query,
        page,
        _number(row.get("clicks")) or 0.0,
        _number(row.get("impressions")) or 0.0,
        _number(row.get("position")) or 0.0,
    )


def mark_gsc_overlap(keywords: list[dict], gsc_rows: list[dict]) -> list[dict]:
    """P1-05: cruza keywords con Search Console y asigna `coverage_status`.

    Se agrega por query (normalizada igual que las keywords): clics e impresiones
    sumados, posición media ponderada por impresiones y `existing_url` = la página
    con más clics (empate: más impresiones).

    - "ya_cubierta": impresiones > 0 y posición media <= 10
    - "parcial": impresiones > 0 y posición media entre 10.01 y 30
    - "libre": sin datos en GSC o posición > 30

    Args:
        keywords: Salida de `normalize_and_deduplicate_keywords`.
        gsc_rows: Filas de `get_search_analytics` con dimensiones `query` y `page`.
    """
    agg: dict[str, dict[str, Any]] = {}
    for row in gsc_rows:
        parsed = _gsc_row(row)
        if not parsed:
            continue
        query, page, clicks, impressions, position = parsed
        bucket = agg.setdefault(dedupe_key(query), {"clicks": 0.0, "impressions": 0.0, "pos_x_impr": 0.0, "pages": {}})
        bucket["clicks"] += clicks
        bucket["impressions"] += impressions
        bucket["pos_x_impr"] += position * impressions
        if page:
            p_clicks, p_impr = bucket["pages"].get(page, (0.0, 0.0))
            bucket["pages"][page] = (p_clicks + clicks, p_impr + impressions)

    result = []
    for kw in keywords:
        out = dict(kw)
        match = agg.get(kw.get("dedupe_key") or dedupe_key(kw.get("keyword", "")))
        if not match or match["impressions"] <= 0:
            out.update(coverage_status="libre", existing_url=None,
                       gsc_clicks=0, gsc_impressions=0, gsc_position=None)
        else:
            position = round(match["pos_x_impr"] / match["impressions"], 2)
            pages = match["pages"]
            best_page = min(pages, key=lambda p: (-pages[p][0], -pages[p][1], p)) if pages else None
            if position <= COVERED_MAX_POSITION:
                status = "ya_cubierta"
            elif position <= PARTIAL_MAX_POSITION:
                status = "parcial"
            else:
                status = "libre"
            out.update(coverage_status=status, existing_url=best_page,
                       gsc_clicks=int(match["clicks"]), gsc_impressions=int(match["impressions"]),
                       gsc_position=position)
        result.append(out)
    return result


# ----------------------------------------------------------- P1-08 / P1-09 ----

DEFAULT_WEIGHTS = {
    "volumen": 0.30,
    "dificultad": 0.20,
    "intencion": 0.25,
    "datos_propios": 0.10,
    "valor_negocio": 0.15,
}
_WEIGHT_ALIASES = {
    "volumen": {"volumen", "volumen_busqueda", "volume"},
    "dificultad": {"dificultad", "dificultad_invertida", "difficulty"},
    "intencion": {"intencion", "intent"},
    "datos_propios": {"datos_propios", "own_data"},
    "valor_negocio": {"valor_negocio", "valor_de_negocio", "business_value"},
}
INTENT_VALUES = {
    "transaccional": 1.0, "transactional": 1.0,
    "comercial": 0.8, "commercial": 0.8,
    "informacional": 0.5, "informativa": 0.5, "informational": 0.5,
    "navegacional": 0.2, "navigational": 0.2,
}


def _resolve_weights(raw: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    if not raw:
        return dict(DEFAULT_WEIGHTS), []
    weights = {factor: 0.0 for factor in DEFAULT_WEIGHTS}
    ignored = []
    for key, value in raw.items():
        factor = next((f for f, aliases in _WEIGHT_ALIASES.items() if normalize_text(key).replace(" ", "_") in aliases), None)
        number = _number(value)
        if factor is None or number is None:
            ignored.append(key)
            continue
        weights[factor] = number
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Los pesos de scoring suman 0: revisa la pestaña Parámetros.")
    return {f: w / total for f, w in weights.items()}, ignored


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _factors(kw: dict, max_log_volume: float, business_value_max: float) -> tuple[dict[str, float], list[str]]:
    missing = []

    volume = _volume(kw)
    if volume is None:
        missing.append("search_volume")
    f_volume = math.log1p(volume) / max_log_volume if volume and max_log_volume > 0 else 0.0

    difficulty = _number(kw.get("difficulty"))
    if difficulty is None:
        missing.append("difficulty")
    f_difficulty = 1 - _clamp(difficulty / 100) if difficulty is not None else 0.0

    intent = kw.get("intent")
    f_intent = _number(intent)
    if f_intent is None:
        f_intent = INTENT_VALUES.get(normalize_text(intent or ""))
    if f_intent is None:
        missing.append("intent")
        f_intent = 0.0

    own = kw.get("has_own_data")
    if own is None:
        missing.append("has_own_data")
    f_own = 1.0 if own is True else _clamp(_number(own) or 0.0)

    business = _number(kw.get("business_value"))
    if business is None:
        missing.append("business_value")
    f_business = _clamp(business / business_value_max) if business is not None else 0.0

    factors = {
        "volumen": f_volume,
        "dificultad": f_difficulty,
        "intencion": _clamp(f_intent),
        "datos_propios": f_own,
        "valor_negocio": f_business,
    }
    return factors, missing


def score_and_route_keywords(keywords: list[dict], params: dict) -> dict:
    """P1-08/P1-09: score ponderado (0-100) y switch crear vs. optimizar.

    Factores normalizados a 0-1: volumen (log, relativo al máximo del lote),
    dificultad invertida (1 - difficulty/100), intención (transaccional 1,
    comercial 0.8, informacional 0.5, navegacional 0.2), datos propios
    (has_own_data) y valor de negocio (business_value / business_value_max, por
    defecto 5). Un dato faltante aporta 0 y queda listado en `missing_fields`.

    Ruteo:
    - create_new_url: estado "libre" con score >= min_keyword_score y volumen >= min_search_volume.
    - optimize_existing: estado "parcial" o "ya_cubierta" (flujo 2).
    - below_threshold: "libre" que no alcanza los umbrales (con motivo; no se pierden).

    Args:
        keywords: Salida de `mark_gsc_overlap` (requiere `coverage_status`).
        params: `parameters` de `read_parameters_and_brief` (scoring_weights,
            min_keyword_score, min_search_volume; opcional business_value_max).
    """
    params = params.get("parameters", params)
    weights, ignored = _resolve_weights(params.get("scoring_weights") or {})
    min_score = float(params.get("min_keyword_score") or 0.0)
    min_volume = float(params.get("min_search_volume") or 0.0)
    business_value_max = float(params.get("business_value_max") or 5.0)

    missing_status = [kw.get("keyword") for kw in keywords if kw.get("coverage_status") is None]
    if missing_status:
        raise ValueError(f"Keywords sin coverage_status (corre mark_gsc_overlap antes): {missing_status[:5]}")

    max_log_volume = max((math.log1p(v) for kw in keywords if (v := _volume(kw))), default=0.0)

    scored = []
    for kw in keywords:
        factors, missing = _factors(kw, max_log_volume, business_value_max)
        score = round(100 * sum(weights[f] * factors[f] for f in weights), 2)
        scored.append({
            **kw,
            "score": score,
            "score_breakdown": {f: round(v, 4) for f, v in factors.items()},
            "missing_fields": missing,
        })
    scored.sort(key=lambda k: (-k["score"], k.get("keyword", "")))

    create, optimize, below = [], [], []
    for kw in scored:
        if kw["coverage_status"] in ("parcial", "ya_cubierta"):
            optimize.append({**kw, "route": "optimize_existing"})
            continue
        reasons = []
        if kw["score"] < min_score:
            reasons.append(f"score {kw['score']} < {min_score}")
        if (_volume(kw) or 0.0) < min_volume:
            reasons.append(f"volumen {_volume(kw) or 0} < {min_volume}")
        if reasons:
            below.append({**kw, "route": "below_threshold", "route_reason": "; ".join(reasons)})
        else:
            create.append({**kw, "route": "create_new_url"})

    return {
        "create_new_url": create,
        "optimize_existing": optimize,
        "below_threshold": below,
        "weights_used": {f: round(w, 4) for f, w in weights.items()},
        "ignored_weights": ignored,
        "summary": {
            "total": len(scored),
            "create_new_url": len(create),
            "optimize_existing": len(optimize),
            "below_threshold": len(below),
        },
    }


# ----------------------------------------------------------------- P1-12 ----

def slugify(text: Any) -> str:
    """ASCII en minúsculas separado por guiones (ñ -> n)."""
    text = _strip_accents(str(text).lower(), keep_enie=False)
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def _host(url: str) -> str | None:
    host = urlparse(url).hostname if "://" in url else None
    return host[4:] if host and host.startswith("www.") else host


def _normalize_path(url: str) -> str:
    path = urlparse(url.strip()).path if "://" in url else url.strip().split("?")[0].split("#")[0]
    path = "/" + unquote(path).lower().strip("/")
    return path.rstrip("/") or "/"


def _truncate_leaf(leaf: str, budget: int) -> str:
    if budget <= 0:
        return ""
    if len(leaf) <= budget:
        return leaf
    cut = leaf[:budget]
    if leaf[budget] != "-" and "-" in cut:
        cut = cut[: cut.rfind("-")]
    return cut.strip("-")


def _pattern_fields(pattern: str) -> list[str]:
    return [name for _, name, _, _ in string.Formatter().parse(pattern) if name]


def nodes_from_pr02(plan: dict) -> list[dict]:
    """Convierte la salida de PR-02 (`hubs` + `paginas`) a nodos hub/pattern/variables."""
    patterns = {_normalize_path(h["url"]): h.get("patron") for h in plan.get("hubs", []) if h.get("url")}
    nodes = []
    for page in plan.get("paginas", []):
        hub = page.get("hub", "")
        nodes.append({
            **page,
            "hub": hub,
            "pattern": patterns.get(_normalize_path(hub)) if hub else None,
            "variables": page.get("variables") or {},
            "keyword": page.get("keyword_principal", ""),
        })
    return nodes


def generate_and_validate_slugs(
    url_tree: list[dict] | dict, existing_site_urls: list[str], max_length: int = 75,
    allowed_hubs: list[str] | None = None,
) -> list[dict]:
    """P1-12: genera la URL de cada página por fórmula y valida duplicados y colisiones.

    URL = "/" + hub (cada segmento con slugify) + "/" + slugify(pattern con variables).
    Siempre subcarpeta: un hub en otro host (subdominio) se marca como inválido.
    `max_length` aplica al path completo; se recorta el último segmento en un guion.

    Args:
        url_tree: Uno de dos formatos:
            - Lista de nodos {"hub": "Seguros de Auto" | "/seguros/auto",
              "pattern": "seguro de {vehiculo} en {ciudad}", "variables": {...}}.
              Sin `pattern` se usa `keyword`.
            - Salida de PR-02: {"hubs": [{"url", "patron"}], "paginas": [{"url_propuesta",
              "keyword_principal", "keywords_secundarias", "hub", "variables"}]}. El
              patrón se toma del hub de cada página.
        existing_site_urls: URLs o paths ya publicados en el sitio.
        max_length: Largo máximo del path (incluye "/").
        allowed_hubs: Hubs que admite la plataforma (en Shopify, "/collections" y
            "/pages"). Vacío o None: hubs libres.

    Returns:
        Cada página con `slug`, `url_path`, `slug_issues` y `slug_valid`. En formato
        PR-02 agrega `coincide_con_propuesta` (compara con `url_propuesta`).
        Issues: hub_invalido, variable_faltante, slug_vacio, excede_largo,
        duplicado_en_plan, colision_url_existente, hub_no_permitido.
    """
    nodes = nodes_from_pr02(url_tree) if isinstance(url_tree, dict) else url_tree
    site_hosts = {h for u in existing_site_urls if u and (h := _host(u.strip()))}
    existing = {_normalize_path(u) for u in existing_site_urls if u and u.strip()}
    allowed = {"/" + "/".join(slugify(seg) for seg in h.strip("/").split("/")) for h in allowed_hubs or []}

    results = []
    for node in nodes:
        issues = []
        hub_raw = str(node.get("hub", "")).strip()
        if "://" in hub_raw:
            # URL completa: válida solo si es el mismo host del sitio
            if site_hosts and _host(hub_raw) not in site_hosts:
                issues.append("hub_invalido")
            hub_raw = urlparse(hub_raw).path
        elif "." in hub_raw.strip("/").split("/")[0]:
            issues.append("hub_invalido")  # "autos.acme.cl": subdominio
        hub_segments = [s for s in (slugify(seg) for seg in hub_raw.strip("/").split("/")) if s]
        if not hub_segments:
            issues.append("hub_invalido")
        elif allowed and "/" + "/".join(hub_segments) not in allowed:
            issues.append("hub_no_permitido")

        pattern = node.get("pattern")
        variables = node.get("variables") or {}
        if pattern:
            missing = [f for f in _pattern_fields(pattern) if f not in variables]
            if missing:
                issues.append("variable_faltante")
            leaf_text = pattern.format_map({**{f: "" for f in missing}, **variables})
        else:
            leaf_text = node.get("keyword", "")

        leaf = slugify(leaf_text)
        prefix = "/" + "/".join(hub_segments) + "/"
        leaf = _truncate_leaf(leaf, max_length - len(prefix))
        if not leaf:
            issues.append("excede_largo" if len(prefix) >= max_length else "slug_vacio")
        url_path = prefix + leaf
        result = {**node, "slug": leaf, "url_path": url_path, "slug_issues": issues}
        if node.get("url_propuesta"):
            result["coincide_con_propuesta"] = _normalize_path(node["url_propuesta"]) == _normalize_path(url_path)
        results.append(result)

    counts: dict[str, int] = {}
    for r in results:
        counts[r["url_path"]] = counts.get(r["url_path"], 0) + 1
    for r in results:
        if counts[r["url_path"]] > 1:
            r["slug_issues"].append("duplicado_en_plan")
        if _normalize_path(r["url_path"]) in existing:
            r["slug_issues"].append("colision_url_existente")
        r["slug_valid"] = not r["slug_issues"]
    return results


# ------------------------------------------------------- voseo / promesas ----

# Formas de voseo frecuentes en copy comercial (con tilde, que las distingue del tuteo).
VOSEO_FORMS = frozenset("""
vos sos tenés podés querés sabés sentís venís decís hacés ponés salís elegís vivís pedís
necesitás buscás pagás contratás cotizás ahorrás encontrás comparás usás manejás viajás
cotizá contratá elegí descubrí conocé comprá pedí asegurá protegé encontrá aprovechá consultá
llamá escribí ingresá completá solicitá accedé mirá hacé tené vení decí poné salí andá viajá
ahorrá compará disfrutá empezá simulá calculá revisá visitá dejá probá buscá renová sumá
registrá pagá olvidá animá contactá descargá reservá agendá informá
""".split())

UNSUPPORTED_CLAIMS = (
    "el mejor", "la mejor", "los mejores", "las mejores", "mejor precio",
    "garantizado", "garantizada", "garantizados", "garantizamos", "100%",
    "lider", "lideres", "numero 1", "n°1", "#1", "el mas barato", "la mas barata",
    "mas barato", "mas barata", "precio mas bajo", "insuperable", "imbatible",
)


def detect_voseo(text: str) -> list[str]:
    """Formas de voseo presentes en `text` (minúsculas, sin repetir, en orden)."""
    found: list[str] = []
    for word in re.findall(r"[a-záéíóúñü]+", str(text).lower()):
        if word in VOSEO_FORMS and word not in found:
            found.append(word)
    return found


def _claims_text(text: str) -> str:
    return re.sub(r"\s+", " ", _strip_accents(str(text).lower(), keep_enie=True))


def detect_unsupported_claims(text: str, source_text: str = "") -> list[str]:
    """Superlativos y promesas comerciales del texto que no aparecen literalmente en la fuente."""
    target, source = _claims_text(text), _claims_text(source_text)
    found = []
    for claim in UNSUPPORTED_CLAIMS:
        pattern = rf"(?<![a-z0-9ñ]){re.escape(claim)}(?![a-z0-9ñ])"
        if re.search(pattern, target) and not re.search(pattern, source):
            found.append(claim)
    return found


def unsupported_numbers(text: str, source_text: str) -> list[str]:
    """Cifras del texto cuyos dígitos no aparecen como cifra en la fuente."""
    _, numbers = _source_index(source_text)
    found = []
    for raw in re.findall(r"\d[\d.,]*\d|\d", str(text)):
        if re.sub(r"\D", "", raw) not in numbers and raw not in found:
            found.append(raw)
    return found


# ----------------------------------------------------------------- P1-16 ----

TITLE_MAX_LENGTH = 60
META_MAX_LENGTH = 155


def validate_title_and_meta(
    title: str,
    meta_description: str,
    keyword: str,
    source_text: str | None = None,
    max_title_length: int = TITLE_MAX_LENGTH,
    max_meta_length: int = META_MAX_LENGTH,
) -> dict:
    """P1-16: valida un Title y una Meta Description.

    Reglas: title de 1 a `max_title_length` caracteres que empieza con la keyword
    principal (comparación normalizada, tolera singular/plural); meta de 1 a
    `max_meta_length` caracteres; sin voseo. Con `source_text` (datos propios de la
    fila) además: toda cifra debe estar en la fila y no se aceptan superlativos o
    promesas ("el mejor", "garantizado", "100%", "líder", "el más barato"...) que
    no vengan literalmente en ella.

    Returns:
        {"valid", "issues", "title_length", "meta_length"}. Issues: title_vacio,
        title_largo, keyword_no_al_inicio, meta_vacia, meta_larga, voseo:<forma>,
        cifra_no_respaldada:<cifra>, promesa_no_respaldada:<frase>.
    """
    title = (title or "").strip()
    meta = (meta_description or "").strip()
    issues = []
    if not title:
        issues.append("title_vacio")
    elif len(title) > max_title_length:
        issues.append("title_largo")
    if title and not dedupe_key(title).startswith(dedupe_key(keyword)):
        issues.append("keyword_no_al_inicio")
    if not meta:
        issues.append("meta_vacia")
    elif len(meta) > max_meta_length:
        issues.append("meta_larga")

    combined = f"{title}\n{meta}"
    issues += [f"voseo:{w}" for w in detect_voseo(combined)]
    if source_text is not None:
        issues += [f"cifra_no_respaldada:{n}" for n in unsupported_numbers(combined, source_text)]
        issues += [f"promesa_no_respaldada:{c}" for c in detect_unsupported_claims(combined, source_text)]
    return {"valid": not issues, "issues": issues, "title_length": len(title), "meta_length": len(meta)}


# ----------------------------------------------------------------- P1-18 ----

_ACCENT_CLASSES = {"a": "[aáàäâ]", "e": "[eéèëê]", "i": "[iíìïî]", "o": "[oóòöô]", "u": "[uúùüû]", "n": "[nñ]"}
_NUMBER_TOKEN = re.compile(r"\d[\d.,]*\d|\d")


def _flexible_phrase(value: str) -> str:
    """Regex que encuentra `value` sin importar tildes, ñ, mayúsculas ni espacios múltiples."""
    base = _strip_accents(value.lower(), keep_enie=False)
    parts = []
    for ch in base:
        if ch.isspace():
            if not parts or parts[-1] != r"\s+":
                parts.append(r"\s+")
        else:
            parts.append(_ACCENT_CLASSES.get(ch, re.escape(ch)))
    return "".join(parts)


def _flexible_number(digits: str) -> str:
    """Regex de una cifra con separadores opcionales: "13990" ~ "13.990" ~ "13,990"."""
    return r"(?<![\d.,])(?:\$\s?)?" + r"[.,]?".join(digits) + r"(?![\d]|[.,]\d)"


def mask_row_values(text: str, fila: dict) -> str:
    """P1-18: enmascara los datos propios de la fila antes de medir similitud.

    Cada valor de la fila se reemplaza por `[VAR_<COLUMNA>]` (sin importar tildes,
    mayúsculas ni separadores de miles) y las cifras restantes por `[NUM]`. Así dos
    páginas que solo cambian los datos (una copia estructural) quedan casi
    idénticas, y la similitud mide la redacción y no el tema.
    """
    masked = str(text)
    replacements: list[tuple[int, str, str]] = []  # (largo del valor, patrón, marcador)
    for key, value in fila.items():
        if value is None or isinstance(value, bool):
            continue
        placeholder = f"[VAR_{_normalize_key_upper(key)}]"
        raw = f"{value:g}" if isinstance(value, float) else str(value)
        if re.search(r"[A-Za-zÀ-ÿ]", raw) and len(raw.strip()) >= 3:
            replacements.append((len(raw.strip()), _flexible_phrase(raw.strip()), placeholder))
        for number in _NUMBER_TOKEN.findall(raw):
            digits = re.sub(r"\D", "", number)
            if digits:
                replacements.append((len(number), _flexible_number(digits), placeholder))
    # Primero los patrones más largos ("Viña del Mar" antes que "Viña")
    for _, pattern, placeholder in sorted(replacements, key=lambda r: -r[0]):
        masked = re.sub(rf"(?<![\wÀ-ÿ]){pattern}(?![\wÀ-ÿ])", placeholder, masked, flags=re.IGNORECASE)
    masked = re.sub(r"(?<![\w\[])(?:\$\s?)?(?:" + _NUMBER_TOKEN.pattern + r")(?![\w\]])", "[NUM]", masked)
    return masked


def _normalize_key_upper(key: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", _strip_accents(str(key).upper(), keep_enie=False)).strip("_")


def _shingles(text: str, n: int) -> set[tuple[str, ...]]:
    """n-gramas de palabras; los marcadores [VAR_X] y [NUM] cuentan como una palabra."""
    words = re.findall(r"\[var_[a-z0-9_]+\]|\[num\]|[a-z0-9ñ]+",
                       _strip_accents(str(text).lower(), keep_enie=True))
    return {tuple(words[i:i + n]) for i in range(max(len(words) - n + 1, 0))}


def compute_shingle_similarity_matrix(texts: list[str], n: int = 3) -> list[list[float]]:
    """P1-18 (léxico): Jaccard de n-gramas de palabras entre textos (diagonal 1.0).

    Sobre texto enmascarado, una copia que solo cambia datos comparte casi todos
    sus n-gramas (~1.0), mientras dos textos redactados por separado comparten
    pocos aunque sigan la misma estructura.
    """
    sets = [_shingles(t, n) for t in texts]
    size = len(texts)
    matrix = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i, size):
            union = sets[i] | sets[j]
            sim = len(sets[i] & sets[j]) / len(union) if union else 0.0
            matrix[i][j] = matrix[j][i] = round(sim, 6)
    return matrix


def compute_cosine_similarity_matrix(vectors: list[list[float]]) -> list[list[float]]:
    """P1-18: matriz de similitud coseno (simétrica, diagonal 1.0, redondeo a 6 decimales).

    Un vector nulo tiene similitud 0 con todos (incluido él mismo).
    """
    norms = [math.sqrt(sum(x * x for x in v)) for v in vectors]
    n = len(vectors)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            if norms[i] == 0 or norms[j] == 0:
                sim = 0.0
            else:
                sim = sum(a * b for a, b in zip(vectors[i], vectors[j])) / (norms[i] * norms[j])
            matrix[i][j] = matrix[j][i] = round(sim, 6)
    return matrix


# ----------------------------------------------------------------- P1-21 ----

JSONLD_REQUIRED_FIELDS = {
    "Product": ("name",),
    "LocalBusiness": ("name", "address"),
    "Service": ("name", "provider"),
    "Article": ("headline",),
    "BreadcrumbList": ("itemListElement",),
}
# Identificadores, URLs y vocabulario controlado de schema.org (códigos, no hechos)
_JSONLD_SKIP_SOURCE_CHECK = {
    "@context", "@type", "@id", "url", "item", "image", "position", "sameAs",
    "priceCurrency", "priceType", "unitText", "unitCode", "availability", "itemCondition",
    "inLanguage", "addressCountry",
}
_NUMERIC_VALUE = re.compile(r"[\d\s.,$%+-]+")


def _source_index(source_text: str) -> tuple[set[str], set[str]]:
    tokens = set(normalize_text(source_text).split())
    numbers = {re.sub(r"\D", "", n) for n in re.findall(r"\d[\d.,]*\d|\d", source_text)}
    return tokens, numbers


def _value_supported(value: Any, tokens: set[str], numbers: set[str]) -> bool:
    """Número: sus dígitos aparecen como cifra en la fuente ("12990" ~ "$12.990").
    Texto: cada palabra de 3+ letras aparece en la fuente."""
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        value = f"{value:g}"
    text = str(value)
    if _NUMERIC_VALUE.fullmatch(text):
        digits = re.sub(r"\D", "", text)
        return not digits or digits in numbers
    return all(t in tokens for t in normalize_text(text).split() if len(t) > 2)


def _scalar_values(node: Any, key: str | None = None):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _scalar_values(v, k)
    elif isinstance(node, list):
        for v in node:
            yield from _scalar_values(v, key)
    elif node is not None and key not in _JSONLD_SKIP_SOURCE_CHECK:
        yield key, node


def validate_json_ld_schema(jsonld: dict, tipo_schema: str, source_text: str | None = None) -> dict:
    """P1-21: valida un bloque JSON-LD de schema.org.

    Reglas: `@context` de schema.org, `@type` igual a `tipo_schema` y dentro de los
    tipos permitidos, campos obligatorios presentes y no nulos, y en
    BreadcrumbList cada elemento con `position`, `name` e `item`. Si se entrega
    `source_text` (fila + contenido visible), cada valor debe estar respaldado:
    los números por sus dígitos ("12990" ~ "$12.990") y los textos con todas sus
    palabras presentes. Se omiten identificadores, URLs y vocabulario controlado
    (priceCurrency, unitText, availability...).

    Returns:
        {"valid", "issues", "campos_faltantes"}. Issues: tipo_no_permitido,
        context_invalido, tipo_distinto, campo_obligatorio_faltante:<campo>,
        breadcrumb_item_invalido:<i>, valor_no_respaldado:<campo>.
    """
    issues: list[str] = []
    missing: list[str] = []
    if tipo_schema not in JSONLD_REQUIRED_FIELDS:
        issues.append("tipo_no_permitido")
    if "schema.org" not in str(jsonld.get("@context", "")):
        issues.append("context_invalido")
    if jsonld.get("@type") != tipo_schema:
        issues.append("tipo_distinto")

    for field in JSONLD_REQUIRED_FIELDS.get(tipo_schema, ()):
        value = jsonld.get(field)
        if value in (None, "", [], {}):
            missing.append(field)
            issues.append(f"campo_obligatorio_faltante:{field}")

    if tipo_schema == "BreadcrumbList":
        for i, item in enumerate(jsonld.get("itemListElement") or []):
            if not isinstance(item, dict) or any(item.get(k) in (None, "") for k in ("position", "name", "item")):
                issues.append(f"breadcrumb_item_invalido:{i}")

    if source_text is not None:
        tokens, numbers = _source_index(source_text)
        for key, value in _scalar_values(jsonld):
            if not _value_supported(value, tokens, numbers):
                issues.append(f"valor_no_respaldado:{key}")

    return {"valid": not issues, "issues": issues, "campos_faltantes": missing}


# ------------------------------------------------------------- P1-23 / 24 ----

ANCHOR_MIN_WORDS = 2
ANCHOR_MAX_WORDS = 6
MAX_SIBLING_LINKS = 4


def validate_internal_links(
    links: list[dict],
    content_text: str,
    source_url: str,
    hub_url: str,
    sibling_urls: list[str],
    existing_urls: list[str],
    anchors_usados: list[str],
) -> dict:
    """P1-23/24: valida enlaces internos propuestos por PR-06.

    Cada enlace se acepta solo si: `parrafo_literal` aparece tal cual en el
    contenido, el `anchor` tiene 2 a 6 palabras y está dentro del párrafo, el
    anchor no se usó antes (ni en `anchors_usados` ni en esta propuesta), el
    destino es el hub, una hermana o una URL existente, no es la propia página y
    no se repite. Se acepta 1 enlace al hub y hasta 4 a hermanas (en orden).

    Returns:
        {"valid_links", "rejected": [{"link", "issues"}], "hub_links",
        "sibling_links", "cumple_minimos"} donde `cumple_minimos` exige 1 enlace al
        hub y al menos 2 a hermanas (si existen al menos 2 hermanas).
    """
    hub = _normalize_path(hub_url) if hub_url else None
    siblings = {_normalize_path(u) for u in sibling_urls} - {_normalize_path(source_url)}
    existing = {_normalize_path(u) for u in existing_urls}
    used = {normalize_text(a) for a in anchors_usados}
    source = _normalize_path(source_url)

    valid, rejected = [], []
    seen_dest: set[str] = set()
    hub_count = sibling_count = 0
    for link in links:
        issues = []
        dest = _normalize_path(str(link.get("destino", "")))
        paragraph = str(link.get("parrafo_literal", "")).strip()
        anchor = str(link.get("anchor", "")).strip()
        anchor_norm = normalize_text(anchor)

        if not paragraph or paragraph not in content_text:
            issues.append("parrafo_no_literal")
        if not ANCHOR_MIN_WORDS <= len(anchor_norm.split()) <= ANCHOR_MAX_WORDS:
            issues.append("anchor_largo_invalido")
        if anchor_norm and anchor_norm not in normalize_text(paragraph):
            issues.append("anchor_fuera_del_parrafo")
        if anchor_norm in used:
            issues.append("anchor_repetido")
        if dest == source:
            issues.append("autoenlace")
        elif dest in seen_dest:
            issues.append("destino_repetido")

        kind = "hub" if dest == hub else "hermana" if dest in siblings else "existente" if dest in existing else None
        if kind is None and "autoenlace" not in issues:
            issues.append("destino_no_permitido")
        if kind == "hub" and hub_count >= 1:
            issues.append("excede_enlaces_hub")
        if kind == "hermana" and sibling_count >= MAX_SIBLING_LINKS:
            issues.append("excede_enlaces_hermanas")

        if issues:
            rejected.append({"link": link, "issues": issues})
            continue
        valid.append({**link, "tipo_destino": kind})
        used.add(anchor_norm)
        seen_dest.add(dest)
        hub_count += kind == "hub"
        sibling_count += kind == "hermana"

    min_siblings = min(2, len(siblings))
    return {
        "valid_links": valid,
        "rejected": rejected,
        "hub_links": hub_count,
        "sibling_links": sibling_count,
        "cumple_minimos": (hub_count == 1 or not hub) and sibling_count >= min_siblings,
    }


SEO_DETERMINISTIC_TOOLS = [
    FunctionTool(normalize_and_deduplicate_keywords),
    FunctionTool(mark_gsc_overlap),
    FunctionTool(score_and_route_keywords),
    FunctionTool(generate_and_validate_slugs),
    FunctionTool(validate_title_and_meta),
    FunctionTool(compute_cosine_similarity_matrix),
    FunctionTool(validate_json_ld_schema),
    FunctionTool(validate_internal_links),
]
