"""Lectura del sitio del cliente: plataforma, inventario de URLs y páginas.

El agente conversacional usa esto para entender el sitio antes de proponer
páginas nuevas: qué plataforma es (y qué URLs admite), qué colecciones o
secciones ya existen y qué dice una página puntual.

Dos capas:
- Funciones puras (`detect_platform`, `parse_sitemap`, `extract_page`,
  `url_rules`, `match_existing_pages`): sin red, fáciles de testear.
- `SiteReader`: descarga con `httpx`, respeta robots.txt y arma el inventario.

Solo se lee HTML tal como lo entrega el servidor (sin ejecutar JavaScript).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import gzip
import re
import xml.etree.ElementTree as ET
from collections import Counter
from html.parser import HTMLParser
from typing import Any, ClassVar
from urllib.parse import unquote, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx

from src.tools.seo_deterministic import dedupe_key, normalize_text

USER_AGENT = "D2B-SEO-Agent/1.0 (+https://d2b.cl)"
TIMEOUT_S = 10.0
MAX_SITEMAP_FILES = 25
MAX_URLS = 5000
MAX_COLLECTION_PAGES = 10  # /collections.json de Shopify, 250 por página
MAX_TEXT_CHARS = 3000
MAX_LINKS = 60
MAX_BODY_BYTES = 8_000_000
MAX_RETRY_WAIT_S = 5.0  # espera máxima ante un 429 antes de reintentar una vez
SITEMAP_CONCURRENCY = 3
MIN_TEXT_CHARS = 200  # menos que esto en una página con scripts: contenido por JS


# Palabras que no distinguen una página de otra al comparar keywords con URLs
STOPWORDS = frozenset({
    "de", "del", "la", "las", "el", "los", "en", "y", "o", "para", "por", "con",
    "sin", "a", "al", "un", "una", "unos", "unas",
})


# ------------------------------------------------------------- plataforma ----

def detect_platform(html: str, headers: dict[str, str] | httpx.Headers) -> str:
    """Detecta la plataforma del sitio por cabeceras y huellas en el HTML.

    Args:
        html: HTML de la página de inicio.
        headers: Cabeceras HTTP de esa respuesta.

    Returns:
        "shopify", "woocommerce", "wordpress", "vtex", "magento", "wix" o
        "desconocida".
    """
    names = {k.lower(): str(v).lower() for k, v in dict(headers).items()}
    text = (html or "").lower()
    generator = ""
    match = re.search(
        r'<meta[^>]+name=["\']generator["\'][^>]+content=["\']([^"\']+)', text
    )
    if match:
        generator = match.group(1)

    if any(k.startswith("x-shopify") for k in names) or "cdn.shopify.com" in text:
        return "shopify"
    if "powered-by" in names and "shopify" in names["powered-by"]:
        return "shopify"
    if "woocommerce" in text or "woocommerce" in generator:
        return "woocommerce"
    if "wp-content" in text or "wordpress" in generator:
        return "wordpress"
    if "vtex" in text or any(k.startswith("x-vtex") for k in names):
        return "vtex"
    if "magento" in text or "mage/" in text:
        return "magento"
    if "wix.com" in generator or "static.wixstatic.com" in text:
        return "wix"
    return "desconocida"


def url_rules(platform: str) -> dict[str, Any]:
    """Qué URLs se pueden crear en la plataforma y qué hub corresponde por intención.

    En Shopify no hay subcarpetas propias: una página nueva vive en
    `/collections/{handle}` (colección filtrada, para intención transaccional o
    comercial) o en `/pages/{handle}` (contenido, para informacional).

    Args:
        platform: Valor devuelto por `detect_platform`.

    Returns:
        {"subcarpetas_libres": bool, "hubs_permitidos": [...],
         "hub_por_intencion": {"transaccional": ..., "informacional": ...},
         "nota": str}. Con subcarpetas libres, las listas van vacías.
    """
    if platform == "shopify":
        return {
            "subcarpetas_libres": False,
            "hubs_permitidos": ["/collections", "/pages"],
            "hub_por_intencion": {
                "transaccional": "/collections",
                "informacional": "/pages",
            },
            "nota": (
                "Shopify no admite subcarpetas propias: transaccional o comercial "
                "va en /collections/{handle}, informacional en /pages/{handle}."
            ),
        }
    return {
        "subcarpetas_libres": True,
        "hubs_permitidos": [],
        "hub_por_intencion": {},
        "nota": "Se pueden crear subcarpetas propias para los hubs.",
    }


def hub_for_intent(rules: dict[str, Any], intent: str | None) -> str | None:
    """Hub que exige la plataforma para una intención; None si los hubs son libres.

    Args:
        rules: Salida de `url_rules`.
        intent: Intención de la keyword principal (transaccional, informacional...).

    Returns:
        El hub fijo o None. Sin intención conocida se usa el transaccional,
        porque las páginas programáticas de ecommerce suelen ser de listado.
    """
    by_intent = rules.get("hub_por_intencion") or {}
    if rules.get("subcarpetas_libres", True) or not by_intent:
        return None
    normalized = normalize_text(intent or "")
    if normalized.startswith(("inform", "naveg")):
        return by_intent.get("informacional")
    return by_intent.get("transaccional")


# ---------------------------------------------------------------- sitemap ----

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_sitemap(content: bytes | str) -> tuple[list[str], list[str]]:
    """Lee un sitemap o un índice de sitemaps.

    Args:
        content: XML (o XML comprimido con gzip).

    Returns:
        (urls_de_paginas, sitemaps_hijos). Un XML inválido devuelve dos listas
        vacías.
    """
    raw = content.encode() if isinstance(content, str) else content
    if raw[:2] == b"\x1f\x8b":
        try:
            raw = gzip.decompress(raw)
        except (OSError, EOFError):
            return [], []
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return [], []
    pages: list[str] = []
    children: list[str] = []
    is_index = _local(root.tag) == "sitemapindex"
    for node in root:
        loc = next((c.text for c in node if _local(c.tag) == "loc" and c.text), None)
        if not loc:
            continue
        (children if is_index else pages).append(loc.strip())
    return pages, children


# ----------------------------------------------------------------- página ----

class _PageParser(HTMLParser):
    """Extrae title, meta description, encabezados, texto visible y enlaces.

    El menú, la cabecera y el pie (`nav`, `header`, `footer`) no entran al texto
    ni a los encabezados, porque se repiten en todas las páginas; sus enlaces sí
    se guardan, porque muestran la estructura del sitio. Si la página tiene
    `<main>`, el texto sale solo de ahí (muchos temas arman el menú con
    elementos propios que no son `nav`).
    """

    SKIP: ClassVar[set[str]] = {"script", "style", "noscript", "svg", "template", "iframe"}
    CHROME: ClassVar[set[str]] = {"nav", "header", "footer"}
    BLOCK: ClassVar[set[str]] = {"p", "div", "li", "br", "h1", "h2", "h3", "h4", "section", "article",
             "td", "tr", "header", "footer", "nav", "main"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.meta_description = ""
        self.h1: list[str] = []
        self.h2: list[str] = []
        self.links: list[str] = []
        self.scripts = 0
        self._chunks: list[str] = []
        self._main_chunks: list[str] = []
        self._main_depth = 0
        self._skip_depth = 0
        self._chrome_depth = 0
        self._heading: str | None = None
        self._heading_text: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k: v or "" for k, v in attrs}
        if tag == "script":
            self.scripts += 1
        if tag in self.SKIP:
            self._skip_depth += 1
            return
        if tag in self.CHROME:
            self._chrome_depth += 1
        if tag == "main":
            self._main_depth += 1
        if tag == "title":
            self._in_title = True
        elif tag == "meta" and attr.get("name", "").lower() == "description":
            self.meta_description = attr.get("content", "").strip()
        elif tag == "a" and attr.get("href"):
            self.links.append(attr["href"])
        elif tag in ("h1", "h2") and not self._chrome_depth:
            self._heading, self._heading_text = tag, []
        if tag in self.BLOCK:
            self._chunks.append("\n")
            if self._main_depth:
                self._main_chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag in self.CHROME:
            self._chrome_depth = max(0, self._chrome_depth - 1)
        if tag == "main":
            self._main_depth = max(0, self._main_depth - 1)
        if tag == "title":
            self._in_title = False
        elif tag == self._heading:
            text = " ".join("".join(self._heading_text).split())
            if text:
                (self.h1 if tag == "h1" else self.h2).append(text)
            self._heading = None

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
            return
        if self._chrome_depth:
            return
        if self._heading:
            self._heading_text.append(data)
        self._chunks.append(data)
        if self._main_depth:
            self._main_chunks.append(data)

    @property
    def text(self) -> str:
        main_text = "".join(self._main_chunks)
        source = main_text if main_text.strip() else "".join(self._chunks)
        lines = (" ".join(line.split()) for line in source.splitlines())
        return "\n".join(line for line in lines if line)


def same_site(url: str, domain: str) -> bool:
    """True si `url` es del dominio (con o sin www, o un subdominio de él)."""
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    return bool(host) and (host == domain or host.endswith(f".{domain}"))


def extract_page(html: str, url: str, domain: str) -> dict[str, Any]:
    """Resume una página para que el modelo la entienda sin leer el HTML.

    Args:
        html: HTML de la página.
        url: URL final (después de redirecciones), para resolver enlaces relativos.
        domain: Dominio del cliente sin www; filtra los enlaces internos.

    Returns:
        {"title", "meta_description", "h1", "h2", "texto", "texto_recortado",
         "enlaces_internos", "avisos"}.
    """
    parser = _PageParser()
    try:
        parser.feed(html or "")
        parser.close()
    except Exception:  # noqa: BLE001, S110 - HTML roto: se usa lo que alcanzó a leer
        pass
    internal: list[str] = []
    for href in parser.links:
        absolute = urljoin(url, href).split("#", 1)[0]
        if absolute.startswith("http") and same_site(absolute, domain):
            path = urlparse(absolute).path or "/"
            if path not in internal:
                internal.append(path)
    text = parser.text
    warnings = []
    if len(text) < MIN_TEXT_CHARS and parser.scripts >= 3:
        warnings.append(
            "La página casi no trae texto en el HTML: el contenido probablemente se "
            "carga con JavaScript y no lo puedo leer."
        )
    return {
        "title": " ".join(parser.title.split()),
        "meta_description": parser.meta_description,
        "h1": parser.h1[:5],
        "h2": parser.h2[:20],
        "texto": text[:MAX_TEXT_CHARS],
        "texto_recortado": len(text) > MAX_TEXT_CHARS,
        "enlaces_internos": internal[:MAX_LINKS],
        "avisos": warnings,
    }


# --------------------------------------------- cruce keyword vs. existente ----

def _tokens(text: Any) -> set[str]:
    """Palabras significativas, sin tildes y en singular (vía `dedupe_key`)."""
    words = dedupe_key(str(text).replace("-", " ").replace("_", " ")).split()
    return {w for w in words if w not in STOPWORDS and not w.isdigit()}


def _page_label(url: str) -> str:
    """Último segmento del path como texto ("/collections/pediatria" -> "pediatria")."""
    segments = [s for s in urlparse(url).path.split("/") if s]
    return unquote(segments[-1]) if segments else ""


def site_pages(inventory: dict[str, Any]) -> list[dict[str, str]]:
    """Páginas del inventario como {"url", "titulo"}: colecciones primero, luego sitemap.

    Args:
        inventory: Salida de `SiteReader.explore`.

    Returns:
        Lista sin URLs repetidas.
    """
    seen: set[str] = set()
    pages: list[dict[str, str]] = []
    for item in inventory.get("colecciones") or []:
        url = item.get("url") or ""
        if url and url not in seen:
            seen.add(url)
            pages.append({"url": url, "titulo": item.get("titulo") or ""})
    for url in inventory.get("urls") or []:
        if url not in seen:
            seen.add(url)
            pages.append({"url": url, "titulo": ""})
    return pages


def _fixed_tokens(keyword: str, groups: list[list[str]]) -> set[str]:
    """Palabras que comparten todas las keywords del grupo de `keyword` (la parte fija).

    En "libros de pediatría" / "libros de cardiología" la parte fija es "libro";
    lo que distingue la página es "pediatria". Un grupo de 1 keyword no tiene
    parte fija.
    """
    key = dedupe_key(keyword)
    for group in groups:
        if len(group) < 2 or key not in {dedupe_key(k) for k in group}:
            continue
        common = set.intersection(*(_tokens(k) for k in group))
        return common
    return set()


def match_existing_pages(
    keywords: list[str],
    pages: list[dict[str, str]],
    groups: list[list[str]] | None = None,
) -> dict[str, dict[str, str]]:
    """Busca, para cada keyword, una página del sitio que ya cubre lo mismo.

    Hay coincidencia cuando las palabras que distinguen la keyword (sin la parte
    fija de su grupo ni palabras vacías) están todas en el título o en el último
    segmento de la URL de la página. Así "libros de pediatría" coincide con
    `/collections/pediatria`, pero "libros de cirugía veterinaria" no coincide
    con `/collections/veterinaria`.

    Args:
        keywords: Keywords a revisar.
        pages: Salida de `site_pages`.
        groups: Grupos de keywords del mismo patrón (para conocer la parte fija).

    Returns:
        {keyword: {"url", "titulo"}} solo para las que tienen coincidencia. Si
        varias páginas coinciden, gana la de menos palabras extra (la más
        específica) y, a igualdad, la que viene primero (colecciones).
    """
    indexed = []
    for page in pages:
        label_tokens = _tokens(_page_label(page["url"]))
        title_tokens = _tokens(page.get("titulo") or "")
        if label_tokens or title_tokens:
            indexed.append((page, label_tokens, title_tokens))

    matches: dict[str, dict[str, str]] = {}
    for keyword in keywords:
        fixed = _fixed_tokens(keyword, groups or [])
        distinctive = _tokens(keyword) - fixed
        if not distinctive:
            continue
        best: tuple[int, dict[str, str]] | None = None
        for page, label_tokens, title_tokens in indexed:
            for candidate in (label_tokens, title_tokens):
                if not candidate or not distinctive <= candidate:
                    continue
                extra = len(candidate - distinctive - fixed)
                if best is None or extra < best[0]:
                    best = (extra, page)
        # Más de 2 palabras de sobra ya es otra página (ej. un producto puntual)
        if best is not None and best[0] <= 2:
            matches[keyword] = best[1]
    return matches


# ----------------------------------------------------------------- lector ----

class SiteUnavailable(RuntimeError):
    """El sitio no respondió o bloqueó la lectura."""


class SiteReader:
    """Descarga y resume el sitio del cliente respetando robots.txt.

    Attributes:
        client: Cliente HTTP inyectable (en tests, con `httpx.MockTransport`).
    """

    def __init__(
        self, client: httpx.AsyncClient | None = None, sleep: Any = asyncio.sleep
    ) -> None:
        self._client = client
        self._sleep = sleep
        self._robots_cache: dict[str, tuple[RobotFileParser, list[str]]] = {}

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={"User-Agent": USER_AGENT, "Accept-Language": "es-CL,es;q=0.9"},
                timeout=TIMEOUT_S,
                follow_redirects=True,
            )
        return self._client

    async def _get(self, url: str) -> httpx.Response:
        response = await self._http().get(url)
        if response.status_code == 429:
            # Shopify y otros limitan ráfagas: se respeta Retry-After y se reintenta una vez
            try:
                wait = float(response.headers.get("retry-after") or 1.0)
            except ValueError:
                wait = 1.0
            await self._sleep(min(max(wait, 0.5), MAX_RETRY_WAIT_S))
            response = await self._http().get(url)
        if len(response.content) > MAX_BODY_BYTES:
            raise SiteUnavailable(f"{url} pesa más de {MAX_BODY_BYTES // 1_000_000} MB.")
        return response

    async def _home(self, domain: str) -> tuple[httpx.Response, str]:
        """Prueba https://dominio y https://www.dominio; devuelve la respuesta y la base."""
        last_error = ""
        for candidate in (f"https://{domain}/", f"https://www.{domain}/"):
            try:
                response = await self._get(candidate)
            except httpx.HTTPError as e:
                last_error = f"{type(e).__name__}: {e}"
                continue
            if response.status_code < 400:
                final = response.url
                return response, f"{final.scheme}://{final.host}"
            last_error = f"HTTP {response.status_code}"
            if response.status_code in (401, 403, 429, 503):
                blocked = "cf-ray" in response.headers or "captcha" in response.text.lower()
                motive = "un sistema anti-bots (Cloudflare u otro)" if blocked else "el servidor"
                raise SiteUnavailable(f"El sitio bloqueó la lectura ({last_error}, {motive}).")
        raise SiteUnavailable(f"No pude abrir el sitio ({last_error}).")

    async def _robots(self, base: str) -> tuple[RobotFileParser, list[str]]:
        if base in self._robots_cache:
            return self._robots_cache[base]
        robots = RobotFileParser()
        sitemaps: list[str] = []
        try:
            response = await self._get(f"{base}/robots.txt")
        except (httpx.HTTPError, SiteUnavailable):
            robots.parse([])
            return robots, sitemaps
        lines = response.text.splitlines() if response.status_code == 200 else []
        robots.parse(lines)
        for line in lines:
            if line.lower().startswith("sitemap:"):
                sitemaps.append(line.split(":", 1)[1].strip())
        self._robots_cache[base] = (robots, sitemaps)
        return robots, sitemaps

    async def _sitemap_urls(self, roots: list[str], warnings: list[str]) -> list[str]:
        queue = list(dict.fromkeys(roots))
        visited: set[str] = set()
        urls: list[str] = []
        while queue and len(visited) < MAX_SITEMAP_FILES and len(urls) < MAX_URLS:
            batch = [u for u in queue[:SITEMAP_CONCURRENCY] if u not in visited]
            queue = queue[SITEMAP_CONCURRENCY:]
            visited.update(batch)
            responses = await asyncio.gather(
                *(self._get(u) for u in batch), return_exceptions=True
            )
            for response in responses:
                if isinstance(response, BaseException) or response.status_code != 200:
                    continue
                pages, children = parse_sitemap(response.content)
                urls.extend(pages)
                queue.extend(c for c in children if c not in visited)
        if queue or len(urls) >= MAX_URLS:
            warnings.append(
                f"El sitemap es grande: leí {min(len(urls), MAX_URLS)} URLs de "
                f"{len(visited)} archivos y quedaron más sin leer."
            )
        return list(dict.fromkeys(urls))[:MAX_URLS]

    async def _shopify_collections(self, base: str) -> list[dict[str, Any]]:
        collections: list[dict[str, Any]] = []
        for page in range(1, MAX_COLLECTION_PAGES + 1):
            try:
                response = await self._get(f"{base}/collections.json?limit=250&page={page}")
                if response.status_code != 200:
                    break
                items = response.json().get("collections") or []
            except (httpx.HTTPError, ValueError, SiteUnavailable):
                break
            if not items:
                break
            for item in items:
                handle = item.get("handle")
                if handle:
                    collections.append({
                        "titulo": item.get("title") or handle,
                        "url": f"{base}/collections/{handle}",
                        "productos": item.get("products_count"),
                    })
        return collections

    async def _wordpress_categories(self, base: str, platform: str) -> list[dict[str, Any]]:
        endpoint = (
            "/wp-json/wc/store/v1/products/categories?per_page=100"
            if platform == "woocommerce"
            else "/wp-json/wp/v2/categories?per_page=100"
        )
        try:
            response = await self._get(base + endpoint)
            items = response.json() if response.status_code == 200 else []
        except (httpx.HTTPError, ValueError, SiteUnavailable):
            return []
        if not isinstance(items, list):
            return []
        return [
            {"titulo": i.get("name") or "", "url": i.get("permalink") or i.get("link") or "",
             "productos": i.get("count")}
            for i in items if isinstance(i, dict) and (i.get("permalink") or i.get("link"))
        ]

    async def explore(self, domain: str) -> dict[str, Any]:
        """Arma el inventario del sitio: plataforma, URLs del sitemap y colecciones.

        Args:
            domain: Dominio sin esquema ni www (ej. "libreria-demo.cl").

        Returns:
            {"dominio", "base_url", "plataforma", "reglas_url", "urls",
             "colecciones", "secciones", "avisos", "explorado_en"}.

        Raises:
            SiteUnavailable: Si la página de inicio no responde o está bloqueada.
        """
        home, base = await self._home(domain)
        platform = detect_platform(home.text, home.headers)
        robots, robot_sitemaps = await self._robots(base)
        warnings: list[str] = []

        urls = await self._sitemap_urls(robot_sitemaps or [f"{base}/sitemap.xml"], warnings)
        urls = [u for u in urls if same_site(u, domain)]
        if not urls:
            warnings.append("No encontré sitemap: el inventario sale solo de los enlaces de la portada.")
            urls = [
                f"{base}{path}"
                for path in extract_page(home.text, str(home.url), domain)["enlaces_internos"]
            ]

        collections: list[dict[str, Any]] = []
        if platform == "shopify" and robots.can_fetch(USER_AGENT, f"{base}/collections.json"):
            collections = await self._shopify_collections(base)
        elif platform in ("wordpress", "woocommerce"):
            collections = await self._wordpress_categories(base, platform)
        if platform == "shopify" and not collections:
            collections = [{"titulo": _page_label(u).replace("-", " "), "url": u, "productos": None}
                           for u in urls if "/collections/" in u and "/products/" not in u]

        sections = Counter(
            (urlparse(u).path.strip("/").split("/")[0] or "(inicio)") for u in urls
        )
        return {
            "dominio": domain,
            "base_url": base,
            "plataforma": platform,
            "reglas_url": url_rules(platform),
            "urls": urls,
            "colecciones": collections,
            "secciones": dict(sections.most_common(12)),
            "avisos": warnings,
            "explorado_en": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        }

    async def read_page(self, url: str, domain: str) -> dict[str, Any]:
        """Lee una página del sitio del cliente y devuelve su resumen.

        Args:
            url: URL completa o path ("/collections/pediatria").
            domain: Dominio del cliente sin www; las URLs de otros dominios se rechazan.

        Returns:
            Resumen de `extract_page` más "url" y "estado_http".

        Raises:
            ValueError: Si la URL es de otro dominio o robots.txt la prohíbe.
            SiteUnavailable: Si la página no responde.
        """
        if url.startswith("/"):
            url = f"https://{domain}{url}"
        elif "://" not in url:
            url = f"https://{url}"
        if not same_site(url, domain):
            raise ValueError(f"Solo leo páginas de {domain}.")
        parsed = urlparse(url)
        robots, _ = await self._robots(f"{parsed.scheme}://{parsed.netloc}")
        if not robots.can_fetch(USER_AGENT, url):
            raise ValueError("robots.txt del sitio no permite leer esa página.")
        try:
            response = await self._get(url)
        except httpx.HTTPError as e:
            raise SiteUnavailable(f"No pude abrir {url} ({type(e).__name__}).") from e
        summary = extract_page(response.text, str(response.url), domain)
        return {"url": str(response.url), "estado_http": response.status_code, **summary}
