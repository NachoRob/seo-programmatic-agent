"""Lectura del sitio: plataforma, sitemap, páginas, cruce con keywords y uso en el flujo.

Nada sale a internet: `SiteReader` recibe un `httpx.AsyncClient` con
`MockTransport`. El test `live` explora el sitio de `SEO_AGENT_LIVE_SITE`.
"""

import gzip
import os
import json

import httpx
import pytest

from src.agent.chat_actions import SeoChatActions
from src.agent.orchestrator import SITE_INVENTORY_KEY, RunStatus
from src.tools.seo_deterministic import generate_and_validate_slugs
from src.tools.site_reader import (
    USER_AGENT,
    SiteReader,
    SiteUnavailable,
    detect_platform,
    extract_page,
    hub_for_intent,
    match_existing_pages,
    parse_sitemap,
    same_site,
    site_pages,
    url_rules,
)
from tests.test_chat_and_root_agent import SID as CHAT_SID
from tests.test_chat_and_root_agent import make_service
from tests.test_orchestrator import (  # noqa: F401 - env es fixture
    SID,
    env,
    orchestrator,
)

BASE = "https://libreria-demo.cl"


# ----------------------------------------------------------------- fakes ----

def mock_site(routes: dict[str, httpx.Response | int], calls: list[str] | None = None) -> SiteReader:
    """SiteReader sobre un sitio falso: `routes` mapea URL completa -> respuesta."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if calls is not None:
            calls.append(url)
        assert request.headers["user-agent"] == USER_AGENT
        found = routes.get(url)
        if found is None:
            return httpx.Response(404, text="no existe")
        if isinstance(found, int):
            return httpx.Response(found, text="")
        return found

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    )

    async def no_sleep(_seconds: float) -> None:
        return None

    return SiteReader(client=client, sleep=no_sleep)


def sitemap(urls: list[str], index: bool = False) -> str:
    tag, item = ("sitemapindex", "sitemap") if index else ("urlset", "url")
    body = "".join(f"<{item}><loc>{u}</loc></{item}>" for u in urls)
    return f'<?xml version="1.0"?><{tag} xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{body}</{tag}>'


SHOPIFY_HOME = """<html><head><title>Librería Demo</title>
<link rel="stylesheet" href="https://cdn.shopify.com/s/files/theme.css"></head>
<body><header><a href="/collections/pediatria">Pediatría</a></header>
<main><h1>Libros de medicina</h1><p>Librería médica en Santiago.</p></main></body></html>"""

COLLECTION_PAGE = """<html><head><title>Libros de Pediatría – Librería Demo</title>
<meta name="description" content="Libros de pediatría con despacho a todo Chile."></head>
<body><header><nav><a href="/collections/cardiologia">Cardiología</a><h2>Menú</h2></nav></header>
<main><h1>Colección: Pediatría</h1><h2>62 productos</h2>
<p>Nelson. Tratado de pediatría.</p><a href="/products/nelson">Nelson</a>
<a href="https://otro-sitio.cl/x">fuera</a></main>
<footer><h2>SERVICIO AL CLIENTE</h2>Contacto</footer>
<script>var a=1;</script></body></html>"""


def shopify_routes() -> dict[str, httpx.Response | int]:
    collections_p1 = {"collections": [
        {"handle": "pediatria", "title": "Pediatría", "products_count": 62},
        {"handle": "cardiologia", "title": "Cardiología", "products_count": 80},
    ]}
    return {
        f"{BASE}/": httpx.Response(200, text=SHOPIFY_HOME, headers={"x-shopify-stage": "production"}),
        f"{BASE}/robots.txt": httpx.Response(
            200, text=f"User-agent: *\nDisallow: /checkout\nSitemap: {BASE}/sitemap.xml\n"
        ),
        f"{BASE}/sitemap.xml": httpx.Response(
            200, text=sitemap([f"{BASE}/sitemap_collections.xml", f"{BASE}/sitemap_products.xml.gz"], index=True)
        ),
        f"{BASE}/sitemap_collections.xml": httpx.Response(
            200, text=sitemap([f"{BASE}/collections/pediatria", f"{BASE}/collections/cardiologia",
                               "https://cdn.otro.com/x"])
        ),
        f"{BASE}/sitemap_products.xml.gz": httpx.Response(
            200, content=gzip.compress(sitemap([f"{BASE}/products/nelson"]).encode())
        ),
        f"{BASE}/collections.json?limit=250&page=1": httpx.Response(200, json=collections_p1),
        f"{BASE}/collections.json?limit=250&page=2": httpx.Response(200, json={"collections": []}),
        f"{BASE}/collections/pediatria": httpx.Response(200, text=COLLECTION_PAGE),
    }


# ------------------------------------------------------------ plataforma ----

class TestDetectPlatform:
    """Huellas de cada plataforma."""

    def test_shopify_by_header(self) -> None:
        """Una cabecera x-shopify-* basta."""
        assert detect_platform("<html></html>", {"X-Shopify-Stage": "production"}) == "shopify"

    def test_shopify_by_cdn(self) -> None:
        """El CDN de Shopify en el HTML basta."""
        assert detect_platform(SHOPIFY_HOME, {}) == "shopify"

    def test_woocommerce_before_wordpress(self) -> None:
        """WooCommerce también tiene wp-content: gana la más específica."""
        html = '<link href="/wp-content/plugins/woocommerce/style.css">'
        assert detect_platform(html, {}) == "woocommerce"

    def test_wordpress_by_generator(self) -> None:
        """El meta generator identifica WordPress."""
        html = '<meta name="generator" content="WordPress 6.6">'
        assert detect_platform(html, {}) == "wordpress"

    def test_unknown(self) -> None:
        """Sin huellas conocidas es desconocida."""
        assert detect_platform("<html><body>hola</body></html>", {}) == "desconocida"

    def test_empty_html(self) -> None:
        """HTML vacío no rompe."""
        assert detect_platform("", {}) == "desconocida"


class TestUrlRules:
    """Reglas de URL y hub por intención."""

    def test_shopify_has_fixed_hubs(self) -> None:
        """Shopify solo admite /collections y /pages."""
        rules = url_rules("shopify")
        assert rules["subcarpetas_libres"] is False
        assert rules["hubs_permitidos"] == ["/collections", "/pages"]

    def test_other_platforms_are_free(self) -> None:
        """WordPress y desconocida admiten subcarpetas."""
        assert url_rules("wordpress")["subcarpetas_libres"] is True
        assert url_rules("desconocida")["hubs_permitidos"] == []

    @pytest.mark.parametrize(
        ("intent", "hub"),
        [
            ("transaccional", "/collections"),
            ("comercial", "/collections"),
            ("informativa", "/pages"),
            ("informacional", "/pages"),
            ("navegacional", "/pages"),
            (None, "/collections"),
            ("", "/collections"),
        ],
    )
    def test_shopify_hub_by_intent(self, intent: str | None, hub: str) -> None:
        """Transaccional va a colección, informacional a página; sin intención, colección."""
        assert hub_for_intent(url_rules("shopify"), intent) == hub

    def test_free_platform_has_no_fixed_hub(self) -> None:
        """Con subcarpetas libres el hub lo decide el plan."""
        assert hub_for_intent(url_rules("wordpress"), "transaccional") is None


# --------------------------------------------------------------- sitemap ----

class TestParseSitemap:
    """Sitemaps simples, índices, gzip y XML roto."""

    def test_urlset(self) -> None:
        """Un urlset devuelve páginas y ningún hijo."""
        assert parse_sitemap(sitemap(["https://a.cl/x", "https://a.cl/y"])) == (
            ["https://a.cl/x", "https://a.cl/y"], [])

    def test_index(self) -> None:
        """Un índice devuelve sitemaps hijos."""
        assert parse_sitemap(sitemap(["https://a.cl/s1.xml"], index=True)) == ([], ["https://a.cl/s1.xml"])

    def test_gzip(self) -> None:
        """Un sitemap comprimido se descomprime."""
        raw = gzip.compress(sitemap(["https://a.cl/x"]).encode())
        assert parse_sitemap(raw) == (["https://a.cl/x"], [])

    def test_invalid_xml(self) -> None:
        """XML inválido o gzip roto devuelven listas vacías."""
        assert parse_sitemap("<html>no es xml") == ([], [])
        assert parse_sitemap(b"\x1f\x8bbasura") == ([], [])

    def test_empty(self) -> None:
        """Un urlset sin URLs devuelve vacío."""
        assert parse_sitemap(sitemap([])) == ([], [])


# ---------------------------------------------------------------- página ----

class TestExtractPage:
    """Resumen de una página para el modelo."""

    def test_main_content_without_chrome(self) -> None:
        """El texto y los H2 salen de <main>; menú y pie quedan fuera."""
        page = extract_page(COLLECTION_PAGE, f"{BASE}/collections/pediatria", "libreria-demo.cl")
        assert page["title"] == "Libros de Pediatría – Librería Demo"
        assert page["meta_description"] == "Libros de pediatría con despacho a todo Chile."
        assert page["h1"] == ["Colección: Pediatría"]
        assert page["h2"] == ["62 productos"]
        assert "Nelson. Tratado de pediatría." in page["texto"]
        assert "SERVICIO AL CLIENTE" not in page["texto"]
        assert "Cardiología" not in page["texto"]

    def test_internal_links_keep_menu_and_drop_external(self) -> None:
        """Los enlaces del menú cuentan (muestran la estructura); los externos no."""
        page = extract_page(COLLECTION_PAGE, f"{BASE}/collections/pediatria", "libreria-demo.cl")
        assert page["enlaces_internos"] == ["/collections/cardiologia", "/products/nelson"]

    def test_js_rendered_page_warns(self) -> None:
        """Casi sin texto y con scripts: avisa que el contenido viene por JavaScript."""
        html = "<html><body><div id='app'></div>" + "<script>x()</script>" * 3 + "</body></html>"
        page = extract_page(html, f"{BASE}/", "libreria-demo.cl")
        assert page["texto"] == ""
        assert len(page["avisos"]) == 1

    def test_long_text_is_trimmed(self) -> None:
        """El texto se recorta y se marca como recortado."""
        page = extract_page("<p>" + "palabra " * 2000 + "</p>", f"{BASE}/", "libreria-demo.cl")
        assert len(page["texto"]) == 3000
        assert page["texto_recortado"] is True

    def test_broken_html_does_not_raise(self) -> None:
        """HTML mal cerrado devuelve lo que se alcanzó a leer."""
        page = extract_page("<html><title>Hola</title><main><p>texto <b>sin cerrar", f"{BASE}/", "libreria-demo.cl")
        assert page["title"] == "Hola"
        assert "texto" in page["texto"]


class TestSameSite:
    """Qué URLs se consideran del cliente."""

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://libreria-demo.cl/x", True),
            ("https://www.libreria-demo.cl/x", True),
            ("https://blog.libreria-demo.cl/x", True),
            ("https://libreria-demo.cl.evil.com/x", False),
            ("https://otrolibreria-demo.cl/x", False),
            ("/relativa", False),
        ],
    )
    def test_same_site(self, url: str, expected: bool) -> None:
        """Dominio, www y subdominios sí; dominios parecidos no."""
        assert same_site(url, "libreria-demo.cl") is expected


# ------------------------------------------------- keyword vs. existente ----

PAGES = [
    {"url": f"{BASE}/collections/pediatria", "titulo": "Pediatría"},
    {"url": f"{BASE}/collections/veterinaria", "titulo": "Veterinaria"},
    {"url": f"{BASE}/collections/cirugia", "titulo": "Cirugía"},
    {"url": f"{BASE}/collections/a-cole-burks", "titulo": "A. Cole Burks"},
    {"url": f"{BASE}/products/manual-de-pediatria-practica-edicion-2024", "titulo": ""},
]


class TestMatchExistingPages:
    """Casos detectados en producción con una tienda Shopify."""

    def test_pattern_keywords_match_their_collection(self) -> None:
        """La parte fija del patrón ("libro") no cuenta: pediatría coincide con la colección."""
        kws = ["libros de pediatría", "libros de cardiología", "libros de veterinaria"]
        found = match_existing_pages(kws, PAGES, [kws])
        assert found == {
            "libros de pediatría": PAGES[0],
            "libros de veterinaria": PAGES[1],
        }

    def test_extra_distinctive_word_does_not_match(self) -> None:
        """"cirugía veterinaria" no es la colección "veterinaria" ni "cirugía"."""
        kws = ["libros de cirugía veterinaria", "libros de pediatría"]
        assert "libros de cirugía veterinaria" not in match_existing_pages(kws, PAGES, [kws])

    def test_keyword_alone_matches_title(self) -> None:
        """Sin grupo, una keyword de una palabra coincide igual."""
        assert match_existing_pages(["pediatría"], PAGES) == {"pediatría": PAGES[0]}

    def test_without_group_fixed_words_count(self) -> None:
        """Sin grupo no se sabe que "libro" es la parte fija: no hay coincidencia."""
        assert match_existing_pages(["libros de pediatría"], PAGES) == {}

    def test_product_with_many_extra_words_is_not_a_match(self) -> None:
        """Un producto que menciona la palabra no cubre la keyword (más de 2 palabras extra)."""
        pages = [PAGES[4]]
        assert match_existing_pages(["pediatría"], pages) == {}

    def test_prefers_most_specific_page(self) -> None:
        """Entre varias, gana la de menos palabras de sobra."""
        pages = [{"url": f"{BASE}/collections/pediatria-general", "titulo": ""}, PAGES[0]]
        assert match_existing_pages(["pediatría"], pages)["pediatría"] == PAGES[0]

    def test_empty_inputs(self) -> None:
        """Sin keywords o sin páginas no hay coincidencias."""
        assert match_existing_pages([], PAGES) == {}
        assert match_existing_pages(["pediatría"], []) == {}

    def test_keyword_made_of_stopwords_is_skipped(self) -> None:
        """Una keyword sin palabras significativas no coincide con nada."""
        assert match_existing_pages(["de la"], PAGES) == {}


class TestSitePages:
    """Unión de colecciones y sitemap."""

    def test_collections_first_without_duplicates(self) -> None:
        """Las colecciones van primero y no se repiten con el sitemap."""
        inventory = {
            "colecciones": [{"titulo": "Pediatría", "url": f"{BASE}/collections/pediatria"}],
            "urls": [f"{BASE}/collections/pediatria", f"{BASE}/pages/contacto"],
        }
        assert site_pages(inventory) == [
            {"url": f"{BASE}/collections/pediatria", "titulo": "Pediatría"},
            {"url": f"{BASE}/pages/contacto", "titulo": ""},
        ]

    def test_empty_inventory(self) -> None:
        """Sin inventario no hay páginas."""
        assert site_pages({}) == []


# ------------------------------------------------------- validador de hubs ----

class TestAllowedHubs:
    """`generate_and_validate_slugs` con los hubs que admite la plataforma."""

    def test_shopify_hub_ok_and_collision(self) -> None:
        """/collections es válido; una colección existente choca."""
        nodes = [
            {"hub": "/collections", "pattern": "libros de {especialidad}", "variables": {"especialidad": "Cardiología"}},
            {"hub": "/collections", "pattern": "libros de {especialidad}", "variables": {"especialidad": "Pediatría"}},
        ]
        out = generate_and_validate_slugs(nodes, [f"{BASE}/collections/libros-de-pediatria"],
                                          allowed_hubs=["/collections", "/pages"])
        assert out[0]["url_path"] == "/collections/libros-de-cardiologia"
        assert out[0]["slug_valid"] is True
        assert out[1]["slug_issues"] == ["colision_url_existente"]

    def test_invented_hub_is_not_allowed(self) -> None:
        """El hub inventado de la prueba con Librería Demo queda marcado."""
        nodes = [{"hub": "/libros-de-medicina/", "pattern": "libros de {e}", "variables": {"e": "Cardiología"}}]
        out = generate_and_validate_slugs(nodes, [], allowed_hubs=["/collections", "/pages"])
        assert out[0]["slug_issues"] == ["hub_no_permitido"]

    def test_free_hubs_without_rules(self) -> None:
        """Sin `allowed_hubs` cualquier subcarpeta vale (comportamiento anterior)."""
        nodes = [{"hub": "/libros-de-medicina/", "pattern": "libros de {e}", "variables": {"e": "Cardiología"}}]
        assert generate_and_validate_slugs(nodes, [])[0]["slug_valid"] is True


# --------------------------------------------------------------- lector ----

class TestSiteReaderExplore:
    """Inventario completo contra un sitio falso."""

    async def test_shopify_inventory(self) -> None:
        """Plataforma, sitemap con índice y gzip, colecciones y secciones."""
        inv = await mock_site(shopify_routes()).explore("libreria-demo.cl")
        assert inv["base_url"] == BASE
        assert inv["plataforma"] == "shopify"
        assert inv["reglas_url"]["subcarpetas_libres"] is False
        assert inv["urls"] == [f"{BASE}/collections/pediatria", f"{BASE}/collections/cardiologia",
                               f"{BASE}/products/nelson"]
        assert inv["colecciones"] == [
            {"titulo": "Pediatría", "url": f"{BASE}/collections/pediatria", "productos": 62},
            {"titulo": "Cardiología", "url": f"{BASE}/collections/cardiologia", "productos": 80},
        ]
        assert inv["secciones"] == {"collections": 2, "products": 1}
        assert inv["avisos"] == []
        json.dumps(inv)  # se guarda en Firestore: tiene que ser serializable

    async def test_falls_back_to_www(self) -> None:
        """Si el dominio pelado no responde, prueba con www."""
        routes = {
            "https://www.acme.cl/": httpx.Response(200, text="<a href='/seguros/auto'>Auto</a>"),
        }
        inv = await mock_site(routes).explore("acme.cl")
        assert inv["base_url"] == "https://www.acme.cl"
        assert inv["plataforma"] == "desconocida"
        # Sin sitemap: el inventario sale de los enlaces de la portada
        assert inv["urls"] == ["https://www.acme.cl/seguros/auto"]
        assert any("sitemap" in a for a in inv["avisos"])

    async def test_blocked_site_raises(self) -> None:
        """Un 403 de Cloudflare se informa como bloqueo."""
        blocked = httpx.Response(403, text="Attention Required", headers={"cf-ray": "abc"})
        routes = {"https://acme.cl/": blocked, "https://www.acme.cl/": blocked}
        with pytest.raises(SiteUnavailable, match="anti-bots"):
            await mock_site(routes).explore("acme.cl")

    async def test_unreachable_site_raises(self) -> None:
        """Si ninguna variante responde, falla con un motivo."""
        with pytest.raises(SiteUnavailable, match="No pude abrir"):
            await mock_site({}).explore("acme.cl")

    async def test_rate_limit_is_retried_once(self) -> None:
        """Un 429 se reintenta una vez."""
        calls: list[str] = []
        routes = shopify_routes()
        responses = iter([httpx.Response(429, headers={"retry-after": "1"}), routes[f"{BASE}/"]])

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            if str(request.url) == f"{BASE}/":
                return next(responses)
            return routes.get(str(request.url)) or httpx.Response(404)

        async def no_sleep(_seconds: float) -> None:
            return None

        reader = SiteReader(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), sleep=no_sleep)
        inv = await reader.explore("libreria-demo.cl")
        assert inv["plataforma"] == "shopify"
        assert calls.count(f"{BASE}/") == 2

    async def test_shopify_without_collections_json_uses_sitemap(self) -> None:
        """Si /collections.json no responde, las colecciones salen del sitemap."""
        routes = shopify_routes()
        del routes[f"{BASE}/collections.json?limit=250&page=1"]
        inv = await mock_site(routes).explore("libreria-demo.cl")
        assert [c["url"] for c in inv["colecciones"]] == [f"{BASE}/collections/pediatria",
                                                         f"{BASE}/collections/cardiologia"]


class TestSiteReaderReadPage:
    """Lectura de una página puntual."""

    async def test_reads_path(self) -> None:
        """Una ruta se completa con el dominio."""
        page = await mock_site(shopify_routes()).read_page("/collections/pediatria", "libreria-demo.cl")
        assert page["estado_http"] == 200
        assert page["h1"] == ["Colección: Pediatría"]

    async def test_rejects_other_domains(self) -> None:
        """No navega fuera del sitio del cliente."""
        with pytest.raises(ValueError, match="Solo leo"):
            await mock_site(shopify_routes()).read_page("https://otro-sitio.cl/", "libreria-demo.cl")

    async def test_respects_robots(self) -> None:
        """Una ruta prohibida por robots.txt no se lee."""
        with pytest.raises(ValueError, match="robots"):
            await mock_site(shopify_routes()).read_page("/checkout", "libreria-demo.cl")

    async def test_robots_is_cached(self) -> None:
        """robots.txt se pide una sola vez por sitio."""
        calls: list[str] = []
        reader = mock_site(shopify_routes(), calls)
        await reader.read_page("/collections/pediatria", "libreria-demo.cl")
        await reader.read_page("/collections/pediatria", "libreria-demo.cl")
        assert calls.count(f"{BASE}/robots.txt") == 1


# ----------------------------------------------------- flujo con inventario ----

ACME_INVENTORY = {
    "dominio": "acme.cl",
    "base_url": "https://www.acme.cl",
    "plataforma": "shopify",
    "reglas_url": url_rules("shopify"),
    "urls": ["https://www.acme.cl/collections/temuco", "https://www.acme.cl/pages/contacto"],
    "colecciones": [{"titulo": "Seguro auto Temuco", "url": "https://www.acme.cl/collections/temuco",
                     "productos": 3}],
    "secciones": {"collections": 1, "pages": 1},
    "avisos": [],
}


async def test_phase_a_routes_keyword_with_existing_collection_to_optimize(env):  # noqa: F811
    """Una keyword sin tráfico en Search Console pero con colección va a optimizar esa colección."""
    await env["state_store"].save(SID, SITE_INVENTORY_KEY, ACME_INVENTORY)
    state = await orchestrator(env).run_phase_1_and_2(SID, "hilo-sitio")
    assert state.plataforma == "shopify"
    assert "https://www.acme.cl/collections/temuco" in state.existing_urls
    temuco = next(k for k in state.keywords if k["keyword"] == "seguro auto temuco")
    assert temuco["ruta"] == "optimize_existing"
    assert temuco["existing_url"] == "https://www.acme.cl/collections/temuco"
    santiago = next(k for k in state.keywords if k["keyword"] == "seguro auto santiago")
    assert santiago["ruta"] == "create_new_url"


async def test_phase_a_without_inventory_keeps_previous_behavior(env):  # noqa: F811
    """Sin explorar el sitio, el análisis funciona como antes y no hay plataforma."""
    state = await orchestrator(env).run_phase_1_and_2(SID, "hilo-sin-sitio")
    assert state.plataforma is None
    temuco = next(k for k in state.keywords if k["keyword"] == "seguro auto temuco")
    assert temuco["ruta"] == "create_new_url"


async def test_phase_b_shopify_hub_follows_intent(env):  # noqa: F811
    """En Shopify el hub inventado por el modelo se reemplaza por /collections (transaccional)."""
    await env["state_store"].save(SID, SITE_INVENTORY_KEY, ACME_INVENTORY)
    await orchestrator(env).run_phase_1_and_2(SID, "hilo-b")
    env["sheets"].approve("Keywords", "keyword", {"seguro auto santiago": True, "seguro auto concepcion": True})
    state = await orchestrator(env).run_phase_3(SID, "hilo-b")
    assert state.status == RunStatus.WAITING_URL_APPROVAL
    assert sorted(p["url"] for p in state.plan) == [
        "/collections/seguro-auto-concepcion", "/collections/seguro-auto-santiago",
    ]
    assert all(p["slug_valid"] for p in state.plan)
    payload = env["llm"].payloads("PR-02")[0]
    assert payload["plataforma"] == "shopify"
    assert payload["reglas_url"]["hubs_permitidos"] == ["/collections", "/pages"]


# ------------------------------------------------------------ acciones chat ----

async def test_chat_explore_site_saves_inventory(tmp_path):
    """explore_site devuelve un resumen y deja el inventario para el análisis."""
    service, _ = make_service(tmp_path)
    routes = {k.replace(BASE, "https://acme.cl"): v for k, v in shopify_routes().items()}
    routes["https://acme.cl/robots.txt"] = httpx.Response(200, text="Sitemap: https://acme.cl/sitemap.xml\n")
    routes["https://acme.cl/sitemap.xml"] = httpx.Response(
        200, text=sitemap(["https://acme.cl/collections/pediatria", "https://acme.cl/pages/faq"]))
    act = SeoChatActions(service, reader=mock_site(routes))
    result = await act.explore_site(CHAT_SID)
    assert result["ok"] is True
    assert result["plataforma"] == "shopify"
    assert result["urls_en_sitemap"] == 2
    # Ordenadas por cantidad de productos, con rutas y no URLs completas
    assert result["colecciones"][0] == {"titulo": "Cardiología", "url": "/collections/cardiologia", "productos": 80}
    saved = await service.orchestrator.state_store.load(CHAT_SID, SITE_INVENTORY_KEY)
    assert saved["plataforma"] == "shopify"
    overview = await act.overview(CHAT_SID, "hilo")
    assert overview["sitio_explorado"] == {"plataforma": "shopify", "urls": 2, "colecciones": 2}


async def test_chat_explore_site_blocked_explains_next_step(tmp_path):
    """Si el sitio bloquea, la respuesta dice qué hacer en vez de inventar."""
    service, _ = make_service(tmp_path)
    act = SeoChatActions(service, reader=mock_site({}))
    result = await act.explore_site(CHAT_SID)
    assert result["ok"] is False
    assert result["accesible"] is False
    assert "Search Console" in result["que_hacer"]


async def test_chat_read_page_outside_domain(tmp_path):
    """read_page no lee sitios ajenos al cliente."""
    service, _ = make_service(tmp_path)
    act = SeoChatActions(service, reader=mock_site({}))
    result = await act.read_page(CHAT_SID, "https://otro-sitio.cl/")
    assert result == {"ok": False, "error": "Solo leo páginas de acme.cl."}


# ----------------------------------------------------------------- live ----

@pytest.mark.live
async def test_live_site():
    """Explora un sitio real definido en SEO_AGENT_LIVE_SITE (p.ej. una tienda Shopify)."""
    site = os.environ.get("SEO_AGENT_LIVE_SITE")
    if not site:
        pytest.skip("Define SEO_AGENT_LIVE_SITE para correr este test contra un sitio real")
    reader = SiteReader()
    inv = await reader.explore(site)
    assert inv["plataforma"]
    if inv["colecciones"]:
        page = await reader.read_page(inv["colecciones"][0]["url"], site)
        assert page["h1"]
