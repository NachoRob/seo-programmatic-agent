"""Motor determinista de SEO (P1-03, P1-05, P1-08/09, P1-12)."""

import copy

import pytest

from src.tools.seo_deterministic import (
    SEO_DETERMINISTIC_TOOLS,
    compute_cosine_similarity_matrix,
    detect_unsupported_claims,
    detect_voseo,
    validate_internal_links,
    validate_json_ld_schema,
    validate_title_and_meta,
    dedupe_key,
    generate_and_validate_slugs,
    mark_gsc_overlap,
    normalize_and_deduplicate_keywords,
    normalize_text,
    score_and_route_keywords,
    slugify,
)

# ------------------------------------------------------------------- P1-03 ----

def test_normalize_text_lowercase_accents_spaces_keeps_enie():
    assert normalize_text("  Seguro   de AUTO  en Viña del Mar ") == "seguro de auto en viña del mar"
    assert normalize_text("Cotización ¿online?") == "cotizacion online"
    assert normalize_text("año") != normalize_text("ano")


@pytest.mark.parametrize("plural, singular", [
    ("seguros de autos", "seguro de auto"),
    ("hoteles baratos", "hotel barato"),
    ("luces led", "luz led"),
    ("camiones usados", "camion usado"),
    ("ciudades de chile", "ciudad de chile"),
    ("clases de yoga", "clase de yoga"),
])
def test_dedupe_key_unifies_obvious_plurals(plural, singular):
    assert dedupe_key(plural) == dedupe_key(singular)


def test_dedupe_key_does_not_merge_different_terms():
    assert dedupe_key("seguro de vida") != dedupe_key("seguro de auto")
    assert dedupe_key("mes") == "mes"


def test_normalize_and_deduplicate_keeps_originals_and_sources():
    raw = [
        {"keyword": "Seguro de Auto", "source": "planner", "search_volume": 900},
        {"keyword": "seguros  de autos", "source": "gsc", "search_volume": 1200},
        {"keyword": "SEGURO DE AUTO", "source": "semrush", "search_volume": 1200},
        {"keyword": "Seguro de Vida", "source": "planner"},
        {"keyword": "   ", "source": "planner"},
    ]
    result = normalize_and_deduplicate_keywords(raw)

    assert (result["before_count"], result["after_count"], result["empty_removed"]) == (5, 2, 1)
    auto = result["keywords"][0]
    # Se queda la de mayor volumen; empate -> la primera aparición
    assert auto["keyword"] == "seguros de autos"
    assert auto["original_keyword"] == "seguros  de autos"
    assert auto["source"] == "gsc"
    assert auto["sources"] == ["gsc", "planner", "semrush"]
    assert [v["original_keyword"] for v in auto["variants"]] == [
        "Seguro de Auto", "seguros  de autos", "SEGURO DE AUTO",
    ]
    assert result["keywords"][1]["keyword"] == "seguro de vida"


def test_normalize_and_deduplicate_does_not_mutate_input():
    raw = [{"keyword": "Seguro Auto", "source": "x"}]
    snapshot = copy.deepcopy(raw)
    normalize_and_deduplicate_keywords(raw)
    assert raw == snapshot


# ------------------------------------------------------------------- P1-05 ----

GSC_ROWS = [
    # Formato crudo de get_search_analytics (dimensions=[query, page])
    {"keys": ["seguro de auto", "https://acme.cl/seguros/auto"], "clicks": 30, "impressions": 100, "position": 4.0},
    {"keys": ["seguros de auto", "https://acme.cl/blog/auto"], "clicks": 2, "impressions": 100, "position": 8.0},
    {"keys": ["seguro de viaje", "https://acme.cl/viaje"], "clicks": 1, "impressions": 50, "position": 18.5},
    # Formato aplanado
    {"query": "seguro de mascota", "page": "https://acme.cl/mascota", "clicks": 0, "impressions": 10, "position": 45.0},
    {"query": "seguro de hogar", "page": "https://acme.cl/hogar", "clicks": 0, "impressions": 20, "position": 10.0},
]


def _kws(*names):
    return normalize_and_deduplicate_keywords([{"keyword": n} for n in names])["keywords"]


def test_mark_gsc_overlap_classifies_and_aggregates():
    result = {k["keyword"]: k for k in mark_gsc_overlap(
        _kws("seguro de auto", "seguro de viaje", "seguro de mascota", "seguro de hogar", "seguro de vida"),
        GSC_ROWS,
    )}

    auto = result["seguro de auto"]
    assert auto["coverage_status"] == "ya_cubierta"
    assert auto["gsc_clicks"] == 32 and auto["gsc_impressions"] == 200
    assert auto["gsc_position"] == 6.0  # ponderada por impresiones
    assert auto["existing_url"] == "https://acme.cl/seguros/auto"  # la de más clics

    assert result["seguro de hogar"]["coverage_status"] == "ya_cubierta"  # posición 10 exacta
    assert result["seguro de viaje"]["coverage_status"] == "parcial"
    mascota = result["seguro de mascota"]
    assert mascota["coverage_status"] == "libre"
    assert mascota["existing_url"] == "https://acme.cl/mascota"  # se adjunta igual
    vida = result["seguro de vida"]
    assert (vida["coverage_status"], vida["existing_url"], vida["gsc_position"]) == ("libre", None, None)


@pytest.mark.parametrize("position, status", [(10.0, "ya_cubierta"), (10.01, "parcial"), (30.0, "parcial"), (30.01, "libre")])
def test_mark_gsc_overlap_boundaries(position, status):
    rows = [{"query": "x y z", "page": "/p", "clicks": 1, "impressions": 10, "position": position}]
    assert mark_gsc_overlap(_kws("x y z"), rows)[0]["coverage_status"] == status


# ------------------------------------------------------------ P1-08 / 09 ----

def _routed_input():
    return [
        {"keyword": "seguro auto online", "coverage_status": "libre", "search_volume": 1000,
         "difficulty": 20, "intent": "transaccional", "has_own_data": True, "business_value": 5},
        {"keyword": "que es un deducible", "coverage_status": "libre", "search_volume": 10,
         "difficulty": 80, "intent": "informacional", "has_own_data": False, "business_value": 1},
        {"keyword": "seguro de viaje", "coverage_status": "parcial", "search_volume": 500,
         "difficulty": 50, "intent": "comercial", "has_own_data": False, "business_value": 3},
        {"keyword": "seguro de auto", "coverage_status": "ya_cubierta", "search_volume": 5},
    ]


def test_score_is_deterministic_and_weighted():
    params = {"scoring_weights": {"volumen": 1, "dificultad": 1, "intencion": 1, "datos_propios": 1, "valor_negocio": 1}}
    result = score_and_route_keywords(_routed_input(), params)
    top = result["create_new_url"][0]
    assert top["keyword"] == "seguro auto online"
    # volumen 1.0, dificultad 0.8, intención 1.0, datos 1.0, negocio 1.0 -> 96
    assert top["score"] == 96.0
    assert result["weights_used"] == {f: 0.2 for f in result["weights_used"]}
    assert score_and_route_keywords(_routed_input(), params) == result


def test_route_split_and_thresholds():
    params = {"parameters": {"scoring_weights": {}, "min_keyword_score": 40, "min_search_volume": 50}}
    result = score_and_route_keywords(_routed_input(), params)

    assert [k["keyword"] for k in result["create_new_url"]] == ["seguro auto online"]
    assert {k["keyword"] for k in result["optimize_existing"]} == {"seguro de viaje", "seguro de auto"}
    below = result["below_threshold"]
    assert [k["keyword"] for k in below] == ["que es un deducible"]
    assert "volumen 10.0 < 50.0" in below[0]["route_reason"]
    assert result["summary"] == {"total": 4, "create_new_url": 1, "optimize_existing": 2, "below_threshold": 1}


def test_optimize_existing_never_dropped_and_sorted_by_score():
    result = score_and_route_keywords(_routed_input(), {"min_keyword_score": 100})
    assert [k["keyword"] for k in result["optimize_existing"]] == ["seguro de viaje", "seguro de auto"]
    assert result["create_new_url"] == []


def test_missing_fields_are_reported():
    result = score_and_route_keywords(_routed_input(), {})
    cubierta = next(k for k in result["optimize_existing"] if k["keyword"] == "seguro de auto")
    assert set(cubierta["missing_fields"]) == {"difficulty", "intent", "has_own_data", "business_value"}


def test_sheet_weight_aliases_and_ignored_keys():
    params = {"scoring_weights": {"valor_de_negocio": 2, "volumen": 2, "color": 5}}
    result = score_and_route_keywords(_routed_input(), params)
    assert result["weights_used"]["valor_negocio"] == 0.5
    assert result["weights_used"]["dificultad"] == 0.0
    assert result["ignored_weights"] == ["color"]


def test_score_requires_coverage_status():
    with pytest.raises(ValueError, match="mark_gsc_overlap"):
        score_and_route_keywords([{"keyword": "x"}], {})


def test_zero_weights_rejected():
    with pytest.raises(ValueError, match="suman 0"):
        score_and_route_keywords(_routed_input(), {"scoring_weights": {"volumen": 0}})


# ------------------------------------------------------------------- P1-12 ----

def test_slugify():
    assert slugify("Seguro de Auto en Viña del Mar!") == "seguro-de-auto-en-vina-del-mar"
    assert slugify("  --Árbol  ñandú-- ") == "arbol-nandu"


def test_generate_slugs_from_pattern_and_hub():
    tree = [
        {"hub": "Seguros de Auto", "pattern": "seguro de {vehiculo} en {ciudad}",
         "variables": {"vehiculo": "Camioneta", "ciudad": "Viña del Mar"}},
        {"hub": "/seguros/hogar/", "keyword": "Seguro Hogar Ñuñoa"},
    ]
    result = generate_and_validate_slugs(tree, [])
    assert result[0]["url_path"] == "/seguros-de-auto/seguro-de-camioneta-en-vina-del-mar"
    assert result[0]["slug"] == "seguro-de-camioneta-en-vina-del-mar"
    assert result[1]["url_path"] == "/seguros/hogar/seguro-hogar-nunoa"
    assert all(r["slug_valid"] for r in result)


def test_slug_respects_max_length_on_word_boundary():
    tree = [{"hub": "seguros", "keyword": "seguro automotriz con cobertura total para conductores jovenes en santiago"}]
    result = generate_and_validate_slugs(tree, [], max_length=40)[0]
    assert len(result["url_path"]) <= 40
    assert not result["url_path"].endswith("-")
    assert result["url_path"] == "/seguros/seguro-automotriz-con-cobertura"


def test_duplicates_in_plan_and_collisions_with_site():
    tree = [
        {"hub": "seguros", "keyword": "Seguro de Auto"},
        {"hub": "Seguros", "keyword": "seguro de auto"},
        {"hub": "seguros", "keyword": "seguro de viaje"},
        {"hub": "seguros", "keyword": "seguro de vida"},
    ]
    existing = ["https://www.acme.cl/seguros/seguro-de-viaje/", "/Seguros/Seguro-De-Vida?utm=x"]
    result = generate_and_validate_slugs(tree, existing)

    assert result[0]["slug_issues"] == ["duplicado_en_plan"]
    assert result[1]["slug_issues"] == ["duplicado_en_plan"]
    assert result[2]["slug_issues"] == ["colision_url_existente"]
    assert result[3]["slug_issues"] == ["colision_url_existente"]
    assert not any(r["slug_valid"] for r in result)


@pytest.mark.parametrize("hub", ["autos.acme.cl", "https://autos.acme.cl/seguros", ""])
def test_subdomain_or_empty_hub_is_invalid(hub):
    result = generate_and_validate_slugs([{"hub": hub, "keyword": "x"}], ["https://www.acme.cl/"])[0]
    assert "hub_invalido" in result["slug_issues"]


def test_full_url_hub_on_same_host_is_subfolder():
    result = generate_and_validate_slugs([{"hub": "https://acme.cl/seguros/", "keyword": "Seguro Auto"}],
                                         ["https://www.acme.cl/"])[0]
    assert result["url_path"] == "/seguros/seguro-auto"
    assert result["slug_valid"]


def test_missing_pattern_variable_is_flagged():
    tree = [{"hub": "seguros", "pattern": "seguro en {ciudad}", "variables": {}}]
    assert "variable_faltante" in generate_and_validate_slugs(tree, [])[0]["slug_issues"]


def test_hub_longer_than_max_length():
    result = generate_and_validate_slugs([{"hub": "a" * 80, "keyword": "x"}], [], max_length=75)[0]
    assert "excede_largo" in result["slug_issues"]


# ------------------------------------------------- P1-12 con salida PR-02 ----

PR02_PLAN = {
    "hubs": [{"url": "/seguros-auto/", "patron": "seguro de auto en {ciudad}"},
             {"url": "https://www.acme.cl/seguros-moto/", "patron": "seguro de moto {cilindrada}"}],
    "paginas": [
        {"url_propuesta": "/seguros-auto/seguro-de-auto-en-vina-del-mar", "keyword_principal": "seguro auto viña del mar",
         "keywords_secundarias": ["seguros de auto viña"], "hub": "/seguros-auto/", "variables": {"ciudad": "Viña del Mar"}},
        {"url_propuesta": "/seguros-auto/concepcion", "keyword_principal": "seguro auto concepción",
         "keywords_secundarias": [], "hub": "/seguros-auto", "variables": {"ciudad": "Concepción"}},
        {"url_propuesta": "/seguros-moto/250cc", "keyword_principal": "seguro moto 250cc",
         "keywords_secundarias": [], "hub": "https://www.acme.cl/seguros-moto/", "variables": {}},
    ],
}


def test_generate_slugs_accepts_pr02_output():
    result = generate_and_validate_slugs(PR02_PLAN, ["https://www.acme.cl/seguros-auto/seguro-de-auto-en-concepcion"])
    vina, conce, moto = result

    assert vina["url_path"] == "/seguros-auto/seguro-de-auto-en-vina-del-mar"
    assert vina["coincide_con_propuesta"] is True
    assert vina["keywords_secundarias"] == ["seguros de auto viña"]  # se conservan los campos de PR-02
    assert vina["slug_valid"]

    assert conce["url_path"] == "/seguros-auto/seguro-de-auto-en-concepcion"
    assert conce["coincide_con_propuesta"] is False
    assert conce["slug_issues"] == ["colision_url_existente"]

    # Hub con URL completa del mismo sitio: subcarpeta válida; falta la variable del patrón
    assert moto["url_path"].startswith("/seguros-moto/")
    assert moto["slug_issues"] == ["variable_faltante"]


def test_generate_slugs_pr02_and_node_formats_match():
    nodes = [{"hub": "/seguros-auto/", "pattern": "seguro de auto en {ciudad}", "variables": {"ciudad": "Viña del Mar"}}]
    plan = {"hubs": [PR02_PLAN["hubs"][0]], "paginas": [PR02_PLAN["paginas"][0]]}
    assert generate_and_validate_slugs(nodes, [])[0]["url_path"] == generate_and_validate_slugs(plan, [])[0]["url_path"]


# ------------------------------------------------------------------- P1-16 ----

def test_validate_title_and_meta_ok_and_tolerates_plural():
    result = validate_title_and_meta("Seguros de auto en Viña desde $12.990", "Cotiza con 14 talleres.", "seguro de auto")
    assert result == {"valid": True, "issues": [], "title_length": 37, "meta_length": 23}


def test_validate_title_and_meta_issues():
    result = validate_title_and_meta("Cotiza tu seguro de auto " + "x" * 50, "m" * 156, "seguro de auto")
    assert result["issues"] == ["title_largo", "keyword_no_al_inicio", "meta_larga"]
    assert validate_title_and_meta("", "", "x")["issues"] == ["title_vacio", "meta_vacia"]


def test_detect_voseo_distinguishes_tuteo():
    assert detect_voseo("Cotizá hoy. Vos tenés 14 talleres y sos cliente.") == ["cotizá", "vos", "tenés", "sos"]
    assert detect_voseo("Cotiza hoy, tienes 14 talleres. Está listo y es más barato.") == []


def test_unsupported_claims_unless_literal_in_source():
    text = "El mejor seguro, líder del mercado, 100% garantizado y el más barato"
    assert detect_unsupported_claims(text) == ["el mejor", "garantizado", "100%", "lider", "el mas barato", "mas barato"]
    assert detect_unsupported_claims(text, "Somos líder en la región y el mejor evaluado") == [
        "garantizado", "100%", "el mas barato", "mas barato",
    ]
    assert detect_unsupported_claims("Mejores precios de la semana") == []


def test_validate_title_and_meta_against_row_data():
    fila = '{"precio_desde": "$12.990", "talleres": 14, "lema": "Líder en Viña"}'
    ok = validate_title_and_meta("Seguro auto Viña: líder en Viña", "Desde $12.990 con 14 talleres.", "seguro auto viña", fila)
    assert ok["issues"] == []
    bad = validate_title_and_meta("Seguro auto Viña | El mejor precio", "Cotizá con 98% de aprobación, garantizado.",
                                  "seguro auto viña", fila)
    assert bad["issues"] == ["voseo:cotizá", "cifra_no_respaldada:98", "promesa_no_respaldada:el mejor",
                             "promesa_no_respaldada:mejor precio", "promesa_no_respaldada:garantizado"]


# ------------------------------------------------------------------- P1-18 ----

def test_cosine_similarity_matrix():
    matrix = compute_cosine_similarity_matrix([[1, 0], [1, 1], [0, 0]])
    assert matrix[0][0] == 1.0
    assert matrix[0][1] == matrix[1][0] == round(1 / 2 ** 0.5, 6)
    assert matrix[2] == [0.0, 0.0, 0.0]


# ------------------------------------------------------------------- P1-21 ----

FILA_TEXT = "ACME Seguros. Seguro de auto en Viña del Mar desde $12.990"


def test_validate_json_ld_ok():
    jsonld = {"@context": "https://schema.org", "@type": "Service", "name": "Seguro de auto",
              "provider": {"@type": "Organization", "name": "ACME Seguros"}, "url": "/seguros-auto/x"}
    assert validate_json_ld_schema(jsonld, "Service", FILA_TEXT) == {"valid": True, "issues": [], "campos_faltantes": []}


def test_validate_json_ld_detects_invented_and_missing_values():
    jsonld = {"@context": "http://example.com", "@type": "Product", "name": "Seguro de auto",
              "aggregateRating": {"ratingValue": 4.9}}
    result = validate_json_ld_schema(jsonld, "Service", FILA_TEXT)
    assert result["issues"] == ["context_invalido", "tipo_distinto",
                                "campo_obligatorio_faltante:provider", "valor_no_respaldado:ratingValue"]
    assert "provider" in result["campos_faltantes"]


def test_validate_json_ld_source_check_numbers_words_and_codes():
    source = 'fila: {"precio_desde": "$12.990 mensual", "talleres": 14} Seguro de auto con ACME Seguros'
    base = {"@context": "https://schema.org", "@type": "Service", "name": "Seguro de Auto Mensual",
            "provider": {"@type": "Organization", "name": "ACME Seguros"}}
    ok = {**base, "offers": {"@type": "Offer", "price": "12990", "priceCurrency": "CLP",
                             "unitText": "month", "valueAddedTaxIncluded": None},
          "numberOfItems": 14, "isAccessibleForFree": False}
    assert validate_json_ld_schema(ok, "Service", source)["issues"] == []

    invented = {**base, "name": "El mejor seguro de auto", "offers": {"price": 9990},
                "aggregateRating": {"ratingValue": "4,9"}}
    assert validate_json_ld_schema(invented, "Service", source)["issues"] == [
        "valor_no_respaldado:name", "valor_no_respaldado:price", "valor_no_respaldado:ratingValue",
    ]


def test_validate_json_ld_breadcrumb_and_type_list():
    crumb = {"@context": "https://schema.org", "@type": "BreadcrumbList", "itemListElement": [
        {"@type": "ListItem", "position": 1, "name": "Seguros", "item": "/seguros"},
        {"@type": "ListItem", "position": 2, "name": "Auto"},
    ]}
    assert validate_json_ld_schema(crumb, "BreadcrumbList")["issues"] == ["breadcrumb_item_invalido:1"]
    assert "tipo_no_permitido" in validate_json_ld_schema({"@type": "Event"}, "Event")["issues"]


# ------------------------------------------------------------- P1-23 / 24 ----

CONTENT = "Seguro de auto en Viña\n\nEn Viña del Mar el seguro de auto parte en $12.990 con talleres en convenio.\n\nCotiza"
PARA = "En Viña del Mar el seguro de auto parte en $12.990 con talleres en convenio."


def _link(dest, anchor, para=PARA):
    return {"destino": dest, "parrafo_literal": para, "anchor": anchor, "justificacion": "x"}


def test_validate_internal_links_accepts_and_counts():
    links = [
        _link("/seguros-auto", "seguro de auto"),
        _link("/seguros-auto/concepcion", "talleres en convenio"),
        _link("/seguros-auto/santiago", "parte en $12.990"),
    ]
    result = validate_internal_links(links, CONTENT, "/seguros-auto/vina", "/seguros-auto/",
                                     ["/seguros-auto/vina", "/seguros-auto/concepcion", "/seguros-auto/santiago"],
                                     [], [])
    assert [l["tipo_destino"] for l in result["valid_links"]] == ["hub", "hermana", "hermana"]
    assert result["cumple_minimos"] is True


def test_validate_internal_links_rejections():
    links = [
        _link("/seguros-auto", "seguro de auto"),
        _link("/seguros-auto", "talleres en convenio"),              # segundo al hub
        _link("/seguros-auto/vina", "viña del mar"),                  # autoenlace
        _link("/otro", "talleres en convenio"),                       # destino no permitido
        _link("/seguros-auto/conce", "auto", ),                       # anchor de 1 palabra
        _link("/seguros-auto/conce", "cotiza tu seguro"),             # anchor usado + fuera del párrafo
        _link("/seguros-auto/conce", "seguro de auto", para="otro"),  # párrafo no literal + anchor repetido
    ]
    result = validate_internal_links(links, CONTENT, "/seguros-auto/vina", "/seguros-auto",
                                     ["/seguros-auto/conce"], ["/blog/x"], ["Cotiza tu seguro"])
    issues = [r["issues"] for r in result["rejected"]]
    assert issues[0] == ["destino_repetido", "excede_enlaces_hub"]
    assert issues[1] == ["autoenlace"]
    assert issues[2] == ["destino_no_permitido"]
    assert issues[3] == ["anchor_largo_invalido"]
    assert set(issues[4]) == {"anchor_fuera_del_parrafo", "anchor_repetido"}
    assert set(issues[5]) == {"parrafo_no_literal", "anchor_fuera_del_parrafo", "anchor_repetido"}
    assert result["hub_links"] == 1 and result["sibling_links"] == 0
    assert result["cumple_minimos"] is False


def test_validate_internal_links_max_four_siblings():
    words = ["seguro de auto", "talleres en convenio", "parte en $12.990", "viña del mar el", "con talleres en"]
    siblings = [f"/h/s{i}" for i in range(5)]
    links = [_link(siblings[i], words[i]) for i in range(5)]
    result = validate_internal_links(links, CONTENT, "/h/origen", "/h", siblings, [], [])
    assert result["sibling_links"] == 4
    assert result["rejected"][0]["issues"] == ["excede_enlaces_hermanas"]


# ------------------------------------------------------------- pipeline ----

def test_pipeline_end_to_end():
    raw = [
        {"keyword": "Seguros de Auto", "source": "planner", "search_volume": 2000, "difficulty": 60,
         "intent": "comercial", "business_value": 5},
        {"keyword": "seguro de auto", "source": "gsc"},
        {"keyword": "Seguro auto Viña del Mar", "source": "planner", "search_volume": 300, "difficulty": 10,
         "intent": "transaccional", "has_own_data": True, "business_value": 4},
    ]
    keywords = normalize_and_deduplicate_keywords(raw)["keywords"]
    covered = mark_gsc_overlap(keywords, GSC_ROWS)
    routed = score_and_route_keywords(covered, {"min_keyword_score": 30})

    assert [k["keyword"] for k in routed["optimize_existing"]] == ["seguros de auto"]
    tree = [{"hub": "seguros-auto", "keyword": k["keyword"]} for k in routed["create_new_url"]]
    slugs = generate_and_validate_slugs(tree, ["https://acme.cl/seguros/auto"])
    assert [s["url_path"] for s in slugs] == ["/seguros-auto/seguro-auto-vina-del-mar"]


def test_tools_have_adk_declarations():
    names = {t.name for t in SEO_DETERMINISTIC_TOOLS}
    assert names == {"normalize_and_deduplicate_keywords", "mark_gsc_overlap",
                     "score_and_route_keywords", "generate_and_validate_slugs",
                     "validate_title_and_meta", "compute_cosine_similarity_matrix",
                     "validate_json_ld_schema", "validate_internal_links"}
    for tool in SEO_DETERMINISTIC_TOOLS:
        assert tool._get_declaration().parameters_json_schema["required"]
