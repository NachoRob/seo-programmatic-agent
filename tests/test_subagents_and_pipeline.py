"""Sub-agentes, esquemas, P1-06 y bucle de calidad por lotes (P1-14 a P1-24)."""

import json

import pytest
from pydantic import ValidationError

from src.agent.content_pipeline import ContentPipeline, PageJob, PipelineParams
from src.agent.prompts import LOCALIZATION_RULE, PR02, PR03, PR04, PR05, PROMPTS
from src.agent.schemas import (
    EnlacePropuesto,
    FaqItem,
    IntentClassificationOutput,
    IntentItem,
    PR02LLMOutput,
    PR03Output,
    PR04Output,
    PR05LLMOutput,
    PR05Output,
    PR06Output,
    SeccionItem,
    TitleMetaOutput,
    TitleMetaVariant,
)
from src.agent.subagents import build_subagent, classify_and_cluster_keywords, cluster_by_similarity
from src.config import Settings
from src.tools.seo_deterministic import detect_voseo

HUB = "/seguros-auto"


# ------------------------------------------------------------------ fakes ----

def make_content(job_url: str, version: int = 1, voseo: bool = False) -> PR03Output:
    city = job_url.rsplit("/", 1)[-1]
    cta = "Cotizá ahora" if voseo else "Cotiza ahora"
    return PR03Output(
        h1=f"Seguro de auto en {city} v{version}",
        intro=f"En {city} el seguro de auto parte en $12.990 y tiene 14 talleres en convenio.",
        secciones=[SeccionItem(h2="Coberturas", texto=f"Cobertura total para conductores de {city}.")],
        faq=[FaqItem(pregunta="¿Cuánto cuesta?", respuesta="Desde $12.990.")],
        cta=cta,
        bloques_omitidos=[],
    )


def judge(score: int, **kw) -> PR04Output:
    return PR04Output(score=score, motivo=kw.get("motivo", "ok"),
                      cifras_no_respaldadas=kw.get("cifras", []),
                      frases_genericas=kw.get("genericas", []), cobertura_intencion=True)


class FakeLLM:
    """Responde por prompt_id. `judge_scores[url]` = lista de scores por intento."""

    def __init__(self, judge_scores=None, fail_writer_once=(), voseo_first=(), bad_titles_first=()):
        self.calls: list[tuple[str, dict]] = []
        self.judge_scores = judge_scores or {}
        self.fail_writer_once = set(fail_writer_once)
        self.voseo_first = set(voseo_first)
        self.bad_titles_first = set(bad_titles_first)
        self._attempts: dict[str, int] = {}

    def _url_for(self, payload):
        return payload["fila"]["url"]

    async def generate(self, prompt_id, payload):
        self.calls.append((prompt_id, payload))
        if prompt_id == "PR-03":
            url = self._url_for(payload)
            if url in self.fail_writer_once:
                self.fail_writer_once.discard(url)
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            self._attempts[url] = self._attempts.get(url, 0) + 1
            return make_content(url, self._attempts[url], voseo=url in self.voseo_first and self._attempts[url] == 1)
        if prompt_id == "PR-04":
            url = self._url_for(payload)
            scores = self.judge_scores.get(url, [5])
            score = scores[min(self._attempts[url], len(scores)) - 1]
            return judge(score, cifras=["99% de satisfacción"] if score < 4 else [],
                         genericas=["el mejor seguro del mercado"] if score < 4 else [])
        if prompt_id == "P1-16":
            kw = payload["keyword"]
            url = payload["fila"]["url"]
            if url in self.bad_titles_first and self._attempts[url] == 1:
                return TitleMetaOutput(variantes=[
                    TitleMetaVariant(title=f"{kw} | El mejor precio", meta_description="Garantizado"),
                    TitleMetaVariant(title=f"{kw} con 99% de aprobación", meta_description="x"),
                    TitleMetaVariant(title=f"{kw}: contratá hoy", meta_description="x"),
                ])
            return TitleMetaOutput(variantes=[
                TitleMetaVariant(title="Cotiza hoy: " + kw, meta_description="x"),  # keyword no al inicio
                TitleMetaVariant(title=f"{kw.capitalize()} desde $12.990", meta_description="Desde $12.990 con 14 talleres."),
                TitleMetaVariant(title=kw + " " + "x" * 80, meta_description="y"),  # muy largo
            ])
        if prompt_id == "PR-05":
            fila = payload["fila"]
            return PR05Output(tipo_schema="Service", campos_faltantes=[], jsonld={
                "@context": "https://schema.org", "@type": "Service",
                "name": "Seguro de auto", "provider": {"@type": "Organization", "name": fila["empresa"]},
            })
        if prompt_id == "PR-06":
            intro = payload["contenido"].split("\n\n")[1]
            hermanas = payload["hermanas"]
            enlaces = [EnlacePropuesto(destino=payload["hub"], parrafo_literal=intro,
                                       anchor="seguro de auto", justificacion="hub")]
            enlaces += [EnlacePropuesto(destino=h["url"], parrafo_literal="párrafo inventado",
                                        anchor="talleres en convenio", justificacion="x") for h in hermanas[:1]]
            return PR06Output(enlaces=enlaces, sin_lugar=[])
        raise AssertionError(prompt_id)

    def payloads(self, prompt_id):
        return [p for pid, p in self.calls if pid == prompt_id]


class FakeEmbedder:
    """Vector por h1 (primera línea). `same_as` fuerza textos idénticos."""

    def __init__(self, same_as=None):
        self.same_as = same_as or {}
        self.calls = 0
        self._ids: dict[str, int] = {}

    async def embed(self, texts):
        self.calls += 1
        vectors = []
        for text in texts:
            h1 = text.split("\n\n")[0]
            key = self.same_as.get(h1, h1)
            idx = self._ids.setdefault(key, len(self._ids))
            vec = [0.0] * 64
            vec[idx] = 1.0
            vectors.append(vec)
        return vectors


def job(city, hub=HUB):
    url = f"{hub}/seguro-auto-{city}"
    return PageJob(url=url, keyword=f"seguro auto {city}", hub=hub,
                   fila={"url": url, "ciudad": city, "precio_desde": "$12.990",
                         "talleres_convenio": 14, "empresa": "ACME Seguros"})


def pipeline(llm, embedder=None, **params):
    # Los tests de similitud con FakeEmbedder usan el método embeddings
    params.setdefault("similarity_method", "embeddings")
    params.setdefault("similarity_threshold", 0.9)
    return ContentPipeline(llm, embedder or FakeEmbedder(), PipelineParams(**params),
                           existing_urls=["/seguros/auto"], anchors_usados=["cotiza tu seguro"])


# ------------------------------------------------------- settings / agentes ----

def test_judge_must_differ_from_writer():
    with pytest.raises(ValidationError, match="juez nunca"):
        Settings(writer_model="gemini-2.5-flash", judge_model="gemini-2.5-flash")


def test_subagents_models_temperatures_and_schemas():
    settings = Settings()
    expected = {"PR-01": 0.4, "PR-02": 0.3, "PR-03": 0.6, "PR-04": 0.0, "PR-05": 0.0, "PR-06": 0.2}
    for prompt_id, temp in expected.items():
        agent = build_subagent(PROMPTS[prompt_id], settings)
        assert agent.generate_content_config.temperature == temp
        assert agent.output_schema is PROMPTS[prompt_id].model_schema
    assert build_subagent(PR03, settings).model == settings.writer_model
    assert build_subagent(PR04, settings).model == settings.judge_model
    assert build_subagent(PR04, settings).model != build_subagent(PR03, settings).model
    assert build_subagent(PR02, settings).output_schema is PR02LLMOutput
    assert build_subagent(PR05, settings).output_schema is PR05LLMOutput


def test_instruction_contains_official_system_prompt_and_is_not_templated():
    agent = build_subagent(PR03, Settings())
    text = agent.instruction(None)
    assert text.startswith("Eres redactor SEO. Escribes solo con los datos que recibes.")
    assert "observaciones_juez" in text
    # PR-02 tiene llaves literales: con instruction callable ADK no las interpola
    assert "{vehiculo}" in build_subagent(PR02, Settings()).instruction(None)


def test_prompts_are_neutral_spanish_without_voseo():
    for spec in PROMPTS.values():
        text = (spec.system + spec.rules).replace(LOCALIZATION_RULE, "")
        assert detect_voseo(text) == [], spec.id


def test_generation_prompts_carry_localization_rule():
    for prompt_id in ("PR-03", "P1-16", "PR-06"):
        assert LOCALIZATION_RULE in PROMPTS[prompt_id].rules
        assert "voseo" in PROMPTS[prompt_id].rules


def test_wire_schemas_convert_to_public_contract():
    pr02 = PR02LLMOutput.model_validate({
        "hubs": [{"url": "/seguros-auto/", "patron": "seguro auto {ciudad}"}],
        "paginas": [{"url_propuesta": "/seguros-auto/seguro-auto-vina", "keyword_principal": "seguro auto viña",
                     "keywords_secundarias": [], "hub": "/seguros-auto/",
                     "variables": [{"nombre": "ciudad", "valor": "Viña del Mar"}]}],
    }).to_output()
    assert pr02.paginas[0].variables == {"ciudad": "Viña del Mar"}

    pr05 = PR05LLMOutput(tipo_schema="Article", jsonld_json='{"@type": "Article", "headline": null}',
                         campos_faltantes=["headline"]).to_output()
    assert pr05.jsonld == {"@type": "Article", "headline": None}
    with pytest.raises(ValueError):
        PR05LLMOutput(tipo_schema="Article", jsonld_json="[1]", campos_faltantes=[]).to_output()


def test_schema_constraints():
    with pytest.raises(ValidationError):
        PR04Output(score=6, motivo="", cifras_no_respaldadas=[], frases_genericas=[], cobertura_intencion=True)
    with pytest.raises(ValidationError):
        TitleMetaOutput(variantes=[TitleMetaVariant(title="a", meta_description="b")] * 2)
    with pytest.raises(ValidationError):
        make_content("x").model_copy(update={"faq": []}).model_validate(
            {**make_content("x").model_dump(), "faq": [{"pregunta": "p", "respuesta": "r"}] * 5})


def test_params_from_sheet_and_batch_bounds():
    params = PipelineParams.from_sheet({"parameters": {"min_judge_score": 3, "similarity_threshold": 0.4}})
    assert (params.min_judge_score, params.similarity_threshold, params.max_retries) == (3, 0.4, 2)
    assert PipelineParams().similarity_method == "trigramas"
    with pytest.raises(ValidationError):
        PipelineParams(batch_size=5)
    with pytest.raises(ValidationError):
        PipelineParams(batch_size=25)


# --------------------------------------------------------------- pipeline ----

async def test_happy_path_runs_all_steps():
    llm = FakeLLM()
    report = await pipeline(llm).run([job("vina"), job("concepcion")])

    assert report.summary == {"total": 2, "aprobada": 2, "requiere_revision_humana": 0, "error": 0}
    row = report.rows[0]
    assert row.status == "aprobada" and len(row.intentos) == 1
    # P1-16: se elige la primera variante válida
    assert row.title_meta["seleccionada"]["title"] == "Seguro auto vina desde $12.990"
    assert row.title_meta["variantes"][0]["issues"] == ["keyword_no_al_inicio"]
    # PR-05 + P1-21
    assert row.schema_jsonld["validacion"]["valid"]
    # PR-06 + P1-23/24: el enlace al hub es literal; el de hermana no
    assert [l["tipo_destino"] for l in row.enlaces["valid_links"]] == ["hub"]
    assert row.enlaces["rejected"][0]["issues"][0] == "parrafo_no_literal"
    assert "enlaces_bajo_minimo" in row.issues
    # El juez nunca recibe el payload del redactor sin el contenido generado
    assert llm.payloads("PR-04")[0]["contenido"]["h1"].startswith("Seguro de auto en")


async def test_retry_with_judge_feedback_when_score_below_minimum():
    url = job("vina").url
    llm = FakeLLM(judge_scores={url: [2, 5]})
    report = await pipeline(llm).run([job("vina")])

    row = report.rows[0]
    assert row.status == "aprobada"
    assert [a.score for a in row.intentos] == [2, 5]
    assert row.intentos[0].motivo == "score 2 < 4"
    retry_payload = llm.payloads("PR-03")[1]
    assert retry_payload["intento"] == 2
    assert retry_payload["observaciones_juez"]["cifras_no_respaldadas"] == ["99% de satisfacción"]
    assert retry_payload["observaciones_juez"]["frases_genericas"] == ["el mejor seguro del mercado"]


async def test_exhausted_retries_mark_human_review_without_stopping_batch():
    bad = job("vina")
    llm = FakeLLM(judge_scores={bad.url: [1, 2, 3]})
    report = await pipeline(llm, max_retries=2).run([bad, job("concepcion")])

    failed, ok = report.rows
    assert failed.status == "requiere_revision_humana"
    assert len(failed.intentos) == 3 and failed.contenido is not None
    assert failed.schema_jsonld is None and failed.enlaces is None  # sin pasos posteriores
    assert ok.status == "aprobada"
    assert [p["url"] for p in llm.payloads("PR-05")] == [ok.url]


async def test_judge_audits_selected_title_and_meta_with_context():
    llm = FakeLLM()
    context = {"localizacion": {"pais": "Chile", "idioma": "es-CL", "trato": "tuteo"},
               "memoria_cliente": ["No usar signos de exclamación"]}
    p = ContentPipeline(llm, FakeEmbedder(), PipelineParams(), generation_context=context)
    await p.run([job("vina")])

    judge_payload = llm.payloads("PR-04")[0]
    assert judge_payload["title"] == "Seguro auto vina desde $12.990"
    assert judge_payload["meta_description"] == "Desde $12.990 con 14 talleres."
    assert judge_payload["localizacion"]["idioma"] == "es-CL"
    for prompt_id in ("PR-03", "P1-16"):
        assert llm.payloads(prompt_id)[0]["localizacion"]["pais"] == "Chile"
        assert llm.payloads(prompt_id)[0]["memoria_cliente"] == ["No usar signos de exclamación"]
    assert llm.payloads("PR-06")[0]["localizacion"]["trato"] == "tuteo"


async def test_brand_from_context_reaches_schema_and_validation():
    class BrandLLM(FakeLLM):
        async def generate(self, prompt_id, payload):
            if prompt_id == "PR-05":
                self.calls.append((prompt_id, payload))
                return PR05Output(tipo_schema="Service", campos_faltantes=[], jsonld={
                    "@context": "https://schema.org", "@type": "Service", "name": "Seguro de auto",
                    "provider": {"@type": "Organization", "name": payload["marca"]},
                })
            return await super().generate(prompt_id, payload)

    llm = BrandLLM()
    p = ContentPipeline(llm, FakeEmbedder(), PipelineParams(), generation_context={"marca": "Seguros Andes"})
    row = (await p.run([job("vina")])).rows[0]
    assert llm.payloads("PR-05")[0]["marca"] == "Seguros Andes"
    assert row.schema_jsonld["validacion"]["valid"] is True  # la marca cuenta como fuente


async def test_voseo_in_content_forces_retry_with_validation_feedback():
    target = job("vina")
    llm = FakeLLM(voseo_first=[target.url])
    report = await pipeline(llm).run([target])
    row = report.rows[0]
    assert row.status == "aprobada"
    assert row.intentos[0].motivo == "voseo en contenido: cotizá"
    assert llm.payloads("PR-03")[1]["observaciones_validacion"]["voseo_en_contenido"] == ["cotizá"]


async def test_unsupported_title_claims_force_retry():
    target = job("vina")
    llm = FakeLLM(bad_titles_first=[target.url])
    report = await pipeline(llm).run([target])
    row = report.rows[0]
    assert row.intentos[0].motivo == "sin title/meta válido"
    issues = [v["issues"] for v in llm.payloads("PR-03")[1]["observaciones_validacion"]["title_meta"]]
    assert "promesa_no_respaldada:el mejor" in issues[0]
    assert "promesa_no_respaldada:garantizado" in issues[0]
    assert "cifra_no_respaldada:99" in issues[1]
    assert "voseo:contratá" in issues[2]
    assert llm.payloads("P1-16")[1]["observaciones_validacion"]["title_meta"]
    assert row.status == "aprobada"


async def test_retry_when_similarity_exceeds_threshold():
    first, second = job("vina"), job("concepcion")
    # El primer intento de "concepcion" es idéntico a "vina"
    embedder = FakeEmbedder(same_as={"Seguro de auto en seguro-auto-concepcion v1": "Seguro de auto en seguro-auto-vina v1"})
    llm = FakeLLM()
    report = await pipeline(llm, embedder).run([first, second])

    row = report.rows[1]
    assert row.status == "aprobada"
    assert row.intentos[0].paso is False
    assert row.intentos[0].max_similarity == 1.0
    assert row.intentos[0].similar_a == first.url
    retry = [p for p in llm.payloads("PR-03") if p["keyword"] == second.keyword][1]
    assert retry["observaciones_similitud"]["similar_a"] == first.url
    assert row.intentos[1].max_similarity == 0.0


@pytest.mark.real_masking
async def test_trigram_similarity_catches_structural_copies():
    """Con trigramas sobre texto enmascarado, una copia que solo cambia datos se reintenta."""
    llm = FakeLLM()
    p = ContentPipeline(llm, FakeEmbedder(), PipelineParams())  # defaults: trigramas, 0.5
    report = await p.run([job("vina"), job("concepcion")])
    second = report.rows[1]
    assert p.params.similarity_method == "trigramas" and p.params.similarity_threshold == 0.5
    # El FakeLLM escribe ambas ciudades con la misma plantilla: es una copia estructural
    assert second.intentos[0].paso is False and second.intentos[0].max_similarity > 0.9
    assert second.intentos[0].similar_a == job("vina").url
    assert "observaciones_similitud" in [p for p in llm.payloads("PR-03") if p["keyword"] == "seguro auto concepcion"][1]


@pytest.mark.real_masking
async def test_similarity_compares_masked_text():
    """P1-18 embebe el texto con los datos de la fila enmascarados."""
    class RecordingEmbedder(FakeEmbedder):
        def __init__(self):
            super().__init__()
            self.texts = []

        async def embed(self, texts):
            self.texts += texts
            return await super().embed(texts)

    embedder = RecordingEmbedder()
    await pipeline(FakeLLM(), embedder, similarity_method="embeddings").run([job("vina")])
    text = embedder.texts[0]
    assert "[VAR_CIUDAD]" in text and "[VAR_PRECIO_DESDE]" in text and "[VAR_TALLERES_CONVENIO]" in text
    assert "vina" not in text.lower().replace("[var_", "")


async def test_similarity_only_compares_same_pattern():
    a, b = job("vina", hub="/seguros-auto"), job("vina", hub="/seguros-moto")
    embedder = FakeEmbedder(same_as={})  # mismo h1 -> mismo vector
    report = await pipeline(FakeLLM(), embedder).run([a, b])
    assert [r.status for r in report.rows] == ["aprobada", "aprobada"]
    assert report.rows[1].max_similarity == 0.0


async def test_api_error_consumes_attempt_and_recovers():
    flaky = job("vina")
    llm = FakeLLM(fail_writer_once=[flaky.url])
    report = await pipeline(llm).run([flaky])
    row = report.rows[0]
    assert row.status == "aprobada" and row.error is None
    assert row.intentos[0].motivo.startswith("error: 429")


async def test_batches_respect_batch_size():
    jobs = [job(f"c{i}") for i in range(25)]
    embedder = FakeEmbedder()
    report = await pipeline(FakeLLM(), embedder, batch_size=10).run(jobs)
    assert report.summary["aprobada"] == 25
    assert embedder.calls == 3  # 10 + 10 + 5, un intento por lote


async def test_anchors_are_not_reused_across_rows():
    llm = FakeLLM()
    p = pipeline(llm)
    report = await p.run([job("vina"), job("concepcion")])
    assert "seguro de auto" in p.anchors_usados
    second = report.rows[1]
    assert second.enlaces["valid_links"] == []
    assert "anchor_repetido" in second.enlaces["rejected"][0]["issues"]


# ------------------------------------------------------------------- P1-06 ----

class IntentLLM:
    async def generate(self, prompt_id, payload):
        assert prompt_id == "P1-06"
        mapping = {"seguro auto": "comercial", "cotizar seguro auto": "transaccional",
                   "que es un deducible": "informativa"}
        return IntentClassificationOutput(clasificaciones=[
            IntentItem(keyword=k, intencion=mapping[k]) for k in payload["keywords"] if k in mapping
        ])


class VectorEmbedder:
    def __init__(self, vectors):
        self.vectors = vectors

    async def embed(self, texts):
        return [self.vectors[t] for t in texts]


async def test_classify_and_cluster_keywords():
    keywords = [
        {"keyword": "que es un deducible", "search_volume": 50},
        {"keyword": "seguro auto", "search_volume": 1000},
        {"keyword": "cotizar seguro auto", "search_volume": 300},
        {"keyword": "seguro moto", "search_volume": 100},
    ]
    vectors = {
        "seguro auto": [1.0, 0.0, 0.0],
        "cotizar seguro auto": [0.95, 0.1, 0.0],
        "seguro moto": [0.5, 0.0, 0.8],
        "que es un deducible": [0.0, 1.0, 0.0],
    }
    result = await classify_and_cluster_keywords(keywords, IntentLLM(), VectorEmbedder(vectors), 0.8)

    assert [c["keywords"] for c in result["clusters"]] == [
        ["seguro auto", "cotizar seguro auto"], ["seguro moto"], ["que es un deducible"],
    ]
    assert result["clusters"][0] == {"cluster_id": 1, "lider": "seguro auto",
                                     "keywords": ["seguro auto", "cotizar seguro auto"], "volumen_total": 1300}
    by_kw = {k["keyword"]: k for k in result["keywords"]}
    assert by_kw["cotizar seguro auto"]["intent"] == "transaccional"
    assert by_kw["que es un deducible"]["intent"] == "informativa"
    assert result["sin_clasificar"] == ["seguro moto"]


def test_cluster_by_similarity_is_deterministic():
    vectors = [[1, 0], [0.99, 0.1], [0, 1]]
    assert cluster_by_similarity(["a", "b", "c"], vectors, 0.9) == [[0, 1], [2]]
    assert cluster_by_similarity(["a", "b", "c"], vectors, 0.9) == cluster_by_similarity(["a", "b", "c"], vectors, 0.9)


def test_intent_schema_rejects_values_outside_closed_list():
    with pytest.raises(ValidationError):
        IntentItem(keyword="x", intencion="informacional")


# ------------------------------------------------------------------- live ----

@pytest.mark.live
async def test_live_pipeline_single_row():
    from src.agent.subagents import AdkStructuredLLM, GenAIEmbedder

    settings = Settings()
    row = PageJob(
        url="/seguros-auto/seguro-auto-vina-del-mar",
        keyword="seguro auto viña del mar",
        hub="/seguros-auto",
        fila={"ciudad": "Viña del Mar", "precio_desde": "$12.990 mensual",
              "talleres_convenio": 14, "tiempo_respuesta_siniestro": "48 horas", "empresa": "ACME Seguros"},
        subtemas=["coberturas", "talleres", "precio"],
        tipo_schema_sugerido="Service",
    )
    from src.agent.orchestrator import localization_from_brief

    context = {"localizacion": localization_from_brief({}, settings).model_dump(), "memoria_cliente": []}
    p = ContentPipeline(AdkStructuredLLM(settings), GenAIEmbedder(settings), PipelineParams(max_retries=1),
                        generation_context=context)
    report = await p.run([row])
    result = report.rows[0]
    print("\n", json.dumps(result.model_dump(include={"status", "intentos", "title_meta", "issues"}),
                           ensure_ascii=False, indent=1)[:2500])
    assert result.status in ("aprobada", "requiere_revision_humana")
    assert result.juicio is not None and 1 <= result.juicio.score <= 5
    assert detect_voseo(result.contenido.as_text()) == []
    print("\nINTRO:", result.contenido.intro, "\nCTA:", result.contenido.cta)
