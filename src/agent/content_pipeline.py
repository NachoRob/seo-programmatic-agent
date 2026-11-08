"""Bucle de calidad por lotes del Flujo 1 (P1-14 a P1-24).

Por cada lote (10 a 20 URLs), en paralelo por fila:
1. PR-03 redacta y P1-16 propone title/meta (validados contra la fila).
2. PR-04 (modelo juez distinto) audita contenido + title + meta seleccionados.
3. P1-18: similitud del texto con los datos de la fila enmascarados contra las
   páginas ya aprobadas del mismo patrón (hub), incluidas las aprobadas antes en
   el mismo lote (orden del plan). Método por defecto `trigramas` (Jaccard de
   3-gramas de palabras); alternativa `embeddings` (coseno). Calibración con 10
   ciudades: trigramas separa páginas buenas (0.12-0.35) de copias (0.67-1.0);
   el coseno de embeddings no las separa (ver scripts/calibrate_similarity.py).
4. P1-19: pasa si `score >= min_judge_score`, `max_similarity < similarity_threshold`
   y los chequeos deterministas (sin voseo en el contenido, title/meta válido).
   Si no, se reintenta con las observaciones del juez, de similitud y de
   validación, hasta `max_retries`. Agotados queda `requiere_revision_humana`.
5. Filas aprobadas: PR-05 + P1-21 (JSON-LD) y PR-06 + P1-23/24 (enlazado).
   Un error en una fila o en un paso no detiene el lote.

`generation_context` (p.ej. `localizacion`, `memoria_cliente` y `marca`) se agrega
a los payloads de PR-03 y P1-16; `localizacion` también a PR-04 y PR-06, y `marca`
a PR-05 (provider/brand del JSON-LD).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.agent.prompts import PR03, PR04, PR05, PR06, TITLE_META
from src.agent.schemas import PR03Output, PR04Output
from src.agent.subagents import Embedder, StructuredLLM
from src.tools.seo_deterministic import (
    compute_cosine_similarity_matrix,
    compute_shingle_similarity_matrix,
    detect_voseo,
    mask_row_values,
    validate_internal_links,
    validate_json_ld_schema,
    validate_title_and_meta,
)

logger = logging.getLogger(__name__)

RowStatus = Literal["aprobada", "requiere_revision_humana", "error"]
MAX_EXISTING_URLS_FOR_LINKS = 100


class PipelineParams(BaseModel):
    min_judge_score: float = Field(default=4, ge=1, le=5)
    similarity_threshold: float = Field(default=0.5, gt=0, le=1)
    similarity_method: Literal["trigramas", "embeddings"] = "trigramas"
    max_retries: int = Field(default=2, ge=0)
    batch_size: int = Field(default=15, ge=10, le=20)

    @classmethod
    def from_sheet(cls, parameters: dict, **overrides: Any) -> PipelineParams:
        """Desde `read_parameters_and_brief()["parameters"]`."""
        parameters = parameters.get("parameters", parameters)
        values = {
            "min_judge_score": parameters.get("min_judge_score"),
            "similarity_threshold": parameters.get("similarity_threshold"),
            **overrides,
        }
        return cls(**{k: v for k, v in values.items() if v is not None})


class PageJob(BaseModel):
    """Una fila aprobada del plan (PR-02 + slug validado + datos propios)."""

    url: str
    keyword: str
    keywords_secundarias: list[str] = []
    hub: str
    fila: dict
    subtemas: list[str] = []
    tipo_schema_sugerido: str | None = None


class AttemptLog(BaseModel):
    intento: int
    score: int | None = None
    max_similarity: float | None = None
    similar_a: str | None = None
    paso: bool = False
    motivo: str = ""


class RowResult(BaseModel):
    url: str
    keyword: str
    hub: str
    status: RowStatus
    intentos: list[AttemptLog] = []
    contenido: PR03Output | None = None
    juicio: PR04Output | None = None
    max_similarity: float | None = None
    similar_a: str | None = None
    title_meta: dict | None = None
    schema_jsonld: dict | None = None
    enlaces: dict | None = None
    issues: list[str] = []
    error: str | None = None


class PipelineReport(BaseModel):
    rows: list[RowResult]
    summary: dict[str, int]


class _RowState:
    def __init__(self, job: PageJob):
        self.job = job
        self.result = RowResult(url=job.url, keyword=job.keyword, hub=job.hub, status="error")
        self.feedback: dict[str, Any] = {}


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


class ContentPipeline:
    def __init__(
        self,
        llm: StructuredLLM,
        embedder: Embedder,
        params: PipelineParams | None = None,
        existing_urls: list[str] | None = None,
        anchors_usados: list[str] | None = None,
        generation_context: dict[str, Any] | None = None,
    ):
        self.llm = llm
        self.embedder = embedder
        self.params = params or PipelineParams()
        self.existing_urls = list(existing_urls or [])
        self.anchors_usados = list(anchors_usados or [])
        self.context = dict(generation_context or {})
        # hub -> [(url, firma)]; la firma es el texto enmascarado o su embedding
        self._approved: dict[str, list[tuple[str, Any]]] = {}
        self._jobs: list[PageJob] = []

    async def run(self, jobs: list[PageJob]) -> PipelineReport:
        self._jobs = jobs
        rows: list[RowResult] = []
        for number, batch in enumerate(_chunks(jobs, self.params.batch_size), start=1):
            logger.info("Lote %d: %d filas", number, len(batch))
            rows.extend(await self._run_batch(batch))
        summary = {"total": len(rows)}
        for status in ("aprobada", "requiere_revision_humana", "error"):
            summary[status] = sum(r.status == status for r in rows)
        return PipelineReport(rows=rows, summary=summary)

    # ------------------------------------------- redacción + title/meta + juez ----

    def _localizacion(self) -> dict | None:
        return self.context.get("localizacion")

    async def _attempt(self, state: _RowState, attempt: int) -> tuple[PR03Output, dict, PR04Output]:
        job = state.job
        payload: dict[str, Any] = {
            **self.context,
            "fila": job.fila,
            "keyword": job.keyword,
            "keywords_secundarias": job.keywords_secundarias,
            "subtemas": job.subtemas,
        }
        if attempt > 1:
            payload["intento"] = attempt
            payload.update(state.feedback)
        content = await self.llm.generate(PR03.id, payload)
        title_meta = await self._title_meta(job, content, state.feedback if attempt > 1 else {})
        selected = title_meta["seleccionada"] or title_meta["variantes"][0]
        judgment = await self.llm.generate(PR04.id, {
            "contenido": content.model_dump(),
            "title": selected["title"],
            "meta_description": selected["meta_description"],
            "fila": job.fila,
            "keyword": job.keyword,
            "subtemas": job.subtemas,
            "localizacion": self._localizacion(),
        })
        return content, title_meta, judgment

    async def _title_meta(self, job: PageJob, content: PR03Output, feedback: dict) -> dict:
        payload: dict[str, Any] = {
            **self.context,
            "keyword": job.keyword, "h1": content.h1, "intro": content.intro, "fila": job.fila,
        }
        if "observaciones_validacion" in feedback:
            payload["observaciones_validacion"] = feedback["observaciones_validacion"]
        output = await self.llm.generate(TITLE_META.id, payload)
        source = json.dumps(job.fila, ensure_ascii=False)
        variants = [
            {**v.model_dump(), **validate_title_and_meta(v.title, v.meta_description, job.keyword, source_text=source)}
            for v in output.variantes
        ]
        return {"variantes": variants, "seleccionada": next((v for v in variants if v["valid"]), None)}

    def _max_similarity(self, hub: str, signature: Any) -> tuple[float, str | None]:
        approved = self._approved.get(hub, [])
        if not approved:
            return 0.0, None
        compare = (compute_shingle_similarity_matrix if self.params.similarity_method == "trigramas"
                   else compute_cosine_similarity_matrix)
        matrix = compare([signature] + [v for _, v in approved])
        sims = matrix[0][1:]
        best = max(range(len(sims)), key=lambda i: sims[i])
        return sims[best], approved[best][0]

    async def _run_batch(self, batch: list[PageJob]) -> list[RowResult]:
        states = [_RowState(job) for job in batch]
        pending = list(states)

        for attempt in range(1, self.params.max_retries + 2):
            outcomes = await asyncio.gather(
                *(self._attempt(s, attempt) for s in pending), return_exceptions=True
            )
            candidates, next_pending = [], []
            for state, outcome in zip(pending, outcomes):
                if isinstance(outcome, BaseException):
                    # Un error de API consume un intento; agotados, la fila queda en "error"
                    logger.warning("Fila %s falló en intento %d: %s", state.job.url, attempt, outcome)
                    state.result.status = "error"
                    state.result.error = f"{type(outcome).__name__}: {outcome}"
                    state.result.intentos.append(AttemptLog(intento=attempt, motivo=f"error: {outcome}"))
                    if attempt <= self.params.max_retries:
                        next_pending.append(state)
                    continue
                candidates.append((state, *outcome))

            # P1-18: se comparan textos con los datos de la fila enmascarados
            masked = [mask_row_values(c.as_text(), s.job.fila) for s, c, _, _ in candidates]
            if self.params.similarity_method == "embeddings" and candidates:
                signatures = await self.embedder.embed(masked)
            else:
                signatures = masked

            for (state, content, title_meta, judgment), vector in zip(candidates, signatures):
                similarity, similar_to = self._max_similarity(state.job.hub, vector)
                voseo = detect_voseo(content.as_text())
                score_ok = judgment.score >= self.params.min_judge_score
                similarity_ok = similarity < self.params.similarity_threshold
                reasons = []
                if not score_ok:
                    reasons.append(f"score {judgment.score} < {self.params.min_judge_score:g}")
                if not similarity_ok:
                    reasons.append(f"similitud {similarity:.3f} >= {self.params.similarity_threshold:g} con {similar_to}")
                if voseo:
                    reasons.append(f"voseo en contenido: {', '.join(voseo)}")
                if title_meta["seleccionada"] is None:
                    reasons.append("sin title/meta válido")

                result = state.result
                result.contenido, result.juicio, result.title_meta = content, judgment, title_meta
                result.max_similarity, result.similar_a = similarity, similar_to
                result.error = None
                result.intentos.append(AttemptLog(
                    intento=attempt, score=judgment.score, max_similarity=similarity,
                    similar_a=similar_to, paso=not reasons, motivo="; ".join(reasons) or "ok",
                ))

                if not reasons:
                    result.status = "aprobada"
                    self._approved.setdefault(state.job.hub, []).append((state.job.url, vector))
                elif attempt <= self.params.max_retries:
                    state.feedback = self._feedback(judgment, similarity_ok, similarity, similar_to, voseo, title_meta)
                    next_pending.append(state)
                else:
                    result.status = "requiere_revision_humana"

            pending = next_pending
            if not pending:
                break

        for state in states:
            if state.result.status == "aprobada":
                await self._post_process(state)
        return [s.result for s in states]

    def _feedback(
        self, judgment: PR04Output, similarity_ok: bool, similarity: float, similar_to: str | None,
        voseo: list[str], title_meta: dict,
    ) -> dict:
        feedback: dict[str, Any] = {
            "observaciones_juez": {
                "score": judgment.score,
                "motivo": judgment.motivo,
                "cifras_no_respaldadas": judgment.cifras_no_respaldadas,
                "frases_genericas": judgment.frases_genericas,
                "cobertura_intencion": judgment.cobertura_intencion,
            }
        }
        if not similarity_ok:
            feedback["observaciones_similitud"] = {
                "similar_a": similar_to,
                "similitud": round(similarity, 3),
                "instruccion": "El texto es casi igual al de otra página del patrón. Diferéncialo con los datos propios de `fila`.",
            }
        validation: dict[str, Any] = {}
        if voseo:
            validation["voseo_en_contenido"] = voseo
        if title_meta["seleccionada"] is None:
            validation["title_meta"] = [
                {"title": v["title"], "meta_description": v["meta_description"], "issues": v["issues"]}
                for v in title_meta["variantes"]
            ]
        if validation:
            feedback["observaciones_validacion"] = validation
        return feedback

    # --------------------------------------------------- pasos posteriores ----

    async def _post_process(self, state: _RowState) -> None:
        try:
            await self._schema(state)
        except Exception as e:  # noqa: BLE001 - el lote sigue
            state.result.issues.append(f"error_schema: {e}")
        try:
            await self._links(state)
        except Exception as e:  # noqa: BLE001
            state.result.issues.append(f"error_enlaces: {e}")

    async def _schema(self, state: _RowState) -> None:
        job, content = state.job, state.result.contenido
        text = content.as_text()
        marca = self.context.get("marca")
        output = await self.llm.generate(PR05.id, {
            "fila": job.fila, "contenido": text, "url": job.url,
            "tipo_sugerido": job.tipo_schema_sugerido, "marca": marca,
        })
        check = validate_json_ld_schema(
            output.jsonld, output.tipo_schema,
            source_text="\n".join([json.dumps(job.fila, ensure_ascii=False), text, marca or ""]),
        )
        state.result.schema_jsonld = {**output.model_dump(), "validacion": check}
        if not check["valid"]:
            state.result.issues.append("schema_invalido")

    async def _links(self, state: _RowState) -> None:
        job, content = state.job, state.result.contenido
        text = content.as_text()
        siblings = [j for j in self._jobs if j.hub == job.hub and j.url != job.url]
        existing = self.existing_urls[:MAX_EXISTING_URLS_FOR_LINKS]
        output = await self.llm.generate(PR06.id, {
            "contenido": text,
            "url_origen": job.url,
            "hub": job.hub,
            "hermanas": [{"url": s.url, "keyword": s.keyword} for s in siblings],
            "urls_existentes": existing,
            "anchors_usados": self.anchors_usados,
            "localizacion": self._localizacion(),
        })
        check = validate_internal_links(
            [e.model_dump() for e in output.enlaces], text, job.url, job.hub,
            [s.url for s in siblings], existing, self.anchors_usados,
        )
        self.anchors_usados.extend(link["anchor"] for link in check["valid_links"])
        state.result.enlaces = {**check, "sin_lugar": output.sin_lugar}
        if not check["cumple_minimos"]:
            state.result.issues.append("enlaces_bajo_minimo")
