"""Sub-agentes ADK del Flujo 1, ejecución estructurada y clustering (P1-06).

- `build_subagent(spec)`: `LlmAgent` con `output_schema`, temperatura y modelo por rol.
- `AdkStructuredLLM`: corre un sub-agente con un payload JSON y valida la salida con Pydantic.
- `GenAIEmbedder`: embeddings de Vertex AI (`text-multilingual-embedding-002`).
- `classify_and_cluster_keywords`: P1-06.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import BaseModel

from src.agent.prompts import INTENT, PROMPTS, PromptSpec
from src.agent.schemas import IntentClassificationOutput
from src.config import Settings, get_settings
from src.tools.seo_deterministic import compute_cosine_similarity_matrix

APP_NAME = "seo_programatico"


class StructuredLLM(Protocol):
    async def generate(self, prompt_id: str, payload: dict) -> BaseModel: ...


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


def configure_vertex_env(settings: Settings | None = None) -> None:
    """ADK y google-genai leen Vertex AI desde el entorno (sin pisar lo que ya exista)."""
    settings = settings or get_settings()
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", settings.vertex_project_id)
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", settings.vertex_location)


def model_for(spec: PromptSpec, settings: Settings) -> str:
    return {"writer": settings.writer_model, "judge": settings.judge_model,
            "extractor": settings.extractor_model}[spec.role]


def build_subagent(spec: PromptSpec, settings: Settings | None = None) -> LlmAgent:
    """`LlmAgent` sin herramientas, con salida JSON forzada por `output_schema`."""
    settings = settings or get_settings()
    instruction = spec.instruction()
    return LlmAgent(
        name=spec.agent_name,
        model=model_for(spec, settings),
        description=f"{spec.id} ({spec.step})",
        # Callable: evita que ADK interprete {variables} del texto como estado
        instruction=lambda _ctx: instruction,
        output_schema=spec.model_schema,
        include_contents="none",
        generate_content_config=types.GenerateContentConfig(temperature=spec.temperature),
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )


class AdkStructuredLLM:
    """Corre un sub-agente por llamada (sesión nueva, sin historial) y valida su JSON."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        configure_vertex_env(self.settings)
        self._runners: dict[str, InMemoryRunner] = {}

    def _runner(self, prompt_id: str) -> InMemoryRunner:
        if prompt_id not in self._runners:
            agent = build_subagent(PROMPTS[prompt_id], self.settings)
            self._runners[prompt_id] = InMemoryRunner(agent=agent, app_name=APP_NAME)
        return self._runners[prompt_id]

    async def generate(self, prompt_id: str, payload: dict) -> BaseModel:
        spec = PROMPTS[prompt_id]
        runner = self._runner(prompt_id)
        session = await runner.session_service.create_session(app_name=APP_NAME, user_id="pipeline")
        message = types.Content(role="user", parts=[types.Part(text=json.dumps(payload, ensure_ascii=False))])

        final_text = None
        async for event in runner.run_async(user_id="pipeline", session_id=session.id, new_message=message):
            if event.is_final_response() and event.content and event.content.parts:
                final_text = "".join(p.text or "" for p in event.content.parts)
        if not final_text:
            raise RuntimeError(f"{prompt_id} no devolvió respuesta")
        parsed = spec.model_schema.model_validate_json(final_text)
        return parsed.to_output() if spec.llm_schema else parsed


class GenAIEmbedder:
    """Embeddings de Vertex AI para similitud semántica."""

    BATCH = 100

    def __init__(self, settings: Settings | None = None, client: Any = None):
        self.settings = settings or get_settings()
        if client is None:
            from google import genai
            client = genai.Client(vertexai=True, project=self.settings.vertex_project_id,
                                  location=self.settings.vertex_location)
        self._client = client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH):
            response = await self._client.aio.models.embed_content(
                model=self.settings.embedding_model,
                contents=texts[i:i + self.BATCH],
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
            )
            vectors.extend(list(e.values) for e in response.embeddings)
        return vectors


# ----------------------------------------------------------------- P1-06 ----

def cluster_by_similarity(keywords: list[str], vectors: list[list[float]], threshold: float) -> list[list[int]]:
    """Clustering determinista por líder: cada keyword (en orden) se une al clúster
    cuyo líder tiene mayor similitud >= `threshold`; si no hay, abre uno nuevo."""
    matrix = compute_cosine_similarity_matrix(vectors)
    leaders: list[int] = []
    clusters: list[list[int]] = []
    for i in range(len(keywords)):
        best, best_sim = None, threshold
        for c, leader in enumerate(leaders):
            if matrix[i][leader] >= best_sim:
                best, best_sim = c, matrix[i][leader]
        if best is None:
            leaders.append(i)
            clusters.append([i])
        else:
            clusters[best].append(i)
    return clusters


async def classify_and_cluster_keywords(
    keywords: list[dict],
    llm: StructuredLLM,
    embedder: Embedder,
    similarity_threshold: float = 0.80,
    batch_size: int = 200,
) -> dict:
    """P1-06: agrupa keywords únicas por similitud coseno y clasifica su intención.

    Ordena por volumen descendente (luego alfabético) para que el líder de cada
    clúster sea la keyword de mayor volumen. La intención sale de la lista
    cerrada informativa/comercial/transaccional/navegacional (LLM a temperatura 0)
    y queda en `intent_p106`; `intent` conserva la intención de entrada si ya
    venía. Una keyword que el modelo no devuelve se reporta en `sin_clasificar`.

    Args:
        keywords: Salida de `normalize_and_deduplicate_keywords` (`keyword`, `search_volume`...).
        llm: Ejecutor estructurado (usa el prompt P1-06).
        embedder: Embeddings para la similitud.
        similarity_threshold: Similitud mínima con el líder para unirse a un clúster.

    Returns:
        {"keywords": [... con `cluster_id` e `intent`], "clusters": [{"cluster_id",
        "lider", "keywords", "volumen_total"}], "sin_clasificar": [...]}
    """
    ordered = sorted(keywords, key=lambda k: (-(k.get("search_volume") or 0), k["keyword"]))
    texts = [k["keyword"] for k in ordered]
    vectors = await embedder.embed(texts)
    groups = cluster_by_similarity(texts, vectors, similarity_threshold)

    intents: dict[str, str] = {}
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        result = await llm.generate(INTENT.id, {"keywords": chunk})
        if not isinstance(result, IntentClassificationOutput):
            raise TypeError(f"P1-06 devolvió {type(result).__name__}")
        for item in result.clasificaciones:
            if item.keyword in chunk:
                intents[item.keyword] = item.intencion

    enriched = [dict(k) for k in ordered]
    clusters = []
    for cluster_id, members in enumerate(groups, start=1):
        for idx in members:
            enriched[idx]["cluster_id"] = cluster_id
            # Una intención ya cargada (p.ej. desde la Sheet) manda sobre la del modelo
            enriched[idx]["intent_p106"] = intents.get(texts[idx])
            enriched[idx]["intent"] = enriched[idx].get("intent") or intents.get(texts[idx])
        clusters.append({
            "cluster_id": cluster_id,
            "lider": texts[members[0]],
            "keywords": [texts[i] for i in members],
            "volumen_total": sum((ordered[i].get("search_volume") or 0) for i in members),
        })

    return {
        "keywords": enriched,
        "clusters": clusters,
        "sin_clasificar": [t for t in texts if t not in intents],
    }
