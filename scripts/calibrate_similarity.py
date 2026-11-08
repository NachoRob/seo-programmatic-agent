"""Calibra `umbral_similitud` (P1-19) con contenido real de PR-03.

Genera una página por ciudad (un intento, sin juez), calcula la similitud coseno
entre todas las parejas y simula cuántas páginas pasarían P1-19 con distintos
umbrales (cada página contra las anteriores del mismo patrón, en orden).

    uv run python scripts/calibrate_similarity.py
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.orchestrator import localization_from_brief  # noqa: E402
from src.agent.subagents import AdkStructuredLLM, GenAIEmbedder  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.tools.seo_deterministic import (  # noqa: E402
    compute_cosine_similarity_matrix,
    compute_shingle_similarity_matrix,
    mask_row_values,
)

# Datos ficticios (cliente de prueba)
CITIES = [
    ("Santiago", "$13.990", 41, "24 horas", "Las Condes y Providencia", "robo de espejos", 12, "Región Metropolitana"),
    ("Viña del Mar", "$12.990", 14, "48 horas", "Reñaca y Miraflores", "choques en estacionamientos", 4, "Valparaíso"),
    ("Concepción", "$11.490", 9, "72 horas", "Talcahuano y San Pedro de la Paz", "daños por lluvia", 3, "Biobío"),
    ("Antofagasta", "$14.490", 7, "48 horas", "Centro y La Portada", "rotura de parabrisas por piedras", 2, "Antofagasta"),
    ("Temuco", "$10.990", 6, "72 horas", "Padre Las Casas", "deslizamientos en caminos rurales", 2, "La Araucanía"),
    ("La Serena", "$12.490", 8, "48 horas", "Coquimbo y Las Compañías", "choques por alcance en Avenida del Mar", 3, "Coquimbo"),
    ("Puerto Montt", "$12.790", 5, "96 horas", "Alerce y Puerto Varas", "daños por temporales", 1, "Los Lagos"),
    ("Rancagua", "$11.990", 6, "48 horas", "Machalí", "robo de neumáticos", 2, "O'Higgins"),
    ("Talca", "$11.290", 5, "72 horas", "Maule y San Clemente", "colisiones en Ruta 5", 2, "Maule"),
    ("Iquique", "$14.990", 4, "96 horas", "Alto Hospicio", "daños por arena y sal", 1, "Tarapacá"),
]
THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.92, 0.93, 0.95, 0.97, 0.98, 0.99)


def rows(rich: bool) -> list[dict]:
    out = []
    for city, price, shops, response, zones, claim, branches, region in CITIES:
        fila = {"ciudad": city, "precio_desde": f"{price} mensual", "talleres_convenio": shops,
                "tiempo_respuesta_siniestro": response}
        if rich:
            fila |= {"comunas_cubiertas": zones, "siniestro_mas_frecuente": claim,
                     "sucursales": branches, "region": region}
        out.append(fila)
    return out


def clone(text: str, src: dict, dst: dict, only_city: bool) -> str:
    """Copia mala: el texto de `src` con los datos de `dst` (solo ciudad o todo)."""
    out = text.replace(src["ciudad"], dst["ciudad"])
    if not only_city:
        for key in ("precio_desde", "talleres_convenio", "tiempo_respuesta_siniestro"):
            out = out.replace(str(src[key]).split(" ")[0], str(dst[key]).split(" ")[0])
    return out


def stats(name: str, vectors: list[list[float]], n: int, matrix: list[list[float]] | None = None) -> dict:
    m = matrix or compute_cosine_similarity_matrix(vectors)
    pairs = [m[i][j] for i in range(n) for j in range(i + 1, n)]
    # P1-19: cada página contra las anteriores aprobadas (sin reintentos)
    sim_vs_previous = [max(m[i][:i]) if i else 0.0 for i in range(n)]
    sim = {}
    for t in THRESHOLDS:
        approved: list[int] = []
        for i in range(n):
            if all(m[i][j] < t for j in approved):
                approved.append(i)
        sim[t] = len(approved)
    return {
        "modo": name,
        "pares": {"min": min(pairs), "p50": statistics.median(pairs),
                  "p90": statistics.quantiles(pairs, n=10)[-1], "max": max(pairs)},
        "max_vs_anteriores": [round(s, 3) for s in sim_vs_previous],
        "aprobadas_por_umbral": sim,
    }


async def scenario(name: str, filas: list[dict], llm, embedder, context) -> dict:
    contents = await asyncio.gather(*(
        llm.generate("PR-03", {**context, "fila": f, "keyword": f"seguro auto {f['ciudad'].lower()}",
                               "keywords_secundarias": [], "subtemas": ["coberturas", "talleres", "precio"]})
        for f in filas
    ))
    texts = [c.as_text() for c in contents]
    masked = [mask_row_values(t, f) for t, f in zip(texts, filas)]
    # Copias malas de la página 0 hacia cada otra ciudad
    clones = [(clone(texts[0], filas[0], f, only), f) for f in filas[1:] for only in (False, True)]
    raw_v = await embedder.embed(texts + [c for c, _ in clones])
    mask_v = await embedder.embed(masked + [mask_row_values(c, f) for c, f in clones])
    n = len(filas)

    def clone_sims(vectors):
        sims = compute_cosine_similarity_matrix([vectors[0]] + vectors[n:])[0][1:]
        return {"min": min(sims), "max": max(sims)}

    clone_masked = [mask_row_values(c, f) for c, f in clones]
    shingle = compute_shingle_similarity_matrix(masked + clone_masked)
    shingle_clone = shingle[0][n:]

    return {
        "escenario": name,
        "palabras_promedio": round(statistics.mean(len(t.split()) for t in texts)),
        "trigramas_enmascarado": {**stats("trigramas", [], n, [row[:n] for row in shingle[:n]]),
                                  "copias_malas": {"min": min(shingle_clone), "max": max(shingle_clone)}},
        "sin_enmascarar": {**stats("sin_enmascarar", raw_v[:n], n), "copias_malas": clone_sims(raw_v)},
        "enmascarado": {**stats("enmascarado", mask_v[:n], n), "copias_malas": clone_sims(mask_v)},
    }


async def main() -> None:
    settings = get_settings()
    llm, embedder = AdkStructuredLLM(settings), GenAIEmbedder(settings)
    context = {"localizacion": localization_from_brief({}, settings).model_dump(), "memoria_cliente": [],
               "marca": "ACME Seguros"}
    results = await asyncio.gather(
        scenario("datos_pobres (4 columnas)", rows(False), llm, embedder, context),
        scenario("datos_ricos (8 columnas)", rows(True), llm, embedder, context),
    )
    print(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
