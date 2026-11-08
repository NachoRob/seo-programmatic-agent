"""Línea de comandos del agente de SEO programático.

    uv run python -m src.cli avanzar <sheet>   # corre la etapa que toca y se detiene en la próxima pausa
    uv run python -m src.cli estado <sheet>    # muestra en qué etapa está el run
    uv run python -m src.cli reiniciar <sheet> # empieza un run nuevo (Etapa A) en un hilo nuevo

<sheet> puede ser el ID o la URL completa de la Google Sheet. `--hilo` separa runs
sobre la misma Sheet (por defecto "default").
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import time

from src.agent.orchestrator import RunStatus, SeoProgrammaticOrchestrator
from src.agent.subagents import AdkStructuredLLM, GenAIEmbedder
from src.config import get_settings

NEXT_STEP = {
    RunStatus.WAITING_KEYWORD_APPROVAL: "Marca Aprobado = TRUE en la pestaña Keywords y vuelve a correr `avanzar`.",
    RunStatus.WAITING_URL_APPROVAL: "Marca Aprobado = TRUE en la pestaña Plan de URLs y vuelve a correr `avanzar`.",
    RunStatus.DONE: "Run terminado: revisa las pestañas Resumen, Contenido, On-page, Schema y Enlaces internos.",
}


def sheet_id(value: str) -> str:
    match = re.search(r"/spreadsheets/d/([A-Za-z0-9_-]+)", value)
    return match.group(1) if match else value


def print_state(state) -> None:
    if state is None:
        print("Sin run para esta Sheet/hilo. Usa `avanzar` para empezar.")
        return
    print(f"Run {state.run_id} | estado: {state.status.value} | dominio: {state.dominio}")
    if state.last_error:
        print(f"Último error: {state.last_error}")
    if state.routing_summary:
        print(f"Keywords: {state.routing_summary}")
    if state.plan:
        print(f"URLs planificadas: {len(state.plan)}")
    if state.pipeline_summary:
        print(f"Contenido: {state.pipeline_summary}")
    if state.status in NEXT_STEP:
        print(f"\nSiguiente paso: {NEXT_STEP[state.status]}")


async def run(args) -> None:
    settings = get_settings()
    orchestrator = SeoProgrammaticOrchestrator(AdkStructuredLLM(settings), GenAIEmbedder(settings), settings=settings)
    sid = sheet_id(args.sheet)
    if args.comando == "estado":
        print_state(await orchestrator.get_state(sid, args.hilo))
        return
    if args.comando == "reiniciar":
        args.hilo = f"{args.hilo}-{int(time.time())}"
        print(f"Hilo nuevo: {args.hilo}")
    started = time.time()
    state = await orchestrator.resume(sid, args.hilo)
    print(f"Listo en {time.time() - started:.0f} s.\n")
    print_state(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agente de SEO programático (Flujo 1)")
    parser.add_argument("comando", choices=["avanzar", "estado", "reiniciar"])
    parser.add_argument("sheet", help="ID o URL de la Google Sheet")
    parser.add_argument("--hilo", default="default", help="Identificador del run sobre la misma Sheet")
    parser.add_argument("-v", "--verbose", action="store_true", help="Muestra el progreso por lote")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
