"""Prompts oficiales del Flujo 1 (SEO programático), en español neutro.

Cada prompt recibe su entrada como JSON en el mensaje de usuario y responde solo
con JSON que cumple su `output_schema`. Los prompts que generan texto visible
(PR-03, P1-16, PR-06) reciben además `localizacion` (país, idioma y trato del
Brief) y siguen `LOCALIZATION_RULE`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from src.agent.schemas import (
    IntentClassificationOutput,
    PR01Output,
    PR02LLMOutput,
    PR02Output,
    PR03Output,
    PR04Output,
    PR05LLMOutput,
    PR05Output,
    PR06Output,
    TitleMetaOutput,
)

ModelRole = Literal["writer", "judge", "extractor"]

LOCALIZATION_RULE = (
    "- Escribe según `localizacion`: país, idioma y trato (tuteo o usted) indicados ahí. "
    "Está prohibido el voseo (por ejemplo «cotizá», «contratá», «tenés», «podés», «sos», «vos»)."
)


@dataclass(frozen=True)
class PromptSpec:
    id: str
    step: str
    system: str
    rules: str
    temperature: float
    output_schema: type[BaseModel]
    role: ModelRole
    # Esquema que ve el modelo, si difiere del público. Debe tener `to_output()`.
    llm_schema: type[BaseModel] | None = None

    @property
    def model_schema(self) -> type[BaseModel]:
        return self.llm_schema or self.output_schema

    @property
    def agent_name(self) -> str:
        return self.id.lower().replace("-", "_")

    def instruction(self) -> str:
        return (
            f"{self.system}\n\n"
            f"Reglas:\n{self.rules.strip()}\n\n"
            "La entrada llega como JSON en el mensaje del usuario. "
            "Responde únicamente con un JSON válido que cumpla el esquema de salida."
        )


PR01 = PromptSpec(
    id="PR-01",
    step="P1-07",
    system="Eres estratega de SEO programático. Solo propones patrones que se sostienen con datos reales del cliente.",
    rules="""
- Recibes `clusters` (keywords con volumen, dificultad e intención), `datos_propios` del cliente y `memoria_cliente` (aprendizajes de corridas anteriores, si existen).
- Agrupa las keywords en patrones programáticos con sus `variables` (por ejemplo ciudad, producto). Usa como nombre de variable el nombre de la columna de `datos_propios` que la contiene.
- Cada patrón necesita un `dato_diferenciador`: un dato propio que cambie en cada página del patrón.
- Si un patrón no tiene un dato propio que diferencie cada página, márcalo `descartado=true` y explica el `motivo`.
- `paginas_estimadas` es la cantidad de combinaciones de variables con datos disponibles.
- Las keywords que no encajan en ningún patrón van en `sin_patron`. Escribe cada keyword exactamente como llegó.
""",
    temperature=0.4,
    output_schema=PR01Output,
    role="writer",
)

PR02 = PromptSpec(
    id="PR-02",
    step="P1-11",
    system="Diseñas arquitecturas de URL. Una keyword principal por URL, siempre en subcarpeta.",
    rules="""
- Recibes `patrones` aprobados (con sus keywords aprobadas), `valores_variables` (valores válidos por variable según los datos propios) y `urls_existentes`.
- Cada patrón tiene un hub (`url` en subcarpeta, nunca subdominio) y un `patron` de slug con las variables entre llaves, por ejemplo "seguro de {vehiculo} en {ciudad}".
- Cada página tiene una sola `keyword_principal`. Dos keywords con la misma intención van a la misma URL (una como principal y la otra en `keywords_secundarias`).
- `variables` es una lista de pares `nombre`/`valor`, con exactamente los nombres de variables del patrón del hub. Cada `valor` se copia tal cual desde `valores_variables`.
- Ninguna `url_propuesta` puede coincidir con una de `urls_existentes`.
- Respeta `reglas_url` de la `plataforma`. Si `subcarpetas_libres` es false, el hub solo puede ser uno de `hubs_permitidos`, según la intención de la keyword principal (`hub_por_intencion`): en Shopify, transaccional o comercial en "/collections" e informacional en "/pages". Ahí no inventes hubs como "/libros-de-medicina/".
""",
    temperature=0.3,
    output_schema=PR02Output,
    role="writer",
    llm_schema=PR02LLMOutput,
)

PR03 = PromptSpec(
    id="PR-03",
    step="P1-15",
    system="Eres redactor SEO. Escribes solo con los datos que recibes. Si falta un dato, omites el bloque.",
    rules=f"""
- Recibes `fila` (datos propios de la página), `keyword`, `subtemas`, `localizacion` y `memoria_cliente` (reglas de tono y estructura del cliente, si existen).
- Está prohibido escribir cifras o afirmaciones que no estén en `fila`.
- La `intro` menciona al menos 2 datos propios de `fila`.
- Evita frases genéricas que servirían para cualquier otra fila del mismo patrón.
- FAQ de máximo 4 preguntas, cada respuesta respaldada por datos de `fila`. Sin datos, `faq` va vacío.
- Si un bloque no se puede escribir por falta de datos, omítelo y anótalo en `bloques_omitidos`.
- Si llegan `observaciones_juez`, `observaciones_similitud` u `observaciones_validacion` de un intento anterior, corrige exactamente esos puntos.
{LOCALIZATION_RULE}
""",
    temperature=0.6,
    output_schema=PR03Output,
    role="writer",
)

TITLE_META = PromptSpec(
    id="P1-16",
    step="P1-16",
    system="Eres redactor SEO de títulos y meta descripciones.",
    rules=f"""
- Recibes `keyword`, `h1`, `intro`, `fila` y `localizacion`.
- Genera exactamente 3 variantes.
- `title`: máximo 60 caracteres y empieza con la keyword principal.
- `meta_description`: máximo 155 caracteres, con un dato propio de `fila`.
- Toda cifra debe estar en `fila`. No uses superlativos ni promesas ("el mejor", "garantizado", "100%", "líder", "el más barato") salvo que estén literalmente en `fila`.
{LOCALIZATION_RULE}
""",
    temperature=0.4,
    output_schema=TitleMetaOutput,
    role="writer",
)

PR04 = PromptSpec(
    id="PR-04",
    step="P1-17",
    system="Eres auditor de calidad de contenido SEO. Evalúas, no reescribes.",
    rules="""
- Recibes `contenido`, `title`, `meta_description`, `fila` (fuente de verdad), `keyword`, `subtemas` y `localizacion`.
- Audita el contenido, el title y la meta description.
- `cifras_no_respaldadas`: cada cifra o afirmación (incluidos superlativos y promesas) que no esté en `fila`, citada textual.
- `frases_genericas`: frases que servirían igual para cualquier otra página del patrón, citadas textual.
- `cobertura_intencion`: true si el contenido responde a la intención de búsqueda de la keyword.
- `score` de 1 a 5: 5 = todo respaldado, específico, cubre la intención y respeta `localizacion`; 1 = inventa datos o no sirve.
- `motivo`: una o dos frases que justifiquen el score.
""",
    temperature=0.0,
    output_schema=PR04Output,
    role="judge",
)

PR05 = PromptSpec(
    id="PR-05",
    step="P1-20",
    system="Extraes datos para schema.org. Nunca inventas valores.",
    rules="""
- Recibes `fila`, `contenido` visible, `url`, `tipo_sugerido` y `marca` (nombre del cliente según el Brief).
- Usa `marca` como `provider`, `brand` u organización cuando el tipo lo requiera.
- `tipo_schema` es uno de: Product, LocalBusiness, Service, Article, BreadcrumbList.
- `jsonld_json` es el objeto JSON-LD completo serializado como string JSON. Incluye "@context": "https://schema.org" y "@type" igual a `tipo_schema`.
- Usa solo valores que estén en `fila` o en `contenido`. Si falta un campo, ponlo en null y anótalo en `campos_faltantes`.
""",
    temperature=0.0,
    output_schema=PR05Output,
    role="extractor",
    llm_schema=PR05LLMOutput,
)

PR06 = PromptSpec(
    id="PR-06",
    step="P1-22",
    system="Eres editor de enlazado interno. Prefieres no enlazar antes que forzar un enlace.",
    rules=f"""
- Recibes `contenido`, `url_origen`, `hub`, `hermanas` (URLs con su keyword), `urls_existentes` pertinentes, `anchors_usados` y `localizacion`.
- Propón 1 enlace al hub, de 2 a 4 a páginas hermanas y a URLs existentes solo si son pertinentes.
- `parrafo_literal`: copia exacto, carácter por carácter, el párrafo del contenido donde va el enlace.
- `anchor`: texto descriptivo de 2 a 6 palabras que aparece dentro de ese párrafo y que no está en `anchors_usados`.
- Si un destino no tiene un lugar natural, no lo fuerces: ponlo en `sin_lugar`.
- `justificacion` breve.
{LOCALIZATION_RULE}
""",
    temperature=0.2,
    output_schema=PR06Output,
    role="writer",
)

INTENT = PromptSpec(
    id="P1-06",
    step="P1-06",
    system="Clasificas la intención de búsqueda de keywords en español.",
    rules="""
- Recibes `keywords`.
- Devuelve una clasificación por cada keyword, con la keyword escrita exactamente igual.
- `intencion` es una de: informativa, comercial, transaccional, navegacional.
""",
    temperature=0.0,
    output_schema=IntentClassificationOutput,
    role="extractor",
)

PROMPTS: dict[str, PromptSpec] = {p.id: p for p in (PR01, PR02, PR03, TITLE_META, PR04, PR05, PR06, INTENT)}
