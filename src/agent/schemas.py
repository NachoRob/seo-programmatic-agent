"""Esquemas de salida (output_schema) de los prompts del Flujo 1."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

# ------------------------------------------------------------ PR-01 / P1-07 ----

class PatronItem(BaseModel):
    nombre: str
    variables: list[str]
    keywords: list[str]
    dato_diferenciador: str
    paginas_estimadas: int = Field(ge=0)
    descartado: bool
    motivo: str | None = None


class PR01Output(BaseModel):
    patrones: list[PatronItem]
    sin_patron: list[str]


# ------------------------------------------------------------ PR-02 / P1-11 ----

class HubItem(BaseModel):
    url: str
    patron: str


class PaginaPlanItem(BaseModel):
    url_propuesta: str
    keyword_principal: str
    keywords_secundarias: list[str]
    hub: str
    variables: dict[str, str]


class PR02Output(BaseModel):
    hubs: list[HubItem]
    paginas: list[PaginaPlanItem]


# Esquema "de cable" para el modelo: Gemini devuelve vacío un objeto sin
# propiedades declaradas (dict libre), así que las variables viajan como lista.
class VariableItem(BaseModel):
    nombre: str
    valor: str


class _PaginaPlanWire(BaseModel):
    url_propuesta: str
    keyword_principal: str
    keywords_secundarias: list[str]
    hub: str
    variables: list[VariableItem]


class PR02LLMOutput(BaseModel):
    hubs: list[HubItem]
    paginas: list[_PaginaPlanWire]

    def to_output(self) -> PR02Output:
        return PR02Output(
            hubs=self.hubs,
            paginas=[
                PaginaPlanItem(**p.model_dump(exclude={"variables"}),
                               variables={v.nombre: v.valor for v in p.variables})
                for p in self.paginas
            ],
        )


# ------------------------------------------------------------ PR-03 / P1-15 ----

class SeccionItem(BaseModel):
    h2: str
    texto: str


class FaqItem(BaseModel):
    pregunta: str
    respuesta: str


class PR03Output(BaseModel):
    h1: str
    intro: str
    secciones: list[SeccionItem]
    faq: list[FaqItem] = Field(max_length=4)
    cta: str
    bloques_omitidos: list[str]

    def as_text(self) -> str:
        """Texto visible de la página (para el juez, similitud y enlazado)."""
        parts = [self.h1, self.intro]
        for s in self.secciones:
            parts += [s.h2, s.texto]
        for f in self.faq:
            parts += [f.pregunta, f.respuesta]
        parts.append(self.cta)
        return "\n\n".join(p for p in parts if p)


# ------------------------------------------------------------------- P1-16 ----

class TitleMetaVariant(BaseModel):
    title: str
    meta_description: str


class TitleMetaOutput(BaseModel):
    variantes: list[TitleMetaVariant] = Field(min_length=3, max_length=3)


# ------------------------------------------------------------ PR-04 / P1-17 ----

class PR04Output(BaseModel):
    score: int = Field(ge=1, le=5)
    motivo: str
    cifras_no_respaldadas: list[str]
    frases_genericas: list[str]
    cobertura_intencion: bool


# ------------------------------------------------------------ PR-05 / P1-20 ----

TipoSchema = Literal["Product", "LocalBusiness", "Service", "Article", "BreadcrumbList"]


class PR05Output(BaseModel):
    tipo_schema: TipoSchema
    jsonld: dict
    campos_faltantes: list[str]


class PR05LLMOutput(BaseModel):
    """Esquema "de cable": el JSON-LD viaja como string JSON (objeto libre)."""

    tipo_schema: TipoSchema
    jsonld_json: str = Field(description="Objeto JSON-LD completo serializado como string JSON")
    campos_faltantes: list[str]

    def to_output(self) -> PR05Output:
        jsonld = json.loads(self.jsonld_json)
        if not isinstance(jsonld, dict):
            raise ValueError("jsonld_json debe ser un objeto JSON")
        return PR05Output(tipo_schema=self.tipo_schema, jsonld=jsonld, campos_faltantes=self.campos_faltantes)


# ------------------------------------------------------------ PR-06 / P1-22 ----

class EnlacePropuesto(BaseModel):
    destino: str
    parrafo_literal: str
    anchor: str
    justificacion: str


class PR06Output(BaseModel):
    enlaces: list[EnlacePropuesto]
    sin_lugar: list[str]


# ------------------------------------------------------------------- P1-06 ----

Intencion = Literal["informativa", "comercial", "transaccional", "navegacional"]


class IntentItem(BaseModel):
    keyword: str
    intencion: Intencion


class IntentClassificationOutput(BaseModel):
    clasificaciones: list[IntentItem]
