# Agente de SEO Programático

Agente de IA open source construido con Google ADK y Gemini, en producción en D2B Inteligencia Digital.

Corre el Flujo 1 de SEO programático sobre la Google Sheet de cada cliente: busca y puntúa keywords
con Search Console, propone un plan de URLs y escribe el contenido de cada página (texto, title/meta,
schema y enlaces internos), con dos aprobaciones del equipo en el medio. Se usa desde **Google Chat**.

- **Datos:** Search Console (MCP en Agent Registry), Google Sheets y Gemini en Vertex AI.
- **Calidad:** motor determinista (sin LLM) para scoring, slugs y validaciones; juez Gemini distinto
  del redactor; detector de copias entre páginas del mismo patrón.
- **Producción:** Cloud Run + app de Google Chat, estado en Firestore, registrado en Agent Registry.

## Requisitos

- Un proyecto de Google Cloud con Vertex AI, Firestore, Cloud Run, Google Chat API y Sheets API.
- Un **servidor MCP de Search Console** registrado en Agent Registry (por defecto con el id
  `search-console-readonly`; se cambia con `SEO_AGENT_SEARCH_CONSOLE_MCP`).
- Python 3.11+ y [`uv`](https://docs.astral.sh/uv/).

Copia `.env.example` a `.env` y completa los valores de tu proyecto. En el resto de este README,
`your-gcp-project` y `123456789012` son el ID y el número de **tu** proyecto.

## Setup

```bash
uv sync
gcloud auth application-default login   # ADC para Agent Registry, MCP y Sheets
```

Google bloquea pedir el scope de Sheets con el cliente OAuth de `gcloud`. Por eso Sheets se usa
**suplantando la service account del agente** desde ADC (basta el login normal con `cloud-platform`):

- Cuenta: `seo-programmatic-agent@your-gcp-project.iam.gserviceaccount.com`
  (`Settings.service_account`, sobreescribible con `SEO_AGENT_SERVICE_ACCOUNT`; se usa para Sheets y Google Chat).
- Tu usuario necesita `roles/iam.serviceAccountTokenCreator` sobre esa cuenta.
- **Cada Sheet de cliente debe compartirse con esa cuenta como Editor.**
- La API de Google Sheets está habilitada en `your-gcp-project`.

## Guía de uso (Google Chat)

**Una vez por cliente**
1. En Search Console del cliente → Configuración → Usuarios y permisos → Agregar usuario:
   `search-console-mcp-runner@your-gcp-project.iam.gserviceaccount.com`, permiso **Restringido**.
2. Crea una Google Sheet en blanco y compártela como **Editor** (sin notificar) con
   `seo-programmatic-agent@your-gcp-project.iam.gserviceaccount.com`.

**Después, conversar.** Abre la app **SEO Programático D2B** y escríbele como a una persona
("quiero hacer páginas para ACME"). El agente pide los datos de a uno: la Sheet, el sitio, las
keywords que ya tengan (pegadas en el chat) y los datos reales por variante (una tabla copiada desde
Sheets o Excel). Él llena las pestañas, corre el análisis y explica el resultado. Se elige
conversando ("saca Temuco", "genera todas") y se puede volver a cualquier paso ("cambiemos las
keywords"). Nadie marca `TRUE` en la Sheet: queda como entregable con **Keywords**, **Plan de URLs**,
**Contenido**, **On-page**, **Schema**, **Enlaces internos** y **Descartados**.

## Agente conversacional (`src/agent/chat_agent.py`)

```bash
uv run adk web          # el mismo agente, para probar en local (app "src")
```

- `build_chat_agent()`: `LlmAgent` (Gemini `orchestrator_model`, por defecto `gemini-2.5-pro`) con
  herramientas chicas sobre `src/agent/chat_actions.py`: `use_sheet`, `list_search_console_sites`
  (propiedades a las que la cuenta del MCP tiene acceso, con la sugerida para el cliente), `set_client_info`,
  `review_site` (Search Console 28 días), `explore_site` (plataforma, sitemap y colecciones del sitio;
  el inventario queda guardado y el análisis lo usa), `read_page` (lee una página del cliente), `set_keywords`, `set_own_data`, `analyze_keywords`
  (Etapa A), `plan_urls` (aprobaciones + Etapa B), `generate_content` (aprobaciones + Etapa C),
  `get_overview`, `get_keyword_analysis`, `show_page` y `save_client_preference`.
- Las herramientas devuelven resúmenes en palabras simples (sin `optimize_existing` ni
  `coverage_status`). Volver atrás usa `SeoProgrammaticOrchestrator.rewind()`.
- La instrucción fija el estilo de Google Chat (mensajes cortos, `*negrita*`, `<url|texto>`) y
  `to_chat_text()` convierte el markdown que igual se cuele.

## App de Google Chat (`src/chat_app.py`)

```bash
uv run uvicorn src.chat_app:app --port 8080   # POST /chat (o /) recibe los eventos de Chat
```

- Cada mensaje va al agente. Conversación = el espacio en un mensaje directo, o el hilo en un
  espacio con hilos. El historial se guarda en Firestore (`seo_chats`).
- **Límite de 30 s de Chat:** el turno corre en una tarea y la respuesta HTTP lo espera hasta
  `chat_sync_seconds` (22 s). Si termina antes, responde directo. Si no, responde
  "⏳ Dame un momento…" y publica el resultado después con `spaces.messages.create`. Las
  herramientas largas (`analyze_keywords`, `plan_urls`, `generate_content`) avisan cuánto tardan y
  ese aviso sale al tiro como respuesta.
- Un lock en `seo_locks` evita dos turnos a la vez en la misma conversación ("Sigo trabajando en lo
  anterior…"). Los botones de las tarjetas antiguas se traducen a un mensaje para el agente.

## Persistencia

| Qué | Dónde | Notas |
|---|---|---|
| Estado de cada run (etapa, keywords, plan...) | Firestore `seo_runs/{sesión}` | Comprimido (zlib); sobrevive al reciclaje de Cloud Run aunque pasen días entre aprobaciones. |
| Lock anti doble clic de Google Chat | Firestore `seo_locks/{clave}` | Compartido entre instancias, vence a la hora si una instancia muere. |
| Memoria por cliente (`save_client_seo_preference`, aprendizajes) | Memory Bank de Vertex AI si hay `SEO_AGENT_AGENT_ENGINE_ID`; si no, Firestore `seo_client_memory/{dominio}` | |
| Historial de cada conversación de Google Chat | Firestore `seo_chats/{hash}` | Estado (Sheet de la conversación) + últimos `chat_history_events` eventos, comprimido. |
| Sesiones conversacionales de `adk web` | `uv run adk web --session_service_uri agentengine://<ID> --memory_service_uri agentengine://<ID>` | Requiere Agent Engine; sin él, en memoria. |

`SEO_AGENT_STATE_BACKEND=json` guarda todo en `.seo_agent_state/` (solo desarrollo).

## Despliegue (Cloud Run + Agent Registry)

```bash
./deploy_agent.sh apis       # una vez: habilita Run, Build, Firestore, Vertex, Sheets, Chat...
./deploy_agent.sh iam        # una vez: roles de la service account (requiere owner)
./deploy_agent.sh deploy     # build con Dockerfile (uv) + deploy de src/chat_app.py
./deploy_agent.sh register   # crea/actualiza el agente en Agent Registry
./deploy_agent.sh url        # URL del servicio
```

- Servicio `seo-programmatic-agent` en `your-gcp-project` / `us-central1`, identidad
  `seo-programmatic-agent@your-gcp-project.iam.gserviceaccount.com`, `--no-cpu-throttling`
  (la etapa sigue en background después de responder a Chat), `--min-instances 0`.
- Roles de la service account: `datastore.user`, `aiplatform.user`, `agentregistry.viewer`,
  `logging.logWriter` y `secretmanager.secretAccessor` solo sobre `search-console-mcp-auth-token`.
- `--allow-unauthenticated` porque Google Chat no envía un token de IAM de Cloud Run: la app
  verifica el token propio de Chat (`SEO_AGENT_CHAT_AUDIENCE` = número de proyecto `123456789012`).
- Agent Registry: servicio `seo-programmatic-agent` (agente, `no-spec`) con interfaz
  `http-json` en `<URL>/chat`, junto a los MCP de D2B.

### Conectar la app en Google Chat (consola de GCP)

1. Abre **Google Chat API → Configuración**:
   https://console.cloud.google.com/apis/api/chat.googleapis.com/hangouts-chat?project=your-gcp-project
2. **Deja sin marcar** "Build this Chat app as a Google Workspace add-on" (la app usa el formato
   de eventos clásico de Chat apps).
3. **Información de la app**: nombre `SEO Programático D2B`, URL de avatar (cualquier PNG
   cuadrado público) y descripción `Corre el SEO programático sobre la Google Sheet del cliente`.
4. **Funciones interactivas**: activadas. Marca "Recibir mensajes 1:1" y "Unirse a espacios y
   conversaciones grupales".
5. **Configuración de conexión**: "URL de extremo HTTP" = salida de `./deploy_agent.sh url`
   seguida de `/chat` (por ejemplo `https://seo-programmatic-agent-xxxx.us-central1.run.app/chat`).
6. **Público de autenticación**: "Número de proyecto" (`123456789012`). Debe coincidir con
   `SEO_AGENT_CHAT_AUDIENCE`, que el script ya configura.
7. **Visibilidad**: "Hacer que esta app de Chat esté disponible para personas y grupos
   específicos" → el grupo o los correos del equipo de SEO de d2b.cl.
8. **Guardar**. Si la organización restringe apps de Chat, un admin de Workspace debe
   permitirla en *Admin console → Apps → Google Workspace → Google Chat → Chat apps*.
9. En Google Chat: **Nuevo chat → Buscar apps → "SEO Programático D2B"** (o agrégala a un
   espacio con @). Pega la URL de una Sheet compartida con la service account.

## Probar el agente (CLI)

```bash
uv run python -m src.cli avanzar "<URL o ID de la Sheet>"   # corre la etapa que toca y para en la próxima pausa
uv run python -m src.cli estado  "<URL o ID de la Sheet>"   # en qué etapa está el run
uv run python -m src.cli reiniciar "<URL o ID de la Sheet>" # run nuevo desde la Etapa A
```

Ciclo: `avanzar` (Etapa A) → marcar `Aprobado` en `Keywords` → `avanzar` (Etapa B) → marcar `Aprobado`
en `Plan de URLs` → `avanzar` (Etapa C). El estado queda en `.seo_agent_state/` (o en Vertex si hay
Agent Engine). La Sheet debe estar compartida con la service account del agente.

`scripts/calibrate_similarity.py` mide la similitud entre páginas de un mismo patrón para calibrar
`umbral_similitud`.

## Uso

```python
from google.adk.agents import Agent
from src.config import get_settings
from src.integrations.agent_registry import AgentRegistryLoader
from src.tools.sheets_tool import SHEETS_TOOLS

toolsets = AgentRegistryLoader().load_toolsets()  # search-console-readonly
agent = Agent(
    name="seo_programatico",
    model=get_settings().orchestrator_model,
    instruction="...",
    tools=[*toolsets.values(), *SHEETS_TOOLS],
)
```

Servidor MCP de Sheets standalone (FastMCP): `uv run python -m src.tools.sheets_tool`.

### Autenticación hacia los MCP

La estrategia de autenticación hacia cada MCP se define por servicio en `Settings.mcp_auth`:

| Servicio | Modo | Cómo |
|---|---|---|
| `search-console-readonly` | `secret_token` | Bearer estático (`MCP_AUTH_TOKEN` del servicio) leído de Secret Manager con ADC. Requiere `roles/secretmanager.secretAccessor`. En local se puede sobreescribir con `SEO_AGENT_MCP_TOKEN_<SERVICIO>`. |
| Otro servicio `*.run.app` | `id_token` (default) | ID token con audiencia = URL del servicio, para Cloud Run con IAM. |
| `*.googleapis.com` | — | ADK inyecta el access token de ADC. |

Nota: `McpToolset` de ADK solo llama al `header_provider` si recibe un contexto. Dentro de un
agente siempre lo recibe; para usarlo fuera de un Runner ver `_discovery_context()`.

### Formato de la Sheet del cliente

Pestañas `Parametros` y `Brief` (configurables por env), formato clave/valor (col A / col B):

| Parámetro          | Valor |
|--------------------|-------|
| peso_volumen       | 0.4   |
| peso_intencion     | 0,35  |
| score_minimo_juez  | 8     |
| umbral_similitud   | 0.5   |
| score_minimo_keyword | 40  |
| volumen_minimo     | 50    |

Las claves se normalizan (minúsculas, sin tildes, `_`), así que "Peso Volumen" también funciona.

## Motor determinista (`src/tools/seo_deterministic.py`)

Pasos de la planilla que corren por código (sin LLM, reproducibles). Expuestos en
`SEO_DETERMINISTIC_TOOLS` para ADK.

| Paso | Función | Qué hace |
|---|---|---|
| P1-03 | `normalize_and_deduplicate_keywords` | Minúsculas, sin tildes (conserva ñ), espacios simples. Deduplica con singular/plural obvio del español; entre duplicados queda el de más volumen y todas las variantes en `variants` con `original_keyword` y `source`. |
| P1-05 | `mark_gsc_overlap` | Cruza con filas de `get_search_analytics` (`keys: [query, page]` o `query`/`page`). Agrega por query: clics e impresiones sumados, posición ponderada por impresiones, `existing_url` = página con más clics. `ya_cubierta` ≤ 10 < `parcial` ≤ 30 < `libre`. |
| P1-08/09 | `score_and_route_keywords` | Score 0-100 con pesos de la Sheet (`peso_volumen`, `peso_dificultad`, `peso_intencion`, `peso_datos_propios`, `peso_valor_negocio`; sin pesos usa 30/20/25/10/15). Volumen en escala log relativa al máximo del lote; `business_value` en escala 0-5. Rutea a `create_new_url`, `optimize_existing` o `below_threshold` (con motivo). |
| P1-12 | `generate_and_validate_slugs` | `/{hub}/{slug}` por fórmula desde `pattern` + `variables` (o `keyword`), ASCII con guiones, recorte en guion a `max_length`. Acepta nodos `hub/pattern/variables` o la salida de PR-02 (`hubs` + `paginas`). Marca `duplicado_en_plan`, `colision_url_existente`, `hub_invalido` (subdominio), `variable_faltante`, `excede_largo`. |
| P1-16 | `validate_title_and_meta` | Title ≤ 60 que empieza con la keyword (tolera plural), meta ≤ 155, sin voseo; con la fila como fuente, toda cifra debe estar en ella y no se aceptan superlativos/promesas ("el mejor", "garantizado", "100%", "líder", "el más barato"...) que no vengan literalmente en la fila. |
| P1-18 | `mask_row_values` + `compute_shingle_similarity_matrix` (o `compute_cosine_similarity_matrix`) | Enmascara los datos de la fila (`[VAR_CIUDAD]`, `[NUM]`...) y mide Jaccard de trigramas de palabras. Calibrado: páginas buenas 0.12-0.35, copias estructurales 0.67-1.0 → umbral 0.5. El coseno de embeddings (`SEO_AGENT_SIMILARITY_METHOD=embeddings`) no separa copias de páginas buenas en este caso. |
| P1-21 | `validate_json_ld_schema` | `@context`, `@type` permitido, campos obligatorios por tipo, BreadcrumbList bien formado y valores respaldados por la fila/contenido (números por dígitos, textos por palabras). |
| P1-23/24 | `validate_internal_links` | Párrafo literal, anchor de 2-6 palabras dentro del párrafo y no usado, destino hub/hermana/existente, 1 al hub y máx. 4 hermanas. |

## Sub-agentes y bucle de calidad (`src/agent/`)

| Archivo | Contenido |
|---|---|
| `schemas.py` | `output_schema` de PR-01..PR-06, P1-16 y P1-06. PR-02 y PR-05 tienen además un esquema "de cable" para el modelo (variables como lista `nombre/valor`, JSON-LD como string) porque Gemini devuelve vacío un `dict` libre; se convierten al contrato público con `to_output()`. |
| `prompts.py` | Prompts oficiales con temperatura y rol de modelo. |
| `subagents.py` | `build_subagent` (`LlmAgent` con `output_schema`), `AdkStructuredLLM` (corre y valida), `GenAIEmbedder` y P1-06 `classify_and_cluster_keywords`. |
| `content_pipeline.py` | `ContentPipeline`: lotes de 10-20, PR-03 → PR-04 → similitud (P1-18) → decisión (P1-19) con reintentos, y luego P1-16, PR-05 + P1-21 y PR-06 + P1-23/24. |

Modelos (Vertex AI, `your-gcp-project`): agente conversacional y redactor `gemini-2.5-pro`; juez, extractor y clasificador
`gemini-2.5-flash`; embeddings `text-multilingual-embedding-002`. `Settings` rechaza
`judge_model == writer_model`.

Cada intento: PR-03 redacta, P1-16 propone title/meta (validados contra la fila) y PR-04 audita el
contenido junto con el title y la meta seleccionados. Decisión P1-19: pasa si
`score >= score_minimo_juez` (default 4), la similitud máxima contra las páginas **ya aprobadas del
mismo hub** es `< umbral_similitud` (default 0.5 con trigramas sobre texto enmascarado, calibrado con `scripts/calibrate_similarity.py`), el contenido no tiene voseo y hay al menos un
title/meta válido. Si falla, se reintenta con `observaciones_juez`, `observaciones_similitud` y
`observaciones_validacion` hasta `max_retries` (default 2); después queda `requiere_revision_humana`.
Un error de API consume un intento. Solo las filas aprobadas pasan a schema y enlazado; los anchors
usados se acumulan entre filas.

Prompts en español neutro. PR-03, P1-16, PR-04 y PR-06 reciben `localizacion` (país, idioma y trato
del Brief; por defecto Chile / es-CL / tuteo) con el voseo prohibido.

## Orquestador del Flujo 1 (`src/agent/orchestrator.py`)

```
NEW ──run_phase_1_and_2──▶ WAITING_KEYWORD_APPROVAL ──run_phase_3──▶ WAITING_URL_APPROVAL ──run_phase_4_to_7──▶ DONE
        P1-01..P1-09            (P1-10: marcar Aprobado      P1-11..P1-13    (P1-13: marcar Aprobado        P1-14..P1-25
                                 en "Keywords")                               en "Plan de URLs")
```

```python
from src.agent.orchestrator import SeoProgrammaticOrchestrator
from src.agent.subagents import AdkStructuredLLM, GenAIEmbedder

orq = SeoProgrammaticOrchestrator(AdkStructuredLLM(), GenAIEmbedder())
await orq.resume(spreadsheet_id, thread_id)  # corre la etapa que toca según el estado guardado
```

- Cada etapa valida el estado de entrada; si falla guarda `last_error` y conserva el estado para reintentar.
- **Etapa A**: Brief + Parámetros (P1-01), `Semillas` (P1-03), Search Console 90 días (P1-04/05),
  clústeres e intención (P1-06), PR-01 (P1-07), scoring y ruteo (P1-08/09). Escribe `Keywords`
  y `Descartados`.
- **Etapa B**: lee `Aprobado` en `Keywords` (`TRUE`, casilla, "sí", "x"...; `FALSE` + `Motivo rechazo`
  queda en Descartados). Sin aprobaciones no avanza. PR-02 (P1-11) con los valores reales de
  `Datos propios`, slugs (P1-12) con `url_id` U001... Escribe `Plan de URLs`.
- **Etapa C**: lee `Aprobado` en `Plan de URLs`, cruza cada página con su fila de `Datos propios`
  (por las variables), corre el pipeline y escribe en **una sola** `batchUpdate` las 8 pestañas
  cruzadas por `url_id`: Resumen, Keywords, Plan de URLs, Contenido, On-page, Schema, Enlaces internos
  y Descartados. Guarda aprendizajes en la memoria del cliente.

### Pestañas de entrada

| Pestaña | Formato |
|---|---|
| `Brief` | clave/valor: `sitio` (o `dominio`, `propiedad_gsc` como `sc-domain:...`), `pais`, `idioma`, `trato` (tuteo/usted), `tipo_schema` |
| `Parametros` | clave/valor (ver arriba) |
| `Semillas` | tabla: `keyword`, `fuente`, `volumen`, `dificultad`, opcionales `intencion`, `valor negocio`, `datos propios` |
| `Datos propios` | tabla: una fila por entidad; las columnas que son variables de patrón (p.ej. `ciudad`) se usan para cruzar cada página con su fila |

### Estado y memoria (`src/agent/memory_manager.py`)

- Estado del run: una sesión ADK por (`spreadsheet_id`, `thread_id`) con `run_state` en `session.state`
  (actualizado con `EventActions.state_delta`). Memoria por cliente: `user_id` = dominio.
- Con `SEO_AGENT_AGENT_ENGINE_ID` usa `VertexAiSessionService` + `VertexAiMemoryBankService`.
  Sin Agent Engine, la memoria por cliente queda en Firestore (o en JSON local con
  `SEO_AGENT_STATE_BACKEND=json`).

## Configuración (`src/config.py`)

Variables de entorno: `SEO_AGENT_PROJECT_ID`, `SEO_AGENT_LOCATION`, `SEO_AGENT_SEARCH_CONSOLE_MCP`,
`SEO_AGENT_VERTEX_PROJECT`, `SEO_AGENT_VERTEX_LOCATION`, `SEO_AGENT_ORCHESTRATOR_MODEL`,
`SEO_AGENT_WRITER_MODEL`, `SEO_AGENT_JUDGE_MODEL`, `SEO_AGENT_EXTRACTOR_MODEL`, `SEO_AGENT_EMBEDDING_MODEL`,
`SEO_AGENT_AGENT_ENGINE_ID`, `SEO_AGENT_LOCAL_STATE_DIR`, `SEO_AGENT_PARAMETERS_TAB`, `SEO_AGENT_BRIEF_TAB`.

## Tests

```bash
uv run pytest            # unitarios con mocks
uv run pytest --live -s  # + Search Console, Gemini/Vertex y la Sheet de prueba (~6 min)
```

Los tests `--live` de Sheets usan `SEO_AGENT_TEST_SHEET_ID` (obligatoria para esos tests: una Sheet de prueba compartida con la service account).
El ciclo completo siembra `Brief`, `Parámetros`, `Semillas` y `Datos propios` con un cliente ficticio
solo si esas pestañas están vacías, y escribe las 8 pestañas de salida.

## Licencia

[MIT](LICENSE).
