#!/usr/bin/env bash
# Despliegue del agente de SEO programático (app de Google Chat) en Cloud Run
# y registro en Agent Registry.
#
#   ./deploy_agent.sh apis       # habilita APIs (una vez)
#   ./deploy_agent.sh iam        # roles de la service account (una vez, requiere owner)
#   ./deploy_agent.sh deploy     # build + deploy en Cloud Run
#   ./deploy_agent.sh register   # crea/actualiza el agente en Agent Registry
#   ./deploy_agent.sh all        # todo lo anterior
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-your-gcp-project}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-seo-programmatic-agent}"
SA="${SA:-seo-programmatic-agent@${PROJECT_ID}.iam.gserviceaccount.com}"
MCP_SECRET="${MCP_SECRET:-search-console-mcp-auth-token}"
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
AGENT_ENGINE_ID="${AGENT_ENGINE_ID:-}"   # opcional: Memory Bank de Vertex AI

cd "$(dirname "$0")"

apis() {
  gcloud services enable --project "$PROJECT_ID" \
    run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
    firestore.googleapis.com aiplatform.googleapis.com secretmanager.googleapis.com \
    sheets.googleapis.com chat.googleapis.com agentregistry.googleapis.com \
    iamcredentials.googleapis.com
}

iam() {
  local member="serviceAccount:${SA}"
  for role in roles/datastore.user roles/aiplatform.user roles/agentregistry.viewer roles/logging.logWriter; do
    gcloud projects add-iam-policy-binding "$PROJECT_ID" --member "$member" --role "$role" \
      --condition=None --quiet >/dev/null
    echo "ok  $role"
  done
  # Token estático del MCP de Search Console (solo ese secreto)
  gcloud secrets add-iam-policy-binding "$MCP_SECRET" --project "$PROJECT_ID" \
    --member "$member" --role roles/secretmanager.secretAccessor --quiet >/dev/null
  echo "ok  secretAccessor en $MCP_SECRET"
}

deploy() {
  # Audiencias válidas del token de Chat: número de proyecto o URL del endpoint (las dos
  # URLs que Cloud Run asigna al servicio). Separadas por ";" porque "," separa env vars.
  local urls
  urls="$(gcloud run services describe "$SERVICE" --project "$PROJECT_ID" --region "$REGION" \
    --format='value(metadata.annotations."run.googleapis.com/urls")' 2>/dev/null | tr -d '[]" ' | tr ',' '\n' \
    | sed 's|$|/chat|' | paste -sd ';' -)"
  local audiences="${PROJECT_NUMBER}${urls:+;${urls}}"
  local env_vars="^|^SEO_AGENT_STATE_BACKEND=firestore|SEO_AGENT_CHAT_AUDIENCE=${audiences}"
  env_vars+="|SEO_AGENT_PROJECT_ID=${PROJECT_ID}|SEO_AGENT_SERVICE_ACCOUNT=${SA}"
  [[ -n "$AGENT_ENGINE_ID" ]] && env_vars+="|SEO_AGENT_AGENT_ENGINE_ID=${AGENT_ENGINE_ID}"

  # --allow-unauthenticated: Google Chat no envía un token de IAM de Cloud Run; la app
  #   verifica el token propio de Chat (audiencia = número de proyecto).
  # --no-cpu-throttling: CPU asignada después de responder, para que la etapa que corre
  #   en BackgroundTasks no se corte al devolver "⏳ Procesando...".
  # --cpu-boost: más CPU durante el arranque en frío (medido: 12 a 31 s), que se descuenta
  #   del plazo de 30 s de Google Chat.
  gcloud run deploy "$SERVICE" \
    --project "$PROJECT_ID" --region "$REGION" \
    --source . \
    --service-account "$SA" \
    --no-cpu-throttling \
    --cpu-boost \
    --min-instances 0 --max-instances 3 \
    --cpu 1 --memory 1Gi --timeout 3600 --concurrency 20 \
    --allow-unauthenticated \
    --set-env-vars "$env_vars" \
    --labels "app=seo-programmatic-agent,team=seo"
  echo
  echo "URL: $(service_url)"
}

service_url() {
  gcloud run services describe "$SERVICE" --project "$PROJECT_ID" --region "$REGION" \
    --format='value(status.url)'
}

register() {
  local url
  url="$(service_url)"
  [[ -n "$url" ]] || { echo "Primero despliega el servicio (./deploy_agent.sh deploy)"; exit 1; }
  local args=(
    --project "$PROJECT_ID" --location "$REGION"
    --display-name "D2B - SEO Programático (Flujo 1)"
    --description "Agente de SEO programático: keywords (Search Console + Gemini), plan de URLs y contenido validado sobre la Google Sheet del cliente, con aprobaciones humanas. Endpoint de Google Chat en /chat."
    --interfaces "url=${url}/chat,protocolBinding=http-json"
  )
  if gcloud agent-registry services describe "$SERVICE" --project "$PROJECT_ID" --location "$REGION" >/dev/null 2>&1; then
    gcloud agent-registry services update "$SERVICE" "${args[@]}"
  else
    gcloud agent-registry services create "$SERVICE" "${args[@]}" --agent-spec-type=no-spec
  fi
  gcloud agent-registry services describe "$SERVICE" --project "$PROJECT_ID" --location "$REGION" \
    --format='yaml(name,displayName,interfaces)'
}

case "${1:-}" in
  apis) apis ;;
  iam) iam ;;
  deploy) deploy ;;
  register) register ;;
  url) service_url ;;
  all) apis; iam; deploy; register ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
