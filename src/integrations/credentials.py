"""Credenciales del agente para APIs de Workspace (Sheets, Chat)."""

from __future__ import annotations

import os

import google.auth

from src.config import get_settings

CLOUD_PLATFORM = "https://www.googleapis.com/auth/cloud-platform"


def running_as_agent_identity() -> bool:
    """True en Cloud Run / Agent Runtime, donde ADC ya es la service account del agente."""
    return bool(os.environ.get("K_SERVICE") or os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID"))


def agent_credentials(scopes: list[str] | tuple[str, ...]):
    """Credenciales con `scopes` en nombre de la service account del agente.

    En local suplanta `Settings.service_account` desde ADC (requiere
    `roles/iam.serviceAccountTokenCreator`), así un login de usuario con solo
    `cloud-platform` alcanza. Desplegado, o sin service account configurada, usa ADC.
    """
    settings = get_settings()
    if settings.service_account and not running_as_agent_identity():
        from google.auth import impersonated_credentials

        source, _ = google.auth.default(scopes=[CLOUD_PLATFORM])
        return impersonated_credentials.Credentials(
            source_credentials=source,
            target_principal=settings.service_account,
            target_scopes=list(scopes),
            lifetime=3600,
        )
    credentials, _ = google.auth.default(scopes=list(scopes))
    return credentials
