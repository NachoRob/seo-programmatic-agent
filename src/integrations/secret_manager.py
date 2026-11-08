"""Acceso mínimo a Secret Manager vía REST con ADC (sin dependencias extra)."""

from __future__ import annotations

import base64
from functools import lru_cache

import google.auth
from google.auth.transport.requests import AuthorizedSession

SECRET_MANAGER_URL = "https://secretmanager.googleapis.com/v1"


@lru_cache(maxsize=1)
def _session() -> AuthorizedSession:
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return AuthorizedSession(credentials)


def _check(response, action: str) -> dict:
    if response.status_code >= 400:
        raise RuntimeError(f"Secret Manager: no se pudo {action} ({response.status_code}): {response.text[:200]}")
    return response.json() if response.content else {}


def access_secret(project_id: str, secret: str, version: str = "latest") -> str:
    """Lee un secreto (requiere `roles/secretmanager.secretAccessor`)."""
    url = f"{SECRET_MANAGER_URL}/projects/{project_id}/secrets/{secret}/versions/{version}:access"
    data = _check(_session().get(url, timeout=30), f"leer {secret}")
    return base64.b64decode(data["payload"]["data"]).decode().strip()
