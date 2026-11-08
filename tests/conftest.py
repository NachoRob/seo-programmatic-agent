import os

import pytest

# Google Sheet de prueba para los tests --live (se puede cambiar por entorno)
TEST_SHEET_ID = os.environ.get("SEO_AGENT_TEST_SHEET_ID", "YOUR_TEST_SHEET_ID")


@pytest.fixture
def live_sheet_id() -> str:
    return TEST_SHEET_ID


@pytest.fixture(autouse=True)
def _no_masking_in_fakes(request, monkeypatch):
    """Los LLM falsos generan textos que solo cambian la ciudad: enmascarados serían
    copias. Salvo en tests marcados `real_masking`, P1-18 no enmascara y los
    trigramas dan 0 (los tests de similitud usan el método `embeddings`)."""
    if "real_masking" not in request.keywords:
        monkeypatch.setattr("src.agent.content_pipeline.mask_row_values", lambda text, fila: text)
        # Con textos-plantilla, los trigramas los verían como copias: similitud 0
        monkeypatch.setattr(
            "src.agent.content_pipeline.compute_shingle_similarity_matrix",
            lambda texts, n=3: [[1.0 if i == j else 0.0 for j in range(len(texts))] for i in range(len(texts))],
        )


def pytest_addoption(parser):
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Ejecuta tests contra GCP real (Agent Registry / MCP). Requiere ADC válidas.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    skip_live = pytest.mark.skip(reason="usa --live para correr contra GCP")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
