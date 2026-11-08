# App de Google Chat del agente de SEO programático (Cloud Run).
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Dependencias primero (capa cacheable mientras no cambie uv.lock)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

RUN useradd --create-home --uid 1000 app
USER app

CMD ["sh", "-c", "uvicorn src.chat_app:app --host 0.0.0.0 --port ${PORT} --proxy-headers"]
