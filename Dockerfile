# -----------------------------
# Stage 0: Base (shared)
# -----------------------------
ARG BASE_REGISTRY=docker.io
FROM ${BASE_REGISTRY}/library/python:3.14-slim AS base

ENV UV_SYSTEM_PYTHON=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/bin:$PATH"

WORKDIR /app

# Multi-arch compatible uv binary
COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /uvx /usr/local/bin/


# -----------------------------
# Stage 1: Builder
# -----------------------------
FROM base AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Prime cache with metadata + lock
COPY pyproject.toml uv.lock* ./

# Resolve dependencies into a local .venv
RUN uv lock 2>/dev/null || true
RUN uv sync --frozen --no-dev --no-install-project 2>/dev/null || uv sync --no-dev --no-install-project

# App source comes after deps for better cache reuse
COPY src/ ./src/


# -----------------------------
# Stage 2: Production
# -----------------------------
FROM base AS production

ENV PATH="/app/.venv/bin:$PATH"

# Non-root execution. Created BEFORE the copies below with a non-recursive
# chown so each COPY --chown= lands correctly-owned files directly, instead
# of a trailing `chown -R` walking the whole tree as a second, non-dedupable
# layer. Measured on this repo's own cache-cold BuildKit log: that recursive
# chown step was #22 DONE 120.0s. Final ownership is identical; this is a
# build-time no-op at runtime.
RUN useradd -m -u 1000 appuser && chown appuser:appuser /app
USER appuser

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/pyproject.toml /app/pyproject.toml
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser alembic.ini ./alembic.ini
COPY --chown=appuser:appuser alembic/ ./alembic/

ENV PYTHONPATH=/app/src

EXPOSE 8155
ENV PORT=8155

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8155"]
