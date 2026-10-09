# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS app

# System deps for building native wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Add uv (fast installer)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Helpful uv settings: compile bytecode & avoid hardlinks
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ---- deps layer (cacheable) ----
# Copy only metadata and the lockfile first to maximize Docker layer caching
COPY pyproject.toml README.md uv.lock /app/

# Install *only the locked dependencies* into the system Python; the build
# fails if uv.lock is out of date with pyproject.toml. The system interpreter,
# not a venv, because the compose services bind-mount the repo over /app and
# would hide a /app/.venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv export --locked --no-emit-project --extra dev -o /tmp/requirements.txt && \
    uv pip install --system -r /tmp/requirements.txt

# ---- project install ----
# Now add your source and install the project itself
COPY opendsm/ /app/opendsm/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -e .[dev]

ENV PYTHONPATH=/usr/local/bin:/app
WORKDIR /app
