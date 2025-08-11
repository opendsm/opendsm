# syntax=docker/dockerfile:1.7
FROM python:3.10-slim AS app

# System deps (you had libenchant-2-dev)
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
# Copy only metadata first to maximize Docker layer caching
COPY pyproject.toml README.md /app/
# If you keep a lockfile, copy it too for reproducible installs
# (safe if missing)
COPY uv.lock /app/uv.lock

# Resolve & install *only dependencies* into the system Python
# Using uv pip compile -> requirements.txt for a stable, cacheable layer
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip compile pyproject.toml -o /tmp/requirements.txt && \
    uv pip install --system -r /tmp/requirements.txt

# ---- project install ----
# Now add your source and install the project itself
COPY opendsm/ /app/opendsm/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -e .[dev]

ENV PYTHONPATH=/usr/local/bin:/app
WORKDIR /app
