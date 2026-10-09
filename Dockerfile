# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.12.23@sha256:61d393e44e249f2e4b526b6c7ddcecce245946826e608e11c93ad4f5bba55b21 AS uv
FROM python:3.12-slim AS app

# System deps for building native wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Add uv (fast installer) from the digest-pinned stage above
COPY --from=uv /uv /uvx /bin/

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
    uv export --locked --no-emit-project --extra dev --extra tutorial -o /tmp/requirements.txt && \
    uv pip install --system -r /tmp/requirements.txt

# ---- project install ----
# Now add your source and install the project itself
COPY opendsm/ /app/opendsm/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -e ".[dev,tutorial]"

ENV PYTHONPATH=/usr/local/bin:/app
WORKDIR /app

# Run as an unprivileged user whose UID matches the host user that bind-mounts the
# repository, so files written into the mount stay writable on both sides
ARG UID=1000
RUN useradd --uid ${UID} --create-home --shell /bin/sh app
ENV UV_CACHE_DIR=/home/app/.cache/uv
USER app
