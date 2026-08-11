# syntax=docker/dockerfile:1.9
#
# ChurnGuard — serving image for both the API and the dashboard.
#
# Build:  docker build -t churn-guard:latest .
# Run:    docker run -p 8000:8000 churn-guard:latest
#
# Two ideas drive the structure.
#
# **Multi-stage.** The builder stage installs dependencies and carries the tools
# needed to do so; the runtime stage copies only the finished virtual environment
# and the application. Build tooling never reaches the shipped image, which keeps
# it smaller and removes compilers an attacker could otherwise use.
#
# **Layer ordering for cache reuse.** Docker caches each instruction and reuses it
# until an input changes, then rebuilds everything after. Dependencies are
# installed *before* the source is copied, so editing a Python file rebuilds only
# the last few layers — seconds instead of minutes. Copying source first would
# reinstall every package on every code change.

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.12.3

# =============================================================================
# Stage 1 — builder: resolve and install dependencies
# =============================================================================
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ARG UV_VERSION
# Copy the uv binary from its official image rather than pip-installing it: no
# network dependency at build time beyond the image pull, and the version is
# pinned, so builds stay reproducible.
COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install dependencies alone first, with only the lockfile and manifest mounted.
# --frozen fails the build if uv.lock disagrees with pyproject.toml, so an image
# can never be built from dependencies nobody committed.
# --no-dev drops jupyterlab, mlflow and seaborn: development and training tools
# that a serving container has no use for.
# --no-install-project defers installing our own code, which is what keeps this
# expensive layer cached across source edits.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Now the project itself. Only this layer rebuilds when source changes.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# =============================================================================
# Stage 2 — runtime: the image that actually ships
# =============================================================================
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# A non-root user. If the process is ever compromised, the attacker inherits its
# privileges — root inside a container is a meaningful step toward root on the
# host. This costs nothing and removes an entire class of escalation.
RUN groupadd --gid 1000 app \
 && useradd --uid 1000 --gid app --create-home --shell /bin/bash app

WORKDIR /app

# The finished virtual environment from the builder. uv's tooling, caches and any
# build dependencies stay behind.
COPY --from=builder --chown=app:app /app/.venv /app/.venv

COPY --chown=app:app src ./src
COPY --chown=app:app configs ./configs
COPY --chown=app:app models ./models
COPY --chown=app:app streamlit_app.py pyproject.toml README.md ./
# The operating threshold lives here; the dashboard also reads the sensitivity
# table from it.
COPY --chown=app:app reports/final_evaluation.json ./reports/
# Demo customers for the dashboard's worklist tab.
COPY --chown=app:app data/interim ./data/interim

# Directories the config layer creates on load, made writable for the app user
# up front so startup never fails on a permission error.
RUN mkdir -p /app/logs /app/reports/figures /app/data/raw /app/data/processed \
 && chown -R app:app /app/logs /app/reports /app/data

ENV PATH="/app/.venv/bin:$PATH" \
    # Unbuffered so logs appear immediately in `docker logs` instead of being
    # held in a buffer until the process exits.
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER app

EXPOSE 8000 8501

# Docker restarts or reroutes traffic based on this. start-period covers model
# loading and SHAP explainer construction, which take a few seconds; without it
# the container would be marked unhealthy before it finished starting.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

# Default command runs the API. docker-compose.yml overrides it for the dashboard,
# so both services share one image and one build.
CMD ["uvicorn", "churn_guard.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
