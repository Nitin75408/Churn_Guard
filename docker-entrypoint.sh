#!/bin/sh
# Run the API and the dashboard in one container.
#
# Needed because some hosts expose exactly one port per container. Streamlit
# takes the public port and the dashboard talks to the API over HTTP, exactly as
# it would in a two-container deployment, so the architecture on display is the
# real one rather than a single-process shortcut.
#
# The API binds 0.0.0.0 rather than loopback. Inside a container that is not the
# exposure it looks like: an unpublished port is unreachable from outside no
# matter what it binds, and which ports get published is the actual control.
# Binding loopback only meant the port could never be published at all, which
# broke the CI smoke test that reaches the API from the host.
#
# docker-compose overrides this to run each service separately, which is the
# arrangement a production deployment would use.
set -e

API_PORT="${API_PORT:-8000}"
API_HOST="${API_HOST:-0.0.0.0}"
UI_PORT="${PORT:-7860}"

echo "[entrypoint] starting API on ${API_HOST}:${API_PORT}"
uvicorn churn_guard.api.main:app --host "${API_HOST}" --port "${API_PORT}" &
API_PID=$!

# Wait for the API to answer before starting the dashboard. Without this the
# dashboard caches a failed connection at startup and silently falls back to
# in-process scoring — it would still work, but it would advertise the wrong
# architecture and show a warning banner.
echo "[entrypoint] waiting for the API to become ready..."
i=0
while [ "$i" -lt 60 ]; do
    if python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:${API_PORT}/health', timeout=2)" 2>/dev/null; then
        echo "[entrypoint] API ready after ${i}s"
        break
    fi
    # If uvicorn died, fail loudly now rather than serving a broken dashboard.
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo "[entrypoint] ERROR: the API process exited during startup" >&2
        exit 1
    fi
    i=$((i + 1))
    sleep 1
done

echo "[entrypoint] starting dashboard on 0.0.0.0:${UI_PORT}"
exec streamlit run streamlit_app.py \
    --server.address=0.0.0.0 \
    --server.port="${UI_PORT}" \
    --server.headless=true \
    --browser.gatherUsageStats=false
