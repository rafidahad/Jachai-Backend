#!/bin/sh
set -eu

wait_for_services() {
  python3 - <<'PY'
import os
import socket
import time
from urllib.parse import urlparse


def wait_for_url(name: str, value: str, default_port: int, timeout: float) -> None:
    parsed = urlparse(value)
    host = parsed.hostname
    port = parsed.port or default_port

    if not host:
        raise SystemExit(f"{name} is missing a host in its URL.")

    deadline = time.time() + timeout
    last_error = None

    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=3):
                print(f"{name} is reachable at {host}:{port}")
                return
        except OSError as exc:
            last_error = exc
            time.sleep(2)

    raise SystemExit(f"Timed out waiting for {name} at {host}:{port}: {last_error}")


timeout = float(os.getenv("WAIT_TIMEOUT_SECONDS", "75"))

database_url = os.getenv("DATABASE_URL", "").strip()
redis_url = os.getenv("REDIS_URL", "").strip()

if database_url:
    wait_for_url("DATABASE_URL", database_url, 5432, timeout)

if redis_url:
    wait_for_url("REDIS_URL", redis_url, 6379, timeout)
PY
}

case "${WAIT_FOR_SERVICES:-true}" in
  1|true|TRUE|True|yes|YES|on|ON)
    wait_for_services
    ;;
esac

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

exec uvicorn app.main:app \
  --host "${UVICORN_HOST:-0.0.0.0}" \
  --port "${UVICORN_PORT:-8000}" \
  --workers "${UVICORN_WORKERS:-1}" \
  --log-level "${UVICORN_LOG_LEVEL:-info}" \
  --proxy-headers \
  --forwarded-allow-ips="*"
