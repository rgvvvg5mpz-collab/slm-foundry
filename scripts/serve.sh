#!/usr/bin/env bash
# Start the API and one worker. Ctrl-C stops both.
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="${PYTHONPATH:-}:$PWD/src"
export FOUNDRY_DATABASE_URL="${FOUNDRY_DATABASE_URL:-sqlite:///$PWD/var/foundry.db}"
PY="${PYTHON:-python3}"

mkdir -p var
"$PY" -m foundry.worker --kinds train,eval,generate,judge,assemble &
WORKER=$!
trap 'kill $WORKER 2>/dev/null || true' EXIT

exec "$PY" -m uvicorn foundry.api:app --host 127.0.0.1 --port 8200 "$@"
