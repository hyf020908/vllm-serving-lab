#!/usr/bin/env bash
# Query /v1/models from a local or remote vLLM OpenAI-compatible server.
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}}"

curl -sS "${BASE_URL}/v1/models"
echo
