#!/usr/bin/env bash
# Send one /v1/chat/completions request to a vLLM OpenAI-compatible server.
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen}"
PROMPT="${PROMPT:-请用一句话介绍 vLLM。}"
MAX_TOKENS="${MAX_TOKENS:-128}"

curl -sS "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @- <<JSON
{
  "model": "${SERVED_MODEL_NAME}",
  "messages": [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "${PROMPT}"}
  ],
  "max_tokens": ${MAX_TOKENS},
  "temperature": 0.2
}
JSON
echo
