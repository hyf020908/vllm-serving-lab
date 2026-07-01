#!/usr/bin/env bash
# Send one chat request and print runtime serving advice beside the raw response.
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}}"
MODEL="${MODEL:-${SERVED_MODEL_NAME:-qwen}}"
PROMPT="${PROMPT:-请用一句话介绍 vLLM。}"
MAX_TOKENS="${MAX_TOKENS:-128}"
CONCURRENCY="${CONCURRENCY:-1}"

RESPONSE_FILE="$(mktemp "${TMPDIR:-/tmp}/serving_lab_response.XXXXXX.tmp")"
TIMING_FILE="$(mktemp "${TMPDIR:-/tmp}/serving_lab_timing.XXXXXX.tmp")"
METRICS_FILE="$(mktemp "${TMPDIR:-/tmp}/serving_lab_metrics.XXXXXX.tmp")"

cleanup() {
  rm -f "${RESPONSE_FILE}" "${TIMING_FILE}" "${METRICS_FILE}"
}
trap cleanup EXIT

HTTP_STATUS="$(
  curl -sS -o "${RESPONSE_FILE}" -w "%{http_code} %{time_starttransfer} %{time_total}" \
    "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d @- <<JSON
{
  "model": "${MODEL}",
  "messages": [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "${PROMPT}"}
  ],
  "max_tokens": ${MAX_TOKENS},
  "temperature": 0.2
}
JSON
)"

printf "%s\n" "${HTTP_STATUS}" > "${TIMING_FILE}"
read -r STATUS_CODE TIME_STARTTRANSFER TIME_TOTAL < "${TIMING_FILE}"

curl -sS "${BASE_URL}/metrics" -o "${METRICS_FILE}" || true

TTFT_MS="$(awk -v value="${TIME_STARTTRANSFER}" 'BEGIN { printf "%.0f", value * 1000 }')"
LATENCY_MS="$(awk -v value="${TIME_TOTAL}" 'BEGIN { printf "%.0f", value * 1000 }')"

echo "=== Model Response ==="
cat "${RESPONSE_FILE}"
echo
echo
echo "=== Request Timing ==="
echo "HTTP Status: ${STATUS_CODE}"
echo "TTFT(ms): ${TTFT_MS}"
echo "Total Latency(ms): ${LATENCY_MS}"
echo
echo "=== Serving Lab Advice ==="
.venv/bin/python examples/serving_lab/scripts/performance_advisor.py \
  --metrics-file "${METRICS_FILE}" \
  --latency-ms "${LATENCY_MS}" \
  --ttft-ms "${TTFT_MS}" \
  --prompt-tokens "$(printf "%s" "${PROMPT}" | wc -m | tr -d ' ')" \
  --concurrency "${CONCURRENCY}" \
  --max-tokens "${MAX_TOKENS}"
