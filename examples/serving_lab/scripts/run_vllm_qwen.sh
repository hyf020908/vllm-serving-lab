#!/usr/bin/env bash
# Start a Qwen-family model with vLLM OpenAI-compatible serving.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/path/to/qwen-model}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TP_SIZE="${TP_SIZE:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"

TRUST_ARG=()
if [[ "${TRUST_REMOTE_CODE}" == "true" ]]; then
  TRUST_ARG=(--trust-remote-code)
fi

exec vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype "${DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --tensor-parallel-size "${TP_SIZE}" \
  "${TRUST_ARG[@]}"
