# vLLM Serving Lab

`examples/serving_lab` is a lightweight engineering lab for vLLM deployment,
OpenAI-compatible API checks, streaming checks, metrics snapshots, benchmark
runs, parameter sweeps, and troubleshooting.

This fork does not change vLLM core inference algorithms. It focuses on LLM
Serving engineering: deployment, tests, benchmarking, observability, and
operational debugging.

## What Is Added

- Static model and launch-parameter preflight checks.
- Qwen, DeepSeek, and GLM launch examples.
- `/v1/models` and `/v1/chat/completions` compatibility checks.
- Streaming chat health checks.
- `/metrics` snapshot parsing.
- Lightweight benchmark and parameter sweep command generation.
- Adaptive startup recovery and runtime performance advice.
- Practical docs for parameters, KV Cache, API compatibility, and failures.

## Layout

```text
examples/serving_lab/
├── configs/
├── docs/
├── scripts/
└── tests/
```

## Quick Start

Run static checks before starting a service:

```bash
.venv/bin/python examples/serving_lab/scripts/preflight_check.py \
  --model-path /path/to/model \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --dtype bfloat16 \
  --trust-remote-code
```

Start a Qwen service:

```bash
MODEL_PATH=/path/to/Qwen-7B-Instruct \
SERVED_MODEL_NAME=qwen \
PORT=8000 \
examples/serving_lab/scripts/run_vllm_qwen.sh
```

Check models:

```bash
HOST=127.0.0.1 PORT=8000 examples/serving_lab/scripts/curl_models.sh
```

Check chat completions:

```bash
SERVED_MODEL_NAME=qwen examples/serving_lab/scripts/curl_chat.sh
```

Run a full compatibility check:

```bash
.venv/bin/python examples/serving_lab/scripts/api_compat_check.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen \
  --max-tokens 64
```

Check streaming output:

```bash
.venv/bin/python examples/serving_lab/scripts/stream_chat_test.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen \
  --prompt "请简单介绍一下 vLLM。"
```

Collect a metrics snapshot:

```bash
.venv/bin/python examples/serving_lab/scripts/metrics_snapshot.py \
  --base-url http://127.0.0.1:8000
```

Run a lightweight benchmark:

```bash
.venv/bin/python examples/serving_lab/scripts/benchmark_openai_api.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen \
  --concurrency 8 \
  --num-requests 50 \
  --max-tokens 128 \
  --stream
```

Generate sweep commands:

```bash
.venv/bin/python examples/serving_lab/scripts/parameter_sweep.py \
  --model-path /path/to/model \
  --served-model-name qwen \
  --output sweep.md
```

## Adaptive Startup and Runtime Advice

Generate an adaptive launch report without starting vLLM:

```bash
.venv/bin/python examples/serving_lab/scripts/adaptive_launcher.py \
  --model-path /path/to/model \
  --served-model-name qwen \
  --dry-run
```

Start with adaptive startup recovery:

```bash
.venv/bin/python examples/serving_lab/scripts/adaptive_launcher.py \
  --model-path /path/to/model \
  --served-model-name qwen \
  --port 8000 \
  --max-retries 4
```

Send a chat request and print runtime advice without restarting the service:

```bash
BASE_URL=http://127.0.0.1:8000 MODEL=qwen \
bash examples/serving_lab/scripts/curl_chat_with_advice.sh
```

## Reproduce Results

1. Record hardware, model path, vLLM commit, and launch command.
2. Run preflight checks and save the report.
3. Run API compatibility checks.
4. Run metrics snapshot before and after benchmark.
5. Copy benchmark output into
   [docs/benchmark_report_template.md](docs/benchmark_report_template.md).

## Resume Positioning

This lab is suitable for showing LLM Serving, LLM Infra application-layer work,
model-service deployment, OpenAI-compatible API testing, observability,
benchmarking, and production troubleshooting skills.

More docs:

- [vLLM parameters](docs/vllm_params.md)
- [KV Cache memory](docs/kv_cache_memory.md)
- [API compatibility](docs/api_compatibility.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Adaptive startup and runtime advice](docs/adaptive_startup_and_advice.md)
- [Benchmark report template](docs/benchmark_report_template.md)
