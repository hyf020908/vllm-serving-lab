# Adaptive Startup and Runtime Advice

This serving lab uses two separate adaptive layers:

- Startup recovery: before the service is ready, the launcher may retry with a
  safer vLLM command because no user traffic is being served yet.
- Runtime advice: after the service is ready and responding, the advisor only
  prints warnings and parameter suggestions. It never kills or restarts the
  running server.

This split keeps startup automation useful without surprising users during live
traffic.

## Layer 1: Adaptive Startup Recovery

`adaptive_launcher.py` reads the model `config.json`, starts `vllm serve`, checks
`GET /v1/models`, classifies startup failures from logs, adjusts launch
parameters, and retries up to `--max-retries`.

```bash
.venv/bin/python examples/serving_lab/scripts/adaptive_launcher.py \
  --model-path /workspace/models \
  --served-model-name qwen \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --max-retries 4 \
  --ready-timeout 90 \
  --output-dir examples/serving_lab/runs/adaptive_launcher
```

Dry-run mode checks config parsing and writes the command/report without
starting vLLM:

```bash
.venv/bin/python examples/serving_lab/scripts/adaptive_launcher.py \
  --model-path /workspace/models \
  --served-model-name qwen \
  --dry-run
```

### Config Fields

The launcher extracts:

- `model_type`
- `architectures`
- `num_hidden_layers`
- `hidden_size`
- `num_attention_heads`
- `num_key_value_heads`
- `max_position_embeddings`
- MoE fields: `num_experts`, `num_local_experts`,
  `moe_intermediate_size`, `n_routed_experts`, `num_experts_per_tok`

### Startup Adjustment Rules

| Failure | Automatic action |
| --- | --- |
| Port occupied | Find the next available port |
| `max_model_len` above config context | Fall back to config context length |
| CUDA OOM / KV Cache insufficient | Reduce `max_model_len`, `max_num_seqs`, and `max_num_batched_tokens` |
| dtype incompatible | Fall back `bfloat16 -> float16 -> auto` |
| missing `trust_remote_code` | Add `--trust-remote-code` and record it |
| readiness timeout | Reduce `max_num_seqs` and `max_num_batched_tokens` |
| MoE / architecture / Transformers / backend conflict | Stop retrying and print manual remediation |
| unknown startup failure | Stop and ask the user to inspect logs |

The launcher does not blindly retry architecture, MoE, Transformers, or
FlashAttention/FlashInfer backend errors because those usually require a
specific version, wheel, model revision, or manual backend choice.

### Output Files

Each run writes:

- `final_config.json`: final parameters, model summary, and attempt records.
- `final_command.sh`: copyable command for reproducing the final launch.
- `launch_report.md`: human-readable attempt report with log excerpts.
- `round_<n>.log`: stdout/stderr from each `vllm serve` attempt.

When startup succeeds, the launcher exits and prints the service PID, port,
final command, and report paths.

## Layer 2: Runtime Performance Advice

`performance_advisor.py` analyzes request timing and optional Prometheus metrics
captured from an already running service. It prints advice only.

```bash
.venv/bin/python examples/serving_lab/scripts/performance_advisor.py \
  --metrics-file /tmp/metrics.txt \
  --latency-ms 3200 \
  --ttft-ms 1200 \
  --output-tokens 128 \
  --prompt-tokens 2048 \
  --concurrency 8 \
  --max-tokens 128
```

Supported inputs:

- `--metrics-file`
- `--metrics-text`
- `--latency-ms`
- `--ttft-ms`
- `--output-tokens`
- `--prompt-tokens`
- `--concurrency`
- `--max-tokens`
- `--target latency|throughput|balanced|long-context`

### Runtime Advice Rules

| Signal | Advice direction |
| --- | --- |
| High TTFT | Check prompt length, lower `max_num_batched_tokens`, enable prefix caching, reduce concurrency or split long prompts |
| High TPOT / ITL | Lower concurrency, lower `max_num_seqs`, inspect decode pressure |
| High P95 / P99 | Lower concurrency, limit `max_tokens`, lower `max_num_batched_tokens` |
| High KV Cache usage | Lower `max_model_len`, lower `max_num_seqs`, limit output length |
| Many waiting requests | Increase batch capacity if memory allows, raise `max_num_batched_tokens`, scale out or raise TP |
| Low throughput with headroom | Sweep higher `max_num_seqs` and `max_num_batched_tokens` |

The advisor always includes a no-restart note. Runtime warnings are safe to use
inside curl wrappers and scripts because they do not mutate the live service.

## Curl With Advice

`curl_chat_with_advice.sh` preserves the raw OpenAI-compatible JSON response,
prints curl timing, fetches `/metrics`, and appends advisor output outside the
model response.

```bash
BASE_URL=http://127.0.0.1:8000 \
MODEL=qwen \
PROMPT="请简单介绍一下 vLLM。" \
MAX_TOKENS=128 \
CONCURRENCY=1 \
bash examples/serving_lab/scripts/curl_chat_with_advice.sh
```

The terminal output has three sections:

- `Model Response`: unmodified JSON returned by `/v1/chat/completions`.
- `Request Timing`: HTTP status, TTFT approximation, and total latency.
- `Serving Lab Advice`: status, reasons, suggestions, benchmark command, and
  next parameter direction.

## Benchmark and Resume Usage

For benchmark work:

1. Run `adaptive_launcher.py --dry-run` and save the initial command.
2. Run `adaptive_launcher.py` and save `launch_report.md`.
3. Use `curl_chat_with_advice.sh` for a smoke request and runtime advice.
4. Run `benchmark_openai_api.py` and `parameter_sweep.py` for controlled
   comparisons.
5. Copy the final command, metrics, advice, and conclusions into
   `docs/benchmark_report_template.md`.

For resume positioning, this demonstrates application-layer LLM serving work:
startup diagnostics, safe retry policy, OpenAI-compatible API usage,
Prometheus-based observability, latency analysis, and benchmark-driven parameter
tuning without modifying vLLM core inference algorithms.
