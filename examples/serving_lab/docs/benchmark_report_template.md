# Benchmark Report Template

## Environment

- Hardware:
- GPU:
- Driver / CUDA:
- vLLM version or commit:
- Python environment:

## Model

- Model name:
- Model path:
- dtype:
- tensor parallel size:
- trust remote code:

## Launch Parameters

```bash
vllm serve ...
```

## Adaptive Startup Report

- Dry-run command:
- Final command file:
- Launch report:
- Startup retries:
- Final port:
- Automatic adjustments:

## Workload

- Concurrency:
- Number of requests:
- Input length:
- Output length:
- Stream: yes/no

## Results

| metric | value |
| --- | ---: |
| TTFT | |
| TPOT / ITL | |
| tokens/s | |
| requests/s | |
| P50 latency | |
| P95 latency | |
| P99 latency | |
| KV Cache usage | |
| error rate | |

## Analysis

- Main bottleneck:
- Error summary:
- Latency conclusion:
- Throughput conclusion:

## Parameter Recommendations

- `max_model_len`:
- `max_num_seqs`:
- `max_num_batched_tokens`:
- `gpu_memory_utilization`:
- `tensor_parallel_size`:
- Runtime advisor status:
- Runtime advisor reasons:
- Runtime advisor next parameter direction:
