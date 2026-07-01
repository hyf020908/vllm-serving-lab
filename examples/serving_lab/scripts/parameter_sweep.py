#!/usr/bin/env python
"""Generate vLLM launch and benchmark commands for parameter sweeps."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path


def split_values(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_serve_command(args: argparse.Namespace, combo: dict[str, str]) -> str:
    parts = [
        "vllm serve",
        args.model_path,
        f"--served-model-name {args.served_model_name}",
        f"--host {args.host}",
        f"--port {args.port}",
        f"--max-model-len {combo['max_model_len']}",
        f"--max-num-seqs {combo['max_num_seqs']}",
        f"--max-num-batched-tokens {combo['max_num_batched_tokens']}",
        f"--gpu-memory-utilization {combo['gpu_memory_utilization']}",
        f"--dtype {combo['dtype']}",
        f"--tensor-parallel-size {combo['tensor_parallel_size']}",
    ]
    if args.trust_remote_code:
        parts.append("--trust-remote-code")
    return " ".join(parts)


def build_benchmark_command(args: argparse.Namespace, combo_id: int) -> str:
    return (
        "python examples/serving_lab/scripts/benchmark_openai_api.py "
        f"--base-url http://{args.host}:{args.port} "
        f"--model {args.served_model_name} "
        f"--concurrency {args.benchmark_concurrency} "
        f"--num-requests {args.benchmark_requests} "
        f"--max-tokens {args.benchmark_max_tokens} "
        f"> benchmark_combo_{combo_id}.md"
    )


def generate(args: argparse.Namespace) -> str:
    keys = [
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "gpu_memory_utilization",
        "dtype",
        "tensor_parallel_size",
    ]
    value_lists = [
        split_values(args.max_model_len),
        split_values(args.max_num_seqs),
        split_values(args.max_num_batched_tokens),
        split_values(args.gpu_memory_utilization),
        split_values(args.dtype),
        split_values(args.tensor_parallel_size),
    ]

    lines = ["# vLLM Serving Lab Parameter Sweep", ""]
    lines.append(
        "| id | max_model_len | max_num_seqs | batched_tokens | gpu_util | "
        "dtype | tp | status | notes |"
    )
    lines.append("| ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |")
    combos = []
    for index, values in enumerate(itertools.product(*value_lists), start=1):
        combo = dict(zip(keys, values))
        combos.append((index, combo))
        lines.append(
            "| {id} | {max_model_len} | {max_num_seqs} | "
            "{max_num_batched_tokens} | {gpu_memory_utilization} | "
            "{dtype} | {tensor_parallel_size} | TODO | TODO |".format(
                id=index, **combo
            )
        )

    for index, combo in combos:
        lines.extend(
            [
                "",
                f"## Combo {index}",
                "",
                "Serve:",
                "```bash",
                build_serve_command(args, combo),
                "```",
                "",
                "Benchmark:",
                "```bash",
                build_benchmark_command(args, index),
                "```",
            ]
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/path/to/model")
    parser.add_argument("--served-model-name", default="qwen")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", default="4096,8192")
    parser.add_argument("--max-num-seqs", default="16,32")
    parser.add_argument("--max-num-batched-tokens", default="4096,8192")
    parser.add_argument("--gpu-memory-utilization", default="0.85,0.9")
    parser.add_argument("--dtype", default="bfloat16,float16")
    parser.add_argument("--tensor-parallel-size", default="1")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--benchmark-concurrency", type=int, default=8)
    parser.add_argument("--benchmark-requests", type=int, default=50)
    parser.add_argument("--benchmark-max-tokens", type=int, default=128)
    parser.add_argument("--output")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = generate(args)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
