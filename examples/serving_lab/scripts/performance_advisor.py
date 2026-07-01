#!/usr/bin/env python
"""Runtime performance advisor for a ready vLLM service.

The advisor analyzes request timing and optional Prometheus metrics. It prints
warnings and parameter suggestions only; it never restarts or kills a service.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path


METRIC_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)


@dataclass(frozen=True)
class MetricSample:
    name: str
    labels: dict[str, str]
    value: float


@dataclass
class AdviceReport:
    status: str
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float | str] = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)
    parameter_direction: list[str] = field(default_factory=list)
    benchmark_command: str = ""


def parse_labels(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    labels: dict[str, str] = {}
    for part in raw.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        labels[key.strip()] = value.strip().strip('"')
    return labels


def parse_prometheus_text(text: str) -> dict[str, list[MetricSample]]:
    metrics: dict[str, list[MetricSample]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = METRIC_RE.match(line)
        if not match:
            continue
        sample = MetricSample(
            name=match.group("name"),
            labels=parse_labels(match.group("labels")),
            value=float(match.group("value")),
        )
        metrics.setdefault(sample.name, []).append(sample)
    return metrics


def first_metric_value(
    metrics: dict[str, list[MetricSample]], names: list[str]
) -> float | None:
    for name in names:
        samples = metrics.get(name)
        if samples:
            return samples[0].value
    return None


def histogram_quantile(samples: list[MetricSample], quantile: float) -> float | None:
    buckets: list[tuple[float, float]] = []
    for sample in samples:
        le = sample.labels.get("le")
        if le is None or le == "+Inf":
            continue
        try:
            buckets.append((float(le), sample.value))
        except ValueError:
            continue
    if not buckets:
        return None
    buckets.sort(key=lambda item: item[0])
    total = buckets[-1][1]
    if total <= 0:
        return None
    threshold = total * quantile
    previous_le = 0.0
    previous_count = 0.0
    for upper, count in buckets:
        if count >= threshold:
            bucket_count = count - previous_count
            if bucket_count <= 0:
                return upper
            fraction = (threshold - previous_count) / bucket_count
            return previous_le + (upper - previous_le) * fraction
        previous_le = upper
        previous_count = count
    return buckets[-1][0]


def derive_metric_summary(
    metrics: dict[str, list[MetricSample]], args: argparse.Namespace
) -> dict[str, float | str]:
    summary: dict[str, float | str] = {}
    if args.latency_ms is not None:
        summary["latency_ms"] = args.latency_ms
    if args.ttft_ms is not None:
        summary["ttft_ms"] = args.ttft_ms
    if args.output_tokens is not None:
        summary["output_tokens"] = args.output_tokens
    if args.prompt_tokens is not None:
        summary["prompt_tokens"] = args.prompt_tokens
    if args.concurrency is not None:
        summary["concurrency"] = args.concurrency
    if args.max_tokens is not None:
        summary["max_tokens"] = args.max_tokens

    waiting = first_metric_value(metrics, ["vllm:num_requests_waiting"])
    running = first_metric_value(metrics, ["vllm:num_requests_running"])
    kv_usage = first_metric_value(metrics, ["vllm:gpu_cache_usage_perc"])
    if waiting is not None:
        summary["waiting_requests"] = waiting
    if running is not None:
        summary["running_requests"] = running
    if kv_usage is not None:
        summary["kv_cache_usage"] = kv_usage / 100.0 if kv_usage > 1 else kv_usage

    ttft_sum = first_metric_value(metrics, ["vllm:time_to_first_token_seconds_sum"])
    ttft_count = first_metric_value(metrics, ["vllm:time_to_first_token_seconds_count"])
    if "ttft_ms" not in summary and ttft_sum is not None and ttft_count:
        summary["ttft_ms"] = 1000.0 * ttft_sum / ttft_count

    tpot_sum = first_metric_value(
        metrics, ["vllm:request_time_per_output_token_seconds_sum"]
    )
    tpot_count = first_metric_value(
        metrics, ["vllm:request_time_per_output_token_seconds_count"]
    )
    if tpot_sum is not None and tpot_count:
        summary["tpot_ms"] = 1000.0 * tpot_sum / tpot_count

    latency_buckets = metrics.get("vllm:e2e_request_latency_seconds_bucket", [])
    p95 = histogram_quantile(latency_buckets, 0.95)
    p99 = histogram_quantile(latency_buckets, 0.99)
    if p95 is not None:
        summary["p95_latency_ms"] = p95 * 1000.0
    if p99 is not None:
        summary["p99_latency_ms"] = p99 * 1000.0

    if "tpot_ms" not in summary:
        latency = summary.get("latency_ms")
        ttft = summary.get("ttft_ms")
        output_tokens = summary.get("output_tokens")
        if (
            isinstance(latency, (int, float))
            and isinstance(ttft, (int, float))
            and isinstance(output_tokens, (int, float))
            and output_tokens > 0
            and latency >= ttft
        ):
            summary["tpot_ms"] = (latency - ttft) / output_tokens
    return summary


def add_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def analyze(summary: dict[str, float | str], target: str = "balanced") -> AdviceReport:
    report = AdviceReport(status="OK", metrics=summary.copy())

    def warn(reason: str, critical: bool = False) -> None:
        report.status = "CRITICAL" if critical else (
            "WARN" if report.status == "OK" else report.status
        )
        add_unique(report.reasons, reason)

    ttft = summary.get("ttft_ms")
    if isinstance(ttft, (int, float)) and ttft > 800:
        warn(f"TTFT is high ({ttft:.0f} ms)", critical=ttft > 2000)
        add_unique(report.suggestions, "check whether the prompt is too long")
        add_unique(report.suggestions, "consider lowering --max-num-batched-tokens")
        add_unique(report.suggestions, "consider enabling prefix caching")
        add_unique(report.suggestions, "lower concurrency or split long prompts")
        add_unique(report.parameter_direction, "reduce prefill pressure")

    tpot = summary.get("tpot_ms")
    if isinstance(tpot, (int, float)) and tpot > 80:
        warn(f"TPOT / ITL is high ({tpot:.1f} ms/token)", critical=tpot > 200)
        add_unique(report.suggestions, "lower concurrency")
        add_unique(report.suggestions, "lower --max-num-seqs")
        add_unique(report.suggestions, "check decode-stage GPU pressure")
        add_unique(report.parameter_direction, "reduce decode concurrency")

    p95 = summary.get("p95_latency_ms")
    p99 = summary.get("p99_latency_ms")
    if isinstance(p95, (int, float)) and p95 > 5000:
        warn(f"P95 latency is high ({p95:.0f} ms)", critical=p95 > 15000)
        add_unique(report.suggestions, "lower concurrency")
        add_unique(report.suggestions, "limit max_tokens for long-running requests")
        add_unique(report.suggestions, "lower --max-num-batched-tokens")
    if isinstance(p99, (int, float)) and p99 > 8000:
        warn(f"P99 latency is high ({p99:.0f} ms)", critical=p99 > 20000)
        add_unique(report.suggestions, "cap request length and isolate long requests")

    kv_usage = summary.get("kv_cache_usage")
    if isinstance(kv_usage, (int, float)) and kv_usage > 0.85:
        warn(f"KV Cache usage is high ({kv_usage:.0%})", critical=kv_usage > 0.95)
        add_unique(report.suggestions, "lower --max-model-len")
        add_unique(report.suggestions, "lower --max-num-seqs")
        add_unique(report.suggestions, "limit output length")
        add_unique(report.parameter_direction, "reduce KV Cache footprint")

    waiting = summary.get("waiting_requests")
    if isinstance(waiting, (int, float)) and waiting > 0:
        warn(f"waiting requests are queued ({waiting:g})", critical=waiting >= 8)
        add_unique(
            report.suggestions, "increase throughput-related parameters carefully"
        )
        add_unique(report.suggestions, "try higher --max-num-batched-tokens")
        add_unique(report.suggestions, "scale out or increase --tensor-parallel-size")
        add_unique(
            report.parameter_direction, "increase batch capacity if memory allows"
        )

    prompt_tokens = summary.get("prompt_tokens")
    if isinstance(prompt_tokens, (int, float)) and prompt_tokens > 4096:
        warn(f"prompt is long ({prompt_tokens:g} tokens)")
        add_unique(report.suggestions, "split or summarize long prompts")
    max_tokens = summary.get("max_tokens")
    if isinstance(max_tokens, (int, float)) and max_tokens > 512:
        warn(f"max_tokens is large ({max_tokens:g})")
        add_unique(report.suggestions, "lower max_tokens for latency-sensitive traffic")
    concurrency = summary.get("concurrency")
    if isinstance(concurrency, (int, float)) and concurrency > 16:
        warn(f"concurrency is high ({concurrency:g})")
        add_unique(report.suggestions, "benchmark a lower concurrency level")

    if report.status == "OK":
        add_unique(
            report.reasons, "metrics are within the default serving-lab thresholds"
        )
        add_unique(report.suggestions, "no runtime restart is recommended")
        add_unique(
            report.parameter_direction,
            "raise --max-num-seqs gradually if throughput is low",
        )
        add_unique(
            report.parameter_direction,
            "raise --max-num-batched-tokens gradually if GPU memory has headroom",
        )
    elif target in {"throughput", "balanced"} and not any(
        "KV Cache usage" in reason or "waiting requests" in reason
        for reason in report.reasons
    ):
        add_unique(
            report.parameter_direction,
            "if throughput is low and resources are idle, compare higher "
            "--max-num-seqs and --max-num-batched-tokens with parameter_sweep.py",
        )

    report.benchmark_command = (
        ".venv/bin/python examples/serving_lab/scripts/benchmark_openai_api.py "
        "--base-url http://127.0.0.1:8000 --model <model> "
        "--concurrency <n> --num-requests 50 --max-tokens <tokens> --stream"
    )
    return report


def render_report(report: AdviceReport) -> str:
    lines = [
        f"Status: {report.status}",
        "Reason:",
    ]
    lines.extend(f"- {reason}" for reason in report.reasons)
    lines.append("")
    lines.append("Metric summary:")
    for key in sorted(report.metrics):
        value = report.metrics[key]
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.3f}")
        else:
            lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("Suggested parameters:")
    lines.extend(f"- {item}" for item in report.suggestions)
    lines.append("")
    lines.append("No automatic restart:")
    lines.append("- The service is already responding; this tool only prints advice.")
    lines.append("")
    lines.append("Next benchmark command:")
    lines.append(report.benchmark_command)
    lines.append("")
    lines.append("Next parameter direction:")
    lines.extend(f"- {item}" for item in report.parameter_direction)
    return "\n".join(lines)


def read_metrics(args: argparse.Namespace) -> str:
    pieces: list[str] = []
    if args.metrics_file:
        pieces.append(Path(args.metrics_file).read_text(encoding="utf-8"))
    if args.metrics_text:
        pieces.append(args.metrics_text)
    return "\n".join(pieces)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-file")
    parser.add_argument("--metrics-text")
    parser.add_argument("--latency-ms", type=float)
    parser.add_argument("--ttft-ms", type=float)
    parser.add_argument("--output-tokens", type=int)
    parser.add_argument("--prompt-tokens", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument(
        "--target",
        choices=("latency", "throughput", "balanced", "long-context"),
        default="balanced",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metrics = parse_prometheus_text(read_metrics(args))
    summary = derive_metric_summary(metrics, args)
    report = analyze(summary, args.target)
    print(render_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
