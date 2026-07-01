#!/usr/bin/env python
"""Fetch and summarize a vLLM /metrics Prometheus snapshot."""

from __future__ import annotations

import argparse
import re
import urllib.error
import urllib.request
from dataclasses import dataclass


METRIC_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)


@dataclass(frozen=True)
class MetricSample:
    name: str
    labels: dict[str, str]
    value: float


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


def first_value(metrics: dict[str, list[MetricSample]], names: list[str]) -> str:
    for name in names:
        samples = metrics.get(name)
        if samples:
            return f"{samples[0].value:g}"
    return "N/A"


def summarize_metrics(metrics: dict[str, list[MetricSample]]) -> list[tuple[str, str]]:
    return [
        ("running requests", first_value(metrics, ["vllm:num_requests_running"])),
        ("waiting requests", first_value(metrics, ["vllm:num_requests_waiting"])),
        ("prompt tokens", first_value(metrics, ["vllm:prompt_tokens_total"])),
        ("generation tokens", first_value(metrics, ["vllm:generation_tokens_total"])),
        ("KV cache usage", first_value(metrics, ["vllm:gpu_cache_usage_perc"])),
        ("CPU cache usage", first_value(metrics, ["vllm:cpu_cache_usage_perc"])),
        ("prefix cache hits", first_value(metrics, ["vllm:prefix_cache_hits_total"])),
        (
            "prefix cache queries",
            first_value(metrics, ["vllm:prefix_cache_queries_total"]),
        ),
        (
            "request latency count",
            first_value(metrics, ["vllm:e2e_request_latency_seconds_count"]),
        ),
        (
            "request latency sum",
            first_value(metrics, ["vllm:e2e_request_latency_seconds_sum"]),
        ),
        (
            "time to first token count",
            first_value(metrics, ["vllm:time_to_first_token_seconds_count"]),
        ),
        (
            "time to first token sum",
            first_value(metrics, ["vllm:time_to_first_token_seconds_sum"]),
        ),
        (
            "time per output token count",
            first_value(metrics, ["vllm:request_time_per_output_token_seconds_count"]),
        ),
        (
            "time per output token sum",
            first_value(metrics, ["vllm:request_time_per_output_token_seconds_sum"]),
        ),
    ]


def fetch_metrics(base_url: str, timeout: float) -> str:
    url = f"{base_url.rstrip('/')}/metrics"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    try:
        text = fetch_metrics(args.base_url, args.timeout)
    except urllib.error.URLError as exc:
        print(f"Failed to fetch /metrics: {exc}")
        return 1

    metrics = parse_prometheus_text(text)
    print("vLLM metrics snapshot")
    print("=" * 24)
    for label, value in summarize_metrics(metrics):
        print(f"{label:<32} {value}")
    print()
    print(f"parsed metric families: {len(metrics)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
